import re
from fuzzywuzzy import fuzz
import textwrap
from typing import Optional, Tuple, Union
from pydantic import BaseModel, Field

# from langchain_core.documents import Document
# from langchain_core.prompts.few_shot import FewShotPromptTemplate
# from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import (
    SystemMessage,
    BaseMessage,
)
from langchain_core.prompts.chat import MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser

# from fewshot_data import FewShotItem, example_tasks

PRE_THINK_INSTRUCT = """
        Before starting plan how to proceed step by step and place your thinking between
        [thinking_start] and [thinking_end]-tags. Then follow your plan and return only the expected output.
        """
# For example:

# Example 1:

# [thinking_start] I have a long article about a recent scientific discovery. I will generate a summary
# that highlights the main findings, the method used, and the implications of the discovery. [thinking_end]

# A recent study published in the Journal of Neuroscience revealed that prolonged exposure to blue light
# before sleep can disrupt circadian rhythms and lead to sleep disorders. The research, conducted on mice,
# found that blue light exposure suppressed the production of melatonin, a hormone that regulates sleep-wake cycles.
# These findings suggest that limiting blue light exposure, particularly in the evening, may help improve sleep
# quality and reduce the risk of sleep disorders.

# Example 2:

# [thinking_start] I have a lengthy book review. I will create a summary that captures the main points of the book,
# the author's style, and the reviewer's overall opinion. [thinking_end]

# In 'The Catcher in the Rye', J.D. Salinger explores the disillusionment and alienation of a teenager named Holden Caulfield.
# Through Holden's first-person narrative, the novel delves into themes of adolescence, identity, and the struggle for authenticity.
# The book's raw and unfiltered language captures the angst and confusion of youth, making it a classic of modern literature.

# Example 3:

# [thinking_start] I have several research papers on a specific topic. I will generate a comprehensive report that synthesizes
# the findings from each paper, identifies common themes, and discusses any discrepancies. [thinking_end]

# A comprehensive analysis of the effects of climate change on global food security was conducted by synthesizing data from
# multiple research papers. The synthesis revealed that climate change is likely to have a significant negative impact on food
# security, particularly in low-income countries. The findings indicated that increased temperatures, droughts, and sea-level
# rise will lead to reduced crop yields, increased food prices, and food insecurity. However, the report also highlighted potential
# adaptation strategies and the importance of international cooperation to mitigate the risks associated with climate change.
# """
KEEP_PRE_THINK_TOGETHER = """
        While following your plan don't explain what you are doing.
        """

MAINTAIN_CONTENT_AND_USER_LANGUAGE = """
        Maintain the content and the language of the input and always output in the language used in context and history.
        """

class PromptFormatter(BaseModel):
    system: str = Field(description="The system message template")
    user: str = Field(description="The user message template")
    parser: Optional[BaseOutputParser] = Field(
        description="The parser for response", default=None
    )

    def customize(
        self, system: Optional[str] = None, user: Optional[str] = None
    ) -> "PromptFormatter":
        return PromptFormatter(
            system=system if system is not None else self.system,
            user=user if user is not None else self.user,
            parser=self.parser,
        )

    def format(
        self,
        system_format: Optional[Tuple[str, str]] = None,
        user_format: Optional[Tuple[str, str]] = None,
        system: Optional[str] = None,
        user: Optional[str] = None,
        use_format: bool = False,
    ) -> "PromptFormatter":
        """
        Format the system and user messages with optional prefix and suffix.

        Args:
            system_format (Optional[Tuple[str, str]]): The prefix and suffix for the system message.
            user_format (Optional[Tuple[str, str]]): The prefix and suffix for the user message.
            system (Optional[str]): The system message. If not provided, the default system message is used.
            user (Optional[str]): The user message. If not provided, the default user message is used.
            use_format (bool): Whether to use the provided prefix and suffix.

        Returns:
            PromptFormatter: A new PromptFormatter instance with the formatted messages.
        """
        if system is None:
            system = self.system
        if user is None:
            user = self.user

        if (not use_format) or (system_format is None and user_format is None):
            return PromptFormatter(system=system, user=user)
        else:
            system_prefix, system_suffix = system_format if system_format else ("", "")
            user_prefix, user_suffix = user_format if user_format else ("", "")
            formatted_system = system_prefix + system + system_suffix
            formatted_user = user_prefix + user + user_suffix
            return PromptFormatter(system=formatted_system, user=formatted_user)

    def get_agent_prompt_template(
        self, custom_system: Optional[str] = None, custom_user: Optional[str] = None
    ) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", custom_system or self.system),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("user", custom_user or self.user),
                # MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
            ]
        )
        if self.parser is not None:
            prompt.partial_variables["format_instructions"] = (
                self.parser.get_format_instructions()
            )

        return prompt

    def get_chat_prompt_template(
        self, custom_system: Optional[str] = None, custom_user: Optional[str] = None
    ) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=custom_system or self.system),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                HumanMessagePromptTemplate.from_template(custom_user or self.user),
            ]
        )
        if self.parser is not None and (
            "format_instructions" in self.system or "format_instructions" in self.user
        ):
            prompt.partial_variables["format_instructions"] = (
                self.parser.get_format_instructions()
            )

        return prompt

    def get_prompt_format(self) -> str:
        return self.parser.get_format_instructions()


class TagsParser(BaseOutputParser[bool]):
    """Custom parser to clean specified tag from results."""

    min_len: int = 10
    start_tag: str = "thinking_start"
    end_tag: str = "thinking_end"

    def parse(self, text: Union[str, BaseMessage]) -> tuple[str, str]:
        # print(f"Parsing tags: {text}")
        if isinstance(text, BaseMessage):
            text = text.content

        # Initialize the lists to store the thinking contents and remaining contents
        tag_contents = []
        text_contents = []

        text_contents_joined = ""
        tag_contents_joined = ""

        # Split the text into matches using regex
        matches = re.split(r"([\[{\(][/ ]*[^\[\]]+[\]}\)])", text, 0, re.IGNORECASE)

        # Initialize a flag to indicate whether we're inside a tag block
        in_tag = False

        # Iterate over the matches
        for match in matches:
            match = match.strip()
            # print(f"Match: {match}")
            # If the match is a start tag, set the flag to True
            if fuzz.ratio(match, self.start_tag) > 80:
                in_tag = True
            # If the match is an end tag, set the flag to False
            elif fuzz.ratio(match, self.end_tag) > 80:
                in_tag = False
            # If we're inside a tag block, add the match to the tag contents
            elif in_tag:
                tag_contents.append(match)
            # Otherwise, add the match to the text contents
            else:
                text_contents.append(match)

        text_contents_joined = "\n".join(text_contents).strip()
        tag_contents_joined = "\n".join(tag_contents).strip()

        if len(text_contents_joined) == 0 and len(tag_contents_joined) > 0 or in_tag:
            excpect_msg = textwrap.dedent(
                f"""
                Expected tags {self.start_tag} and {self.end_tag} in pairs.
                But got only tag: {self.start_tag if len(text_contents_joined) == 0 else self.end_tag}.
                Please make sure to close the tags that have been opened.
                """
            )
            raise OutputParserException(excpect_msg)
        elif len(tag_contents_joined) > 0 and len(text_contents_joined) < self.min_len:
            excpect_msg = textwrap.dedent(
                f"""
            Expected a response message at least {self.min_len} characters long
            but got only [thinking_start] .... [thinking_end] response. Please make
            sure that the response is long enough and also contains text outside of the tags.
            """
            )
        elif len(text_contents_joined) < self.min_len:
            excpect_msg = textwrap.dedent(
                f"""
            Expected a response message at least {self.min_len} characters long.
            Please make sure that the response is long enough.
            """
            )

            raise OutputParserException(excpect_msg)

        return text_contents_joined, tag_contents_joined

    @property
    def _type(self) -> str:
        return "tag_output_parser"


# def generate_result_sample(tasks: List[FewShotItem], amount = 3) -> dict:
#     sample = random.sample(tasks, amount)
#     documents = [Document(page_content=task.document) for task in sample]
#     result = "\n".join([f"Task {i+1}:\n{task.result}" for i, task in enumerate(sample)])
#     return {"documents": documents, "result": result}

# def get_journey_format_example(amount=4) -> FewShotPromptTemplate:
#     instruction = textwrap.dedent(
#         """
#         Find tasks from the following documents and create a task list based on it.

#         Documents:
#         {documents}

#         Task:
#         {tasks}
#     """
#     )

#     task_lists = [lambda: generate_result_sample(example_tasks, amount) for _ in range(3)]

#     return FewShotPromptTemplate(
#         examples=task_lists,
#         example_prompt=PromptTemplate(
#             template=instruction,
#             input_variables=["documents", "tasks"],
#         ),
#         suffix=textwrap.dedent(
#             """
#             Documents:
#             {context}
#             """),
#         input_variables=["context"],
#     )
