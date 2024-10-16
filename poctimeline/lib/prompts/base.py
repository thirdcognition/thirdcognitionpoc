from fuzzywuzzy import fuzz
import textwrap
from typing import Dict, List, Optional, Tuple, Union
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
from sqlalchemy import Enum

from lib.helpers.shared import pretty_print

# from fewshot_data import FewShotItem, example_tasks

PRE_THINK_INSTRUCT = """
        Reason through the query inside <thinking> tags, and then provide your final response inside <output> tags.
        If you detect that you made a mistake in your reasoning at any point, correct yourself inside <reflection> tags.
        """
PRE_THINK_TAGS = ["thinking", "reflection"]

ACTOR_INTRODUCTIONS = "You are a world-class AI system, capable of complex reasoning and reflection called Virtual Buddy."
# For example:

# Example 1:

# <thinking> I have a long article about a recent scientific discovery. I will generate a summary
# that highlights the main findings, the method used, and the implications of the discovery. </thinking>

# A recent study published in the Journal of Neuroscience revealed that prolonged exposure to blue light
# before sleep can disrupt circadian rhythms and lead to sleep disorders. The research, conducted on mice,
# found that blue light exposure suppressed the production of melatonin, a hormone that regulates sleep-wake cycles.
# These findings suggest that limiting blue light exposure, particularly in the evening, may help improve sleep
# quality and reduce the risk of sleep disorders.

# Example 2:

# <thinking> I have a lengthy book review. I will create a summary that captures the main points of the book,
# the author's style, and the reviewer's overall opinion. </thinking>

# In 'The Catcher in the Rye', J.D. Salinger explores the disillusionment and alienation of a teenager named Holden Caulfield.
# Through Holden's first-person narrative, the novel delves into themes of adolescence, identity, and the struggle for authenticity.
# The book's raw and unfiltered language captures the angst and confusion of youth, making it a classic of modern literature.

# Example 3:

# <thinking> I have several research papers on a specific topic. I will generate a comprehensive report that synthesizes
# the findings from each paper, identifies common themes, and discusses any discrepancies. </thinking>

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


from html.parser import HTMLParser


class TagHTMLParser(HTMLParser):
    def __init__(self, allowed_tags):
        super().__init__()
        self.allowed_tags = allowed_tags
        self.stack = []
        self.root = None

    def reset(self):
        self.stack = []
        self.root = None
        super().reset()

    def handle_starttag(self, tag, attrs):
        for allowed_tag in self.allowed_tags:
            if fuzz.ratio(tag, allowed_tag) > 80:
                new_node = {"tag": tag, "body": "", "children": []}
                if self.stack:
                    self.stack[-1]["children"].append(new_node)
                else:
                    self.root = new_node
                self.stack.append(new_node)

    def handle_endtag(self, tag):
        if self.stack:
            if fuzz.ratio(tag, self.stack[-1]["tag"]) > 80:
                self.stack.pop()

    def handle_data(self, data):
        if self.stack and len(str(data).strip()) > 0:
            self.stack[-1]["body"] += (
                data.strip() + "\n\n" if isinstance(data, str) else data
            )

    def get_root(self):
        return self.root


def parse_html(html, allowed_tags):
    parser = TagHTMLParser(allowed_tags)
    parser.feed(html)
    return parser.get_root()


class TagsParser(BaseOutputParser[Union[str, Dict]]):
    """Custom parser to clean specified tag from results."""

    min_len: int = Field(default=10)
    tags: List[str] = Field(default_factory=lambda: ["thinking", "reflection"])
    content_tags: List[str] = Field(default_factory=lambda: ["root", "output"])
    optional_tags: Optional[List[str]] = Field(default=None)
    return_tag: bool = Field(default=False)
    all_tags_required: bool = Field(default=False)
    required_output_tag: str = Field(default="output")
    required_output_values: Optional[List[str]] = Field(default=None)

    def get_child_content(self, node, tags=None) -> tuple[int, str]:
        if tags is None:
            tags = self.tags
        content = ""
        child_count = 0
        if node["tag"] in tags:
            content = str(node["body"]).strip() + "\n"
            child_count += 1
        for child in node["children"]:
            child_content = self.get_child_content(child, tags)
            child_count += child_content[0]
            content += child_content[1].strip() + "\n"
        return child_count, content.strip()

    def parse(self, text: Union[str, BaseMessage]) -> Union[str, Dict]:
        tag_html_parser = TagHTMLParser(
            (
                self.tags + self.optional_tags
                if isinstance(self.optional_tags, list)
                else self.tags
            )
            + self.content_tags
        )

        # print(f"Parsing tags: {text}")
        if isinstance(text, BaseMessage):
            text = text.content

        if isinstance(text, str):
            text = text.strip()

        tag_html_parser.feed(
            f"<root>{text}</root>"
            if not text.startswith("<root>") or not text.startswith("<output>")
            else text
        )
        parsed_content = tag_html_parser.get_root()

        pretty_print({"text": text, "parsed": parsed_content}, "Parsed content:")

        tag_content = {}
        tag_children = {}
        # tag_contents_joined = {}

        tags = self.tags + self.content_tags
        if isinstance(self.optional_tags, list):
            tags += self.optional_tags

        for tag in tags:
            if parsed_content is not None and "children" in parsed_content:
                # content[tag] = ""
                # for node in parsed_content["children"]:
                #     if node["tag"] == tag:
                #         content[tag] += (
                child_content = self.get_child_content(
                    parsed_content,
                    [tag],
                    # (
                    #     self.tags + self.optional_tags
                    #     if isinstance(self.optional_tags, list)
                    #     else self.tags
                    # ),
                )
                tag_children[tag] = child_content[0]
                tag_content[tag] = child_content[1].strip()  # + 2 * "\n"
                #         )
                # tag_content[tag] = tag_content[tag].strip()
                if len(tag_content[tag]) == 0:
                    del tag_content[tag]
                # tag_contents_joined[tag] = (
                #     tag_content[tag]
                #     if tag not in tag_contents_joined
                #     else tag_contents_joined[tag] + "\n\n" + tag_content[tag]
                # )

        for tag in self.content_tags:
            if tag in tag_content and (
                self.required_output_values is None or tag != self.required_output_tag
            ):
                del tag_content[tag]
        #     if tag in tag_contents_joined:
        #         del tag_contents_joined[tag]

        text_contents_joined = str(parsed_content["body"]).strip()
        for node in parsed_content["children"]:
            if node["tag"] not in self.tags:
                child_content = self.get_child_content(node, self.content_tags)
                text_contents_joined += child_content[1] + 2 * "\n"
        text_contents_joined = text_contents_joined.strip()

        if (
            self.min_len > 0
            and len(repr(tag_content)) > 2
            and len(text_contents_joined) == 0
        ):
            found_tags = tag_content.keys()

            excpect_msg = textwrap.dedent(
                f"""
            Expected a response but got only:
            {",\n ".join([f"<{tag}> .... </{tag}>" for tag in found_tags])}>
            response{"" if len(found_tags) == 1 else "s"}. Please make sure that the response is long enough
            and also contains text outside of the tags.
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

        if self.all_tags_required:
            missing_tags = []
            for tag in self.tags:
                if (tag not in tag_content or len(tag_content[tag]) == 0) and (
                    tag not in tag_children or tag_children[tag] == 0
                ):
                    missing_tags.append(tag)

            if len(missing_tags) > 0:
                raise OutputParserException(
                    f"Expected a response with all tags: {', '.join(self.tags)}\nMissing tags: {', '.join(missing_tags)}"
                )

        if self.required_output_values is not None:
            output_content = str(tag_content[self.required_output_tag]).strip()
            matched_value = None
            for item in self.required_output_values:
                if fuzz.ratio(output_content, item) > 80:
                    matched_value = item
                    break
            if matched_value is None:
                raise OutputParserException(
                    f"Expected a response with a value similar to one from these: [{','.join(self.required_output_values)}]\n\nGot: {output_content}"
                )
            return matched_value

        if self.return_tag:
            resp = {
                "content": text_contents_joined,
                "tags": {
                    tag: tag_content[tag]
                    for tag in tag_content.keys()
                    if tag in self.tags
                },
                # "tags_joined": tag_contents_joined,
                "parsed": parsed_content,
            }
            pretty_print(resp, "Tag parser response:")
            return resp
        else:
            return text_contents_joined

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
