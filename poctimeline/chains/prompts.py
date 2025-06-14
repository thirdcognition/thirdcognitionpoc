import re
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
    # HumanMessage,
    # ChatMessage,
    BaseMessage,
)
from langchain_core.prompts.chat import MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser

# from fewshot_data import FewShotItem, example_tasks

actor_introductions = "You are a helpful and efficient AI assistant Virtual Buddy."

pre_think_instruct = """
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
keep_pre_think_together = """
        While following your plan don't explain what you are doing.
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
            excpect_msg = textwrap.dedent(f"""
                Expected tags {self.start_tag} and {self.end_tag} in pairs.
                But got only tag: {self.start_tag if len(text_contents_joined) == 0 else self.end_tag}.
                Please make sure to close the tags that have been opened.
                """)
            raise OutputParserException(excpect_msg)
        elif len(tag_contents_joined) > 0 and len(text_contents_joined) < self.min_len:
            excpect_msg = textwrap.dedent(f"""
            Expected a response message at least {self.min_len} characters long
            but got only [thinking_start] .... [thinking_end] response. Please make
            sure that the response is long enough and also contains text outside of the tags.
            """)
        elif len(text_contents_joined) < self.min_len:
            excpect_msg = textwrap.dedent(f"""
            Expected a response message at least {self.min_len} characters long.
            Please make sure that the response is long enough.
            """)

            raise OutputParserException(excpect_msg)

        return text_contents_joined, tag_contents_joined

    @property
    def _type(self) -> str:
        return "tag_output_parser"

text_formatter = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {pre_think_instruct}
        {keep_pre_think_together}
        Rewrite the text specified by the user between the context start and context end in full detail using natural language.
        Don't use html tags or markdown. Remove all mentions of confidentiality. Use only information from the available in the text.
        """
    ),
    user=textwrap.dedent(
        """
        Context start
        {context}
        Context end

        Format the text in the context.
        """
    ),
)
text_formatter.parser = TagsParser(min_len=100)

text_formatter_compress = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {pre_think_instruct}
        {keep_pre_think_together}
        Summarise, compress and reduce the text specified by the user between the context start and context end in retaining details using natural language.
        Don't return the context text as it is, process it to shorter form.
        The context is a part of a longer document. Don't use html tags or markdown. Remove all mentions of confidentiality.
        Use only information from the available in the text.
        """
    ),
    user=textwrap.dedent(
        """
        Context start
        {context}
        Context end

        Format the text in the context.
        """
    ),
)
text_formatter_compress.parser = TagsParser(min_len=100)

text_formatter_guided = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {pre_think_instruct}
        {keep_pre_think_together}
        Rewrite the text between the context start and context end using only information and follow the instructions exactly.
        Don't use html tags or markdown.
        """
    ),
    user=textwrap.dedent(
        """
        Instructions: {question}

        Context start
        {context}
        Context end

        Format the text in the context.
        """
    ),
)
text_formatter_guided.parser = TagsParser(min_len=100)

md_formatter = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {pre_think_instruct}
        {keep_pre_think_together}
        Rewrite the text between the context start and context end using markdown syntax. Use only information from the context.
        Remove all mentions of confidentiality.
        """
    ),
    user=textwrap.dedent(
        """
        Context start
        {context}
        Context end

        Format the text in the context.
        """
    ),
)
md_formatter.parser = TagsParser(min_len=100)

md_formatter_guided = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {pre_think_instruct}
        {keep_pre_think_together}
        Rewrite the text between the context start and context end using markdown syntax. Use only information from the context
        and follow the instructions exactly. Remove all mentions of confidentiality. Follow the instructions exactly.
        """
    ),
    user=textwrap.dedent(
        """
        Instructions: {question}

        Context start
        {context}
        Context end

        Format the text in the context.
        """
    ),
)
md_formatter_guided.parser = TagsParser(min_len=100)

action = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a task completing machine.
        {pre_think_instruct}
        {keep_pre_think_together}
        Use the following pieces of information between the context start and context end to complete the task.
        If you don't know the how, just say that you don't know how.
        """
    ),
    user=textwrap.dedent(
        """
        Task: {action}

        Context start
        {context}
        Context end

        Complete the task with the details from context.
        """
    ),
)
action.parser = TagsParser(min_len=10)

grader = PromptFormatter(
    system=textwrap.dedent(
        f"""
        You are a grader assessing relevance of a retrieved document to a user question.

        If the document contains keywords related to the user question, grade it as relevant.
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        """
    ),
    user=textwrap.dedent(
        """
        Here is the retrieved document:
        {document}

        Here is the user question:
        {question}

        Grade the document based on the question.
        """
    ),
)
grader.parser = JsonOutputParser()

check = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a verification machine. Check that the items matches the format exactly and answer with one of
        the appropriate option only. Do not explain your answer or add anything else after or before the answer.
        """
    ),
    user=textwrap.dedent(
        """
        {format}

        {expected_count}

        Amount of items: {count}
        {context}

        Respond options: {options}

        Do the items match the format?
        Respond with one of the options only.
        Do not explain your answer or add anything else after or before the answer.
        """
    ),
)

question = PromptFormatter(
    system=textwrap.dedent(
        f"""
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context and conversation history to answer the question.
        If you don't know the answer, say that you don't know. Limit your response to three sentences maximum
        and keep the answer concise. Don't reveal that the context is empty, just say you don't know.
        """
    ),
    user=textwrap.dedent(
        """
        Context start
        {context}
        Context end
        Question: {question}
        """
    ),
)

class QuestionClassifierParser(BaseOutputParser[tuple[bool, BaseMessage]]):
    """Custom parser to clean specified tag from results."""

    def parse(self, text: Union[str, BaseMessage]) -> tuple[bool, BaseMessage]:
        # print(f"Parsing tags: {text}")
        if isinstance(text, BaseMessage):
            text = text.content

        # Extract all floats from the text using regular expressions
        # Check if 'yes' exists on the first line of text
        first_line = text.split('\n')[0].strip().lower()
        if 'yes' in first_line:
            return True, text
        elif 'no' in first_line:
            return False, text
        else:
            raise OutputParserException(f"Unexpected response: Expected 'yes' or 'no', but got '{first_line}'.")

    @property
    def _type(self) -> str:
        return "question_classifier_parser"

question_classifier = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a strict message classifier.
        Return "yes" if the message is a question or "no" if the message is not a question.
        Do not add anything else in the response. Just return "yes" or "no".
        """
    ),
    user=textwrap.dedent(
        """
        Message:
        {question}

        Respond with "yes" or "no"
        """
    ),
)
question_classifier.parser = QuestionClassifierParser()

helper = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a startup coach and answer questions thoroughly and exactly.
        {pre_think_instruct}
        Use the info between the context start and context end and previous discussion to answer the question.
        If you don't know the answer, just say that you don't know. Keep the answer concise.
        """
    ),
    user=textwrap.dedent(
        """
        Context start
        {context}
        Context end
        Question: {question}
        """
    ),
)
helper.parser = TagsParser(min_len=10)

chat = PromptFormatter(
    system=textwrap.dedent(
        f"""
        You are replying in a lighthearted and funny way, but don't over do it.
        Use max three sentences maximum and keep the answer concise. You can use history
        to make the answer more relevant but focus on 2-3 latest Human messages.
        Don't explain your responses, apologize or mention that you are an assistant.
        """
    ),
    user=textwrap.dedent(
        """
        {question}
        """
    ),
)
# chat.parser = TagsParser(min_len=10)


hyde = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Given the chat history and the latest user question
        which might reference the chat history,
        formulate a standalone answer which could be a result
        for a search engine query for the question.
        Use maximum of three sentences.
        """
    ),
    user=textwrap.dedent("""{question}"""),
)

hyde_document = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Given the title and the subject generate a short description for a document defined by
        the title. Use maximum of five sentences.
        """
    ),
    user=textwrap.dedent("""{question}"""),
)

summary = PromptFormatter(
    system=textwrap.dedent(
        f"""
        You are an assistant for summarizing texts.
        {pre_think_instruct}
        {keep_pre_think_together}
        Summarize the text between the context start and context end using natural language.
        """
    ),
    user=textwrap.dedent(
        """
        Context start
        {context}
        Context end
        Summarize the text.
        """
    ),
)
summary.parser = TagsParser(min_len=10)

summary_guided = PromptFormatter(
    system=textwrap.dedent(
        f"""
        You are an assistant for summarizing texts.
        {pre_think_instruct}
        {keep_pre_think_together}
        Summarize the text between the context start and context end using natural language
        and follow the instructions exactly.
        """
    ),
    user=textwrap.dedent(
        """
        Context start
        {context}
        Context end
        Instructions: {question}
        Summarize the text.
        """
    ),
)
summary_guided.parser = TagsParser(min_len=10)


class JourneyStep(BaseModel):
    title: str = Field(description="Title for the subject", title="Title")
    description: str = Field(
        description="Description for the subject", title="Description"
    )


class JourneyStepList(BaseModel):
    steps: List[JourneyStep] = Field(description="List of subjects", title="Subjects")


journey_steps = PromptFormatter(
    system=textwrap.dedent(
        f"""
        {actor_introductions}
        Act as a teacher who is planning a curriculum.
        Using the content between context start and end write a list
        with the specified format structure.
        If instructions are provided follow them exactly.
        Only use the information available within the context.
        If there's a history with previous titles or subjects,
        use them to make sure you don't repeat the same subjects.
        Return only the properly formatted JSON object with the formatted data.
        """
    ),
    user=textwrap.dedent(  # Use get_journey_format_example instead
        """
        instuctions start
        {journey_instructions}
        {instructions}
        instructions end

        context start
        {context}
        context end

        ----------------
        format structure start
        {format_instructions}
        format structure end
        ----------------

        Create a list of {amount} subjects.
        If instructions are provided, follow them exactly. If instructions specify
        a topic or subject, make sure the list includes only items which fall within
        within that topic. Make sure the list has exactly {amount} items.
        Format the context data using the format structure.
        Do not add any information to the context or come up with subjects
        not defined within the context.
        If there's a history with previous titles or subjects,
        use them to make sure you don't repeat the same subjects.
        Return only the properly formatted JSON object with the formatted data.
        """
    ),
)
journey_steps.parser = PydanticOutputParser(pydantic_object=JourneyStepList)

journey_step_content = PromptFormatter(
    system=textwrap.dedent(
        f"""
        {actor_introductions}
        Act as a teacher who is planning the content for a class with a specific subject.
        Use blog content style and structure starting with an introduction and synopsis and continuing with clearly sectioned content. Use markdown syntax for formatting.
        Your student is a business graduate who is interested in learning about the subject.
        You only have one student you're tutoring and you are making study materials for them.
        {pre_think_instruct}
        {keep_pre_think_together}
        Create the study material for the student with the following information between context start and end.
        Only use the information available within the context. Do not add or remove information from the context.
        If there's a history with previous titles or subjects, use them to make sure you don't repeat the same subjects.
        If instructions are provided follow them exactly.
        """
    ),
    user=textwrap.dedent(  # Use get_journey_format_example instead
        """
        instuctions start
        {journey_instructions}
        {instructions}
        instructions end

        context start
        {context}
        context end

        Subject:
        {subject}


        Create study materials for the student defined by the subject.
        Use blog content style and structure starting with an introduction and synopsis and continuing with clearly sectioned content. Don't include any other content outside of the subject.
        Only use the information available within the context. Do not add or remove information from the context.
        If instructions are provided, follow them exactly. If instructions specify
        a topic or subject, make sure the list includes only items which fall within
        within that topic.
        The study materials should be exhaustive, detailed and generated from the context.
        If there's a history with previous titles or subjects,
        use them to make sure you don't repeat the sam or subjects.
        If instructions are provided follow them exactly.
        """
    ),
)
journey_step_content.parser = TagsParser(min_len=10)

# journey_step_content_redo = PromptFormatter(
#     system=textwrap.dedent(
#         f"""
#         {actor_introductions}
#         Act as a teacher who is detailing the content for a class with a specific subject.
#         Do not use code, lists, or any markup, markdown or html. Just use natural spoken language divided
#         into a clear structure.
#         Your student is a business graduate who is interested in learning about the subject.
#         You only have one student you're tutoring and you are making material for them.
#         {pre_think_instruct}
#         {keep_pre_think_together}
#         Create the material for the student with the following information between context start and end.
#         Only use the information available within the context. Do not add or remove information from the context.
#         If there's a history with previous titles or subjects,
#         use them to make sure you don't repeat the same subjects.
#         If instructions are provided follow them exactly.
#         The material should be clearly divided into sections defined by 2nd level headers and content for those sections.
#         Use a formal tone, but write the content in an understandable format.
#         """
#     ),
#     user=textwrap.dedent(  # Use get_journey_format_example instead
#         """
#         instuctions start
#         {journey_instructions}
#         {instructions}
#         instructions end

#         context start
#         {context}
#         context end

#         Subject:
#         {subject}


#         Create materials for the student defined by the subject. Don't include any other content outside of the subject.
#         Only use the information available within the context. Do not add or remove information from the context.
#         If instructions are provided, follow them exactly. If instructions specify
#         a topic or subject, make sure the list includes only items which fall within
#         within that topic.
#         The materials should be exhaustive, detailed and generated from the context.
#         If instructions are provided follow them exactly.
#         The generated material should follow a descriptive tutorial style with a clear structure using only
#         the available content.
#         """
#     ),
# )
# journey_step_content_redo.parser = TagsParser(min_len=10)

journey_step_intro = PromptFormatter(
    system=textwrap.dedent(
        f"""
        {actor_introductions}
        Act as a teacher who is writing a brief introduction and a synopsis to a specific subject
        for the student. Use an informal style with and 3 sentences maximum.
        Do not use code, lists, or any markup, markdown or html. Just use natural spoken language.
        Your student is a business graduate who is interested in learning about the subject.
        You only have one student you're tutoring so don't have to address more than one person.
        {pre_think_instruct}
        {keep_pre_think_together}
        If there's a history with previous titles or subjects,
        use them to make sure you don't repeat the same subjects.
        Use the content for the class available between content start and end.
        """
    ),
    user=textwrap.dedent(  # Use get_journey_format_example instead
        """
        instuctions start
        {journey_instructions}
        {instructions}
        instructions end

        Subject:
        {subject}

        content start
        {context}
        content end

        Write an introduction and a synopsis to the subject for the student using only natural language.
        If instructions are provided, follow them exactly. If instructions specify
        a topic or subject, make sure the list includes only items which fall within
        within that topic.
        Don't add anything new to the content, just explain it with 3 sentences maximum.
        """
    ),
)
journey_step_intro.parser = TagsParser(min_len=10)

journey_step_actions = PromptFormatter(
    system=textwrap.dedent(
        f"""
        {actor_introductions}
        Act as a teacher who planning sections of content for teaching the student a specific subject.
        You only have one student you're tutoring so don't have to address more than one person. Also add references to support documents
        with their summary and material to use when teaching the student about the subject.
        {pre_think_instruct}
        {keep_pre_think_together}
        If there's a history with previous titles or subjects,
        use them to make sure you don't repeat the same subjects.
        Use the content for the class available between content start and end.
        """
    ),
    user=textwrap.dedent(  # Use get_journey_format_example instead
        """
        instuctions start
        {journey_instructions}
        {instructions}
        instructions end

        Amount:
        {amount}

        Subject:
        {subject}

        content start
        {context}
        content end

        Write a list of sections with their content to teach the subject to the student.
        Prepare also a list of reference documents and their summary to use with each section when teaching the student about the subject.
        If instructions are provided, follow them exactly. If instructions specify a topic or subject, make sure the list includes only
        items which fall within within that topic. Create at maximum the specified amount of items.
        If there's a history with previous titles or subjects, use them to make sure you don't repeat the same subjects.
        """
    ),
)
journey_step_actions.parser = TagsParser(min_len=10)

journey_step_action_details = PromptFormatter(
    system=textwrap.dedent(
        f"""
        {actor_introductions}
        Act as a teacher who is creating resources and content to support teaching in class
        to use as a base for the discussion and lesson with the student.
        {pre_think_instruct}
        {keep_pre_think_together}
        If there's a history with previous titles or subjects, use them to make sure you don't repeat the same subjects.
        Use the provided resource description and the content available between content start and end to create the resources.
        """
    ),
    user=textwrap.dedent(  # Use get_journey_format_example instead
        """
        instuctions start
        {journey_instructions}
        {instructions}
        instructions end

        Resource description:
        {resource}

        content start
        {context}
        content end

        Prepare max 10 sentences of material as the resource described to be used while teaching a student.
        If instructions are provided, follow them exactly. If instructions specify a topic or subject, make sure the list includes only
        items which fall within within that topic.
        If there's a history with previous titles or subjects,
        use them to make sure you don't repeat the same subjects.
        """
    ),
)
journey_step_action_details.parser = TagsParser(min_len=20)


class ResourceStructure(BaseModel):
    title: str = Field(description="Title for the content to help in the task.", title="Title")
    summary: str = Field(
        description="Most important parts of the document for the step", title="Summary"
    )
    reference: str = Field(
        description="Name of the resource, references or link if available.",
        title="Reference",
    )

class ActionStructure(BaseModel):
    title: str = Field(description="Title for the content", title="Title")
    description: str = Field(
        description="Objective, or target for the content", title="Description"
    )
    resources: List[ResourceStructure] = Field(
        description="List of references to documents or resources to help with the specifics for the content.",
        title="Resources",
    )
    test: str = Field(
        description="Description on how to do a test to verify that the student has succeeded in learning the contents.",
        title="Test",
    )


class SubjectStructure(BaseModel):
    title: str = Field(description="Title of the class", title="Title")
    subject: str = Field(description="Subject of the class", title="Subject")
    intro: str = Field(description="Introduction to the class", title="Intro")
    content: str = Field(description="Detailed content of the class", title="Content")
    actions: List[ActionStructure] = Field(
        description="List of contents to help teaching the subject.",
        title="Content",
    )


journey_structured = PromptFormatter(
    system=textwrap.dedent("""
    Act as a structured data formatter and use specified format instructions exactly
    to format the context data. Return only the JSON object with the formatted data.
    """),
    user=textwrap.dedent("""

    context start
    {context}
    context end
    ----------------
    format instructions start
    {format_instructions}
    format instructions end
    ----------------
    Format the context data using the format instructions.
    Return only the properly formatted JSON object with the formatted data.
    """),
)
journey_structured.parser = PydanticOutputParser(pydantic_object=SubjectStructure)

DEFAULT_PROMPT_FORMATTER = chat

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
