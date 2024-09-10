import textwrap
from typing import Union
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import (
    JsonOutputParser,
    BaseOutputParser,
    PydanticOutputParser,
)
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from lib.models.prompts import TitledSummary
from lib.prompts.base import (
    MAINTAIN_CONTENT_AND_USER_LANGUAGE,
    PromptFormatter,
    KEEP_PRE_THINK_TOGETHER,
    PRE_THINK_INSTRUCT,
    TagsParser,
)

action = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a task completing machine.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
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


class QuestionClassifierParser(BaseOutputParser[tuple[bool, BaseMessage]]):
    """Custom parser to clean specified tag from results."""

    def parse(self, text: Union[str, BaseMessage]) -> tuple[bool, BaseMessage]:
        # print(f"Parsing tags: {text}")
        if isinstance(text, BaseMessage):
            text = text.content

        # Extract all floats from the text using regular expressions
        # Check if 'yes' exists on the first line of text
        first_line = text.split("\n")[0].strip().lower()
        if "yes" in first_line:
            return True, text
        elif "no" in first_line:
            return False, text
        else:
            raise OutputParserException(
                f"Unexpected response: Expected 'yes' or 'no', but got '{first_line}'."
            )

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


summary_with_title = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a structured data formatter for titling and summarizing texts and use specified format instructions exactly
        to format the context data.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        Title and summarize the text between the context start and context end using natural language.
        Return only the JSON object with the formatted data.
        """
    ),
    user=textwrap.dedent(
        """
        context start
        {context}
        context end
        ----------------
        format instructions start
        {format_instructions}
        format instructions end
        ----------------
        Title and summarize the text and return the result using specified JSON structure.
        Return only the properly formatted JSON object with the formatted data.
        """
    ),
)
summary_with_title.parser = PydanticOutputParser(pydantic_object=TitledSummary)

summary = PromptFormatter(
    system=textwrap.dedent(
        f"""
        You are an assistant for summarizing texts.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
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
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
        Summarize the text between the context start and context end using natural language
        and follow the instructions exactly.
        """
    ),
    user=textwrap.dedent(
        """
        Context start
        {context}
        Context end
        Instructions: {instructions}
        Summarize the text.
        """
    ),
)
summary_guided.parser = TagsParser(min_len=10)

combine_description = PromptFormatter(
    system=textwrap.dedent(
        f"""
        You are an assistant for summarizing bullet points.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
        Summarize the bullet points between the context start and context end using natural language into one description.
        """
    ),
    user=textwrap.dedent(
        """
        Context start
        {context}
        Context end
        Summarize the bullet points into one description.
        """
    ),
)
combine_description.parser = TagsParser(min_len=10)

error_retry = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a error fixer. You are given a prompt, a completion and an error message.
        The completion did not satisfy the constraints given in the prompt. Fix the completion
        based on the error.
        """
    ),
    user=textwrap.dedent(
        """
        Prompt:
        {prompt}
        Completion:
        {completion}

        Above, the Completion did not satisfy the constraints given in the Prompt.
        Details: {error}
        Please try again:
        """
    ),
)

structured = PromptFormatter(
    system=textwrap.dedent(
        """
        Act as a structured data formatter and use specified format instructions exactly
        to format the context data. Return only the JSON object with the formatted data.
        If history is available use it as specified.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        """
    ),
    user=textwrap.dedent(
        """
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
        """
    ),
)
