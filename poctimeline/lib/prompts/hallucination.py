import re
import textwrap
from typing import Union
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.messages import BaseMessage

from lib.prompts.base import PromptFormatter


class HallucinationParser(BaseOutputParser[tuple[bool, BaseMessage]]):
    """Custom parser to check for hallucinations"""

    def parse(self, text: Union[str, BaseMessage]) -> tuple[bool, BaseMessage]:
        # print(f"Parsing tags: {text}")
        if isinstance(text, tuple):
            text = text[0]

        if isinstance(text, BaseMessage):
            text = text.content

        # Extract all floats from the text using regular expressions
        floats = [float(n) for n in re.findall(r"[-+]?(?:\d*\.*\d+)", text)]

        if len(floats) == 0:
            raise OutputParserException(
                "Unable to verify if the content is based on context and instructions",
                observation=text,
            )
        if floats[0] == 0.0:
            raise OutputParserException(
                "The generated text is not based on the context or it does not follow instructions.",
                observation=text,
            )

        # raise OutputParserException(f"Unexpected error: Testing errors.")

        return True, text

    @property
    def _type(self) -> str:
        return "hallucination_parser"


hallucination = PromptFormatter(
    system=textwrap.dedent(
        f"""
        You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. Only check for facts do not
        check for formatting issues or that the provided content is correctly formatted in the specified format. Do not check for valid JSON.
        Just check if LLM generation is grounded on facts.

        Give a binary score 1 or 0, where 1 means that the answer is grounded in / supported by the set of facts or history and answers the question.

        After the score write a short explanation if the answer does not pass test. Always return the score of 0 or 1 regardless.

        Example 1:
        1

        Generated content is grounded in facts

        Example 2:
        0

        Generated content is built on information which is not in scope

        Example 3:
        0

        Generated content is guessing
        """
    ),
    user=textwrap.dedent(
        """
        {input}

        LLM generation: {output}

        Return 1 or 0 score and explanation regardless. Do not check for valid JSON.
        """
    ),
)
hallucination.parser = HallucinationParser()
