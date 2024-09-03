import textwrap
from prompts.base import PromptFormatter


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