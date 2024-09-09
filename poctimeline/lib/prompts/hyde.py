import textwrap
from lib.prompts.base import MAINTAIN_CONTENT_AND_USER_LANGUAGE, PromptFormatter


hyde = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Given the chat history and the latest user question
        which might reference the chat history,
        formulate a standalone answer which could be a result
        for a search engine query for the question.
        Use maximum of three sentences.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        """
    ),
    user=textwrap.dedent("""{question}"""),
)

hyde_document = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Given the title and the subject generate a short description for a document defined by
        the title. Use maximum of five sentences.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        """
    ),
    user=textwrap.dedent("""{question}"""),
)
