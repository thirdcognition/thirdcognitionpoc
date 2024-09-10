import textwrap
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from lib.models.sqlite_tables import ParsedConceptList
from lib.prompts.base import (
    KEEP_PRE_THINK_TOGETHER,
    MAINTAIN_CONTENT_AND_USER_LANGUAGE,
    PRE_THINK_INSTRUCT,
    PromptFormatter,
    TagsParser,
)
from lib.prompts.actions import structured

text_formatter = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
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
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
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
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
        Rewrite the text between the context start and context end using only information and follow the instructions exactly.
        Don't use html tags or markdown.
        """
    ),
    user=textwrap.dedent(
        """
        Instructions: {instructions}

        Context start
        {context}
        Context end

        Format the text in the context.
        """
    ),
)
text_formatter_guided.parser = TagsParser(min_len=100)

text_formatter_compress_guided = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
        Summarise, compress and reduce the text specified by the user between the context start and context end in retaining details using natural language.
        Don't return the context text as it is, process it to shorter form.
        The context is a part of a longer document. Don't use html tags or markdown. Remove all mentions of confidentiality.
        Use only information from the available in the text.
        Follow instructions exactly.
        """
    ),
    user=textwrap.dedent(
        """
        Instructions: {instructions}

        Context start
        {context}
        Context end

        Format the text in the context.
        """
    ),
)
text_formatter_compress.parser = TagsParser(min_len=100)

md_formatter = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
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
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
        Rewrite the text between the context start and context end using markdown syntax. Use only information from the context
        and follow the instructions exactly. Remove all mentions of confidentiality. Follow the instructions exactly.
        """
    ),
    user=textwrap.dedent(
        """
        Instructions: {instructions}

        Context start
        {context}
        Context end

        Format the text in the context.
        """
    ),
)
md_formatter_guided.parser = TagsParser(min_len=100)

concept_structured = structured.customize(
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
        Using the context find the available concepts and them using the specified format.
        If there is list categories specified use those to map the concepts from this
        file as much as possible.
        Try to use the existing categories and only make new categories when the existing categories
        specified do not match with the context.
        Format the context data using the format instructions.
        Return only the properly formatted JSON object with the formatted data.
        """
    )
)

concept_structured.parser = PydanticOutputParser(pydantic_object=ParsedConceptList)
