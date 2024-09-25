import textwrap
from lib.prompts.base import (
    KEEP_PRE_THINK_TOGETHER,
    MAINTAIN_CONTENT_AND_USER_LANGUAGE,
    PRE_THINK_INSTRUCT,
    PromptFormatter,
    TagsParser,
)


text_formatter_simple = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        Rewrite the text specified by the user defined in context in full detail.
        Remove all mentions of confidentiality. Use only information from the available in the context.
        """
    ),
    user=textwrap.dedent(
        """
        Context start
        {context}
        Context end

        Rewrite the text in the context in full detail.
        """
    ),
)

text_formatter = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
        Rewrite the text specified by the user defined in context in full detail.
        If available use previous and next page to improve the context.
        Remove all mentions of confidentiality. Use only information from the available in the context.
        """
    ),
    user=textwrap.dedent(
        """
        Previous page ending start
        {prev_page}
        Previous page ending end
        Context start
        {context}
        Context end
        Next page beginning start
        {next_page}
        Next page beginning end

        Rewrite the text in the context in full detail.
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
        Summarise, compress and reduce the text specified by the user between the
        context start and context end in retaining details using natural language.
        Don't return the context text as it is, process it to shorter form.
        The context is a part of a longer document. Don't use html tags or markdown.
        Remove all mentions of confidentiality.
        Use only information from the available in the text.
        If available follow the instructions but don't repeat them in the result.
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
        Previous page ending start
        {prev_page}
        Previous page ending end
        Next page beginning start
        {next_page}
        Next page beginning end
        Context start
        {context}
        Context end

        Format the text in the context.
        """
    ),
)
text_formatter_guided.parser = TagsParser(
    min_len=100,
)

text_formatter_compress_guided = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
        Summarise, compress and reduce the text specified by the user between the
        context start and context end in retaining details using natural language.
        Don't return the context text as it is, process it to shorter form.
        The context is a part of a longer document. Don't use html tags or markdown.
        Remove all mentions of confidentiality.
        Use only information from the available in the text.
        If available follow the instructions but don't repeat them in the result.
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
text_formatter_compress_guided.parser = TagsParser(min_len=100)


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
md_formatter.parser = TagsParser(
    min_len=100,
)

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
md_formatter_guided.parser = TagsParser(
    min_len=100,
)
