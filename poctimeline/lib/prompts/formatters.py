import textwrap
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from lib.models.sqlite_tables import (
    ParsedConceptCategoryTagList,
    ParsedConceptList,
    ParsedConceptStructure,
    ParsedConceptStructureList,
    ParsedUniqueConceptList,
)
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
    system=textwrap.dedent(
        f"""
        Act as a document concept extractor.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        Extract concepts from the text between the context start and context end.
        The concepts are the main ideas, topics, or subjects that are discussed or referred to in the text.
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
        Using the context find the all different concepts and extract them using the specified format.
        Format the context data using the format instructions.
        Return only the properly formatted JSON object with the formatted data.
        """
    ),
)

concept_structured.parser = PydanticOutputParser(pydantic_object=ParsedConceptList)

# Check if there's any more concepts within the content
concept_more = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document concept extractor.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        You are provided with a list of concepts extracted from a document.
        Check if there's any more concepts that are not covered by the existing concepts
        within the text between the context start and context end.
        Extract only the new concepts from the text between the context start and context end.
        The concepts are the main ideas, topics, or subjects that are discussed or referred to in the text.
        """
    ),
    user=textwrap.dedent(
        """
        context start
        {context}
        context end
        ----------------
        existing concepts start
        {existing_concepts}
        existing concepts end
        ----------------
        format instructions start
        {format_instructions}
        format instructions end
        ----------------
        Using the context find the all new and different concepts and extract them using the specified format.
        Don't export existing concepts. If there's no remaining concepts, return an empty list.
        Format the context data using the format instructions.
        Return only the properly formatted JSON object with the formatted data.
        """
    ),
)
concept_more.parser = PydanticOutputParser(pydantic_object=ParsedConceptList)

concept_unique = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a concept unique compiler who is finding all unique concepts from a list.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        You are provided with a list of concepts extracted from a document.
        Check that all the concepts are unique. If there are any duplicates, remove or combine them.
        The concepts are the main ideas, topics, or subjects.
        """
    ),
    user=textwrap.dedent(
        """
        existing concepts start
        {existing_concepts}
        existing concepts end
        ----------------
        format instructions start
        {format_instructions}
        format instructions end
        ----------------
        Using the existing concepts find the all unique concepts and extract them using the specified format.
        Don't export duplicate concepts, combine them if possible and only use existing ids. Don't create new concepts.
        Format the context data using the format instructions.
        Return only the properly formatted JSON object with the formatted data.
        """
    ),
)
concept_unique.parser = PydanticOutputParser(pydantic_object=ParsedUniqueConceptList)

concept_hierarchy = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a concept hierachy compiler who is finding the hierachy of concepts from a list.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        You are provided with a list of concepts extracted from a document.
        Find all connected concepts and define a hierarchy for the concepts.
        The concepts are the main ideas, topics, or subjects.
        """
    ),
    user=textwrap.dedent(
        """
        existing concepts start
        {existing_concepts}
        existing concepts end
        ----------------
        format instructions start
        {format_instructions}
        format instructions end
        ----------------
        Using the existing concepts find the hierachy of concepts and extract it using the specified format.
        Format the context data using the format instructions.
        Return only the properly formatted JSON object with the formatted data.
        """
    ),
)
concept_hierarchy.parser = PydanticOutputParser(pydantic_object=ParsedConceptStructureList)

concept_categories = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a concept category compiler who is finding the different categories for the concepts
        from a list of concepts and assigning them to the concepts.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        You are provided with a list of concepts extracted from a document and a list of existing categories.
        Connect the concepts to existing categories and assign new categories to the concepts if necessary.
        The concepts are the main ideas, topics, or subjects.
        """
    ),
    user=textwrap.dedent(
        """
        existing categories start
        {existing_categories}
        existing categories end
        ----------------
        concepts start
        {concepts}
        concepts end
        ----------------
        format instructions start
        {format_instructions}
        format instructions end
        ----------------
        Using the existing concepts find the hierachy of concepts and extract it using the specified format.
        Format the context data using the format instructions.
        Return only the properly formatted JSON object with the formatted data.
        """
    ),
)
concept_categories.parser = PydanticOutputParser(
    pydantic_object=ParsedConceptCategoryTagList
)
