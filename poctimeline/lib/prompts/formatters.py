import os
import textwrap
import random
from typing import Any, Dict, List
from langchain_core.output_parsers import PydanticOutputParser
from lib.helpers import read_and_load_yaml
from lib.models.sqlite_tables import (
    ParsedConceptTaxonomyList,
    ParsedConceptList,
    ParsedConceptStructureList,
    ParsedUniqueConceptList,
)
from lib.prompts.base import (
    KEEP_PRE_THINK_TOGETHER,
    MAINTAIN_CONTENT_AND_USER_LANGUAGE,
    PRE_THINK_INSTRUCT,
    SUMMARIZE_INSTRUCT,
    SUMMARIZE_INSTRUCT_TAGS,
    PromptFormatter,
    TagsParser,
)
from lib.prompts.actions import structured

current_dir = os.path.dirname(os.path.abspath(__file__))

taxonomy_examples_yaml: Dict[str, Any] = read_and_load_yaml(
    os.path.join(current_dir, "taxonomy_examples.yaml")
)
taxonomy_template = taxonomy_examples_yaml["template"]

# Randomly pick 10 examples
TAXONOMY = "Template structure: \n" + textwrap.indent(
    f"{taxonomy_template}\n\n"
    + "\n".join(
        [
            f"Example {i+1}:\n{example}"
            for i, example in enumerate(
                random.sample(list(taxonomy_examples_yaml["examples"].values()), 10)
            )
        ]
    ),
    8 * " ",
)
TAXONOMY_TAGS: List[str] = taxonomy_examples_yaml["tags"]
TAXONOMY_OPTIONAL_TAGS: List[str] = taxonomy_examples_yaml["optional_tags"]
TAXONOMY_ALL_TAGS: List[str] = TAXONOMY_TAGS + TAXONOMY_OPTIONAL_TAGS

text_formatter_simple = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
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

text_formatter = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {SUMMARIZE_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
        Rewrite the text specified by the user between the context start and context end in full detail using natural language.
        Don't use html tags or markdown. Remove all mentions of confidentiality. Use only information from the available in the text.
        """
    ),
    user=textwrap.dedent(
        """
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
text_formatter.parser = TagsParser(
    min_len=100,
    tags=SUMMARIZE_INSTRUCT_TAGS,
    optional_tags=["thinking", "reflection"],
    all_tags_required=True,
    return_tag=True,
)

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
        {SUMMARIZE_INSTRUCT}
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
    tags=SUMMARIZE_INSTRUCT_TAGS,
    optional_tags=["thinking", "reflection"],
    all_tags_required=True,
    return_tag=True,
)

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
        {SUMMARIZE_INSTRUCT}
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
    tags=SUMMARIZE_INSTRUCT_TAGS,
    optional_tags=["thinking", "reflection"],
    all_tags_required=True,
    return_tag=True,
)

md_formatter_guided = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document formatter.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {SUMMARIZE_INSTRUCT}
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
    tags=SUMMARIZE_INSTRUCT_TAGS,
    optional_tags=["thinking", "reflection"],
    all_tags_required=True,
)

concept_taxonomy = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document concept extractor.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
        Extract content taxonomy category tags from the context the provided template structure.
        The category tags are used to organize the ideas, topics, or subjects that can be found from the context.
        Prioritize using existing tags instead of providing new ones. Do not export or rewrite the existing taxonomy,
        only use it as a reference to define new taxonomy.

        Use the following templates and examples as guide:

        {TAXONOMY}
        """
    ),
    user=textwrap.dedent(
        """
        existing taxonomy start
        {existing_taxonomy}
        existing taxonomy end
        ----------------
        Context start
        {context}
        Context end

        Do not include, export, rewrite, or modify the existing taxonomy.
        Only use the existing taxonomy as a reference to define new taxonomy.
        """
    ),
)
concept_taxonomy.parser = TagsParser(
    min_len=0,
    tags=TAXONOMY_TAGS,
    optional_tags=["thinking", "reflection"] + TAXONOMY_OPTIONAL_TAGS,
    all_tags_required=True,
    return_tag=True,
)

concept_taxonomy_refine = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a taxonomy combiner.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
        Combine and refine the taxonomy defined within
        the new taxonomy items where possible.
        The existing taxonomy should be used as a reference to
        define new taxonomy and cannot be modified.

        Use the following templates and examples as guide:

        {TAXONOMY}
        """
    ),
    user=textwrap.dedent(
        """
        existing taxonomy start
        {existing_taxonomy}
        existing taxonomy end
        ----------------
        new taxonomy items start
        {new_taxonomy_items}
        new taxonomy items end

        Do not include, export, rewrite, or modify the existing taxonomy.
        Only use the existing taxonomy as a reference to when combining and
        refining the new taxonomy items.
        The new taxonomy items should still always follow the example format.
        """
    ),
)
concept_taxonomy_refine.parser = TagsParser(
    min_len=0,
    tags=TAXONOMY_TAGS,
    optional_tags=["thinking", "reflection"] + TAXONOMY_OPTIONAL_TAGS,
    all_tags_required=True,
    return_tag=True,
)

concept_structured = structured.customize(
    system=textwrap.dedent(
        f"""
        Act as a document concept extractor.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        Extract concepts fitting the specified taxonomy from the text between the context start and context end.
        Use specified taxonomy to categorize the concepts.
        The concepts are the main ideas, topics, or subjects that can be found from the context.
        """
    ),
    user=textwrap.dedent(
        """
        context start
        {context}
        context end
        ----------------
        taxonomy start
        {taxonomy}
        taxonomy end
        ----------------
        format instructions start
        {format_instructions}
        format instructions end
        ----------------
        Using the context find all concepts fitting the specified taxonomy and extract
        them using the specified format.
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
        You are provided with a list of concepts and a taxonomy extracted from a document.
        Check if there's any more concepts that are not covered by the existing concepts
        within the text between the context start and context end.
        Extract only new concepts fitting the specified taxonomy from the text between the context start and context end.
        Use specified taxonomy to categorize the concepts.
        The concepts are the main ideas, topics, or subjects that can be found from the context.
        """
    ),
    user=textwrap.dedent(
        """
        context start
        {context}
        context end
        ----------------
        taxonomy start
        {taxonomy}
        taxonomy end
        ----------------
        existing concepts start
        {existing_concepts}
        existing concepts end
        ----------------
        format instructions start
        {format_instructions}
        format instructions end
        ----------------
        Using the context find all new and undefined concepts fitting the specified taxonomy and extract
        them using the specified format.
        Don't export existing concepts. If you cannot find any new concepts, return an empty list.
        Format the context data using the format instructions.
        Return only the properly formatted JSON object with the formatted data.
        """
    ),
)
concept_more.parser = PydanticOutputParser(pydantic_object=ParsedConceptList)

concept_unique = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a concept compiler who is finding all unique concepts from a list.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        Check that all the concepts within the list are unique.
        If there are any duplicate or overlapping concepts combine them.
        """
    ),
    user=textwrap.dedent(
        """
        existing concepts start
        {existing_concepts}
        existing concepts end
        ----------------
        new concepts start
        {new_concepts}
        new concepts end
        ----------------
        format instructions start
        {format_instructions}
        format instructions end
        ----------------
        Using the concepts find the all unique concepts and extract them using the specified format.
        Don't export duplicate concepts, combine them if possible and only use existing ids where defined.
        Don't create any new concepts.
        Format the resulting concepts using the format instructions.
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
concept_hierarchy.parser = PydanticOutputParser(
    pydantic_object=ParsedConceptStructureList
)

concept_taxonomy_structured = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a taxonomy builder and a concept taxonomy compiler who is building and combining
        a structured format out of new and existing taxonomy categories.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        Connect the concepts to the taxonomy using tags and ids and define a hierarchy for the taxonomy.
        Prefer using existing taxonomy over new taxonomy.
        Do not create new concepts or taxonomy categories which are not defined in the
        content. Only use the available information and items.
        Use the specified format and return your output in JSON only.
        """
    ),
    user=textwrap.dedent(
        """
        existing taxonomy start
        {existing_taxonomy}
        existing taxonomy end
        ----------------
        new taxonomy start
        {new_taxonomy}
        new taxonomy end
        ----------------
        concepts start
        {concepts}
        concepts end
        ----------------
        format instructions start
        {format_instructions}
        format instructions end
        ----------------
        Using the existing and new taxonomy combine them into a structured format.
        Connect the concepts to the taxonomy using tags and concept ids
        and define a hierarchy for the taxonomy.
        Prefer using existing taxonomy over new taxonomy.
        Format the context data using the format instructions.
        Return only the properly formatted JSON object with the formatted data.
        """
    ),
)
concept_taxonomy_structured.parser = PydanticOutputParser(
    pydantic_object=ParsedConceptTaxonomyList
)
