import os
import textwrap
import random
from typing import Any, Dict, List
from lib.helpers import read_and_load_yaml
from langchain_core.output_parsers import PydanticOutputParser
from lib.models.sqlite_tables import ParsedTaxonomyList
from lib.prompts.base import (
    KEEP_PRE_THINK_TOGETHER,
    MAINTAIN_CONTENT_AND_USER_LANGUAGE,
    PRE_THINK_INSTRUCT,
    PromptFormatter,
    TagsParser,
)

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


taxonomy = PromptFormatter(
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
taxonomy.parser = TagsParser(
    min_len=0,
    tags=TAXONOMY_TAGS,
    optional_tags=["thinking", "reflection"] + TAXONOMY_OPTIONAL_TAGS,
    all_tags_required=True,
    return_tag=True,
)

taxonomy_refine = PromptFormatter(
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
taxonomy_refine.parser = TagsParser(
    min_len=0,
    tags=TAXONOMY_TAGS,
    optional_tags=["thinking", "reflection"] + TAXONOMY_OPTIONAL_TAGS,
    all_tags_required=True,
    return_tag=True,
)

taxonomy_structured = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a taxonomy compiler who is building a structured format
        out of new and existing taxonomy categories.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        Define a hierarchy for the taxonomy using
        taxonomy ids. Prefer using existing taxonomy over new taxonomy.
        Where parent taxonomy is needed for hierarchy create new taxonomy
        items to connect existing taxonomy together.
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
        format instructions start
        {format_instructions}
        format instructions end
        ----------------
        Using the existing and new taxonomy combine them into a structured format.
        Prefer using existing taxonomy over new taxonomy but if not available
        create new taxonomy items to connect existing taxonomy together.
        Format the context data using the format instructions.
        Return only the properly formatted JSON object with the formatted data.
        """
    ),
)
taxonomy_structured.parser = PydanticOutputParser(pydantic_object=ParsedTaxonomyList)
