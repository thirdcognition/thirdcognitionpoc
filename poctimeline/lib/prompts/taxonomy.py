import os
import textwrap
import random
from typing import Any, Dict, List
from lib.helpers.shared import read_and_load_yaml
from langchain_core.output_parsers import PydanticOutputParser
from lib.models.taxonomy import Taxonomy, TaxonomyStructureList
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

taxonomy_hierarchy = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a taxonomy hierachy compiler who is finding the hierachy of taxonomy items from a list.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        You are provided with a list of taxonomy items extracted from a document.
        Find all connected taxonomy items and define a hierarchy for the taxonomy items.
        If there are taxonomy items which are relatively similar, combine them using joined list.
        """
    ),
    user=textwrap.dedent(
        """
        taxonomy items start
        {hierarchy_items}
        taxonomy items end
        ----------------
        format instructions start
        {format_instructions}
        format instructions end
        ----------------
        Using the taxonomy items find the hierachy of taxonomy items and extract it using the specified format.
        Format the context data using the format instructions.
        Return only the properly formatted JSON object with the formatted data.
        """
    ),
)
taxonomy_hierarchy.parser = PydanticOutputParser(pydantic_object=TaxonomyStructureList)

taxonomy_combiner = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a taxonomy combiner which is combining the taxonomy list into a single taxonomy.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        You are provided with a list of taxonomy items extracted from a document.
        Combine all taxonomy items into one taxonomy using the defined format.
        """
    ),
    user=textwrap.dedent(
        """
        taxonomy items start
        {joined_items}
        taxonomy items end
        ----------------
        format instructions start
        {format_instructions}
        format instructions end
        ----------------
        Using the taxonomy items create a new taxonomy which covers all the details from the taxonomy items.
        Format the context data using the format instructions.
        Return only the properly formatted JSON object with the formatted data.
        """
    ),
)
taxonomy_combiner.parser = PydanticOutputParser(pydantic_object=Taxonomy)
