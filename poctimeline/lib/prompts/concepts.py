import textwrap
from langchain_core.output_parsers import PydanticOutputParser
from lib.models.concepts import (
    ConceptData,
    ConceptList,
    ConceptStructureList,
)
from lib.prompts.base import (
    MAINTAIN_CONTENT_AND_USER_LANGUAGE,
    PromptFormatter,
)
from lib.prompts.actions import structured

concept_structured = structured.customize(
    system=textwrap.dedent(
        f"""
        Act as a document concept extractor.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        Extract concepts fitting the specified taxonomy and instructions from
        the text between the context start and context end.
        Use specified taxonomy to categorize the concepts.
        Concept is any idea, topic, or subject that can be found from the context.
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

concept_structured.parser = PydanticOutputParser(pydantic_object=ConceptList)

# Check if there's any more concepts within the content
concept_more = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a document concept extractor.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        You are provided with a list of concepts and a taxonomy extracted from a document.
        Check if there's any more concepts within the context that are not covered by the existing concepts.
        Extract only new concepts fitting the specified taxonomy.
        Use specified taxonomy to categorize the concepts.
        Concept is any idea, topic, or subject that can be found from the context.
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
concept_more.parser = PydanticOutputParser(pydantic_object=ConceptList)

# concept_unique = PromptFormatter(
#     system=textwrap.dedent(
#         f"""
#         Act as a concept compiler who is finding all unique concepts from a list.
#         {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
#         Check that all the concepts within the list are unique.
#         If there are any duplicate or overlapping concepts combine them.
#         """
#     ),
#     user=textwrap.dedent(
#         """
#         existing concepts start
#         {existing_concepts}
#         existing concepts end
#         ----------------
#         new concepts start
#         {new_concepts}
#         new concepts end
#         ----------------
#         format instructions start
#         {format_instructions}
#         format instructions end
#         ----------------
#         Using the concepts find the all unique concepts and extract them using the specified format.
#         Don't export duplicate concepts, combine them if possible and only use existing ids where defined.
#         Don't create any new concepts.
#         Format the resulting concepts using the format instructions.
#         Return only the properly formatted JSON object with the formatted data.
#         """
#     ),
# )
# concept_unique.parser = PydanticOutputParser(pydantic_object=ParsedUniqueConceptList)

concept_hierarchy = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a concept hierachy compiler who is finding the hierachy of concepts from a list.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        You are provided with a list of concepts extracted from a document.
        Find all connected concepts and define a hierarchy for the concepts.
        The concept is any idea, topic, or subject that has been found from the document.
        If there are concepts which are directly overlapping, you can join them.
        """
    ),
    user=textwrap.dedent(
        """
        existing concepts start
        {hierarchy_items}
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
    pydantic_object=ConceptStructureList
)

concept_combiner = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a concept combiner which is combining multiple concepts into a single concept.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        You are provided with a set of concepts extracted from a document.
        Combine all concepts into one topic using the defined format.
        """
    ),
    user=textwrap.dedent(
        """
        concepts start
        {joined_items}
        concepts end
        ----------------
        format instructions start
        {format_instructions}
        format instructions end
        ----------------
        Using the concepts create a new concept which combines the concepts.
        Format the resulting concepts using the format instructions.
        Return only the properly formatted JSON object with the formatted data.
        """
    ),
)
concept_combiner.parser = PydanticOutputParser(pydantic_object=ConceptData)