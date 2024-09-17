import asyncio
import io
import operator
from typing import Annotated, Dict, List, Literal, Set, TypedDict, Union
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.constants import Send

from lib.chains.base_parser import get_text_from_completion
from lib.chains.init import get_chain
from lib.graphs.process_text import clean_dividers, get_source_content
from lib.helpers import pretty_print
from lib.models.prompts import TitledSummary
from lib.db_tools import (
    db_commit,
    delete_db_concept,
    get_db_concepts,
    get_db_concept_taxonomy,
    get_existing_concept_ids,
    update_concept_category,
    update_db_concept_rag,
)
from lib.document_tools import (
    a_semantic_splitter,
    split_text,
)
from lib.models.sqlite_tables import (
    ConceptTaxonomy,
    ConceptDataTable,
    ParsedConcept,
    ParsedConceptIds,
    ParsedConceptTaxonomyList,
    ParsedConceptList,
    ConceptData,
    ParsedConceptStructure,
    ParsedConceptStructureList,
    ParsedUniqueConceptList,
    SourceContentPage,
    SourceReference,
    convert_concept_category_tag_to_dict,
    convert_taxonomy_dict_to_tag_simple_structure_string,
    convert_taxonomy_dict_to_tag_structure_string,
    convert_taxonomy_tags_to_dict,
)
from lib.prompts.formatters import TAXONOMY_ALL_TAGS


class FindConceptsState(TypedDict):
    categories: List[str]
    filename: str
    url: str
    source: str
    concepts: List[ConceptData]
    found_concepts: Annotated[list, operator.add]
    new_concepts: List[ParsedConcept]
    collected_concepts: List[ConceptData]
    content_result: Dict
    contents: List[Union[Document, str]]
    semantic_contents: List[Document]
    source_content_topics: List[SourceContentPage]
    reformatted_txt: Annotated[list, operator.add]
    collapsed_reformatted_txt: List[Document]
    summary: str
    topics: Set[str]
    instructions: str
    get_source_complete: bool = False
    split_reformatted_complete: bool = False
    search_concepts_complete: bool = False
    collapse_concepts_complete: bool = False
    concept_complete: bool = False
    rag_complete: bool = False
    taxonomy: Dict


class FindConceptsConfig(TypedDict):
    instructions: str = None
    update_rag: bool = False


class ProcessConceptsState(TypedDict):
    categories: List[str]
    filename: str
    url: str
    topic: SourceContentPage
    taxonomy: Dict


_new_ids = {}
_divider = "±!add!±"


def get_taxonomy_item_list(categories: List[str], reset=False) -> List[ConceptTaxonomy]:
    concepts = get_db_concept_taxonomy(categories=categories, reset=reset)
    all_concepts: List[ConceptTaxonomy] = []

    for category, concept_list in concepts.items():
        for concept in concept_list:
            all_concepts.append(concept)
    return all_concepts


def get_specific_tag(items, tag="category_tag") -> List[dict]:
    found_items = []
    for item in items:
        if item["tag"] == tag:
            found_items.append(item)
        if 0 < len(item["children"]):
            found_items.extend(get_specific_tag(item["children"], tag))
    return found_items


# async def split_reformatted_content(state: FindConceptsState):
#     snippets: List[Document] = []
#     flat_contents: List[Document] = []
#     for item in state["source_content_topics"]:
#         if isinstance(item, Document):
#             flat_contents.append(item)
#         elif isinstance(item, List):
#             flat_contents.extend(item)
#         else:
#             # print(f"{state["contents"]}")
#             raise ValueError(f"Unknown type {type(item)}: {item=}")
#     for page, content in enumerate(flat_contents):
#         if isinstance(content, Document):
#             content = content.page_content

#         # print(f"Snippet {page}:\n{content[:100]}...")

#         split = await a_semantic_splitter(
#             content, threshold_type="gradient", breakpoint_threshold=90
#         )
#         snippets += [
#             Document(
#                 page_content=clean_dividers(doc), metadata={"snippet": i, "page": page}
#             )
#             for i, doc in enumerate(split)
#         ]


#     return {
#         "split_reformatted_complete": True,
#         "snippets": snippets,
#     }


async def search_for_taxonomy(state: FindConceptsState, config: RunnableConfig):
    existing_concept_taxonomy: List[ConceptTaxonomy] = get_taxonomy_item_list(
        reset=True, categories=state["categories"]
    )

    pretty_print(existing_concept_taxonomy, "Existing taxonomy")

    taxonomy_ids = [taxonomy_item.id for taxonomy_item in existing_concept_taxonomy]

    existing_taxonomy = ""
    existing_taxonomy += "\n\n".join(
        [
            convert_taxonomy_dict_to_tag_simple_structure_string(
                convert_concept_category_tag_to_dict(v)
            )
            for v in existing_concept_taxonomy
        ]
        if 0 < len(existing_concept_taxonomy)
        else []
    )

    topics = state["source_content_topics"]

    nex_taxonomy_str = ""
    new_taxonomy_items = []
    cat_for_id = "-".join(state["categories"]).replace(" ", "-").lower()
    for topic in topics:
        topic_taxonomy = await get_chain("concept_taxonomy").ainvoke(
            {
                "existing_taxonomy": existing_taxonomy + "\n" + nex_taxonomy_str,
                "context": f"{topic.topic}: \n\n{topic.page_content}",
            }
        )
        found_new_taxonomy_items = get_specific_tag(
            topic_taxonomy["parsed"]["children"] or []
        )
        if len(found_new_taxonomy_items) == 0:
            print(
                f"No new category taxonomy found for topic: {topic.page_content[:100]}..."
            )
            continue
        if len(found_new_taxonomy_items) > 0:
            for item in found_new_taxonomy_items:
                try:
                    new_taxonomy = convert_taxonomy_tags_to_dict(
                        item, TAXONOMY_ALL_TAGS
                    )
                    new_id = (
                        new_taxonomy["category_tag"]["id"]
                        if "id" in new_taxonomy["category_tag"]
                        and new_taxonomy["category_tag"]["id"] not in taxonomy_ids
                        else cat_for_id
                        + "-"
                        + get_unique_id(
                            new_taxonomy["category_tag"]["taxonomy"], taxonomy_ids
                        )
                    )
                    taxonomy_ids.append(new_id)
                    new_taxonomy["category_tag"]["id"] = new_id
                    parent_id = (
                        new_taxonomy["category_tag"]["parent_id"]
                        if "parent_id" in new_taxonomy["category_tag"]
                        else cat_for_id
                        + "-"
                        + get_unique_id(
                            new_taxonomy["category_tag"]["parent_taxonomy"], []
                        )
                    )
                    if parent_id in taxonomy_ids:
                        new_taxonomy["category_tag"]["parent_id"] = parent_id
                    new_taxonomy_items.append(new_taxonomy)
                    nex_taxonomy_str += (
                        convert_taxonomy_dict_to_tag_simple_structure_string(
                            new_taxonomy
                        )
                        + "\n"
                    )
                except Exception as e:
                    print(f"Error parsing taxonomy: {e}")
                    pretty_print(item, "Error parsing taxonomy", force=True)
                    continue

    global _new_ids
    _new_ids[state["filename"] if "filename" in state else state["url"]] = []

    return {
        "search_for_taxonomy_complete": True,
        "taxonomy": new_taxonomy_items,
    }


async def map_search_concepts(state: FindConceptsState):
    taxonomy: List[ConceptTaxonomy] = get_taxonomy_item_list(
        categories=state["categories"]
    )

    new_taxonomy: Dict[str, Dict] = state["taxonomy"]
    state_taxonomy = [
        convert_taxonomy_dict_to_tag_simple_structure_string(
            convert_concept_category_tag_to_dict(item)
        )
        for item in taxonomy
    ] + [
        convert_taxonomy_dict_to_tag_simple_structure_string(item)
        for item in new_taxonomy
    ]

    pretty_print(new_taxonomy, "Taxonomy items")

    return [
        Send(
            "search_concepts",
            {
                "url": state["url"] if "url" in state else None,
                "filename": state["filename"] if "filename" in state else None,
                "categories": state["categories"],
                "topic": topic,
                "taxonomy": state_taxonomy,
            },
        )
        for topic in state["source_content_topics"]
    ]


def get_unique_id(id_str: str, existing_ids: List[str]):
    id = f"{id_str.lower().strip().replace(' ', '_')}"
    if id not in existing_ids:
        return id
    id_index = 0
    while True:
        id_index += 1
        new_id = f"{id}_{id_index}"
        if new_id not in existing_ids:
            return new_id


async def search_concepts(state: ProcessConceptsState, config: RunnableConfig):
    topic: SourceContentPage = state["topic"]
    ident = (
        f"File: {state['filename']} --\tPage: {topic.page_content} --\tTopic: {topic.topic_index}"
        if state["filename"]
        else f"URL: {state['url']} --\tPage: {topic.page_content} --\tTopic: {topic.topic_index}"
    )
    taxonomy = "\n".join(state["taxonomy"] or "")

    params = {
        "context": ident
        + f"{topic.topic}: {get_text_from_completion(topic.page_content)}",
        "taxonomy": taxonomy,
    }

    new_concepts: List[ParsedConcept] = []

    # print(f"{ident}: Finding concepts in")
    more_concepts: ParsedConceptList = await get_chain("concept_structured").ainvoke(
        params
    )
    if isinstance(more_concepts, AIMessage):
        print("Retry concept_structured once")
        more_concepts: ParsedConceptList = await get_chain(
            "concept_structured"
        ).ainvoke(params)

    max_repeat = 1
    repeat = 0
    try:
        while len(more_concepts.concepts) > 0:
            repeat += 1

            new_concepts += more_concepts.concepts
            if repeat > max_repeat:
                break
            params = {
                "context": params["context"],
                "taxonomy": taxonomy,
                "existing_concepts": "\n".join(
                    f"{concept.title.replace('\n', ' ').strip()}:\n"
                    f"Summary: {concept.summary.replace('\n', ' ').strip()}\n"
                    f"Taxonomy IDs: {', '.join(concept.taxonomy).replace('\n', ' ').strip()}\n"
                    f"Content:\n{concept.content.strip()}\n"
                    for concept in new_concepts
                ),
            }
            # print(f"{ident}: Finding more concepts after found: {len(more_concepts.concepts)} {repeat=}")
            more_concepts: ParsedConceptList = await get_chain("concept_more").ainvoke(
                params
            )
            if isinstance(more_concepts, AIMessage):
                print("Retry concept_more once")
                more_concepts: ParsedConceptList = await get_chain(
                    "concept_more"
                ).ainvoke(params)
    except Exception as e:
        print(e)

    global _new_ids
    global_new_ids = _new_ids[
        state["filename"] if "filename" in state else state["url"]
    ]
    new_ids = []
    cat_for_id = "-".join(state["categories"]).replace(" ", "-").lower()
    for concept in new_concepts:
        concept.id = get_unique_id(
            cat_for_id + "-" + concept.title,
            get_existing_concept_ids() + new_ids + global_new_ids,
        )
        new_ids.append(concept.id)
        concept.page_number = topic.page_number

    global_new_ids += new_ids
    params = {
        "existing_concepts": "\n".join(
            f"Id({concept.id}): {concept.title.replace('\n', ' ').strip()}\n"
            f"Summary: {concept.summary.replace('\n', ' ').strip()}\n"
            f"Taxonomy IDs: {', '.join(concept.taxonomy).replace('\n', ' ').strip()}\n"
            f"Content:\n{concept.content.strip()}\n"
            for concept in new_concepts
        )
    }
    # print(f"{ident}: Finding unique concepts in")
    unique_concept_ids: ParsedUniqueConceptList = await get_chain(
        "concept_unique"
    ).ainvoke(params)
    if isinstance(unique_concept_ids, AIMessage):
        print("Retry concept_unique once")
        unique_concept_ids: ParsedUniqueConceptList = await get_chain(
            "concept_unique"
        ).ainvoke(params)
    global _divider

    concepts: Dict[str, ParsedConcept] = {}
    for combined_concept in unique_concept_ids.concepts:
        if combined_concept.id in concepts.keys():
            continue
        for concept in new_concepts:
            if concept.id == combined_concept.id:
                concept.summary = combined_concept.summary.replace("\n", " ").strip()
                concept.title = combined_concept.title.replace("\n", " ").strip()
                concepts[combined_concept.id] = concept

    for combined_concept in unique_concept_ids.concepts:
        for concept in new_concepts:
            if (
                combined_concept.id in concepts.keys()
                and combined_concept.combined_ids is not None
                and concept.id in combined_concept.combined_ids
            ):
                concepts[combined_concept.id].content += (
                    "\n" + _divider + "\n" + concept.content.strip()
                )
                concepts[combined_concept.id].taxonomy = list(
                    set(concepts[combined_concept.id].taxonomy + concept.taxonomy)
                )

    return {"found_concepts": list(concepts.values())}


async def combine_concepts(state: FindConceptsState, config: RunnableConfig):
    found_concepts = state["found_concepts"]
    new_concepts = []
    for item in found_concepts:
        if isinstance(item, ParsedConcept):
            new_concepts.append(item)
        if isinstance(item, List):
            new_concepts.extend(item)

    return {"search_concepts_complete": True, "new_concepts": new_concepts}


# def unwrap_concept_hierarchy(
#     structure: Union[
#         ParsedConceptStructure, List[ParsedConceptStructure], ParsedConceptStructureList
#     ]
# ) -> Dict[str, List]:
#     concept_children: Dict[str, List[str]] = {}
#     if isinstance(structure, ParsedConceptStructureList):
#         structure = structure.structure

#     if isinstance(structure, ParsedConceptStructure) and structure.children is not None and len(structure.children) > 0:
#         if structure.id not in concept_children.keys():
#             concept_children[structure.id] = []
#         for child in structure.children:
#             concept_children[structure.id].append(child.id)
#             concept_children.update(unwrap_concept_hierarchy(child))

#     return concept_children


def unwrap_concept_hierarchy(
    structure: ParsedConceptStructureList,
) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}

    def traverse(node: ParsedConceptStructure, parent_id=None):
        if parent_id:
            if parent_id not in result:
                result[parent_id] = []
            result[parent_id].append(node.id)

        for child in node.children:
            traverse(child, node.id)

    for node in structure.structure:
        traverse(node)

    return result


async def collapse_concepts(state: FindConceptsState, config: RunnableConfig):
    concepts: List[ConceptData] = []  # state["concepts"] if "concepts" in state else []
    all_concept_ids = get_existing_concept_ids(refresh=True)
    found_concepts: List[ParsedConcept] = state["new_concepts"]
    source = state["filename"] if "filename" in state else state["url"]

    if found_concepts is not None and len(found_concepts) > 0:
        found_concepts_by_id: Dict[str, ParsedConcept] = {
            concept.id: concept for concept in found_concepts
        }

        existing_taxonomy_items: List[ConceptTaxonomy] = get_taxonomy_item_list(
            reset=True, categories=state["categories"]
        )
        existing_taxonomy_by_id: Dict[str, ConceptTaxonomy] = {
            concept.id: concept for concept in existing_taxonomy_items
        }

        existing_taxonomy = (
            [convert_concept_category_tag_to_dict(v) for v in existing_taxonomy_items]
            if 0 < len(existing_taxonomy_items)
            else []
        )

        new_concept_taxonomy: List[Dict] = state["taxonomy"]
        previous_concept_data: List[ConceptDataTable] = get_db_concepts(source=source)
        previous_concepts: Dict[str, ConceptData] = {}
        if len(previous_concept_data) > 0:
            for concept in previous_concept_data:
                previous_concepts[concept.id] = ConceptData(**concept.concept_contents)

        get_unique_concepts = lambda: get_chain("concept_unique").ainvoke(
            {
                "existing_concepts": "Previously defined concepts:\n"
                + "\n".join(
                    f"Id({concept.id}): {concept.title}\nSummary: {concept.summary}\nTaxonomy IDs: {', '.join(concept.taxonomy)}"
                    for concept in previous_concepts.values()
                )
                + "\n\nNewly defined concepts:\n"
                + "\n".join(
                    f"Id({concept.id}): {concept.title}\nSummary: {concept.summary}\nTaxonomy IDs: {', '.join(concept.taxonomy)}"
                    for concept in found_concepts
                )
            }
        )

        # print(f"Found {len(found_concepts)} concepts, optimizing for unique concepts.")
        unique_concepts: ParsedUniqueConceptList = await get_unique_concepts()
        if isinstance(unique_concepts, AIMessage):
            print("\n\nRetrying get unique concepts...")
            unique_concepts = await get_unique_concepts()
        # await get_chain(
        #     "concept_unique"
        # ).ainvoke(
        #     {
        #         "existing_concepts": "Previously defined concepts:\n"
        #         + "\n".join(
        #             f"Id({concept.id}): {concept.title}\nSummary: {concept.summary}\nTaxonomy IDs: {', '.join(concept.taxonomy)}"
        #             for concept in previous_concepts.values()
        #         )
        #         + "\n\nNewly defined concepts:\n"
        #         + "\n".join(
        #             f"Id({concept.id}): {concept.title}\nSummary: {concept.summary}\nTaxonomy IDs: {', '.join(concept.taxonomy)}"
        #             for concept in found_concepts
        #         )
        #     }
        # )

        pretty_print(unique_concepts, "New unique concepts")

        concept_hierarchy_task = lambda: get_chain("concept_hierarchy").ainvoke(
            {
                "existing_concepts": "\n".join(
                    f"Id({concept.id}): {concept.title}\n{concept.summary}"
                    for concept in unique_concepts.concepts
                )
            }
        )

        concept_taxonomy_task = lambda: get_chain(
            "concept_taxonomy_structured"
        ).ainvoke(
            {
                "existing_taxonomy": "\n".join(
                    convert_taxonomy_dict_to_tag_simple_structure_string(item)
                    for item in existing_taxonomy
                ),
                "new_taxonomy": "\n".join(
                    convert_taxonomy_dict_to_tag_simple_structure_string(item)
                    for item in new_concept_taxonomy
                ),
                "concepts": "\n".join(
                    f"Id({concept.id}): {concept.title}\nSummary: {concept.summary}\nTaxonomy IDs: {', '.join(concept.taxonomy)}"
                    for concept in unique_concepts.concepts
                ),
            }
        )

        # : ParsedConceptStructure
        # concept_hierarchy_task = get_chain("concept_hierarchy").ainvoke(
        #     {
        #         "existing_concepts": "\n".join(
        #             f"Id({concept.id}): {concept.title}\n{concept.summary}"
        #             for concept in unique_concepts.concepts
        #         )
        #     }
        # )
        # # : ParsedConceptTaxonomyList
        # concept_taxonomy_task = get_chain("concept_taxonomy_structured").ainvoke(
        #     {
        #         "existing_taxonomy": "\n".join(
        #             convert_taxonomy_dict_to_tag_structure_string(item)
        #             for item in existing_taxonomy
        #         ),
        #         "new_taxonomy": "\n".join(
        #             convert_taxonomy_dict_to_tag_structure_string(item)
        #             for item in new_concept_taxonomy
        #         ),
        #         "concepts": "\n".join(
        #             f"Id({concept.id}): {concept.title}\nSummary: {concept.summary}\nTaxonomy IDs: {', '.join(concept.taxonomy)}"
        #             for concept in unique_concepts.concepts
        #         ),
        #     }
        # )

        concept_taxonomy: ParsedConceptTaxonomyList
        concept_hierarchy: ParsedConceptStructureList
        concept_taxonomy, concept_hierarchy = await asyncio.gather(
            concept_taxonomy_task(), concept_hierarchy_task()
        )
        global _divider

        # Retry once if fail.
        if isinstance(concept_taxonomy, AIMessage):
            print("\n\nRetrying concept taxonomy...")
            concept_taxonomy = await concept_taxonomy_task()

        if isinstance(concept_hierarchy, AIMessage):
            print("\n\nRetrying concept hierarchy...")
            concept_hierarchy = await concept_hierarchy_task()

        pretty_print(concept_taxonomy, "Concept taxonomy")
        pretty_print(concept_hierarchy, "Concept hierarchy")

        unwrapped_hierarchy = unwrap_concept_hierarchy(concept_hierarchy)
        inverted_hierarchy: Dict[str, str] = {}
        for key, value in unwrapped_hierarchy.items():
            for item in value:
                inverted_hierarchy[item] = key

        # pretty_print(unwrapped_hierarchy, "Unwrapped hierarchy")
        # pretty_print(inverted_hierarchy, "Inverted hierarchy")

        existing_concept_category_ids = list(existing_taxonomy_by_id.keys())
        new_category_ids = []
        cat_for_id = "-".join(state["categories"]).replace(" ", "-").lower()
        all_new_ids = list(
            set([category.id.strip() for category in concept_taxonomy.taxonomy])
        )
        all_possible_ids = list(set(existing_concept_category_ids + new_category_ids))
        for category in concept_taxonomy.taxonomy:
            category.id = (
                category.id
                or get_unique_id(
                    cat_for_id + "-" + category.taxonomy,
                    list(existing_concept_category_ids) + new_category_ids,
                )
            ).strip()
            new_id = category.id
            changes = False

            if new_id in existing_concept_category_ids:
                changes = (
                    changes
                    or existing_taxonomy_by_id[new_id].tag != category.tag
                    or existing_taxonomy_by_id[new_id].title != category.title
                    or existing_taxonomy_by_id[new_id].description
                    != category.description
                    or existing_taxonomy_by_id[new_id].taxonomy != category.taxonomy
                    or existing_taxonomy_by_id[new_id].parent_taxonomy
                    != category.parent_taxonomy
                )
                existing_taxonomy_by_id[new_id].tag = category.tag.strip()
                existing_taxonomy_by_id[new_id].title = category.title.replace(
                    "\n", " "
                ).strip()
                existing_taxonomy_by_id[new_id].description = (
                    category.description.replace("\n", " ").strip()
                )
                existing_taxonomy_by_id[new_id].taxonomy = category.taxonomy.strip()
                existing_taxonomy_by_id[new_id].parent_taxonomy = (
                    (category.parent_taxonomy.strip())
                    if category.parent_taxonomy
                    else None
                )
            else:
                new_category_ids.append(new_id)
                parent_id = (
                    category.parent_id.strip()
                    if category.parent_id is not None
                    and category.parent_id.strip() in all_possible_ids
                    else None
                )
                changes = True
                existing_taxonomy_by_id[new_id] = ConceptTaxonomy(
                    id=new_id,
                    parent_id=parent_id,
                    title=category.title.replace("\n", " ").strip(),
                    tag=category.tag.strip(),
                    description=category.description.replace("\n", " ").strip(),
                    taxonomy=category.taxonomy.strip(),
                    parent_taxonomy=(
                        category.parent_taxonomy.strip()
                        if category.parent_taxonomy
                        else None
                    ),
                    type=category.type.strip(),
                )

            if category.parent_id:
                parent_id = (
                    category.parent_id.strip()
                    if category.parent_id is not None
                    and category.parent_id.strip() in all_possible_ids
                    else existing_taxonomy_by_id[new_id].parent_id
                )
                changes = (
                    changes or existing_taxonomy_by_id[new_id].parent_id != parent_id
                )
                existing_taxonomy_by_id[new_id].parent_id = parent_id

            if changes:
                pretty_print(existing_taxonomy_by_id[new_id], "Updated category")
                # print(f"\n\n\nNew/Updated category tag:\n\n{existing_taxonomy_by_id[new_id].model_dump_json(indent=4)}")
                update_concept_category(
                    existing_taxonomy_by_id[new_id], categories=state["categories"]
                )

        removed_ids = []

        all_ids = list(
            set(
                [concept.id for concept in unique_concepts.concepts]
                + list(previous_concepts.keys())
            )
        )

        all_taxonomy_ids = list(existing_taxonomy_by_id.keys())

        # Create new concepts
        for sum_concept in unique_concepts.concepts:
            id = sum_concept.id
            old_concept_item = False
            new_concept: ConceptData
            if id is None or id not in all_concept_ids:
                concept = (
                    found_concepts_by_id[id]
                    if id is not None and id in found_concepts_by_id.keys()
                    else sum_concept
                )
                new_id = concept.id or cat_for_id + "-" + get_unique_id(
                    concept.title, all_ids
                )
                new_concept = ConceptData(
                    id=new_id,
                    parent_id=(
                        inverted_hierarchy[new_id]
                        if new_id in inverted_hierarchy.keys()
                        else None
                    ),
                    title=sum_concept.title,
                    summary=sum_concept.summary,
                    contents=(
                        [concept.summary]
                        if isinstance(concept, ParsedConceptIds)
                        else concept.content.split(_divider)
                    ),
                    references=[
                        SourceReference(
                            source=source,
                            page_number=(
                                1
                                if isinstance(concept, ParsedConceptIds)
                                else concept.page_number
                            ),
                        )
                    ],
                    taxonomy=[],
                    sources=[source],
                    children=[
                        child_id
                        for child_id in unwrapped_hierarchy.get(id, [])
                        if child_id in all_ids
                    ],
                )
            elif id in found_concepts_by_id.keys():
                concept = found_concepts_by_id[id]
                new_concept = previous_concepts[id]
                new_concept.contents.extend(concept.content.split(_divider))
                new_concept.children = list(
                    set(
                        new_concept.children
                        + [
                            child_id
                            for child_id in unwrapped_hierarchy.get(id, [])
                            if child_id in all_ids
                        ]
                    )
                )
                new_concept.taxonomy = [
                    taxonomy_id
                    for taxonomy_id in list(
                        set(new_concept.taxonomy + sum_concept.taxonomy)
                    )
                    if taxonomy_id in all_taxonomy_ids
                ]

                references = [
                    ref for ref in new_concept.references if ref.source == source
                ]
                if concept.page_number not in [ref.page_number for ref in references]:
                    new_concept.references.append(
                        SourceReference(
                            source=source,
                            page_number=concept.page_number,
                        )
                    )
            elif id in previous_concepts.keys():
                old_concept_item = True
                new_concept = previous_concepts[id]

            if sum_concept.combined_ids is not None:
                for id in sum_concept.combined_ids:
                    if old_concept_item:
                        removed_ids.append(new_concept.id)
                    if id in previous_concepts.keys():
                        removed_ids.append(id)
                        new_concept.contents.extend(previous_concepts[id].contents)
                        for ref in previous_concepts[id].references:
                            if f"{ref.source}_{(ref.page_number or 0)}" not in [
                                f"{ref.source}_{(ref.page_number or 0)}"
                                for ref in new_concept.references
                            ]:
                                new_concept.references.append(ref)
                        new_concept.sources = list(
                            set(new_concept.sources + previous_concepts[id].sources)
                        )
                        new_concept.children.append(previous_concepts[id].id)
                        new_concept.parent_id = (
                            new_concept.parent_id or previous_concepts[id].parent_id
                        )
                    if id in found_concepts_by_id.keys():
                        concept = found_concepts_by_id[id]
                        new_concept.contents.extend(concept.content.split(_divider))
                        references = [
                            ref
                            for ref in new_concept.references
                            if ref.source == source
                        ]
                        if concept.page_number not in [
                            ref.page_number for ref in references
                        ]:
                            new_concept.references.append(
                                SourceReference(
                                    source=source,
                                    page_number=concept.page_number,
                                )
                            )

            for taxonomy_item in concept_taxonomy.taxonomy:
                if new_concept.id in taxonomy_item.connected_concepts:
                    if taxonomy_item.id not in new_concept.taxonomy:
                        new_concept.taxonomy.append(taxonomy_item.id)

            concepts.append(new_concept)
        for removed_id in removed_ids:
            if removed_id in previous_concepts.keys():
                del previous_concepts[removed_id]
                delete_db_concept(removed_id, commit=False)

        db_commit()

    return {
        "concept_complete": True,
        "collapse_concepts_complete": True,
        "collected_concepts": concepts,
    }


async def collapse_concept(i: int, concepts: List[ConceptData], concept: ConceptData):
    joined = "\n".join(concept.contents)

    if len(concept.contents) > 1:
        reformat = get_text_from_completion(
            await get_chain("text_formatter_simple").ainvoke({"context": joined})
        )
        concept.contents = [reformat]
        joined = reformat

    if len(joined) < 200:
        concept.summary = concept.summary or joined
    else:
        contents = split_text(joined)
        if len(concept.contents) > 1 or concept.summary is None:
            summary: TitledSummary = (
                await get_chain("summary_with_title").ainvoke({"context": joined})
                if len(contents) == 1
                else await get_chain("summary_documents_with_title").ainvoke(
                    {"context": contents}
                )
            )
            concept.title = summary.title
            concept.summary = summary.summary

    concepts[i] = concept


async def should_rag(state: FindConceptsState, config: RunnableConfig):
    if config["configurable"]["update_rag"]:
        return "rag_concepts"
    else:
        return END


async def finalize_concepts(state: FindConceptsState, config: RunnableConfig):
    collected_concepts = state["collected_concepts"]
    tasks = []
    for i, concept in enumerate(collected_concepts):
        tasks.append(collapse_concept(i, collected_concepts, concept))

    await asyncio.gather(*tasks)

    return {
        "collected_concepts": collected_concepts,
        "finalize_concepts_complete": True,
    }


async def rag_concepts(state: FindConceptsState, config: RunnableConfig):
    update_db_concept_rag(
        state["filename"] or state["url"],
        state["categories"],
        concepts=state["collected_concepts"] if "collected_concepts" in state else None,
    )

    return {"rag_complete": True}


find_concepts_graph = StateGraph(FindConceptsState, FindConceptsConfig)
find_concepts_graph.add_node("get_source_content", get_source_content)
# find_concepts_graph.add_node("split_reformatted_content", split_reformatted_content)
find_concepts_graph.add_node("search_for_taxonomy", search_for_taxonomy)
find_concepts_graph.add_node("search_concepts", search_concepts)
find_concepts_graph.add_node("combine_concepts", combine_concepts)
find_concepts_graph.add_node("collapse_concepts", collapse_concepts)
find_concepts_graph.add_node("finalize_concepts", finalize_concepts)
find_concepts_graph.add_node("rag_concepts", rag_concepts)


find_concepts_graph.add_edge(START, "get_source_content")
# find_concepts_graph.add_edge("get_source_content", "split_reformatted_content")
# find_concepts_graph.add_edge("split_reformatted_content", "search_for_taxonomy")
find_concepts_graph.add_edge("get_source_content", "search_for_taxonomy")
find_concepts_graph.add_conditional_edges(
    "search_for_taxonomy", map_search_concepts, ["search_concepts"]
)

find_concepts_graph.add_edge("search_concepts", "combine_concepts")
find_concepts_graph.add_edge("combine_concepts", "collapse_concepts")
find_concepts_graph.add_edge("collapse_concepts", "finalize_concepts")
find_concepts_graph.add_conditional_edges("finalize_concepts", should_rag)
find_concepts_graph.add_edge("rag_concepts", END)

find_concepts = find_concepts_graph.compile()
