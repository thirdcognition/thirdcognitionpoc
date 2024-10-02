import asyncio
import itertools
import operator
from typing import Annotated, Dict, List, TypedDict

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.constants import Send

from lib.chains.hierarchy_compiler import get_hierarchy
from lib.chains.init import get_chain
from lib.db.concept import (
    delete_db_concept,
    get_db_concepts,
    get_existing_concept_ids,
    update_db_concept,
)
from lib.models.user import user_db_commit
from lib.db.taxonomy import get_taxonomy_item_list
from lib.helpers.shared import (
    get_unique_id,
    pretty_print,
)

from lib.models.taxonomy import (
    TaxonomyDataTable,
    convert_taxonomy_to_json_string,
)
from lib.models.concepts import (
    ConceptData,
    ConceptDataTable,
    ConceptList,
    get_concept_str,
)

from lib.models.topics import get_topic_str, split_topics


class FindConceptsState(TypedDict):
    categories: List[str]
    filename: str
    url: str
    source: str
    content_topics: List[Dict]
    instructions: str
    taxonomy: Dict
    # Generated
    concepts: List[ConceptDataTable]
    found_concepts: Annotated[list, operator.add]
    new_concepts: List[ConceptDataTable]
    collected_concepts: List[ConceptDataTable]
    # semantic_contents: List[Document]
    search_concepts_complete: bool = False
    combine_concepts_complete: bool = False
    collapse_concepts_complete: bool = False


class FindConceptsConfig(TypedDict):
    instructions: str = None


class ProcessConceptsState(TypedDict):
    categories: List[str]
    filename: str
    url: str
    topics: List[Dict]
    taxonomy: List[TaxonomyDataTable]


_new_ids = {}


async def map_search_concepts(state: FindConceptsState):
    user_db_commit()
    taxonomy: List[TaxonomyDataTable] = get_taxonomy_item_list(
        categories=state["categories"], reset=True
    )

    if 0 == len(taxonomy):
        user_db_commit()
        taxonomy = get_taxonomy_item_list(
            categories=state["categories"], reset=True
        )
        if 0 == len(taxonomy):
            raise Exception("No taxonomy found")

    return [
        Send(
            "search_concepts",
            {
                "url": state["url"] if "url" in state else None,
                "filename": state["filename"] if "filename" in state else None,
                "categories": state["categories"],
                "topics": topics,
                "taxonomy": taxonomy,
            },
        )
        for topics in split_topics(state["content_topics"])
    ]

async def search_concepts(state: ProcessConceptsState, config: RunnableConfig):
    topics: List[Dict] = state["topics"]
    content = ""

    content = get_topic_str(topics, as_array=False)
    taxonomy = convert_taxonomy_to_json_string(state["taxonomy"])

    params = {
        "context": content,
        "taxonomy": taxonomy,
    }

    new_concepts: List[ConceptData] = []

    # print(f"{ident}: Finding concepts in")
    more_concepts: ConceptList = await get_chain("concept_structured").ainvoke(params)
    if isinstance(more_concepts, AIMessage):
        print("Retry concept_structured once")
        await asyncio.sleep(30)
        more_concepts: ConceptList = await get_chain("concept_structured").ainvoke(
            params
        )

    max_repeat = 1
    repeat = 0
    try:
        while len(more_concepts.concepts) > 0:
            repeat += 1

            new_concepts += more_concepts.concepts
            if repeat > max_repeat:
                break
            params = {
                "context": content,
                "taxonomy": taxonomy,
                "existing_concepts": get_concept_str(new_concepts),

            }
            more_concepts: ConceptList = await get_chain("concept_more").ainvoke(params)
            if isinstance(more_concepts, AIMessage):
                print("Retry concept_more once")
                await asyncio.sleep(30)
                more_concepts: ConceptList = await get_chain("concept_more").ainvoke(
                    params
                )
    except Exception as e:
        print(e)

    global _new_ids
    global_new_ids = _new_ids.get(
        state["filename"] if "filename" in state else state["url"], []
    )
    _new_ids[state["filename"] if "filename" in state else state["url"]] = (
        global_new_ids
    )
    new_ids = []
    cat_for_id = "-".join(state["categories"]).replace(" ", "-").lower()
    for concept in new_concepts:
        concept.id = get_unique_id(
            cat_for_id + "-" + concept.title,
            get_existing_concept_ids() + new_ids + global_new_ids,
        )
        new_ids.append(concept.id)

    global_new_ids += new_ids

    return {"found_concepts": [concept.to_concept_data_table() for concept in new_concepts]}


async def combine_concepts(state: FindConceptsState, config: RunnableConfig):
    found_concepts = state["found_concepts"]
    new_concepts = []
    for item in found_concepts:
        if isinstance(item, List):
            new_concepts.extend(item)
        elif item is not None:
            new_concepts.append(item)

    return {"search_concepts_complete": True, "combine_concepts_complete": True, "new_concepts": new_concepts}


async def collapse_concepts(state: FindConceptsState, config: RunnableConfig):
    new_concepts: List[ConceptDataTable] = state["new_concepts"]

    source = state["filename"] if "filename" in state else state["url"]

    existing_concept_data: List[ConceptDataTable] = get_db_concepts(source=source)
    existing_concepts_by_id: Dict[str, ConceptDataTable] = {}
    existing_concept_ids: List[str] = []
    existing_concepts = []
    if len(existing_concept_data) > 0:
        for concept in existing_concept_data:
            concept.taxonomy = sorted(concept.taxonomy)
            existing_concepts_by_id[concept.id] = concept
        existing_concept_ids = sorted(
            existing_concepts_by_id.keys(),
            key=lambda x: (
                "-".join(existing_concepts_by_id[x].taxonomy),
                existing_concepts_by_id[x].id,
            ),
        )
        existing_concepts = [existing_concepts_by_id[id] for id in existing_concept_ids]

    for concept in new_concepts:
        concept.taxonomy = sorted(concept.taxonomy)
    new_concepts = sorted(new_concepts, key=lambda x: ("-".join(x.taxonomy), x.id))
    new_concepts_by_id: Dict[str, ConceptDataTable] = {
        concept.id: concept for concept in new_concepts
    }
    new_concept_ids = sorted(
        new_concepts_by_id.keys(),
        key=lambda x: (
            "-".join(new_concepts_by_id[x].taxonomy),
            new_concepts_by_id[x].id,
        ),
    )

    all_concepts = existing_concepts + new_concepts
    all_concept_ids = [concept.id for concept in all_concepts]
    all_concepts_by_id = {concept.id: concept for concept in all_concepts}

    joined_concepts: List[ConceptData]
    unwrapped_hierarchy, inverted_hierarchy, joined_concepts, removed_ids = (
        await get_hierarchy(
            all_concepts,
            valid_ids=all_concept_ids,
            hierarchy_chain_name="concept_hierarchy",
            hierarchy_item_formatter=lambda x: get_concept_str(x, one_liner=True),
            join_chain_name="concept_combiner",
            join_item_formatter=lambda x: get_concept_str(x),
        )
    )

    joined_ids = [concept.id for concept in joined_concepts]

    for removed_id in list(set(removed_ids + joined_ids)):
        if removed_id in all_concepts_by_id:
            del all_concepts_by_id[removed_id]
        if removed_id in existing_concept_ids:
            delete_db_concept(removed_id, False)
        if removed_id in new_concept_ids:
            new_concepts = [
                concept for concept in new_concepts if concept.id != removed_id
            ]

        new_concept_ids = [concept.id for concept in new_concepts]

        for concept in joined_concepts:
            all_concepts_by_id[concept.id] = concept
            if concept.id not in new_concept_ids:
                new_concept_ids.append(concept.id)
                new_concepts.append(concept.to_concept_data_table())

        for concept in all_concepts_by_id.values():
            if concept.id in inverted_hierarchy.keys():
                concept.parent_id = inverted_hierarchy[concept.id]
            else:
                concept.parent_id = None

            update_db_concept(concept, categories=state["categories"], commit=False)

        user_db_commit()

    return {
        "collapse_concepts_complete": True,
        "collected_concepts": new_concepts,
    }


find_concepts_graph = StateGraph(FindConceptsState, FindConceptsConfig)
find_concepts_graph.add_node("search_concepts", search_concepts)
find_concepts_graph.add_node("combine_concepts", combine_concepts)
find_concepts_graph.add_node("collapse_concepts", collapse_concepts)
# find_concepts_graph.add_node("finalize_concepts", finalize_concepts)

find_concepts_graph.add_conditional_edges(
    START, map_search_concepts, ["search_concepts"]
)

find_concepts_graph.add_edge("search_concepts", "combine_concepts")
find_concepts_graph.add_edge("combine_concepts", "collapse_concepts")
find_concepts_graph.add_edge("collapse_concepts", END)
# find_concepts_graph.add_edge("collapse_concepts", "finalize_concepts")
# find_concepts_graph.add_edge("finalize_concepts", END)

find_concepts = find_concepts_graph.compile()
