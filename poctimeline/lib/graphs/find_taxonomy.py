import asyncio
import operator
from typing import Annotated, Dict, List, Set, TypedDict
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.constants import Send

from lib.chains.hierarchy_compiler import get_hierarchy
from lib.chains.init import get_chain
from lib.models.user import user_db_commit
from lib.db.taxonomy import (
    delete_db_taxonomy,
    get_taxonomy_item_list,
    handle_new_taxonomy_item,
    update_db_taxonomy,
)
from lib.helpers.shared import (
    get_id_str,
    get_specific_tag,
    pretty_print,
)
from lib.load_env import SETTINGS

from lib.models.taxonomy import (
    Taxonomy,
    TaxonomyDataTable,
    convert_taxonomy_to_tag_structure_string,
    convert_taxonomy_to_json_string,
)
from lib.models.topics import get_topic_str, split_topics


class FindTaxonomyState(TypedDict):
    categories: List[str]
    filename: str
    url: str
    source: str
    content_topics: List[Dict]
    instructions: str
    #generated
    new_taxonomy: Annotated[list[Taxonomy], operator.add]
    # combined_taxonomy: List[Taxonomy]
    results: List[TaxonomyDataTable]
    #state
    search_taxonomy_complete: bool = False
    combine_taxonomy_items_complete: bool = False


class FindTaxonomyConfig(TypedDict):
    instructions: str = None


class ProcessTaxonomyState(TypedDict):
    categories: List[str]
    existing_taxonomy_ids: List[str]
    existing_taxonomy: str
    topics: List[Dict]


_new_ids = {}
_divider = "±!s!±"


async def search_taxonomy(state: ProcessTaxonomyState, config: RunnableConfig):
    existing_taxonomy: str = state["existing_taxonomy"]
    topics: List[Dict] = state["topics"]
    taxonomy_ids = state["existing_taxonomy_ids"]

    new_taxonomy_items:List[TaxonomyDataTable] = []
    cat_for_id = get_id_str(state["categories"])

    content = get_topic_str(topics, as_json=False, as_tags=True)
    # "\n\n".join(
    #     [f"{topic["topic"]}: \n\n{topic["page_content"]}" for topic in topics]
    # )
    params = {
        "existing_taxonomy": existing_taxonomy,
        "context": content,
    }
    topic_taxonomy = await get_chain("taxonomy").ainvoke(params)
    if isinstance(topic_taxonomy, AIMessage):
        print("Retrying search taxonomy once")
        await asyncio.sleep(30)
        topic_taxonomy = await get_chain("taxonomy").ainvoke(params)
    found_new_taxonomy_items = get_specific_tag(
        topic_taxonomy["parsed"]["children"] or []
    )
    if len(found_new_taxonomy_items) > 0:
        for item in found_new_taxonomy_items:
            new_taxonomy = handle_new_taxonomy_item(item, taxonomy_ids, cat_for_id)
            if new_taxonomy:
                taxonomy_ids.append(new_taxonomy.id)
                new_taxonomy_items.append(new_taxonomy.to_taxonomy_data_table())

    return {
        "search_for_taxonomy_complete": True,
        "new_taxonomy": [new_taxonomy_items],
    }

async def map_search_taxonomy(state: FindTaxonomyState):
    user_db_commit()
    existing_taxonomy: List[TaxonomyDataTable] = get_taxonomy_item_list(categories=state["categories"], reset=True)
    existing_taxonomy_str = convert_taxonomy_to_tag_structure_string(existing_taxonomy)
    # "\n\n".join(
    #     [
    #         convert_taxonomy_to_tag_structure_string(
    #             convert_taxonomy_to_dict(v)
    #         )
    #         for v in existing_taxonomy
    #     ]
    #     if 0 < len(existing_taxonomy)
    #     else []
    # )
    existing_taxonomy_ids = [taxonomy_item.id for taxonomy_item in existing_taxonomy]
    # pretty_print(existing_taxonomy_str, "Taxonomy items")

    return [
        Send(
            "search_taxonomy",
            {
                "url": state["url"] if "url" in state else None,
                "filename": state["filename"] if "filename" in state else None,
                "categories": state["categories"],
                "existing_taxonomy": existing_taxonomy_str,
                "existing_taxonomy_ids": existing_taxonomy_ids,
                "topics": topics,
            },
        )
        for topics in split_topics(state["content_topics"])
    ]

def taxonomy_formatter(taxonomy: List[TaxonomyDataTable], show_description=False):
    return [
        convert_taxonomy_to_json_string(
            item, show_description=show_description
        )
        for item in taxonomy
    ]

async def combine_taxonomy_items(state: FindTaxonomyState):
    # cat_for_id = get_id_str(state["categories"])
    # existing_taxonomy, existing_taxonomy_ids = get_existing_taxonomy(
    #     state["categories"]
    # )

    # max_chars = SETTINGS.default_llms.instruct.char_limit // 2
    existing_taxonomy: List[TaxonomyDataTable] = get_taxonomy_item_list(categories=state["categories"])
    existing_taxonomy_ids = [taxonomy_item.id for taxonomy_item in existing_taxonomy]
    found_taxonomy:List[TaxonomyDataTable] = [
        taxonomy
        for taxonomy_items in state["new_taxonomy"]
        for taxonomy in taxonomy_items
    ]
    found_taxonomy_ids = [taxonomy_item.id for taxonomy_item in found_taxonomy]


    valid_ids = [taxonomy.id for taxonomy in existing_taxonomy + found_taxonomy]
    all_taxonomy_by_id = {taxonomy.id: taxonomy for taxonomy in existing_taxonomy + found_taxonomy}

    joined_taxonomy: List[Taxonomy]
    unwrapped_hierarchy, inverted_hierarchy, joined_taxonomy, removed_ids = (
        await get_hierarchy(
            existing_taxonomy + found_taxonomy,
            valid_ids=valid_ids,
            hierarchy_chain_name="taxonomy_hierarchy",
            hierarchy_item_formatter=lambda x: taxonomy_formatter(x, False),
            join_chain_name="taxonomy_combiner",
            join_item_formatter=lambda x: taxonomy_formatter(x, True),
        )
    )
    joined_ids = [taxonomy.id for taxonomy in joined_taxonomy]

    for removed_id in list(set(removed_ids + joined_ids)):
        if removed_id in all_taxonomy_by_id:
            del all_taxonomy_by_id[removed_id]
        if removed_id in existing_taxonomy_ids:
            delete_db_taxonomy(removed_id, False)
        if removed_id in found_taxonomy_ids:
            found_taxonomy = [taxonomy for taxonomy in found_taxonomy if taxonomy.id != removed_id]

    found_taxonomy_ids = [taxonomy.id for taxonomy in found_taxonomy]

    for taxonomy in joined_taxonomy:
        all_taxonomy_by_id[taxonomy.id] = taxonomy
        if taxonomy.id not in found_taxonomy_ids:
            found_taxonomy_ids.append(taxonomy.id)
            found_taxonomy.append(taxonomy.to_taxonomy_data_table())

    for taxonomy in all_taxonomy_by_id.values():
        if taxonomy.id in inverted_hierarchy.keys():
            taxonomy.parent_id = inverted_hierarchy[taxonomy.id]
            taxonomy.parent_taxonomy = all_taxonomy_by_id[taxonomy.parent_id].taxonomy if taxonomy.parent_id in all_taxonomy_by_id else None
        else:
            taxonomy.parent_id = None
            taxonomy.parent_taxonomy = None

        update_db_taxonomy(taxonomy, categories=state["categories"], commit=False)

    return {
        "search_taxonomy_complete": True,
        "combine_taxonomy_items_complete": True,
        "results": found_taxonomy,
        # "taxonomy_ids": found_taxonomy_ids,
    }

find_taxonomy_graph = StateGraph(FindTaxonomyState, FindTaxonomyConfig)
find_taxonomy_graph.add_node("search_taxonomy", search_taxonomy)
find_taxonomy_graph.add_node("combine_taxonomy_items", combine_taxonomy_items)

find_taxonomy_graph.add_conditional_edges(
    START, map_search_taxonomy, ["search_taxonomy"]
)

find_taxonomy_graph.add_edge("search_taxonomy", "combine_taxonomy_items")
find_taxonomy_graph.add_edge("combine_taxonomy_items", END)


find_taxonomy = find_taxonomy_graph.compile()
