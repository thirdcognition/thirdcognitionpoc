import asyncio
import operator
from typing import Annotated, Dict, List, Set, TypedDict
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.constants import Send

from lib.chains.init import get_chain
from lib.db.sqlite import db_commit
from lib.db.taxonomy import (
    delete_db_taxonomy,
    get_taxonomy_item_list,
    handle_new_taxonomy_item,
    update_db_taxonomy,
)
from lib.helpers import (
    get_id_str,
    get_specific_tag,
    get_unique_id,
    pretty_print,
)
from lib.load_env import SETTINGS

from lib.models.taxonomy import (
    Taxonomy,
    ParsedTaxonomy,
    ParsedTaxonomyList,
    convert_taxonomy_to_dict,
    convert_taxonomy_dict_to_taxonomy,
    convert_taxonomy_dict_to_tag_simple_structure_string,
)
from lib.models.source import split_topics


class FindTaxonomyState(TypedDict):
    categories: List[str]
    filename: str
    url: str
    source: str
    content_topics: List[Dict]
    instructions: str
    #generated
    new_taxonomy: Annotated[list, operator.add]
    combined_taxonomy: List[Dict]
    results: List[Taxonomy]
    #state
    taxonomy_complete: bool = False


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

    new_taxonomy_items = []
    cat_for_id = get_id_str(state["categories"])

    content = "\n\n".join(
        [f"{topic["topic"]}: \n\n{topic["page_content"]}" for topic in topics]
    )
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
                taxonomy_ids.append(new_taxonomy["category_tag"]["id"])
                new_taxonomy_items.append(new_taxonomy)

    return {
        "search_for_taxonomy_complete": True,
        "new_taxonomy": [new_taxonomy_items],
    }


def get_existing_taxonomy(categories: List[str]):
    taxonomy: List[Taxonomy] = get_taxonomy_item_list(categories=categories)
    existing_taxonomy = "\n\n".join(
        [
            convert_taxonomy_dict_to_tag_simple_structure_string(
                convert_taxonomy_to_dict(v)
            )
            for v in taxonomy
        ]
        if 0 < len(taxonomy)
        else []
    )
    existing_taxonomy_ids = [taxonomy_item.id for taxonomy_item in taxonomy]

    return existing_taxonomy, existing_taxonomy_ids


async def map_search_taxonomy(state: FindTaxonomyState):

    existing_taxonomy, existing_taxonomy_ids = get_existing_taxonomy(
        state["categories"]
    )
    pretty_print(existing_taxonomy, "Taxonomy items")

    return [
        Send(
            "search_taxonomy",
            {
                "url": state["url"] if "url" in state else None,
                "filename": state["filename"] if "filename" in state else None,
                "categories": state["categories"],
                "existing_taxonomy": existing_taxonomy,
                "existing_taxonomy_ids": existing_taxonomy_ids,
                "topics": topics,
            },
        )
        for topics in split_topics(state["content_topics"])
    ]


async def combine_taxonomy_items(state: FindTaxonomyState):
    cat_for_id = get_id_str(state["categories"])
    existing_taxonomy, existing_taxonomy_ids = get_existing_taxonomy(
        state["categories"]
    )

    max_chars = SETTINGS.default_llms.instruct.char_limit // 2
    found_taxonomy = [
        taxonomy
        for taxonomy_items in state["new_taxonomy"]
        for taxonomy in taxonomy_items
    ]
    nex_taxonomy_str = "\n".join(
        [
            convert_taxonomy_dict_to_tag_simple_structure_string(nex_taxonomy)
            for nex_taxonomy in found_taxonomy
        ]
    )
    new_taxonomy_items = []
    taxonomy_ids = [taxonomy_item for taxonomy_item in existing_taxonomy_ids]
    if len(nex_taxonomy_str) > max_chars and len(found_taxonomy) > 10:
        all_new_items = _divider.join(
            [
                convert_taxonomy_dict_to_tag_simple_structure_string(item, True)
                for item in found_taxonomy
            ]
        )

        params = {
            "existing_taxonomy": existing_taxonomy,
            "new_taxonomy_items": all_new_items.replace(_divider, "\n\n"),
        }
        topic_taxonomy = await get_chain("taxonomy_refine").ainvoke(params)
        if isinstance(topic_taxonomy, AIMessage):
            print("Retrying taxonomy_refine once")
            await asyncio.sleep(30)
            topic_taxonomy = await get_chain("taxonomy_refine").ainvoke(params)

        combined_taxonomy_items = get_specific_tag(
            topic_taxonomy["parsed"]["children"] or []
        )
        print(
            f"Reduced the taxonomy items ({len(found_taxonomy)}) by: {len(found_taxonomy) - len(combined_taxonomy_items)}"
        )

        if len(combined_taxonomy_items) > 0:
            for item in combined_taxonomy_items:
                new_taxonomy = handle_new_taxonomy_item(item, taxonomy_ids, cat_for_id)
                if new_taxonomy:
                    taxonomy_ids.append(new_taxonomy["category_tag"]["id"])
                    new_taxonomy_items.append(new_taxonomy)

    return {
        "combined_taxonomy": new_taxonomy_items or found_taxonomy,
        "taxonomy_ids": taxonomy_ids,
    }


async def handle_combining_taxonomy(
    new_taxonomy: Taxonomy,
    old_taxonomy: Taxonomy,
    categories: List[str],
    all_possible_taxonomy_ids: List,
):
    changes = (
        old_taxonomy.tag != new_taxonomy.tag
        or old_taxonomy.title != new_taxonomy.title
        or old_taxonomy.description != new_taxonomy.description
        or old_taxonomy.taxonomy != new_taxonomy.taxonomy
        or old_taxonomy.parent_taxonomy != new_taxonomy.parent_taxonomy
    )

    if not changes:
        return

    new_id = new_taxonomy.id
    parent_id = new_taxonomy.parent_id
    old_id = old_taxonomy.id
    cat_for_id = get_id_str(categories)
    params = {
        "existing_taxonomy": convert_taxonomy_dict_to_tag_simple_structure_string(
            convert_taxonomy_to_dict(old_taxonomy),
            show_description=True,
        ),
        "new_taxonomy_items": convert_taxonomy_dict_to_tag_simple_structure_string(
            convert_taxonomy_to_dict(new_taxonomy),
            show_description=True,
        ),
    }
    refined_taxonomy = await get_chain("taxonomy_refine").ainvoke(params)
    if isinstance(refined_taxonomy, AIMessage):
        print("Retrying taxonomy_refine once")
        await asyncio.sleep(30)
        refined_taxonomy = await get_chain("taxonomy_refine").ainvoke(params)

    combined_taxonomy_items = get_specific_tag(
        refined_taxonomy["parsed"]["children"] or []
    )
    new_taxonomy_items = []
    if len(combined_taxonomy_items) == 0:
        print("Combined taxonomy items is empty, deleting...")
        delete_db_taxonomy(old_id, commit=False)
    elif len(combined_taxonomy_items) > 1:

        delete_db_taxonomy(new_id, commit=False)
        all_possible_taxonomy_ids.remove(new_id)
        for concept_data in combined_taxonomy_items:
            new_taxonomy_dict = handle_new_taxonomy_item(
                concept_data, all_possible_taxonomy_ids, cat_for_id
            )
            new_taxonomy = convert_taxonomy_dict_to_taxonomy(new_taxonomy_dict)
            new_taxonomy_items.append(new_taxonomy)
            update_db_taxonomy(new_taxonomy, categories=categories, commit=False)
            all_possible_taxonomy_ids.append(new_taxonomy.id)
        pretty_print(new_taxonomy_items, "Combined taxonomy items", force=True)
    else:
        pretty_print(
            combined_taxonomy_items,
            "Combined into one taxonomy item",
        )

        new_taxonomy_dict = handle_new_taxonomy_item(
            combined_taxonomy_items[0],
            all_possible_taxonomy_ids,
            cat_for_id,
            existing_id=new_id,
            existing_parent_id=parent_id,
        )
        if new_taxonomy_dict["category_tag"]["id"] != new_id:
            delete_db_taxonomy(new_id, commit=False)
            all_possible_taxonomy_ids.remove(new_id)

        new_taxonomy = convert_taxonomy_dict_to_taxonomy(new_taxonomy_dict)
        update_db_taxonomy(new_taxonomy, categories=categories, commit=False)

    return new_taxonomy_items if len(combined_taxonomy_items) > 1 else [new_taxonomy]


async def taxonomy_task(existing_taxonomy, new_taxonomy):
    return await get_chain("taxonomy_structured").ainvoke(
        {
            "existing_taxonomy": "\n".join(
                [
                    convert_taxonomy_dict_to_tag_simple_structure_string(
                        item, show_description=True
                    )
                    for item in existing_taxonomy
                ]
            ),
            "new_taxonomy": "\n".join(
                [
                    convert_taxonomy_dict_to_tag_simple_structure_string(
                        item, show_description=True
                    )
                    for item in new_taxonomy
                ]
            ),
        }
    )


def parsed_to_taxonomy(
    taxonomy: ParsedTaxonomy, new_id: str = None, parent_id: str = None
):
    return Taxonomy(
        id=new_id or taxonomy.id,
        parent_id=parent_id or taxonomy.parent_id,
        title=taxonomy.title.replace("\n", " ").strip(),
        tag=taxonomy.tag.strip() if taxonomy.tag is not None else None,
        description=taxonomy.description.replace("\n", " ").strip(),
        taxonomy=taxonomy.taxonomy.strip(),
        parent_taxonomy=(
            taxonomy.parent_taxonomy.strip() if taxonomy.parent_taxonomy else None
        ),
        type=taxonomy.type.strip(),
    )


async def collapse_taxonomy(state: FindTaxonomyState, config: RunnableConfig):
    existing_taxonomy_items: List[Taxonomy] = get_taxonomy_item_list(
        reset=True, categories=state["categories"]
    )
    existing_taxonomy_by_id: Dict[str, Taxonomy] = {
        taxonomy.id: taxonomy for taxonomy in existing_taxonomy_items
    }

    existing_taxonomy = (
        [convert_taxonomy_to_dict(v) for v in existing_taxonomy_items]
        if 0 < len(existing_taxonomy_items)
        else []
    )

    new_taxonomy: List[Dict] = state["combined_taxonomy"]

    refined_taxonomy: ParsedTaxonomyList = await taxonomy_task(
        existing_taxonomy, new_taxonomy
    )
    if isinstance(refined_taxonomy, AIMessage):
        print("\n\nRetrying concept taxonomy...")
        refined_taxonomy = await taxonomy_task(existing_taxonomy, new_taxonomy)

    pretty_print(refined_taxonomy, "Concept taxonomy", force=True)

    new_taxonomy_ids = []

    existing_taxonomy_ids = list(existing_taxonomy_by_id.keys())
    all_possible_taxonomy_ids = list(set(existing_taxonomy_ids + new_taxonomy_ids))
    tasks = []
    cat_for_id = get_id_str(state["categories"])
    new_taxonomy_items = []
    for taxonomy in refined_taxonomy.taxonomy:
        taxonomy.id = (
            taxonomy.id
            or get_unique_id(
                cat_for_id + "-" + taxonomy.taxonomy,
                list(existing_taxonomy_ids) + new_taxonomy_ids,
            )
        ).strip()
        new_id = taxonomy.id
        changes = False
        parent_id = (
            taxonomy.parent_id.strip()
            if taxonomy.parent_id is not None
            and taxonomy.parent_id.strip() in all_possible_taxonomy_ids
            else (
                existing_taxonomy_by_id[new_id].parent_id
                if new_id in existing_taxonomy_by_id
                else None
            )
        )
        if parent_id and new_id in existing_taxonomy_by_id:
            changes = changes or existing_taxonomy_by_id[new_id].parent_id != parent_id

        new_taxonomy_item = False
        new_taxonomy = parsed_to_taxonomy(taxonomy, new_id=new_id, parent_id=parent_id)
        if new_id not in existing_taxonomy_ids:
            new_taxonomy_ids.append(new_id)
            new_taxonomy_items.append(new_taxonomy)
            all_possible_taxonomy_ids.append(new_id)

            changes = True
            new_taxonomy_item = True
            pretty_print(
                new_taxonomy,
                "Updated taxonomy" if not new_taxonomy_item else "New taxonomy",
            )
            # print(f"\n\n\nNew/Updated category tag:\n\n{existing_taxonomy_by_id[new_id].model_dump_json(indent=4)}")
            update_db_taxonomy(
                new_taxonomy,
                categories=state["categories"],
                commit=False,
            )
        else:
            tasks.append(
                handle_combining_taxonomy(
                    new_taxonomy,
                    existing_taxonomy_by_id[new_id],
                    state["categories"],
                    all_possible_taxonomy_ids,
                )
            )

    new_taxonomy_items.extend(await asyncio.gather(*tasks))

    db_commit()

    return {
        "results": new_taxonomy_items,
        "taxonomy_complete": True,
    }


find_taxonomy_graph = StateGraph(FindTaxonomyState, FindTaxonomyConfig)
find_taxonomy_graph.add_node("search_taxonomy", search_taxonomy)
find_taxonomy_graph.add_node("combine_taxonomy_items", combine_taxonomy_items)
find_taxonomy_graph.add_node("collapse_taxonomy", collapse_taxonomy)


# find_taxonomy_graph.add_edge(START, "search_taxonomy")
find_taxonomy_graph.add_conditional_edges(
    START, map_search_taxonomy, ["search_taxonomy"]
)

find_taxonomy_graph.add_edge("search_taxonomy", "combine_taxonomy_items")
find_taxonomy_graph.add_edge("combine_taxonomy_items", "collapse_taxonomy")
find_taxonomy_graph.add_edge("collapse_taxonomy", END)

find_taxonomy = find_taxonomy_graph.compile()
