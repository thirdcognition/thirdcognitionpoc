import asyncio
from typing import Callable, Dict, List, Optional
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from lib.chains.init import get_chain
from lib.helpers import pretty_print
from lib.load_env import SETTINGS


hierarchy_task = lambda chain_name, item, item_formatter: get_chain(chain_name).ainvoke(
    {
        "hierarchy_items": item if item_formatter is None else item_formatter(item),
    }
)

join_hierarchy_task = lambda chain_name, item, item_formatter: get_chain(
    chain_name
).ainvoke(
    {
        "joined_items": item if item_formatter is None else item_formatter(item),
    }
)


class HierarchyNode(BaseModel):
    id: str
    joined: Optional[List[str]] = None
    children: Optional[List["HierarchyNode"]] = None


def unwrap_hierarchy(
    structure: List, valid_ids: List[str] = None
) -> Dict[str, List[HierarchyNode]]:
    result: Dict[str, List[HierarchyNode]] = {}

    def traverse(node, parent_id=None):
        if isinstance(node, dict):
            node_id = node.get("id")
            children = node.get("children", [])
            joined = node.get("joined", [])
        else:
            node_id = getattr(node, "id", None)
            children = getattr(node, "children", [])
            joined = getattr(node, "joined", [])

        hierarchy_children = []
        for child in children:
            if valid_ids is None or node_id in valid_ids:
                hierarchy_children.append(traverse(child, node_id))

        if len(joined) > 0 and node_id not in joined:
            joined.append(node_id)

        if len(joined) > 1 and valid_ids is not None:
            joined = [item for item in joined if item in valid_ids]
            if len(joined) < 2:
                joined = []
        else:
            joined = []

        hierarchy_node = HierarchyNode(
            id=node_id, joined=joined, children=hierarchy_children
        )

        if parent_id:
            if parent_id not in result:
                result[parent_id] = []
            result[parent_id].append(hierarchy_node)

        return hierarchy_node

    for node in structure:
        traverse(node)

    return result


def flatten_hierarchy(
    structure: List, valid_ids: List[str] = None
) -> Dict[str, HierarchyNode]:
    result: Dict[str, HierarchyNode] = {}

    def traverse(node):
        if isinstance(node, dict):
            node_id = node.get("id")
            children = node.get("children", [])
            joined = node.get("joined", [])
        else:
            node_id = getattr(node, "id", None)
            children = getattr(node, "children", [])
            joined = getattr(node, "joined", [])

        if node_id in result:
            node_id = f"{node_id}_{len(result)}"

        if len(joined) > 0 and node_id not in joined:
            joined.append(node_id)

        if len(joined) > 1 and valid_ids is not None:
            joined = [item for item in joined if item in valid_ids]
            if len(joined) < 2:
                joined = []
        else:
            joined = []

        hierarchy_children = []
        for child in children:
            if valid_ids is None or node_id in valid_ids:
                hierarchy_children.append(traverse(child))

        hierarchy_node = HierarchyNode(
            id=node_id, joined=joined, children=hierarchy_children
        )
        result[node_id] = hierarchy_node
        return hierarchy_node

    for node in structure:
        traverse(node)

    return result


def join_items(items: List, item_formatter: Callable, splitter="\n\n") -> List[str]:
    item_strings = item_formatter(items)

    if len(splitter.join(item_strings)) > SETTINGS.default_llms.instruct.char_limit:
        new_item_strings = []
        current_str = ""
        items_left = len(item_strings)
        item_index = 0
        while items_left > 0:
            item_index += 1
            items_left -= 1
            try:
                item_str = item_strings[item_index]
                if (
                    len(current_str) + len(item_str)
                    > SETTINGS.default_llms.instruct.char_limit
                ):
                    new_item_strings.append(current_str)
                    current_str = ""
                current_str += item_str + splitter
            except IndexError:
                print(f"IndexError: {item_index} , {items_left} > {len(item_strings)}")
                continue
        new_item_strings.append(current_str)
        item_strings = new_item_strings
    else:
        item_strings = [splitter.join(item_strings)]

    return item_strings


async def execute_chain(chain_name, items, task_func: Callable, item_formatter: Callable = None):
    tasks = [
        task_func(chain_name, item, item_formatter) for i, item in enumerate(items)
    ]

    results = await asyncio.gather(*tasks)

    retry_tasks = {}
    for i, result in enumerate(results):
        if isinstance(result, AIMessage):
            print(f"\n\nRetrying hierarchy request {i}...")
            retry_tasks[i] = task_func(chain_name, items[i], item_formatter)

    if len(retry_tasks.keys()) > 0:
        retry_results = {}
        for retry_item in retry_tasks.keys():
            retry_results[retry_item] = await retry_tasks[retry_item]
        for i, result in enumerate(results):
            if i in retry_results.keys():
                results[i] = retry_results[i]

    return results


async def get_hierarchy(
    items: list,
    hierarchy_item_formatter: Callable,
    hierarchy_chain_name: str,
    valid_ids: List[str] = None,
    join_chain_name: str = None,
    join_item_formatter: Callable = None,
) -> tuple[Dict[str, list], Dict[str, str], list, list[str]]:

    if len(items) < 2:
        return {}, {}, [], []

    item_strings = join_items(items, hierarchy_item_formatter)

    results = await execute_chain(hierarchy_chain_name, item_strings, hierarchy_task)

    hierarchy = []
    for result in results:
        hierarchy.extend(getattr(result, "structure", []))

    flat_hierarchy = flatten_hierarchy(hierarchy, valid_ids)

    unwrapped_hierarchy = unwrap_hierarchy(hierarchy, valid_ids)
    inverted_hierarchy: Dict[str, str] = {}
    for key, value in unwrapped_hierarchy.items():
        for item in value:
            inverted_hierarchy[item.id] = key

    joined_ids = {}
    for key, value in flat_hierarchy.items():
        if 0 < len(value.joined):
            joined_ids[key] = value.joined

    items_by_id = {}
    for item in items:
        if isinstance(item, dict):
            item_id = item.get("id")
        else:
            item_id = getattr(item, "id", item)

        items_by_id[item_id] = item

    new_items: list = []
    removed_items: list[str] = []
    if join_item_formatter is not None and join_chain_name is not None:
        item_strings = []
        for key, ids in joined_ids.items():
            removed_items.extend(ids)
            joined_items = [items_by_id[id] for id in ids]
            item_strings.append(join_items(joined_items, join_item_formatter))

        new_items = await execute_chain(
            join_chain_name, item_strings, join_hierarchy_task
        )

    return unwrapped_hierarchy, inverted_hierarchy, new_items, removed_items
