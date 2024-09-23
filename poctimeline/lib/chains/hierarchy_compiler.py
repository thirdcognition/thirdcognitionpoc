import asyncio
from typing import Callable, Dict, List
from langchain_core.messages import AIMessage

from lib.chains.init import get_chain
from lib.helpers import unwrap_hierarchy
from lib.load_env import SETTINGS


hierarchy_task = lambda chain_name, item_str: get_chain(chain_name).ainvoke(
    {
        "hierarchy_items": item_str,
    }
)


async def get_hierarchy(
    items: list, get_item_str: Callable, chain_name: str, valid_ids: List[str] = None
) -> tuple[Dict[str, list], Dict[str, str], Dict[str, list]]:

    if len(items) < 2:
        return {}, {}, {}

    item_strings = get_item_str(items)

    if len("\n\n".join(item_strings)) > SETTINGS.default_llms.instruct.char_limit:
        new_item_strings = []
        current_str = ""
        items_left = len(items)
        item_index = 0
        while items_left > 0:
            item_index += 1
            items_left -= 1
            item_str = item_strings[item_index]
            if (
                len(current_str) + len(item_str)
                > SETTINGS.default_llms.instruct.char_limit
            ):
                new_item_strings.append(current_str)
                current_str = ""
            current_str += item_str + "\n\n"
        new_item_strings.append(current_str)
        item_strings = new_item_strings
    else:
        item_strings = ["\n\n".join(item_strings)]

    tasks = [
        hierarchy_task(chain_name, item_str)
        for i, item_str in enumerate(item_strings)
    ]

    results = await asyncio.gather(*tasks)

    retry_tasks = {}
    for i, result in enumerate(results):
        if isinstance(result, AIMessage):
            print(f"\n\nRetrying hierarchy request {i}...")
            retry_tasks[i] = hierarchy_task(chain_name, item_strings[i])

    if len(retry_tasks.keys()) > 0:
        retry_results = {}
        for retry_item in retry_tasks.keys():
            retry_results[retry_item] = await retry_tasks[retry_item]
        for i, result in enumerate(results):
            if i in retry_results.keys():
                results[i] = retry_results[i]

    hierarchy = []
    for result in results:
        hierarchy.extend(getattr(result, "structure", []))

    unwrapped_hierarchy = unwrap_hierarchy(hierarchy, valid_ids)
    inverted_hierarchy: Dict[str, str] = {}
    for key, value in unwrapped_hierarchy.items():
        for item in value:
            inverted_hierarchy[item] = key

    flat_hierarchy_set: Dict[str, set] = {}
    max_depth = 100
    for key, value in inverted_hierarchy.items():
        print("parent", value, "child", key)
        max_depth_step = 0
        while value in inverted_hierarchy and max_depth_step < max_depth:
            print("parent", value, "child", key)
            max_depth_step += 1
            value = inverted_hierarchy[value]
        if value not in flat_hierarchy_set:
            flat_hierarchy_set[value] = set()
        flat_hierarchy_set[value].add(key)

    flat_hierarchy = {key: list(value) for key, value in flat_hierarchy_set.items()}

    return unwrapped_hierarchy, inverted_hierarchy, flat_hierarchy
