from functools import cache
import json
import os
from enum import Enum


from lib.load_env import SETTINGS

journey_template_dir = os.path.join(
    SETTINGS.file_repository_path, "journey_structures_json"
)


## Symbols from https://www.w3schools.com/charsets/
class ActionSymbol(Enum):
    selected = "&#9673;"
    unselected = "&#9678;"
    up_down = "&#8597;"
    below = "&#8615;"
    image = "&#9968;"
    edit = "&#9881;"
    open = "&#9660;"
    closed = "&#9654;"
    add = "&plus;"
    remove = "&minus;"
    delete = "&#9747;"
    duplicate = "&#x2398;"

@cache
def get_available_journey_template_roles(as_str:bool=False) -> dict[list] | str:
    global journey_template_dir
    with open(
        os.path.join(journey_template_dir, "knowledge_services_roles.json"), "r"
    ) as f:
        roles: dict = json.load(f)

    filtered_roles = {}
    filtered_str = ''
    mapping = get_journey_template_mapping()
    for category in roles:
        for role in roles[category]:
            cat_id, role_id = match_title_to_cat_and_id(role)
            if role_id in mapping.keys():
                if filtered_roles.get(category) is None:
                    filtered_str += f'\nCategory: {category}\n'
                    filtered_roles[category] = []
                filtered_roles[category].append(role)
                filtered_str += f'Role: {role}\n'

    if as_str:
        return filtered_str

    return filtered_roles


@cache
def get_journey_template_index() -> list:
    global journey_template_dir
    with open(os.path.join(journey_template_dir, "structured/index.json"), "r") as f:
        return json.load(f)


@cache
def get_journey_template_mapping() -> dict:
    global journey_template_dir
    with open(os.path.join(journey_template_dir, "structured/mappings.json"), "r") as f:
        return json.load(f)


def match_title_to_cat_and_id(title: str) -> tuple[str, str]:
    for cat in get_journey_template_index():
        for item in cat["children"]:
            if item["title"] == title:
                return cat["key"], item["key"]
    return None


@cache
def load_journey_template(item_id: str) -> dict:
    global journey_template_dir
    filepath = get_journey_template_mapping().get(item_id)
    if filepath:
        with open(
            os.path.join(journey_template_dir, "structured", f"{filepath}"), "r"
        ) as f:
            return json.load(f)
