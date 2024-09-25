import json
import pprint as pp
import re
from typing import Any, Callable, Dict, List, Optional, Union
from pydantic import BaseModel
import streamlit as st
import yaml
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnableSequence,
    RunnableWithMessageHistory,
)
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from lib.load_env import DEBUGMODE
from lib.db.sqlite import Base


def print_params(msg="", params=""):
    if DEBUGMODE:
        if msg:
            print(f"\n\n\n{msg}")
        if params:
            print(f"'\n\n{pp.pformat(params).replace('\\n', '\n')}\n\n")


def pretty_print(obj, msg=None, force=DEBUGMODE):
    if force:
        if msg:
            print(f"\n\n\n{msg}\n")
        else:
            print(f"\n\n\n{type(obj)}\n")
        if obj is None:
            print("obj = None")
        elif isinstance(obj, BaseModel):
            print(obj.model_dump_json(indent=2))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                print(f"\n{i}:\n")
                if isinstance(item, BaseModel):
                    print(item.model_dump_json(indent=2))
                    print("\n\n")
                else:
                    pp.pprint(item)
        else:
            pp.pprint(obj)
        print("\n\n")


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = {}

    if session_id not in st.session_state["chat_history"]:
        st.session_state["chat_history"][session_id] = ChatMessageHistory()
    return st.session_state["chat_history"][session_id]


def get_chain_with_history(chain_id: str, chain: RunnableSequence):
    if "history_chains" not in st.session_state:
        st.session_state["history_chains"] = {}
    if chain_id in st.session_state["history_chains"]:
        return st.session_state["history_chains"][chain_id]

    history_chain = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        output_messages_key="answer",
        history_messages_key="chat_history",
    )

    st.session_state["history_chains"][chain_id] = history_chain
    return history_chain


def read_and_load_yaml(file_path):
    with open(file_path, "r") as file:
        content = file.read().replace("\t", "    ")
        data = yaml.safe_load(content)
    return data


def get_text_from_completion(completion):
    completion_content = repr(completion)
    if isinstance(completion, List) and isinstance(completion[0], Document):
        completion_content = "\n\n".join(
            [doc.page_content.strip() for doc in completion]
        )
    elif isinstance(completion, tuple):
        if isinstance(completion[0], bool):
            completion_content = completion[1].strip()
        elif len(completion) == 2:
            completion_content = (
                f"<thinking> {completion[1].strip()} </thinking>"
                if len(completion[1].strip()) > 0
                else ""
            ) + f"{completion[0].strip()}"
        else:
            completion_content = completion[0].strip()
    elif isinstance(completion, BaseMessage):
        completion_content = completion.content.strip()
    elif isinstance(completion, Document):
        completion_content = completion.page_content
    elif isinstance(completion, BaseModel):
        completion_content = completion.model_dump_json()
    elif isinstance(completion, dict) and "content" in completion.keys():
        completion_content = str(completion["content"]).strip()
    elif isinstance(completion, str):
        completion_content = completion.strip()

    return completion_content

def get_number(page_number):
    if isinstance(page_number, (int, float)):
        return int(page_number)
    elif isinstance(page_number, str) and page_number.isdigit():
        return int(page_number)
    elif ", " in page_number:
        page_number = page_number.split(", ")[0]
        if page_number.isdigit():
            return int(page_number)
    return 0

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                items.extend(flatten_dict({str(i): item}, new_key).items())
        else:
            try:
                if not isinstance(v, (int, float, str, bool)):
                    v = json.dumps(v)
                items.append((new_key, v))
            except Exception as e:
                print(f"Error flattening key '{new_key}': {e}")
                continue
    return dict(items)

def combine_metadata(docs: List[Document]) -> Dict[str, Any]:
    combined_metadata = {k: v for k, v in docs[0].metadata.items()}
    for doc in docs[1:]:
        for k, v in doc.metadata.items():
            if k in combined_metadata and combined_metadata[k] is not None:
                if isinstance(combined_metadata[k], dict):
                    combined_metadata[k].update(v)
                elif isinstance(combined_metadata[k], list):
                    if isinstance(v, list):
                        combined_metadata[k].extend(v)
                    else:
                        combined_metadata[k].append(v)
                elif isinstance(combined_metadata[k], str):
                    combined_metadata[k] += f", {repr(v)}"
                else:
                    combined_metadata[k] = f"{repr(combined_metadata[k])}, {repr(v)}"
            else:
                combined_metadata[k] = v

    return combined_metadata


def get_specific_tag(items, tag="category_tag") -> List[dict]:
    found_items = []
    for item in items:
        if item["tag"] == tag:
            found_items.append(item)
        if 0 < len(item["children"]):
            found_items.extend(get_specific_tag(item["children"], tag))
    return found_items


def convert_tags_to_dict(input_dict, tags, output_tag="item"):
    output_dict = {output_tag: {}}

    for child in input_dict["children"]:
        if child["tag"] in tags:
            if child["tag"] in output_dict[output_tag]:
                if isinstance(output_dict[output_tag][child["tag"]], list):
                    output_dict[output_tag][child["tag"]].append(child["body"].strip())
                else:
                    output_dict[output_tag][child["tag"]] = [
                        output_dict[output_tag][child["tag"]],
                        child["body"].strip(),
                    ]
            else:
                output_dict[output_tag][child["tag"]] = child["body"].strip()

    return output_dict


def get_id_str(item):
    if isinstance(item, list):
        item = "-".join(item)
    if isinstance(item, dict):
        item = "-".join(item.values())

    item = re.sub(
        r"[\'\(\)\"]", "", item
    )  # remove single quotes, parentheses, and double quotes
    item = re.sub(
        r"[\n\t]+", " ", item
    )  # replace newline and tab characters with a space
    item = re.sub(r"\s+", " ", item)  # replace multiple whitespaces with a single space
    item = item.replace(" ", "-")  # replace spaces with hyphens
    item = re.sub(r"-+", "-", item)  # replace multiple hyphens with a single hyphen
    item = item.lower()  # convert to lowercase
    item = item.strip("-")  # remove leading and trailing hyphens
    item = item.strip()  # remove leading and trailing hyphens
    return item


def get_unique_id(id_str: str, existing_ids: List[str]):
    id = get_id_str(id_str)
    if id not in existing_ids:
        return id
    id_index = 0
    max = len(existing_ids) + 10
    while True and id_index < max:
        id_index += 1
        new_id = f"{id}_{id_index}"
        if new_id not in existing_ids:
            return new_id
    return f"{id}_{id_index}"


def get_item_str(
    items: Union[Any, List],
    as_json: bool = False,
    as_tags: bool = False,
    as_array: bool = False,
    key_names: List[str] = [
        "id",
    ],
    key_mapping: Dict[str, str] = {},
    select_keys: List[str] = None,
    item_str="item",
    one_liner=False,
    show_empty_keys=False,
) -> Union[str, List[str]]:
    if not isinstance(items, list):
        items = [items]

    ret_str = []
    for item in items:
        item_dict = {}
        if isinstance(item, str):
            ret_str.append(item)
            continue
        if isinstance(item, BaseModel):
            item = item.model_dump()
        if isinstance(item, Base):
            item = item.__dict__

        for key in key_names:
            if (select_keys is None or key in select_keys) and key in item.keys():
                value = item[key]
                if value is None and not show_empty_keys:
                    continue
                elif value is None and not as_json:
                    value = "None"

                if isinstance(value, Document):
                    value = value.page_content

                if not as_json and not isinstance(
                    value, (str, int, float, bool, type(None))
                ):
                    value = yaml.dump(value)

                if isinstance(value, str):
                    if one_liner:
                        value = value.replace("\n", " ")
                        value = re.sub(r"\s+", " ", value)
                    value = value.strip()

                item_dict[
                    key if key not in key_mapping.keys() else key_mapping[key]
                ] = value
        if as_json:
            item_str = json.dumps(item_dict, indent=2 if not one_liner else None)
        elif as_tags:
            item_str = ""
            for key in item_dict.keys():
                if isinstance(item_dict[key], str) and "\n" in item_dict[key]:
                    item_str += f"<{key}>\n{item_dict[key]}\n</{key}>\n"
                else:
                    item_str += f"<{key}>{item_dict[key]}</{key}> "
                    if not one_liner:
                        item_str += "\n"
        else:
            item_str = ""
            for key in item_dict.keys():
                if isinstance(item_dict[key], str) and "\n" in item_dict[key]:
                    item_str += str(key).capitalize() + ":\n" + item_dict[key] + "\n"
                else:
                    item_str += f"{str(key).capitalize()}: {item_dict[key]} "
                    if not one_liner:
                        item_str += "\n"

            item_str = item_str.strip()
        if as_tags:
            ret_str.append(
                (
                    ("<" + item_str + ">{}</" + item_str + ">")
                    if one_liner
                    else ("<" + item_str + ">\n{}\n</" + item_str + ">\n\n")
                ).format(item_str)
            )
        else:
            ret_str.append(item_str)

    if as_array:
        return ret_str
    if as_json:
        if len(items) == 1:
            return ret_str[0]
        else:
            return ("[{}]" if one_liner else "[\n{}\n]\n\n").format(", ".join(ret_str))
    if as_tags:
        if len(items) == 1:
            return ret_str[0]
        else:
            return (
                ("<" + item_str + "s>{}</" + item_str + "s>")
                if one_liner
                else ("<" + item_str + "s>\n{}\n</" + item_str + "s>\n\n")
            )

    if len(items) == 1:
        return ret_str[0]
    else:
        return (" " if one_liner else "\n\n").join(ret_str)
