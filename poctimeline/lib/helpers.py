import pprint as pp
from typing import Dict, List
from pydantic import BaseModel
import streamlit as st
import yaml
from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnableSequence,
    RunnableWithMessageHistory,
)
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from lib.load_env import DEBUGMODE

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


def parse_content_dict(data):
    result = []
    for item in data["children"]:
        if item["tag"] == "output":
            topic = next(
                (
                    child["body"]
                    for child in item["children"]
                    if child["tag"] == "topic"
                ),
                None,
            )
            content = item["body"]
            summary = next(
                (
                    child["body"]
                    for child in item["children"]
                    if child["tag"] == "summary"
                ),
                None,
            )
            result.append({"topic": topic, "content": content, "summary": summary})
        else:
            result.extend(parse_content_dict(item))
    return result


def prepare_contents(
    content: str, prev_page: str = "", next_page: str = "", max_length: int = 1000
):
    content = [content] if not isinstance(content, list) else content
    next_page = [next_page] if not isinstance(next_page, list) else next_page
    prev_page = [prev_page] if not isinstance(prev_page, list) else prev_page
    content = "\n".join(
        [
            ((item.page_content if isinstance(item, Document) else item))
            for item in content
        ]
    )
    next_page = "\n".join(
        [
            (item.page_content if isinstance(item, Document) else repr(item))
            for item in next_page
        ]
    )
    next_page = next_page[:max_length] if len(next_page) > max_length else next_page
    prev_page = "\n".join(
        [
            (item.page_content if isinstance(item, Document) else repr(item))
            for item in prev_page
        ]
    )
    prev_page = prev_page[-max_length:] if len(prev_page) > max_length else prev_page

    return (content, prev_page, next_page)


def get_specific_tag(items, tag="category_tag") -> List[dict]:
    found_items = []
    for item in items:
        if item["tag"] == tag:
            found_items.append(item)
        if 0 < len(item["children"]):
            found_items.extend(get_specific_tag(item["children"], tag))
    return found_items

def get_id_str(item):
    if isinstance(item, list):
        item = "-".join(item)
    if isinstance(item, dict):
        item = "-".join(item.values())
    return item.replace(" ", "-").lower()

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



def unwrap_hierarchy(
    structure: List,
) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}

    def traverse(node, parent_id=None):
        if isinstance(node, dict):
            node_id = node.get("id")
            children = node.get("children", [])
        else:
            node_id = getattr(node, "id", None)
            children = getattr(node, "children", [])

        if parent_id:
            if parent_id not in result:
                result[parent_id] = []
            result[parent_id].append(node_id)

        for child in children:
            traverse(child, node_id)

    for node in structure:
        traverse(node)

    return result



