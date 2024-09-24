import pprint as pp
import re
from typing import Callable, Dict, List, Optional, Union
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
from lib.models.source import ParsedTopic


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
            instruct = next(
                (
                    child["body"]
                    for child in item["children"]
                    if child["tag"] == "instruct"
                ),
                None,
            )
            id = next(
                (child["body"] for child in item["children"] if child["tag"] == "id"),
                None,
            )
            result.append(
                {
                    "topic": topic,
                    "content": content,
                    "summary": summary,
                    "id": id,
                    "instruct": instruct,
                }
            )
        else:
            result.extend(parse_content_dict(item))
    return result


def create_doc_from_list_with_metadata(
    content: list,
    max_length=None,
    crop_length: int = None,
    crop_from_start: bool = True,
    format_callback: Callable = None,
) -> List[Document]:
    if isinstance(content[0], Document):
        metadata = combine_metadata(content)
        for key in metadata.keys():
            if isinstance(metadata[key], str):
                metadata[key] = ", ".join(set(metadata[key].split(", ")))
    elif isinstance(content, Document):
        content = [content]
        metadata = content[0].metadata
    else:
        metadata = {}
    results: List[Document] = None
    result: str = None

    if max_length is not None and crop_length is None:
        cur_str = ""
        results = []
        ith = 0
        for item in content:
            item_str = (
                format_callback(item)
                if format_callback is not None
                else (
                    f"{(item.metadata['title']+':\n') if 'title' in item.metadata else ''}{item.page_content}"
                    if isinstance(item, Document)
                    else repr(item)
                )
            )
            if len(cur_str) + len(item_str) > max_length:
                results.append(
                    Document(
                        page_content=cur_str, metadata={**metadata, "cut_index": ith}
                    )
                )
                cur_str = ""
                ith += 1
            cur_str += item_str + "\n"
        if len(cur_str) > 0:
            results.append(
                Document(page_content=cur_str, metadata={**metadata, "cut_index": ith})
            )
    else:
        result = "\n".join(
            [
                (
                    format_callback(item)
                    if format_callback is not None
                    else (
                        f"{(item.metadata['title']+':\n') if 'title' in item.metadata else ''}{item.page_content}"
                        if isinstance(item, Document)
                        else repr(item)
                    )
                )
                for item in content
                if item is not None
                and (
                    isinstance(item, Document)
                    and len(item.page_content) > 0
                    or len(item) > 0
                )
            ]
        )

    if crop_length is not None and len(result) > crop_length:
        if crop_from_start:
            result = result[:crop_length]
        else:
            result = result[-crop_length:]

    if result is not None:
        result = Document(result, metadata=metadata or {})
        results = [result]

    return results


def combine_metadata(docs: List[Document]) -> Dict[str, str]:
    combined_metadata = {k: str(v) for k, v in docs[0].metadata.items()}
    for doc in docs[1:]:
        for k, v in doc.metadata.items():
            if k in combined_metadata:
                combined_metadata[k] += f", {v}"
            else:
                combined_metadata[k] = str(v)

    return combined_metadata


def get_topic_doc_context(doc: Document):
    metadata = doc.metadata
    if metadata and len(metadata.keys()) > 0:
        file_info = f"File: {metadata['source']}\n" if "source" in metadata else ""
        pages_info = f"Pages: {metadata['page']}\n" if "page" in metadata else ""
        topic_indices_info = (
            f"Topic indices: {metadata['topic_index']}\n"
            if "topic_index" in metadata
            else ""
        )
        instructions_info = (
            f"Instructions:\n{metadata['instruct']}\n" if "instruct" in metadata else ""
        )

        content = file_info + pages_info + topic_indices_info + instructions_info

        if len(content) > 0:
            content = "Metadata:\n" + content + "\n"
    else:
        content = ""

    content += "Content:\n" + doc.page_content

    return content


def get_topic_document(item: Dict, page: Dict, instructions: str):
    return Document(
        (f"{item['topic'].strip()}:\n" if "topic" in item and item["topic"] else "")
        + get_text_from_completion(item["document"]),
        metadata={
            "instruct": item.get("instruct", ""),
            "page": page["page"],
            "page_index": item["topic_index"],
            "topic_index": f'{page["page"]}_{item["topic_index"]}',
            "topic": item["topic"] if "topic" in item else "",
            "instructions": instructions,
        },
    )


def get_topic_item(topic: dict, state: dict, metadata: dict, i=0):
    item = {
        "id": (
            str(state["page" if "page" in state else "source"])
            + "_"
            + str(i + 1)
            + "_"
            + (get_id_str(topic["id"]) if "id" in topic else get_id_str(topic["topic"]))
        ),
        "index": state["index"] if "index" in state else None,
        "page": (
            ", ".join(
                [str(state["page"])]
                if not isinstance(state["page"], list)
                else state["page"]
            )
            if "page" in state
            else None
        ),
        "topic_index": i + 1,
        "source": metadata.get("source", ""),
        "topic": topic["topic"],
        "instruct": topic["instruct"] if "instruct" in topic else None,
        "summary": topic["summary"],
        "document": topic["content"].strip(),
    }
    document = get_topic_document(
        item,
        {"page": state["page"], "topic_index": i},
        instructions=state["instructions"],
    )
    item["document"] = document

    return item


def parse_tag_items(
    response: Union[Dict, ParsedTopic], state: Dict, content_metadata: Dict = None
):
    if isinstance(response, ParsedTopic):
        topic = {
            "id": response.id,
            "topic": response.topic,
            "instruct": response.instruct,
            "summary": response.summary,
            "content": response.document,
        }
        item_state = {
            "index": (
                (response.page[0] if isinstance(response.page, list) else response.page)
                - 1
                if response.page
                else None
            ),
            "page": response.page,
            "instructions": (
                state["instructions"]
                if state is not None and "instructions" in state
                else None
            ),
        }
        metadata = {
            "source": response.source,
            "page": response.page,
        }

        item = get_topic_item(
            topic,
            item_state,
            metadata,
            response.topic_index[0] if response.topic_index else 0,
        )
        return [item]

    metadata = {}
    if isinstance(state.get("content"), Document):
        metadata = content_metadata.copy()

    tags = None

    if "filename" in state and state["filename"] is not None:
        metadata["source"] = state["filename"]
    if "page" in state and state["page"] is not None:
        metadata["page"] = state["page"]
    if "url" in state and state["url"] is not None:
        metadata["source"] = state["url"]

    if "tags" in response:
        tags = response["tags"]
        if "thinking" in tags:
            metadata["thinking"] = tags["thinking"]

    if "parsed" in response:
        parsed_content = parse_content_dict(response["parsed"])
        items = []
        for i, topic in enumerate(parsed_content):
            item = get_topic_item(topic, state, metadata, i)
            items.append(item)
    else:
        doc = Document(
            page_content=get_text_from_completion(response), metadata=metadata
        )
        topics = []
        summaries = []
        if tags is not None:
            if "topic" in tags:
                topics = str(tags["topic"]).split("\n\n")
            if "summary" in tags:
                summaries = str(tags["summary"]).split("\n\n")

        items = [{"document": doc, "topic": topics, "summary": summaries}]

    return items


def prepare_contents(
    content: List[Document],
    prev_page: List[Document] = None,
    next_page: List[Document] = None,
    max_length: int = 1000,
    format_callback: Callable = None,
):
    content = [content] if not isinstance(content, list) else content
    next_page = [next_page] if not isinstance(next_page, list) else next_page
    prev_page = [prev_page] if not isinstance(prev_page, list) else prev_page

    content = create_doc_from_list_with_metadata(
        content, format_callback=format_callback
    )[0]
    next_page = (
        create_doc_from_list_with_metadata(
            next_page, crop_length=max_length, format_callback=format_callback
        )[0]
        if next_page is not None
        else ""
    )
    prev_page = (
        create_doc_from_list_with_metadata(
            prev_page,
            crop_length=max_length,
            crop_from_start=False,
            format_callback=format_callback,
        )[0]
        if prev_page is not None
        else ""
    )

    return (content, prev_page, next_page)


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
