from datetime import datetime
from enum import Enum
import json
import textwrap
from typing import Callable, Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field
import sqlalchemy as sqla
from sqlalchemy.ext.mutable import MutableList

from langchain_core.documents import Document
from lib.db.sqlite import Base
from lib.helpers import (
    combine_metadata,
    get_id_str,
    get_item_str,
    get_text_from_completion,
)
from lib.load_env import SETTINGS
from lib.models.reference import Reference, ReferenceType


class TopicDataTable(Base):
    __tablename__ = SETTINGS.topics_tablename

    id = sqla.Column(sqla.String, primary_key=True)
    references = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    page_content = sqla.Column(sqla.String)
    page_number = sqla.Column(sqla.Integer)
    topic_index = sqla.Column(sqla.Integer)
    doc_metadata = sqla.Column(sqla.JSON)
    topic = sqla.Column(sqla.String)
    instruct = sqla.Column(sqla.String)
    summary = sqla.Column(sqla.String)
    category_tags = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    chroma_collections = sqla.Column(
        MutableList.as_mutable(sqla.PickleType), default=[]
    )
    chroma_ids = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])


class TopicModel(BaseModel):
    id: str
    page_content: str
    page_number: int
    topic_index: int
    doc_metadata: Dict
    topic: str
    instruct: str
    summary: str


def topic_to_dict(topic: TopicDataTable) -> Dict:
    return {
        "topic": topic.topic,
        "page_content": topic.page_content,
        "page_number": topic.page_number,
        "topic_index": topic.topic_index,
        "metadata": topic.doc_metadata,
    }


def split_topics(
    topics: Union[List[TopicModel], List[Dict]],
    char_count: int = SETTINGS.default_llms.instruct.char_limit // 2,
) -> List[List[Union[TopicModel, Dict]]]:
    topic_lists = []
    topic_list = []
    cur_content = ""
    for topic in topics:
        topic_list.append(topic)
        if isinstance(topic, TopicModel):
            cur_content += topic.page_content
        elif isinstance(topic, dict) and "page_content" in topic:
            cur_content += (
                topic["page_content"].page_content
                if isinstance(topic["page_content"], Document)
                else topic["page_content"]
            )
        if len(cur_content) > char_count:
            topic_lists.append(topic_list)
            topic_list = []
            cur_content = ""

    if len(topic_list) > 0:
        topic_lists.append(topic_list)
    return topic_lists


class ParsedTopic(BaseModel):
    id: str = Field(
        description="Human readable topic ID with letters, numbers and _-characters. If available, use previously defined id.",
        title="Id",
    )
    topic: str = Field(
        description="Topic title that covers the content",
        title="Topic",
    )
    summary: str = Field(
        description="Summary of the content for the topic",
        title="Summary",
    )
    document: str = Field(
        description="Formatted content for the topic in full detail.",
        title="Document",
    )
    topic_index: list[Union[list[int], int]] = Field(
        description="Topic index within the page it was found from. When joining include all pages",
        title="Topic Index",
    )
    references: list[Reference] = Field(
        description="References to the content for the topic",
        title="References",
    )
    # page: Optional[Union[list[int], int]] = Field(
    #     description="Page from which the topic was uncovered. When joining include all pages.",
    #     title="Page",
    # )
    instruct: Optional[str] = Field(
        description="Instructions on how to interpret the content for the topic",
        title="Instruct",
    )
    # source: Optional[Union[list[str], str]] = Field(
    #     description="Source of the content for the topic",
    #     title="Source",
    # )


class ParsedTopicStructure(BaseModel):
    id: str = Field(
        description="Topic ID",
        title="Id",
    )
    children: List["ParsedTopicStructure"] = Field(
        description="A list of children using the defined cyclical structure of ParsedTopicStructure(id, children: List[ParsedTopicStructure], joined: List[str]).",
        title="Children",
    )
    joined: List[str] = Field(
        description="A list of Topic IDs that have been used to build the topic.",
        title="Combined topic IDs",
    )


class ParsedTopicStructureList(BaseModel):
    structure: List[ParsedTopicStructure] = Field(
        description="A list of topics identified in the context", title="Topics"
    )


def create_topic_doc_from_list_with_metadata(
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
        metadata = content.metadata
        content = [content]
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


def prepare_topic_contents(
    content: List[Document],
    prev_page: List[Document] = None,
    next_page: List[Document] = None,
    max_length: int = 1000,
    format_callback: Callable = None,
):
    content = [content] if not isinstance(content, list) else content
    next_page = [next_page] if not isinstance(next_page, list) else next_page
    prev_page = [prev_page] if not isinstance(prev_page, list) else prev_page

    content = create_topic_doc_from_list_with_metadata(
        content, format_callback=format_callback
    )[0]
    next_page = (
        create_topic_doc_from_list_with_metadata(
            next_page, crop_length=max_length, format_callback=format_callback
        )[0]
        if next_page is not None
        else ""
    )
    prev_page = (
        create_topic_doc_from_list_with_metadata(
            prev_page,
            crop_length=max_length,
            crop_from_start=False,
            format_callback=format_callback,
        )[0]
        if prev_page is not None
        else ""
    )

    return (content, prev_page, next_page)


def parse_topic_dict(data):
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
            result.extend(parse_topic_dict(item))
    return result


def get_topic_str(
    topics: List, one_liner=False, as_json=True, as_array=True, as_tags=False
):
    key_names = [
        "id",
        "references",
        "topic_index",
        "source",
        "topic",
        "instruct",
        "summary",
        "page_content",
        "document",
    ]
    key_mapping = {"page_content": "document"}
    if one_liner:
        select_keys = [
            "id",
            "references",
            "topic_index",
            "source",
            "topic",
            "instruct",
            "summary",
        ]
    else:
        select_keys = None

    return get_item_str(
        topics,
        as_json=as_json,
        as_array=as_array,
        as_tags=as_tags,
        key_names=key_names,
        key_mapping=key_mapping,
        select_keys=select_keys,
        one_liner=one_liner,
    )


def get_topic_doc_context(doc: Document):
    metadata = doc.metadata
    if metadata and len(metadata.keys()) > 0:
        file_info = f"Source: {metadata['source']}\n" if "source" in metadata else ""
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
            "references": item["references"] if "references" in item else "",
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
                else [str(p) for p in state["page"]]
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
        "references": topic["references"] if "references" in topic else None,
    }
    document = get_topic_document(
        item,
        {"page": state["page"], "topic_index": i},
        instructions=state["instructions"],
    )
    item["document"] = document

    return item


def parse_topic_items(
    response: Union[Dict, ParsedTopic], state: Dict, content_metadata: Dict = None
):
    if isinstance(response, ParsedTopic):
        topic = {
            "id": response.id,
            "topic": response.topic,
            "instruct": response.instruct,
            "summary": response.summary,
            "content": response.document,
            "references": response.references,
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
        source_ref = None
        source = state.get("filename") or state.get("url")
        for ref in response.references:
            if ref.type == ReferenceType.source:
                if source is not None and source != ref.id:
                    continue
                source_ref = ref
                break
        if source_ref is not None:
            metadata = {
                "source": source_ref.id,
                "page": source_ref.index,
            }
        else:
            metadata = {}

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
        parsed_content = parse_topic_dict(response["parsed"])
        items = []
        for i, topic in enumerate(parsed_content):
            item = get_topic_item(topic, state, metadata, i)
            items.append(item)
    else:
        doc = Document(
            page_content=get_text_from_completion(response), metadata=metadata
        )
        topic = ""
        summary = ""
        if tags is not None:
            if "topic" in tags:
                topic = (
                    str(tags["topic"]).replace("\n", " ").strip()
                    if tags["topic"]
                    else ""
                )
            if "summary" in tags:
                summary = (
                    str(tags["summary"]).replace("\n", " ").strip()
                    if tags["summary"]
                    else ""
                )

        items = [{"document": doc, "topic": topic, "summary": summary}]

    return items


def dict_to_topic_data_table(
    data_dict, category_tags=[], chroma_collections=[], chroma_ids=[]
):
    if isinstance(data_dict, dict):
        return TopicDataTable(
            id=data_dict["id"],
            references=(
                (
                    data_dict["references"]
                    if isinstance(data_dict["references"], list)
                    else [data_dict["references"]]
                )
                if "references" in data_dict
                else []
            ),
            page_content=get_text_from_completion(data_dict["page_content"]),
            page_number=data_dict["page"],
            topic_index=data_dict["topic_index"],
            doc_metadata=data_dict["metadata"],
            topic=data_dict["topic"],
            instruct=data_dict["instruct"],
            summary=data_dict["summary"],
            category_tags=category_tags,
            chroma_collections=chroma_collections,
            chroma_ids=chroma_ids,
        )
    else:
        print("topic dict", type(data_dict), repr(data_dict))
        raise ValueError("Input must be a dictionary")
