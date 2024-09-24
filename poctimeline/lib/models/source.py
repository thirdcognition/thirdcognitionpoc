from datetime import datetime
from enum import Enum
import textwrap
from typing import Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field
import sqlalchemy as sqla
from sqlalchemy.ext.mutable import MutableList

from langchain_core.documents import Document
from lib.db.sqlite import Base
from lib.load_env import SETTINGS


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
    page: Optional[Union[list[int], int]] = Field(
        description="Page from which the topic was uncovered. When joining include all pages.",
        title="Page",
    )
    instruct: Optional[str] = Field(
        description="Instructions on how to interpret the content for the topic",
        title="Instruct",
    )
    source: Optional[Union[list[str], str]]  = Field(
        description="Source of the content for the topic",
        title="Source",
    )


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
        description="A list of topics identified in the context", title="Concepts"
    )


class SourceDataTable(Base):
    __tablename__ = SETTINGS.file_tablename

    # id = Column(Integer, primary_key=True)
    source = sqla.Column(sqla.String, primary_key=True)
    type = sqla.Column(sqla.PickleType)
    texts = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    category_tags = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    last_updated = sqla.Column(sqla.DateTime)
    chroma_collections = sqla.Column(
        MutableList.as_mutable(sqla.PickleType), default=[]
    )
    chroma_ids = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    disabled = sqla.Column(sqla.Boolean, default=False)
    edited_content = sqla.Column(sqla.Text)
    file_data = sqla.Column(sqla.BLOB)
    source_contents = sqla.Column(sqla.PickleType, default=None)
    source_concepts = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])


class SourceContentPage(BaseModel):
    page_content: str
    page_number: int
    topic_index: int
    metadata: Dict
    topic: str
    instruct: str
    summary: str
    id: str
    chroma_collections: Optional[List[str]]
    chroma_ids: Optional[List[str]]


class SourceContents(BaseModel):
    topic: str
    summary: str
    formatted_content: str
    all_topics: Set[str]
    formatted_topics: List[SourceContentPage]
    # concepts: List[ConceptData]
    # concept_summaries: Dict[str, str] = Field(
    #     description="A dictionary of summaries for each concept where key is the concept id and value is the summary of all contents related to that concept",
    #     title="Summaries",
    # )


class SourceType(Enum):
    url = "url"
    file = "file"


class SourceData(BaseModel):
    source: str
    type: SourceType
    texts: List[str] = []
    category_tags: List[str] = []
    last_updated: Optional[datetime] = None
    chroma_collections: List[str] = []
    chroma_ids: List[str] = []
    disabled: bool = False
    edited_content: Optional[str] = None
    file_data: Optional[bytes] = None
    source_contents: Optional[SourceContents] = None
    source_concepts: Optional[List[str]] = None


def topic_to_dict(topic: SourceContentPage) -> Dict:
    return {
        "topic": topic.topic,
        "page_content": topic.page_content,
        "page_number": topic.page_number,
        "topic_index": topic.topic_index,
        "metadata": topic.metadata,
    }


def split_topics(
    topics: Union[List[SourceContentPage], List[Dict]],
    char_count: int = SETTINGS.default_llms.instruct.char_limit // 2,
) -> List[List[Union[SourceContentPage, Dict]]]:
    topic_lists = []
    topic_list = []
    cur_content = ""
    for topic in topics:
        topic_list.append(topic)
        if isinstance(topic, SourceContentPage):
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


def get_topic_str(
    items: List,
    all_details: bool = False,
    as_array: bool = False,
) -> str:
    ret_str = []
    for item in items:
        if all_details:
            item_str = "{\n"
            if "id" in item:
                item_str += f'"id": "{item["id"].replace("\n", " ").strip()}",\n'
            if "page" in item:
                item_str += f'"page": {item["page"]},\n'
            if "topic_index" in item:
                item_str += f'"topic_index": {item["topic_index"]},\n'
            if "source" in item:
                item_str += (
                    f'"source": "{item["source"].replace("\n", " ").strip()}",\n'
                )
            if "topic" in item:
                item_str += f'"topic": "{item["topic"].replace("\n", " ").strip()}",\n'
            if "instruct" in item:
                item_str += (
                    f'"instruct": "{item["instruct"].replace("\n", " ").strip()}",\n'
                )
            if "summary" in item:
                item_str += (
                    f'"summary": "{item["summary"].replace("\n", " ").strip()}",\n'
                )
            if "page_content" in item:
                item_str += f'"document": "{item["page_content"].replace("\n", " ").strip()}",\n'
            item_str = item_str.rstrip(",\n") + "\n}"
        else:
            item_str = (
                f"TopicID({item['id'].replace('\n', ' ').strip()}) "
                + f"{item['topic'].replace('\n', ' ').strip()}: "
                + f"{item['summary'].replace('\n', ' ').strip()}"
            )

        ret_str.append(item_str)

    if as_array:
        return ret_str
    return "\n\n".join(ret_str)
