from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field
import sqlalchemy as sqla
from sqlalchemy.ext.mutable import MutableList

from langchain_core.documents import Document
from lib.db.sqlite import Base
from lib.load_env import SETTINGS


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
    summary: str
    id: str
    chroma_collections: Optional[List[str]]
    chroma_ids: Optional[List[str]]


class SourceContents(BaseModel):
    topics: Set[str]
    formatted_topics: List[SourceContentPage]
    formatted_content: str
    summary: str
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


