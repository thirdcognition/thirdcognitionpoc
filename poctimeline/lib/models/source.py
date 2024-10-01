from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Union
from pydantic import BaseModel
import sqlalchemy as sqla
from sqlalchemy.ext.mutable import MutableList
from langchain_core.documents import Document
from lib.db.sqlite import Base
from lib.load_env import SETTINGS


class SourceType(Enum):
    url = "url"
    file = "file"


class SourceDataTable(Base):
    __tablename__ = SETTINGS.source_tablename

    source = sqla.Column(sqla.String, primary_key=True)
    type = sqla.Column(sqla.Enum(SourceType))
    texts = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    category_tags = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    last_updated = sqla.Column(sqla.DateTime, nullable=True)
    chroma_collections = sqla.Column(
        MutableList.as_mutable(sqla.PickleType), default=[]
    )
    chroma_ids = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    disabled = sqla.Column(sqla.Boolean, default=False)
    edited_content = sqla.Column(sqla.Text, nullable=True)
    file_data = sqla.Column(sqla.BLOB, nullable=True)
    source_contents = sqla.Column(sqla.PickleType, nullable=True)
    source_topics = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    source_concepts = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])

class SourcePage(BaseModel):
    content: str
    page_metadata: Dict[str, Union[str, int, None]]

class SourceContents(BaseModel):
    topic: str
    summary: str
    pages: List[SourcePage]
    formatted_content: str