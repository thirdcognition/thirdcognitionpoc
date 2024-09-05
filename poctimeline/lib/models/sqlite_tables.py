from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import sqlalchemy as sqla
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.mutable import MutableList

from lib.load_env import SETTINGS

Base = declarative_base()

class SourceType(Enum):
    url = "url"
    file = "file"

class Reference(BaseModel):
    source: str = Field(
        description="The name of the file if applicable", title="Source"
    )
    source_type: SourceType = Field(
        description="Type of the source. Can be 'url' or 'file'.", title="Source Type"
    )
    page_number: Optional[int] = Field(
        description="The page number of the file", title="Page Number"
    )

class CategoryTag(BaseModel):
    tag: str = Field(
        description="The tag for the category", title="Tag"
    )
    description: str = Field(
        description="The description of the category", title="Description"
    )

class SourceConcept(BaseModel):
    id: str = Field(
        description="An human readable id for this concept using letters and _", title="Id"
    )
    title: str = Field(
        description="A human readable title for this concept", title="Title"
    )
    content: str = Field(
        description="Detailed and descriptive content in written format based on the context and identified concept. Should contain all relevant information in a readable format.", title="Contents"
    )
    reference: Reference = Field(
        description= "A reference to the file and page number where this concept was identified", title="Reference"
    )
    category: List[CategoryTag] = Field(
        description="A list of category tags that can be used to group this concept with similar concepts", title="Category"
    )


class SourceConceptList(BaseModel):
    concepts: List[SourceConcept] = Field(
        description="A list of concepts identified in the context", title="Concepts"
    )

class SourceContents(BaseModel):
    formatted_content: str
    summary: str
    concepts: List[SourceConcept]
    concept_summaries: Dict[str, str] = Field(
        description="A dictionary of summaries for each concept where key is the concept id and value is the summary of all contents related to that concept", title="Summaries"
    )

class SourceData(BaseModel):
    source: str
    type: SourceType
    texts: List[str] = []
    category_tag: List[str] = []
    last_updated: Optional[datetime] = None
    chroma_collection: List[str] = []
    chroma_ids: List[str] = []
    disabled: bool = False
    edited_content: Optional[str] = None
    file_data: Optional[bytes] = None
    source_contents: Optional[SourceContents] = None

class SourceDataTable(Base):
    __tablename__ = SETTINGS.file_tablename

    # id = Column(Integer, primary_key=True)
    source = sqla.Column(sqla.String, primary_key=True)
    type = sqla.Column(sqla.PickleType)
    texts = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    category_tag = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    last_updated = sqla.Column(sqla.DateTime)
    chroma_collection = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    chroma_ids = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    disabled = sqla.Column(sqla.Boolean, default=False)
    edited_content = sqla.Column(sqla.Text)
    file_data = sqla.Column(sqla.BLOB)
    source_contents = sqla.Column(sqla.PickleType, default=None)


# Define a new class for JourneyDataTable with list_name as primary key
class JourneyDataTable(Base):
    __tablename__ = SETTINGS.journey_tablename

    # id = Column(Integer, primary_key=True)
    journeyname = sqla.Column(sqla.String, primary_key=True)
    files = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    subjects = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    chroma_collection = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    disabled = sqla.Column(sqla.Boolean, default=False)
    title = sqla.Column(sqla.String, default="")
    summary = sqla.Column(sqla.Text, default="")
    instructions = sqla.Column(sqla.Text, default="")
    last_updated = sqla.Column(sqla.DateTime, default=None)
