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


class ParsedConceptCategoryTag(BaseModel):
    parent_id: Optional[str] = Field(
        description=f"A previously defined concept category tag id if available. If not available, leave blank. This is used to create a hierarchy of tags. The parent_id should be the id of the parent tag.",
        title="Parent Id",
    )
    id: Optional[str] = Field(
        description="A previously defined concept category tag id if available. If not available, leave blank.",
        title="Id",
    )
    tag: str = Field(
        description="The tag for the category using letters, _ and -", title="Tag"
    )
    title: str = Field(description="A title for this concept", title="Title")
    description: str = Field(
        description="The description of the category", title="Description"
    )
    connected_concepts: List[str] = Field(
        description="A list of concept ids that are connected to this tag",
        title="Connected Concepts",
    )


class ParsedConceptCategoryTagList(BaseModel):
    tags: List[ParsedConceptCategoryTag] = Field(
        description="A list of unique concept category tags identified from the provided concepts",
        title="Concepts",
    )


class ParsedConcept(BaseModel):
    content: str = Field(
        description="Detailed and descriptive content in written format based on the context and identified concept. Should contain all relevant information in a readable format.",
        title="Contents",
    )
    title: str = Field(
        description="A human readable title for this concept", title="Title"
    )
    summary: str = Field(
        description="A short summary of the concept with 1-2 sentences of the content.",
        title="Summary",
    )
    tags: List[str] = Field(
        description="A list of tags that this concepts could belong to", title="Tags"
    )
    id: Optional[str] = Field(
        description="An human readable id for this concept using letters and _ if available. If not available, leave blank.",
        title="Id",
    )
    page_number: Optional[int] = Field(
        description="Page number where this concept was identified",
        title="Reference",
    )


class ParsedConceptList(BaseModel):
    concepts: List[ParsedConcept] = Field(
        description="A list of concepts identified in the context", title="Concepts"
    )


class ParsedConceptIds(BaseModel):
    id: Optional[str] = Field(
        description="An human readable id for this concept using letters and _ if available. If not available, leave blank.",
        title="Id",
    )
    combined_ids: Optional[List[str]] = Field(
        description="List of IDs that were combined into this 1 concept. If not available, leave blank.",
        title="Combined IDs",
    )
    summary: str = Field(
        description="A short summary of the concept with 1-2 sentences of the content.",
        title="Summary",
    )
    title: str = Field(
        description="A human readable title for this concept", title="Title"
    )
    tags: List[str] = Field(
        description="A list of tags that this concepts could belong to", title="Tags"
    )


class ParsedUniqueConceptList(BaseModel):
    concepts: List[ParsedConceptIds] = Field(
        description="A list of concepts identified in the context", title="Concepts"
    )


class ParsedConceptStructure(BaseModel):
    id: str = Field(
        description="Concept ID",
        title="Id",
    )
    children: List["ParsedConceptStructure"] = Field(
        description="A list of children using the defined cyclical structure of ParsedConceptStructure(id, children: List[ParsedConceptStructure]).",
        title="Structure",
    )

class ParsedConceptStructureList(BaseModel):
    structure: List[ParsedConceptStructure] = Field(
        description="A list of concepts identified in the context", title="Concepts"
    )

class ConceptCategoryTag(BaseModel):
    parent_id: Optional[str] = Field(
        description="Id of the parent concept category tag if available. If not available, leave blank.",
        title="Parent Id",
    )
    id: Optional[str] = Field(
        description="An human readable id for this concept using letters and _",
        title="Id",
    )
    tag: str = Field(description="The tag for the category", title="Tag")
    title: str = Field(description="A title for this concept", title="Title")
    description: str = Field(
        description="The description of the category", title="Description"
    )


class SourceReference(BaseModel):
    source: str = Field(
        description="The name of the file if applicable", title="Source"
    )
    page_number: Optional[int] = Field(
        description="The page number of the file", title="Page Number"
    )


class ConceptData(BaseModel):
    id: str = Field(
        description="An human readable id for this concept using letters and _",
        title="Id",
    )
    parent_id: Optional[str] = Field(
        description="An human readable id for the parent concept using letters and _",
        title="Parent Id",
    )
    children: Optional[List[str]] = Field(
        description="A list of children concepts ids", title="Children"
    )
    title: Optional[str] = Field(
        description="A human readable title for this concept", title="Title"
    )
    summary: Optional[str] = Field(
        description="A summary of all contents related to this concept",
        title="Summary",
        default=None,
    )
    contents: List[str] = Field(
        description="Detailed and descriptive content in written format based on the context and identified concept. Should contain all relevant information in a readable format.",
        title="Contents",
    )
    sources: List[str] = Field(
        description="A list of sources ids where this concept was identified",
        title="Sources",
    )
    references: List[SourceReference] = Field(
        description="A reference to the source and page number where this concept was identified",
        title="Reference",
    )
    tags: List[str] = Field(
        description="A list of concept category tag ids that this concept belongs to",
        title="Category",
    )


class SourceContents(BaseModel):
    formatted_content: str
    summary: str
    # concepts: List[ConceptData]
    # concept_summaries: Dict[str, str] = Field(
    #     description="A dictionary of summaries for each concept where key is the concept id and value is the summary of all contents related to that concept",
    #     title="Summaries",
    # )


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


class ConceptDataTable(Base):
    __tablename__ = SETTINGS.concepts_tablename

    # id = Column(Integer, primary_key=True)
    id = sqla.Column(sqla.String, primary_key=True)
    parent_id = sqla.Column(sqla.String)
    concept_contents = sqla.Column(sqla.PickleType, default=None)
    category_tags = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    sources = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    last_updated = sqla.Column(sqla.DateTime)
    chroma_collections = sqla.Column(
        MutableList.as_mutable(sqla.PickleType), default=[]
    )
    chroma_ids = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    disabled = sqla.Column(sqla.Boolean, default=False)


class ConceptCategoryDataTable(Base):
    __tablename__ = SETTINGS.concept_category_tags_tablename

    # id = Column(Integer, primary_key=True)
    id = sqla.Column(sqla.String, primary_key=True)
    parent_id = sqla.Column(sqla.String)
    concept_category_tag = sqla.Column(sqla.PickleType, default=None)
    category_tags = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    last_updated = sqla.Column(sqla.DateTime)
    disabled = sqla.Column(sqla.Boolean, default=False)


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


# Define a new class for JourneyDataTable with list_name as primary key
class JourneyDataTable(Base):
    __tablename__ = SETTINGS.journey_tablename

    # id = Column(Integer, primary_key=True)
    journeyname = sqla.Column(sqla.String, primary_key=True)
    files = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    subjects = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    chroma_collections = sqla.Column(
        MutableList.as_mutable(sqla.PickleType), default=[]
    )
    disabled = sqla.Column(sqla.Boolean, default=False)
    title = sqla.Column(sqla.String, default="")
    summary = sqla.Column(sqla.Text, default="")
    instructions = sqla.Column(sqla.Text, default="")
    last_updated = sqla.Column(sqla.DateTime, default=None)
