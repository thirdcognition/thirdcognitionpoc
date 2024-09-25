import textwrap
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
import sqlalchemy as sqla
from sqlalchemy.ext.mutable import MutableList

from lib.db.sqlite import Base
from lib.helpers import get_item_str, pretty_print
from lib.load_env import SETTINGS


class ConceptDataTable(Base):
    __tablename__ = SETTINGS.concepts_tablename

    # id = Column(Integer, primary_key=True)
    id = sqla.Column(sqla.String, primary_key=True)
    parent_id = sqla.Column(sqla.String)
    concept_contents = sqla.Column(sqla.PickleType, default=None)
    taxonomy = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    category_tags = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    sources = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    last_updated = sqla.Column(sqla.DateTime)
    chroma_collections = sqla.Column(
        MutableList.as_mutable(sqla.PickleType), default=[]
    )
    chroma_ids = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    disabled = sqla.Column(sqla.Boolean, default=False)


# class ParsedConcept(BaseModel):
#     content: str = Field(
#         description="Detailed and descriptive content in written format based on the context and identified concept. Should contain all relevant information in a readable format.",
#         title="Contents",
#     )
#     title: str = Field(
#         description="A human readable title for this concept", title="Title"
#     )
#     summary: str = Field(
#         description="A short summary of the concept with 1-2 sentences of the content.",
#         title="Summary",
#     )
#     taxonomy: List[str] = Field(
#         description="A list of taxonomy ids that this concept belongs to. Use only existing taxonomy ids for the items",
#         title="Taxonomy",
#     )
#     id: Optional[str] = Field(
#         description="An human readable id for this concept using letters and _ if available. If not available, leave blank.",
#         title="Id",
#     )
#     parent_id: Optional[str] = Field(
#         description="An human readable id for this concept using letters and _ if available. If not available, leave blank.",
#         title="Id",
#     )
#     page_number: Optional[int] = Field(
#         description="Page number where this concept was identified",
#         title="Reference",
#     )


# class ParsedConceptList(BaseModel):
#     concepts: List[ParsedConcept] = Field(
#         description="A list of concepts identified in the context", title="Concepts"
#     )


# class ParsedConceptIds(BaseModel):
#     id: Optional[str] = Field(
#         description="An human readable id for this concept using letters and _ if available. If not available, leave blank.",
#         title="Id",
#     )
#     parent_id: Optional[str] = Field(
#         description="An human readable id for the parent concept using letters and _ if available. If not available, leave blank.",
#         title="Parent Id",
#     )
#     combined_ids: Optional[List[str]] = Field(
#         description="List of IDs that were combined into this 1 concept. If not available, leave blank.",
#         title="Combined IDs",
#     )
#     summary: str = Field(
#         description="A short summary of the concept with 1-2 sentences of the content.",
#         title="Summary",
#     )
#     title: str = Field(
#         description="A human readable title for this concept", title="Title"
#     )
#     taxonomy: List[str] = Field(
#         description="A list of category taxonomy ids that the combined concepts could belong to. Use only existing taxonomy ids for the items",
#         title="Taxonomy",
#     )


# class ParsedUniqueConceptList(BaseModel):
#     concepts: List[ParsedConceptIds] = Field(
#         description="A list of concepts identified in the context", title="Concepts"
#     )


class ConceptStructure(BaseModel):
    id: str = Field(
        description="Concept ID",
        title="Id",
    )
    children: List["ConceptStructure"] = Field(
        description="A list of children using the defined cyclical structure of ConceptStructure(id, children: List[ConceptStructure], joined: List[str]).",
        title="Children",
    )
    joined: List[str] = Field(
        description="A list of Concept IDs that have been used to build the concept.",
        title="Combined concept IDs",
    )


class ConceptStructureList(BaseModel):
    structure: List[ConceptStructure] = Field(
        description="A list of concepts identified in the context", title="Concepts"
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
    title: str = Field(
        description="A human readable title for this concept", title="Title"
    )
    summary: str = Field(
        description="A summary of all contents related to this concept",
        title="Summary",
        default=None,
    )
    content: str = Field(
        description="Detailed and descriptive content in written format based on the context and identified concept. Should contain all relevant information in a readable format.",
        title="Contents",
    )
    sources: Union[List[str], str] = Field(
        description="A list of sources ids where this concept was identified",
        title="Sources",
    )
    references: List[SourceReference] = Field(
        description="A list of references to the source and page number where this concept was identified",
        title="Reference",
    )
    taxonomy: List[str] = Field(
        description="A list of taxonomy category ids that this concept belongs to",
        title="taxonomy",
    )

class ConceptList(BaseModel):
    concepts: List[ConceptData] = Field(
        description="A list of concepts identified in the context", title="Concepts"
    )

def get_concept_str(concepts: List, one_liner=False, as_json=True, as_array=True, as_tags=False):
    key_names = [
        "id",
        "parent_id",
        "title",
        "summary",
        "content",
        "sources",
        "references",
        "taxonomy",
    ]
    if one_liner:
        select_keys = [
            "id",
            "parent_id",
            "title",
            "summary",
            "sources",
            "references",
            "taxonomy",
        ]
    else:
        select_keys = None

    return get_item_str(
        concepts,
        as_json=as_json,
        as_array=as_array,
        as_tags=as_tags,
        key_names=key_names,
        select_keys=select_keys,
        one_liner=one_liner,
    )
