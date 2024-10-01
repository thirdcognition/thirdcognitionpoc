from datetime import datetime
import textwrap
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
import sqlalchemy as sqla
from sqlalchemy.ext.mutable import MutableList

from lib.db.sqlite import Base
from lib.helpers import get_item_str, pretty_print
from lib.load_env import SETTINGS
from lib.models.reference import Reference

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
    references: List[Reference] = Field(
        description="A list of references to topics, concepts or sources where this concept was identified",
        title="Reference",
    )
    taxonomy: List[str] = Field(
        description="A list of taxonomy category ids that this concept belongs to",
        title="taxonomy",
    )
    def to_concept_data_table(self) -> "ConceptDataTable":
        return ConceptDataTable.from_concept_data(self)


class ConceptDataTable(Base):
    __tablename__ = SETTINGS.concepts_tablename

    id = sqla.Column(sqla.String, primary_key=True)
    parent_id = sqla.Column(sqla.String)
    title = sqla.Column(sqla.String)
    summary = sqla.Column(sqla.String)
    content = sqla.Column(sqla.String)
    taxonomy = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    category_tags = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    references = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    last_updated = sqla.Column(sqla.DateTime)
    chroma_collections = sqla.Column(
        MutableList.as_mutable(sqla.PickleType), default=[]
    )
    chroma_ids = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    disabled = sqla.Column(sqla.Boolean, default=False)

    @classmethod
    def from_concept_data(cls, concept_data: ConceptData) -> "ConceptDataTable":
        return cls(
            id=concept_data.id,
            parent_id=concept_data.parent_id,
            title=concept_data.title,
            summary=concept_data.summary,
            content=concept_data.content,
            taxonomy=concept_data.taxonomy,
            references=concept_data.references,
            last_updated=datetime.now(),
        )

    def to_concept_data(self) -> ConceptData:
        return ConceptData(
            id=self.id,
            parent_id=self.parent_id,
            title=self.title,
            summary=self.summary,
            content=self.content,
            taxonomy=self.taxonomy,
            references=[Reference(**ref) for ref in self.references],
        )

    def copy(self, **kwargs) -> "ConceptDataTable":
        return ConceptDataTable(
            id=kwargs.get('id', self.id),
            parent_id=kwargs.get('parent_id', self.parent_id),
            title=kwargs.get('title', self.title),
            summary=kwargs.get('summary', self.summary),
            content=kwargs.get('content', self.content),
            taxonomy=kwargs.get('taxonomy', self.taxonomy.copy()) if self.taxonomy is not None else [],
            category_tags=kwargs.get('category_tags', self.category_tags.copy() if self.category_tags is not None else []),
            references=kwargs.get('references', self.references.copy() if self.references is not None else []),
            last_updated=kwargs.get('last_updated', self.last_updated),
            chroma_collections=kwargs.get('chroma_collections', self.chroma_collections.copy() if self.chroma_collections is not None else []),
            chroma_ids=kwargs.get('chroma_ids', self.chroma_ids.copy() if self.chroma_ids is not None else []),
            disabled=kwargs.get('disabled', self.disabled),
        )

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
        "references",
        "taxonomy",
    ]
    if one_liner:
        select_keys = [
            "id",
            "parent_id",
            "title",
            "summary",
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
