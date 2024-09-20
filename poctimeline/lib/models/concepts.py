import textwrap
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import sqlalchemy as sqla
from sqlalchemy.ext.mutable import MutableList

from lib.db.sqlite import Base
from lib.helpers import pretty_print
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
    taxonomy: List[str] = Field(
        description="A list of taxonomy ids that this concept belongs to. Use only existing taxonomy ids for the items",
        title="Taxonomy",
    )
    id: Optional[str] = Field(
        description="An human readable id for this concept using letters and _ if available. If not available, leave blank.",
        title="Id",
    )
    parent_id: Optional[str] = Field(
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
    parent_id: Optional[str] = Field(
        description="An human readable id for the parent concept using letters and _ if available. If not available, leave blank.",
        title="Parent Id",
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
    taxonomy: List[str] = Field(
        description="A list of category taxonomy ids that the combined concepts could belong to. Use only existing taxonomy ids for the items",
        title="Taxonomy",
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
    taxonomy: List[str] = Field(
        description="A list of taxonomy category ids that this concept belongs to",
        title="taxonomy",
    )


def get_concept_str(
    concepts: List,
    as_array: bool = False,
    summary: bool = False,
    content: bool = False,
    taxonomy: bool = False,
    page_number: bool = False,
    combined_ids: bool = False,
    sources: bool = False,
    children: bool = False,
    references: bool = False,
) -> str:
    ret_str = []
    for concept in concepts:
        if isinstance(concept, ConceptData):
            ret_str.append(
                (f"ParentID({concept.parent_id}) " if concept.parent_id else "")
                + f"ID({concept.id}) "
                + (
                    f"ChildIDs({', '.join(concept.children)}) "
                    if children and concept.children and len(concept.children) > 0
                    else ""
                )
                + (
                    f"TaxonomyIDs({', '.join(concept.taxonomy)}) "
                    if taxonomy and concept.taxonomy and len(concept.taxonomy) > 0
                    else ""
                )
                + (
                    f"Sources({', '.join(concept.sources)}) "
                    if sources and concept.sources and len(concept.sources) > 0
                    else ""
                )
                + (
                    f"References({', '.join([f'{reference.source}:{reference.page_number}' for reference in concept.references])}) "
                    if references and concept.references and len(concept.references) > 0
                    else ""
                )
                + f"\n{concept.title.strip()}:"
                + (
                    f"\nSummary: {concept.summary.replace('\n', ' ').strip()}\n"
                    if summary and concept.summary
                    else ""
                )
                + (
                    f"\nContent:\n{'\n'.join(concept.contents).strip()}"
                    if content and concept.contents
                    else ""
                )
            )
        if isinstance(concept, ParsedConcept):
            ret_str.append(
                (f"ParentID({concept.parent_id}) " if concept.parent_id else "")
                + f"ID({concept.id}) "
                + (
                    f"Page({concept.page_number}) "
                    if page_number and concept.page_number
                    else ""
                )
                + (
                    f"TaxonomyIDs({', '.join(concept.taxonomy)}) "
                    if taxonomy and concept.taxonomy and len(concept.taxonomy) > 0
                    else ""
                )
                + f"\n{concept.title.strip()}:"
                + (
                    f"\nSummary: {concept.summary.replace('\n', ' ').strip()}\n"
                    if summary and concept.summary
                    else ""
                )
                + (
                    f"\nContent:\n{concept.content.strip()}"
                    if content and concept.content
                    else ""
                )
            )
        if isinstance(concept, ParsedConceptIds):
            ret_str.append(
                (f"ParentID({concept.parent_id}) " if concept.parent_id else "")
                + f"ID({concept.id}) "
                + (
                    f"CombinedIDs({', '.join(concept.combined_ids)}) "
                    if combined_ids
                    and concept.combined_ids
                    and len(concept.combined_ids) > 0
                    else ""
                )
                + (
                    f"TaxonomyIDs({', '.join(concept.taxonomy)}) "
                    if taxonomy and concept.taxonomy and len(concept.taxonomy) > 0
                    else ""
                )
                + f"\n{concept.title.strip()}:"
                + (
                    f"\nSummary: {concept.summary.replace('\n', ' ').strip()}\n"
                    if summary and concept.summary
                    else ""
                )
            )
    if as_array:
        return ret_str
    return "\n\n".join(ret_str)
