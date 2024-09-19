from datetime import datetime
from enum import Enum
import textwrap
from typing import Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field
import sqlalchemy as sqla
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.mutable import MutableList
from langchain_core.documents import Document
from lib.helpers import pretty_print
from lib.load_env import SETTINGS

Base = declarative_base()


class SourceType(Enum):
    url = "url"
    file = "file"


class ParsedTaxonomy(BaseModel):
    parent_id: Optional[str] = Field(
        description=f"A previously defined category taxonomy id if available. If not available, leave blank. This is used to create a hierarchy of taxonomy categories. The parent_id should be the id of the parent taxonomy.",
        title="Parent Id",
    )
    id: Optional[str] = Field(
        description="A previously defined category taxonomy id if available. If not available, leave blank.",
        title="Id",
    )
    taxonomy: str = Field(
        description="The taxonomy of the category. This is used to determine the structure for the taxonomy.",
        title="Taxonomy",
    )
    parent_taxonomy: Optional[str] = Field(
        description="The parent taxonomy of the category. This is used to determine the structure for the taxonomy tree.",
        title="Parent Taxonomy",
    )
    type: str = Field(
        description="The type of the category. This is used to determine the type of the category. For example, a category could be a subject, a topic, a skill, etc.",
        title="Type",
    )
    tag: str = Field(
        description="A generalized tag for this taxonomy which can be used to group similar items and areas together.",
        title="Tag",
    )
    title: str = Field(description="A title for this taxonomy category", title="Title")
    description: str = Field(
        description="The description of the taxonomy category", title="Description"
    )


class ParsedTaxonomyList(BaseModel):
    taxonomy: List[ParsedTaxonomy] = Field(
        description="A list of new category taxonomy build from the provided list of taxonomy and concepts. Each taxonomy category should be generic and applicable for varied subjects.",
        title="Taxonomy",
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


class Taxonomy(BaseModel):
    parent_id: Optional[str] = Field(
        description="Id of the parent concept category tag if available.",
        title="Parent Id",
    )
    id: Optional[str] = Field(
        description="An human readable id for this concept using letters and _",
        title="Id",
    )
    tag: str = Field(description="The tag for the category", title="Tag")
    type: str = Field(description=f"The type of the category.", title="Type")
    title: str = Field(description="A title for this concept", title="Title")
    description: str = Field(
        description="The description of the category", title="Description"
    )
    taxonomy: str = Field(
        description="The taxonomy of the category.",
        title="Taxonomy",
    )
    parent_taxonomy: Optional[str] = Field(
        description="The parent taxonomy of the category.",
        title="Parent Taxonomy",
    )


def convert_taxonomy_to_dict(taxonomy: Taxonomy) -> dict:
    return {
        "category_tag": {
            "title": taxonomy.title,
            "taxonomy": taxonomy.taxonomy,
            "parent_taxonomy": taxonomy.parent_taxonomy,
            "tag": taxonomy.tag,
            "type": taxonomy.type,
            "id": taxonomy.id,
            "parent_id": taxonomy.parent_id,
            "description": taxonomy.description,
        }
    }


def convert_taxonomy_dict_to_tag_structure_string(data: dict) -> str:
    category_tag_data: Dict = data["category_tag"]
    tag_structure = f"""
    <category_tag>
        {f'<id>{category_tag_data["id"]}</id>' if 'id' in category_tag_data.keys() else ''}
        {f'<parent_id>{category_tag_data["parent_id"]}</parent_id>' if 'parent_id' in category_tag_data.keys() else ''}
        <title>{category_tag_data.get("title", "")}</title>
        <taxonomy>{category_tag_data.get("taxonomy", "")}</taxonomy>
        {f'<parent_taxonomy>{category_tag_data["parent_taxonomy"]}</parent_taxonomy>' if "parent_taxonomy" in category_tag_data.keys() else ''}
        <tag>{category_tag_data.get("tag", "")}</tag>
        <type>{category_tag_data.get("type", "")}</type>
        <description>{category_tag_data.get("description", "")}</description>
    </category_tag>
    """
    return textwrap.dedent(tag_structure)


def convert_taxonomy_dict_to_tag_simple_structure_string(
    data: dict, show_description: bool = False, children: List[str] = None
) -> str:
    category_tag_data: Dict = data["category_tag"]
    tag_structure = (
        (
            f'parent_id({category_tag_data["parent_id"]}) > '
            if "parent_id" in category_tag_data
            else ""
        )
        + (f'id({category_tag_data["id"]}) ' if "id" in category_tag_data else "")
        + (
            f'parent_taxonomy({category_tag_data["parent_taxonomy"]}) > '
            if "parent_taxonomy" in category_tag_data
            else ""
        )
        + f'taxonomy({category_tag_data.get("taxonomy", "")}) '
        + f'tag({category_tag_data.get("tag", "")}) '
        + (f'connected_concepts({", ".join(children)}) ' if children else "")
        + f'type({category_tag_data.get("type", "")}): '
        + f'{category_tag_data.get("title", "")}'
        + (
            f'\n{str(category_tag_data.get("description", "")).replace("\n", " ")}\n\n'
            if show_description
            else ""
        )
    )
    return tag_structure


def convert_taxonomy_tags_to_dict(input_dict, tags):
    output_dict = {"category_tag": {}}

    for child in input_dict["children"]:
        if child["tag"] in tags:
            output_dict["category_tag"][child["tag"]] = child["body"].strip()

    return output_dict


def convert_taxonomy_dict_to_taxonomy(data: dict) -> Taxonomy:
    category_tag_data: Dict = data.get("category_tag", {})
    if not category_tag_data:
        raise ValueError("Invalid data format. 'category_tag' key not found.")
    return Taxonomy(
        id=category_tag_data.get("id", ""),
        parent_id=category_tag_data.get("parent_id", ""),
        title=category_tag_data.get("title", ""),
        taxonomy=category_tag_data.get("taxonomy", ""),
        parent_taxonomy=category_tag_data.get("parent_taxonomy", ""),
        tag=category_tag_data.get("tag", ""),
        description=category_tag_data.get("description", ""),
        type=category_tag_data.get("type", ""),
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


class SourceContentPage(BaseModel):
    page_content: str
    page_number: int
    topic_index: int
    metadata: Dict
    topic: str
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


class TaxonomyDataTable(Base):
    __tablename__ = SETTINGS.concept_taxonomys_tablename

    # id = Column(Integer, primary_key=True)
    id = sqla.Column(sqla.String, primary_key=True)
    parent_id = sqla.Column(sqla.String)
    concept_taxonomy = sqla.Column(sqla.PickleType, default=None)
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
