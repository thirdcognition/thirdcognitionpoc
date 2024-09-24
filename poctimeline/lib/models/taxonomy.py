import json
import textwrap
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import sqlalchemy as sqla
from sqlalchemy.ext.mutable import MutableList

from lib.db.sqlite import Base
from lib.helpers import pretty_print
from lib.load_env import SETTINGS


class TaxonomyDataTable(Base):
    __tablename__ = SETTINGS.concept_taxonomys_tablename

    # id = Column(Integer, primary_key=True)
    id = sqla.Column(sqla.String, primary_key=True)
    parent_id = sqla.Column(sqla.String)
    concept_taxonomy = sqla.Column(sqla.PickleType, default=None)
    category_tags = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    last_updated = sqla.Column(sqla.DateTime)
    disabled = sqla.Column(sqla.Boolean, default=False)


# class ParsedTaxonomy(BaseModel):
#     parent_id: Optional[str] = Field(
#         description=f"A previously defined category taxonomy id if available. If not available, leave blank. This is used to create a hierarchy of taxonomy categories. The parent_id should be the id of the parent taxonomy.",
#         title="Parent Id",
#     )
#     id: Optional[str] = Field(
#         description="A previously defined category taxonomy id if available. If not available, leave blank.",
#         title="Id",
#     )
#     taxonomy: str = Field(
#         description="The taxonomy of the category. This is used to determine the structure for the taxonomy.",
#         title="Taxonomy",
#     )
#     parent_taxonomy: Optional[str] = Field(
#         description="The parent taxonomy of the category. This is used to determine the structure for the taxonomy tree.",
#         title="Parent Taxonomy",
#     )
#     type: Optional[str] = Field(
#         description="The type of the category. This is used to determine the type of the category. For example, a category could be a subject, a topic, a skill, etc.",
#         title="Type",
#     )
#     tag: Optional[str] = Field(
#         description="A generalized tag for this taxonomy which can be used to group similar items and areas together.",
#         title="Tag",
#     )
#     title: str = Field(description="A title for this taxonomy category", title="Title")
#     description: str = Field(
#         description="The description of the taxonomy category", title="Description"
#     )


class Taxonomy(BaseModel):
    parent_id: Optional[str] = Field(
        description="Parent Id if available.",
        title="Parent Id",
    )
    id: Optional[str] = Field(
        description="An human readable id for this taxonomy using letters and _",
        title="Id",
    )
    tag: Optional[str] = Field(description="The tag for taxonomy", title="Tag")
    type: str = Field(description=f"The type of taxonomy.", title="Type")
    title: str = Field(description="A title for this taxonomy", title="Title")
    description: str = Field(
        description="The description of taxonomy", title="Description"
    )
    taxonomy: str = Field(
        description="The taxonomy classification.",
        title="Taxonomy classification",
    )
    parent_taxonomy: Optional[str] = Field(
        description="The parent taxonomy classification if available.",
        title="Parent taxonomy classification",
    )


class TaxonomyList(BaseModel):
    taxonomy: List[Taxonomy] = Field(
        description="A list of new taxonomy build from the provided list of topics and contents.",
        title="Taxonomy",
    )


class TaxonomyStructure(BaseModel):
    id: str = Field(
        description="Taxonomy ID",
        title="Id",
    )
    children: List["TaxonomyStructure"] = Field(
        description="A list of children using the defined cyclical structure of ParsedTaxonomyStructure(id, children: List[ParsedTaxonomyStructure], joined: List[str]).",
        title="Children",
    )
    joined: List[str] = Field(
        description="A list of Taxonomy IDs that have been used to build the topic.",
        title="Combined topic IDs",
    )


def convert_taxonomy_dict_to_tag_structure_string(data: dict) -> str:
    category_tag_data: Dict = data["category_tag"]
    tag_structure = f"""
    <category_tag>
        {f'<id>{category_tag_data["id"]}</id>' if 'id' in category_tag_data.keys() and category_tag_data['id'] is not None else ''}
        {f'<parent_id>{category_tag_data["parent_id"]}</parent_id>' if 'parent_id' in category_tag_data.keys() and category_tag_data['parent_id'] is not None else ''}
        <title>{category_tag_data.get("title", "")}</title>
        <taxonomy>{category_tag_data.get("taxonomy", "")}</taxonomy>
        {f'<parent_taxonomy>{category_tag_data["parent_taxonomy"]}</parent_taxonomy>' if "parent_taxonomy" in category_tag_data.keys() and category_tag_data['parent_taxonomy'] is not None else ''}
        <tag>{category_tag_data.get("tag", "")}</tag>
        <type>{category_tag_data.get("type", "")}</type>
        <description>{category_tag_data.get("description", "")}</description>
    </category_tag>
    """
    return textwrap.dedent(tag_structure)


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


def convert_taxonomy_to_json_string(
    taxonomy: Taxonomy, show_description: bool = False, children: List[str] = None
) -> str:
    tag_structure = {
        "id": taxonomy.id,
        "taxonomy_classification": taxonomy.taxonomy,
        "tag": taxonomy.tag,
        "type": taxonomy.type,
        "title": taxonomy.title,
    }
    if taxonomy.parent_id is not None:
        tag_structure["parent_id"] = taxonomy.parent_id
    if taxonomy.parent_taxonomy is not None:
        tag_structure["parent_taxonomy_classification"] = taxonomy.parent_taxonomy
    if children is not None:
        tag_structure["connected_concepts"] = "[{}]".format(", ".join(children))
    if show_description and taxonomy.description:
        tag_structure["description"] = taxonomy.description
    return json.dumps(tag_structure)


def convert_taxonomy_dict_to_tag_simple_structure_string(
    data: dict, show_description: bool = False, children: List[str] = None
) -> str:
    category_tag_data: Dict = data["category_tag"]
    tag_structure = (
        (
            f'parent_id({category_tag_data["parent_id"]}) > '
            if "parent_id" in category_tag_data
            and category_tag_data["parent_id"] is not None
            else ""
        )
        + (f'id({category_tag_data["id"]}) ' if "id" in category_tag_data else "")
        + (
            f'parent_taxonomy({category_tag_data["parent_taxonomy"]}) > '
            if "parent_taxonomy" in category_tag_data
            and category_tag_data["parent_taxonomy"] is not None
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


# def convert_parsed_taxonomy_to_taxonomy(parsed_taxonomy: ParsedTaxonomy) -> Taxonomy:
#     return Taxonomy(
#         parent_id=parsed_taxonomy.parent_id,
#         id=parsed_taxonomy.id,
#         tag=parsed_taxonomy.tag,
#         type=parsed_taxonomy.type,
#         title=parsed_taxonomy.title,
#         description=parsed_taxonomy.description,
#         taxonomy=parsed_taxonomy.taxonomy,
#         parent_taxonomy=parsed_taxonomy.parent_taxonomy
#     )


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
