from datetime import datetime
from functools import cache
from typing import Dict, List, Union
import sqlalchemy as sqla

import streamlit as st
from lib.models.user import user_db_get_session

from lib.helpers import convert_tags_to_dict, get_id_str, get_unique_id, pretty_print
from lib.models.taxonomy import TaxonomyDataTable, Taxonomy
from lib.prompts.taxonomy import TAXONOMY_ALL_TAGS


def db_taxonomy_exists(tag_id: str) -> bool:
    return (
        user_db_get_session()
        .query(sqla.exists().where(TaxonomyDataTable.id == tag_id))
        .scalar()
    )


@cache
def get_taxonomy_by_id(
    concept_category_id: str,
) -> TaxonomyDataTable:
    return (
        user_db_get_session()
        .query(TaxonomyDataTable)
        .filter(TaxonomyDataTable.id == concept_category_id)
        .first()
    )


def get_db_taxonomy(
    categories: List[str] = None, reset: bool = False
) -> Dict[str, List[TaxonomyDataTable]]:
    db_taxonomy: Dict[str, List[TaxonomyDataTable]] = None
    if "db_taxonomy" not in st.session_state or reset:
        found_taxonomy = (
            user_db_get_session().query(TaxonomyDataTable).distinct().all()
        )

        if categories is not None:
            found_taxonomy = [
                taxonomy
                for taxonomy in found_taxonomy
                if any(cat in taxonomy.category_tags for cat in categories)
            ]

        if "db_taxonomy" not in st.session_state or reset:
            db_taxonomy = {}
        else:
            db_taxonomy = st.session_state.db_taxonomy
        for taxonomy in found_taxonomy:
            categories = taxonomy.category_tags
            for ict in categories:
                if ict not in db_taxonomy:
                    db_taxonomy[ict] = []
                concept_taxonomy = (
                    taxonomy  # Taxonomy(**taxonomy.concept_taxonomy.__dict__)
                )
                concept_taxonomy.id = taxonomy.id
                db_taxonomy[ict].append(concept_taxonomy)

        st.session_state.db_taxonomy = db_taxonomy
    else:
        db_taxonomy = st.session_state.db_taxonomy

    return db_taxonomy


def delete_db_taxonomy(taxonomy_id: str, commit: bool = True):
    instance = (
        user_db_get_session()
        .query(TaxonomyDataTable)
        .where(TaxonomyDataTable.id == taxonomy_id)
        .first()
    )
    user_db_get_session().delete(instance)
    if commit:
        user_db_get_session().commit()


def update_db_taxonomy(
    taxonomy: Union[TaxonomyDataTable, Taxonomy],
    categories=List[str],
    commit: bool = True,
):
    if db_taxonomy_exists(taxonomy.id):
        # print(f"\n\nUpdate existing taxonomy:\n\n{taxonomy.model_dump_json(indent=4)}")
        existing_taxonomy = (
            user_db_get_session()
            .query(TaxonomyDataTable)
            .filter(TaxonomyDataTable.id == taxonomy.id)
            .first()
        )
        if taxonomy.parent_id is not None:
            existing_taxonomy.parent_id = taxonomy.parent_id
        if taxonomy.tag is not None:
            existing_taxonomy.tag = taxonomy.tag
        if taxonomy.type is not None:
            existing_taxonomy.type = taxonomy.type
        if taxonomy.title is not None:
            existing_taxonomy.title = taxonomy.title
        if taxonomy.description is not None:
            existing_taxonomy.description = taxonomy.description
        if taxonomy.taxonomy is not None:
            existing_taxonomy.taxonomy = taxonomy.taxonomy
        if taxonomy.parent_taxonomy is not None:
            existing_taxonomy.parent_taxonomy = taxonomy.parent_taxonomy
        # existing_taxonomy.concept_taxonomy = taxonomy
        existing_taxonomy.category_tags = list(
            set(categories + existing_taxonomy.category_tags)
        )
        existing_taxonomy.last_updated = datetime.now()
    else:
        # print(f"\n\nCreate new tag:\n\n{tag.model_dump_json(indent=4)}")
        new_taxonomy = TaxonomyDataTable(
            parent_id=taxonomy.parent_id,
            id=taxonomy.id,
            tag=taxonomy.tag,
            type=taxonomy.type,
            title=taxonomy.title,
            description=taxonomy.description,
            taxonomy=taxonomy.taxonomy,
            parent_taxonomy=taxonomy.parent_taxonomy,
            category_tags=categories,
            last_updated=datetime.now(),
        )
        user_db_get_session().add(new_taxonomy)

    if commit:
        user_db_get_session().commit()


def get_taxonomy_item_list(
    categories: List[str], reset=False
) -> List[TaxonomyDataTable]:
    taxonomy: Dict = get_db_taxonomy(categories=categories, reset=reset)
    all_taxonomy: List[TaxonomyDataTable] = []

    for taxonomy_list in taxonomy.values():
        for taxonomy in taxonomy_list:
            all_taxonomy.append(taxonomy)
    return all_taxonomy


def handle_new_taxonomy_item(
    item, taxonomy_ids: List, cat_for_id: str, existing_id=None, existing_parent_id=None
) -> Taxonomy:
    try:
        new_taxonomy_dict = convert_tags_to_dict(
            item, TAXONOMY_ALL_TAGS, "category_tag"
        )
        new_item = new_taxonomy_dict["category_tag"]
        new_id = (
            (
                new_item["id"]
                if ("id" in new_item and new_item["id"] not in taxonomy_ids)
                else (
                    get_unique_id(
                        cat_for_id + "-" + get_id_str(new_item["taxonomy"]),
                        taxonomy_ids,
                    )
                )
            )
            if existing_id is None
            else existing_id
        )

        new_item["id"] = new_id
        parent_id = (
            (
                new_item["parent_id"]
                if ("parent_id" in new_item)
                else (
                    get_unique_id(
                        cat_for_id + "-" + get_id_str(new_item["parent_taxonomy"]),
                        [],
                    )
                )
            )
            if existing_parent_id is None
            else existing_parent_id
        )
        if parent_id in taxonomy_ids:
            new_item["parent_id"] = parent_id

        new_taxonomy = Taxonomy(
            parent_id=parent_id if parent_id in taxonomy_ids else None,
            id=new_id,
            tag=new_item["tag"] if "tag" in new_item else "",
            type=new_item["type"] if "type" in new_item else "",
            title=new_item["title"] if "title" in new_item else "",
            description=new_item["description"] if "description" in new_item else "",
            taxonomy=new_item["taxonomy"],
            parent_taxonomy=(
                new_item["parent_taxonomy"] if "parent_taxonomy" in new_item else ""
            ),
        )

        return new_taxonomy

    except Exception as e:
        print(f"Error parsing taxonomy: {e}")
        pretty_print(item, "Error parsing taxonomy", force=True)
