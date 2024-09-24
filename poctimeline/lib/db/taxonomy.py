from datetime import datetime
from functools import cache
from typing import Dict, List
import sqlalchemy as sqla

import streamlit as st
from lib.db.sqlite import db_session

from lib.helpers import convert_tags_to_dict, get_id_str, get_unique_id, pretty_print
from lib.models.taxonomy import TaxonomyDataTable, Taxonomy
from lib.prompts.taxonomy import TAXONOMY_ALL_TAGS


def db_taxonomy_exists(tag_id: str) -> bool:
    return (
        db_session().query(sqla.exists().where(TaxonomyDataTable.id == tag_id)).scalar()
    )


@cache
def get_taxonomy_by_id(
    concept_category_id: str,
) -> TaxonomyDataTable:
    return (
        db_session()
        .query(TaxonomyDataTable)
        .filter(TaxonomyDataTable.id == concept_category_id)
        .first()
    )


def get_db_taxonomy(
    categories: List[str] = None, reset: bool = False
) -> Dict[str, List[Taxonomy]]:
    db_concept_taxonomy: Dict[str, List[Taxonomy]] = None
    if "db_concept_taxonomy" not in st.session_state or reset:
        found_categories = db_session().query(TaxonomyDataTable).distinct().all()

        if categories is not None:
            found_categories = [
                category
                for category in found_categories
                if any(cat in category.category_tags for cat in categories)
            ]

        if "db_concept_taxonomy" not in st.session_state or reset:
            db_concept_taxonomy = {}
        else:
            db_concept_taxonomy = st.session_state.db_taxonomy
        for category in found_categories:
            file_categories = category.category_tags
            for ict in file_categories:
                if ict not in db_concept_taxonomy:
                    db_concept_taxonomy[ict] = []
                concept_taxonomy = Taxonomy(**category.concept_taxonomy.__dict__)
                concept_taxonomy.id = category.id
                db_concept_taxonomy[ict].append(concept_taxonomy)

        st.session_state.db_taxonomy = db_concept_taxonomy
    else:
        db_concept_taxonomy = st.session_state.db_taxonomy

    # if isinstance(categories, str):
    #     categories = [categories]
    # if categories:
    #     new_db_concept_taxonomy = {}
    #     for cat in categories:
    #         new_db_concept_taxonomy.update(
    #             {k: v for k, v in db_concept_taxonomy.items() if cat in v.category_tags}
    #         )
    #     db_concept_taxonomy = new_db_concept_taxonomy
    return db_concept_taxonomy


def delete_db_taxonomy(taxonomy_id: str, commit: bool = True):
    instance = (
        db_session()
        .query(TaxonomyDataTable)
        .where(TaxonomyDataTable.id == taxonomy_id)
        .first()
    )
    db_session().delete(instance)
    if commit:
        db_session().commit()


def update_db_taxonomy(taxonomy: Taxonomy, categories=List[str], commit: bool = True):
    if db_taxonomy_exists(taxonomy.id):
        # print(f"\n\nUpdate existing taxonomy:\n\n{taxonomy.model_dump_json(indent=4)}")
        concept_category = (
            db_session()
            .query(TaxonomyDataTable)
            .filter(TaxonomyDataTable.id == taxonomy.id)
            .first()
        )
        concept_category.concept_taxonomy = taxonomy
        concept_category.category_tags = list(
            set(categories + concept_category.category_tags)
        )
        concept_category.last_updated = datetime.now()
    else:
        # print(f"\n\nCreate new tag:\n\n{tag.model_dump_json(indent=4)}")
        concept_category = TaxonomyDataTable(
            id=taxonomy.id,
            parent_id=taxonomy.parent_id,
            concept_taxonomy=taxonomy,
            category_tags=categories,
            last_updated=datetime.now(),
        )
        db_session().add(concept_category)

    if commit:
        db_session().commit()


def get_taxonomy_item_list(categories: List[str], reset=False) -> List[Taxonomy]:
    concepts = get_db_taxonomy(categories=categories, reset=reset)
    all_concepts: List[Taxonomy] = []

    for category, concept_list in concepts.items():
        for concept in concept_list:
            all_concepts.append(concept)
    return all_concepts


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
