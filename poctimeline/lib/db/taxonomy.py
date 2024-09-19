from datetime import datetime
from functools import cache
from typing import Dict, List
import sqlalchemy as sqla

import streamlit as st
from lib.db.sqlite import db_session

from lib.models.sqlite_tables import (
    ConceptTaxonomyDataTable,
    ConceptTaxonomy,
)


def db_concept_taxonomy_exists(tag_id: str) -> bool:
    return (
        db_session()
        .query(sqla.exists().where(ConceptTaxonomyDataTable.id == tag_id))
        .scalar()
    )


@cache
def get_concept_taxonomy_by_id(
    concept_category_id: str,
) -> ConceptTaxonomyDataTable:
    return (
        db_session()
        .query(ConceptTaxonomyDataTable)
        .filter(ConceptTaxonomyDataTable.id == concept_category_id)
        .first()
    )


def get_db_concept_taxonomy(
    categories: List[str] = None, reset: bool = False
) -> Dict[str, List[ConceptTaxonomy]]:
    db_concept_taxonomy: Dict[str, List[ConceptTaxonomy]] = None
    if "db_concept_taxonomy" not in st.session_state or reset:
        found_categories = db_session().query(ConceptTaxonomyDataTable).distinct().all()

        if categories is not None:
            found_categories = [
                category
                for category in found_categories
                if any(cat in category.category_tags for cat in categories)
            ]

        if "db_concept_taxonomy" not in st.session_state or reset:
            db_concept_taxonomy = {}
        else:
            db_concept_taxonomy = st.session_state.db_concept_taxonomy
        for category in found_categories:
            file_categories = category.category_tags
            for ict in file_categories:
                if ict not in db_concept_taxonomy:
                    db_concept_taxonomy[ict] = []
                concept_taxonomy = ConceptTaxonomy(**category.concept_taxonomy.__dict__)
                concept_taxonomy.id = category.id
                db_concept_taxonomy[ict].append(concept_taxonomy)

        st.session_state.db_concept_taxonomy = db_concept_taxonomy
    else:
        db_concept_taxonomy = st.session_state.db_concept_taxonomy

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


def delete_db_concept_taxonomy(taxonomy_id: str, commit: bool = True):
    instance = (
        db_session()
        .query(ConceptTaxonomyDataTable)
        .where(ConceptTaxonomyDataTable.id == taxonomy_id)
        .first()
    )
    db_session().delete(instance)
    if commit:
        db_session().commit()


def update_concept_taxonomy(
    taxonomy: ConceptTaxonomy, categories=List[str], commit: bool = True
):
    if db_concept_taxonomy_exists(taxonomy.id):
        # print(f"\n\nUpdate existing taxonomy:\n\n{taxonomy.model_dump_json(indent=4)}")
        concept_category = (
            db_session()
            .query(ConceptTaxonomyDataTable)
            .filter(ConceptTaxonomyDataTable.id == taxonomy.id)
            .first()
        )
        concept_category.concept_taxonomy = taxonomy
        concept_category.category_tags = list(
            set(categories + concept_category.category_tags)
        )
        concept_category.last_updated = datetime.now()
    else:
        # print(f"\n\nCreate new tag:\n\n{tag.model_dump_json(indent=4)}")
        concept_category = ConceptTaxonomyDataTable(
            id=taxonomy.id,
            parent_id=taxonomy.parent_id,
            concept_taxonomy=taxonomy,
            category_tags=categories,
            last_updated=datetime.now(),
        )
        db_session().add(concept_category)

    if commit:
        db_session().commit()
