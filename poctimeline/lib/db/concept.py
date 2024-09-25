from datetime import datetime
from functools import cache
from typing import List, Union
import sqlalchemy as sqla

from lib.db.rag import get_chroma_collections, update_rag
from lib.db.sqlite import db_session
from lib.document_tools import get_concept_rag_chunks

from lib.models.taxonomy import (
    Taxonomy,
)
from lib.models.concepts import (
    ConceptDataTable,
    ConceptData,
)
from lib.models.source import (
    SourceDataTable,
)
from lib.helpers import pretty_print

existing_concept_ids = []


def db_concept_id_exists(concept_id: str) -> bool:
    return (
        db_session()
        .query(sqla.exists().where(ConceptDataTable.id == concept_id))
        .scalar()
    )


@cache
def get_concept_by_id(concept_id: str) -> ConceptDataTable:
    return (
        db_session()
        .query(ConceptDataTable)
        .filter(ConceptDataTable.id == concept_id)
        .first()
    )


def get_existing_concept_ids(refresh: bool = False) -> List[str]:
    global existing_concept_ids
    if refresh or len(existing_concept_ids) == 0:
        existing_concept_ids = [
            concept.id for concept in db_session().query(ConceptDataTable).all()
        ]
    return existing_concept_ids


def get_db_concepts(
    id: str = None,
    source: str = None,
    categories: List[str] = None,
    taxonomy: Taxonomy = None,
) -> Union[ConceptDataTable, List[ConceptDataTable]]:
    if id is not None:
        return (
            db_session()
            .query(ConceptDataTable)
            .filter(ConceptDataTable.id == id)
            .first()
        )

    concepts = db_session().query(ConceptDataTable).all()

    if source is not None:
        # print(f"Filtering concepts by source: {source}")
        # pretty_print(
        #     [{"sources": concept.sources, "id": concept.id} for concept in concepts],
        #     force=True,
        # )
        concepts = [concept for concept in concepts if source in concept.sources]
        # pretty_print(concepts, "filtered concepts", force=True)
    if taxonomy is not None:
        concepts = [
            concept for concept in concepts if str(taxonomy.id) in concept.taxonomy
        ]
    if categories is not None:
        concepts = [
            concept
            for concept in concepts
            if set(categories).issubset(concept.category_tags)
        ]

    return concepts

def update_db_concept(concept: ConceptData, categories=List[str], commit: bool = True):
    if db_concept_id_exists(concept.id):
        # print(f"\n\nUpdate existing concept:\n\n{concept.model_dump_json(indent=4)}")
        db_concept = (
            db_session()
            .query(ConceptDataTable)
            .filter(ConceptDataTable.id == concept.id)
            .first()
        )
        db_concept.parent_id = concept.parent_id
        db_concept.concept_contents = concept
        db_concept.category_tags = list(
            set(categories + db_concept.category_tags)
        )
        db_concept.taxonomy = concept.taxonomy
        new_sources = concept.sources if isinstance(concept.sources, list) else [concept.sources]
        db_concept.sources = list(set(new_sources + db_concept.sources))
        db_concept.last_updated = datetime.now()


    if commit:
        db_session().commit()

def delete_db_concept(concept_id: str, commit: bool = True):
    instance = (
        db_session()
        .query(ConceptDataTable)
        .where(ConceptDataTable.id == concept_id)
        .first()
    )
    chroma_collections = instance.chroma_collections
    chroma_ids = instance.chroma_ids
    if len(chroma_ids) > 0:
        for collection in chroma_collections:
            try:
                vectorstore = get_chroma_collections(collection)
                vectorstore.delete(chroma_ids)
            except Exception as e:
                print(e)
    db_session().delete(instance)
    if commit:
        db_session().commit()


def update_db_concept_rag(
    source: str,
    categories: List[str],
    concepts: List[ConceptData] = None,
):
    category_id = "-".join(categories)
    defined_concept_ids = [concept.id for concept in concepts] if concepts else []

    existing_concepts = (
        db_session()
        .query(ConceptDataTable)
        .filter(
            ConceptDataTable.id.in_(defined_concept_ids),
        )
        .distinct()
        .all()
    )
    existing_concepts = [
        concept
        for concept in existing_concepts
        if set(categories).issubset(concept.category_tags)
    ]

    rag_chunks = []
    rag_ids = []
    rag_metadatas = []

    concepts_by_id = {}
    new_ids = []
    old_ids = [concept.id for concept in existing_concepts]
    if concepts is not None:
        resp: List[tuple[ConceptData, List, List, List]] = get_concept_rag_chunks(
            category_id=category_id, concepts=concepts
        )
        for concept, concept_chunks, concept_ids, concept_metadatas in resp:
            rag_chunks.extend(concept_chunks)
            rag_ids.extend(concept_ids)
            rag_metadatas.extend(concept_metadatas)
            concepts_by_id[str(concept.id)] = {"concept": concept, "rag_ids": rag_ids}
            if str(concept.id) not in old_ids:
                new_ids.append(str(concept.id))

    existing_chroma_ids = []
    existing_chroma_collections = []
    handled_concepts = []

    concept_ids = []

    if existing_concepts is not None:
        for concept in existing_concepts:
            existing_chroma_ids.extend(concept.chroma_ids)
            existing_chroma_collections.extend(concept.chroma_collections)

            if concept.id in concepts_by_id:
                concept_ids.append(concept.id)
                new_concept: ConceptData = concepts_by_id[str(concept.id)]["concept"]
                parent_id = concept.parent_id or new_concept.parent_id
                if parent_id is not concept.parent_id:
                    if parent_id not in old_ids or parent_id not in new_ids:
                        parent_concept = db_concept_id_exists(parent_id)
                    else:
                        parent_concept = True
                    if parent_concept or parent_id == new_concept.id:
                        print(f"Parent concept with id {parent_id} does not exist.")
                        parent_id = None

                concept.concept_contents = new_concept
                concept.taxonomy = new_concept.taxonomy
                concept.parent_id = parent_id
                concept.chroma_ids = concepts_by_id[str(concept.id)]["rag_ids"]
                concept.chroma_collections = [
                    "rag_" + cat + "_concept" for cat in categories
                ]
                concept.category_tags = categories
                concept.last_updated = datetime.now()
                concept.sources = list(set(list(concept.sources) + [source]))
                handled_concepts.append(concept.id)

    if concepts is not None and len(concepts) > 0:
        for concept in concepts:
            if concept.id not in handled_concepts:
                parent_id = concept.parent_id
                if parent_id not in old_ids or parent_id not in new_ids:
                    parent_concept = db_concept_id_exists(parent_id)
                else:
                    parent_concept = True
                if parent_concept or parent_id == concept.id:
                    print(f"Parent concept with id {parent_id} does not exist.")
                    parent_id = None
                new_id = concept.id
                if db_concept_id_exists(concept.id):
                    # all_concept_ids = get_existing_concept_ids(True)
                    matching_ids = sorted(
                        set(
                            [
                                concept_id
                                for concept_id in existing_concept_ids
                                if concept.id in concept_id
                            ]
                        )
                    )
                    new_id = concept.id + "-" + str(len(matching_ids))
                    print(
                        f"Concept with id {concept.id} already exists. Changing id to {new_id}"
                    )
                    defined_concept_ids.remove(concept.id)
                    defined_concept_ids.append(new_id)
                concept_ids.append(new_id)
                new_sources = concept.sources if isinstance(concept.sources, list) else [concept.sources]
                new_concept = ConceptDataTable(
                    id=new_id,
                    parent_id=parent_id,
                    concept_contents=concept,
                    taxonomy=concept.taxonomy,
                    category_tags=categories,
                    sources=list(set(new_sources + [source])),
                    chroma_ids=concepts_by_id[str(concept.id)]["rag_ids"],
                    chroma_collections=[
                        "rag_" + cat + "_concept" for cat in categories
                    ],
                    last_updated=datetime.now(),
                )
                db_session().add(new_concept)

    existing_chroma_ids = None if len(existing_chroma_ids) == 0 else existing_chroma_ids
    existing_chroma_collections = (
        None if len(existing_chroma_collections) == 0 else existing_chroma_collections
    )

    if len(rag_chunks) > 0:
        update_rag(
            categories,
            rag_ids,
            rag_chunks,
            rag_metadatas,
            existing_chroma_ids,
            existing_chroma_collections,
            type="concept",
        )

    existing_source = (
        db_session()
        .query(SourceDataTable)
        .filter(SourceDataTable.source == source)
        .first()
    )

    if existing_source is not None:
        existing_source.source_concepts = list(
            set(list(existing_source.source_concepts) + concept_ids)
        )
        existing_source.last_updated = datetime.now()

    db_session().commit()
