from datetime import datetime
from functools import cache
import io
import os
from typing import Dict, List, Union
import sqlalchemy as sqla
from sqlalchemy.orm import sessionmaker
import chromadb

from langchain_chroma import Chroma
from chromadb.utils.embedding_functions import create_langchain_embedding
from chromadb.config import Settings as ChromaSettings
import streamlit as st
from lib.chains.init import get_embeddings
from lib.document_tools import get_source_rag_chunks, get_concept_rag_chunks
from lib.load_env import SETTINGS
from lib.models.journey import JourneyModel
from lib.models.sqlite_tables import (
    Base,
    ConceptCategoryDataTable,
    ConceptTaxonomy,
    ConceptDataTable,
    ConceptData,
    SourceContents,
    SourceData,
    SourceDataTable,
    JourneyDataTable,
    SourceType,
)
from lib.chains.prompt_generator import CustomPrompt, CustomPromptContainer

chroma_client = None
database_session = None


def init_db():
    global database_session
    if database_session is None:
        engine = sqla.create_engine("sqlite:///{}".format(SETTINGS.sqlite_db))
        Base.metadata.create_all(engine)
        DatabaseSession = sessionmaker(bind=engine)
        database_session = DatabaseSession()

    return database_session


def get_db_sources(
    reset=False, source=None, categories: List[str] = None
) -> Dict[str, SourceData]:
    db_sources: Dict[str, SourceData] = None
    if (
        "db_sources" not in st.session_state
        or reset
        or source != None
        and source not in st.session_state.db_sources.keys()
    ):
        if source is not None:
            sources = database_session.query(SourceDataTable).filter(
                SourceDataTable.source == source
            )
        else:
            sources = list(database_session.query(SourceDataTable).all())

        if categories is not None:
            sources = [
                source
                for source in sources
                if any(cat in source.category_tags for cat in categories)
            ]

        if "db_sources" not in st.session_state or reset:
            db_sources = {}
        else:
            db_sources = st.session_state.db_sources

        for source in sources:
            db_sources[source.source] = SourceData(**source.__dict__)

        st.session_state.db_sources = db_sources
    else:
        db_sources = st.session_state.db_sources

    if isinstance(categories, str):
        categories = [categories]
    if categories:
        new_db_sources = {}
        for cat in categories:
            new_db_sources.update(
                {k: v for k, v in db_sources.items() if cat in v.category_tags}
            )
        db_sources = new_db_sources
    return db_sources


def db_commit():
    database_session.commit()
    get_db_sources(reset=True)


def delete_db_source(filename: str, commit: bool = True):
    instance = (
        database_session.query(SourceDataTable)
        .where(SourceDataTable.source == filename)
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
    database_session.delete(instance)
    if commit:
        db_commit()


def db_source_exists(filename: str) -> bool:
    return database_session.query(
        sqla.exists().where(SourceDataTable.source == filename)
    ).scalar()


def save_db_source(
    filename,
    texts: List[str],
    categories: List[str] = None,
    collections: List[str] = None,
    uploaded_file: io.BytesIO = None,
):
    if db_source_exists(filename):
        # If the file exists, get the row and update its text field
        existing_file = (
            database_session.query(SourceDataTable)
            .filter(SourceDataTable.source == filename)
            .first()
        )

        existing_file.texts = texts  # Update the text field with the new content
        existing_file.category_tags = categories or existing_file.category_tags
        existing_file.last_updated = datetime.now()
        existing_file.file_data = uploaded_file.getvalue() or existing_file.file_data
        existing_file.chroma_collections = (
            collections or existing_file.chroma_collections
        )
    else:
        # If the file does not exist, create a new row
        file = SourceDataTable(
            source=filename,
            type=SourceType.file,
            texts=texts,
            category_tags=categories,
            chroma_collections=collections,
            last_updated=datetime.now(),
            file_data=uploaded_file.getvalue() if uploaded_file else None,
        )
        database_session.add(file)

    database_session.commit()


def update_rag(
    categories: List[str],
    rag_ids: List[str],
    rag_split: List[str],
    rag_metadatas: List[str],
    existing_ids: List[str] = None,
    existing_collections: List[str] = None,
):
    if (
        existing_ids is not None
        and existing_collections is not None
        and len(existing_ids) > 0
    ):
        for collection in existing_collections:
            try:
                vectorstore = get_chroma_collections(collection)
                vectorstore.delete(existing_ids)
            except Exception as e:
                print(e)

    collections = []

    for cat in categories:
        collections.append("rag_" + cat)
        vectorstore = get_chroma_collections(
            "rag_" + cat
        )  # get_vectorstore("rag_" + cat)
        store_complete = False
        retries = 0
        while not store_complete and retries < 3:
            if retries > 0:
                vectorstore.delete(rag_ids)
            retries += 1
            vectorstore.add(ids=rag_ids, documents=rag_split, metadatas=rag_metadatas)
            rag_items = vectorstore.get(
                rag_ids,
                include=["embeddings", "documents", "metadatas"],
            )
            store_complete = True

            for rag_id in rag_ids:
                if rag_id not in rag_items["ids"]:
                    store_complete = False
                    print(f"{rag_id} not in {rag_items['ids']} - retrying...")
                    break


def db_concept_id_exists(concept_id: str) -> bool:
    return database_session.query(
        sqla.exists().where(ConceptDataTable.id == concept_id)
    ).scalar()


existing_concept_ids = []


def get_existing_concept_ids(refresh: bool = False) -> List[str]:
    global existing_concept_ids
    if refresh or len(existing_concept_ids) == 0:
        existing_concept_ids = [
            concept.id for concept in database_session.query(ConceptDataTable).all()
        ]
    return existing_concept_ids


def db_concept_category_tag_id_exists(tag_id: str) -> bool:
    return database_session.query(
        sqla.exists().where(ConceptCategoryDataTable.id == tag_id)
    ).scalar()


def get_db_concept_taxonomy(
    categories: List[str] = None, reset: bool = False
) -> Dict[str, List[ConceptTaxonomy]]:
    db_concept_taxonomy: Dict[str, List[ConceptTaxonomy]] = None
    if "db_concept_taxonomy" not in st.session_state or reset:
        found_categories = (
            database_session.query(ConceptCategoryDataTable).distinct().all()
        )

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
                concept_taxonomy = ConceptTaxonomy(
                    **category.concept_category_tag.__dict__
                )
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


def update_concept_category(tag: ConceptTaxonomy, categories=List[str]):
    if db_concept_category_tag_id_exists(tag.id):
        # print(f"\n\nUpdate existing tag:\n\n{tag.model_dump_json(indent=4)}")
        concept_category = (
            database_session.query(ConceptCategoryDataTable)
            .filter(ConceptCategoryDataTable.id == tag.id)
            .first()
        )
        concept_category.concept_category_tag = tag
        concept_category.category_tags = list(
            set(categories + concept_category.category_tags)
        )
        concept_category.last_updated = datetime.now()
    else:
        # print(f"\n\nCreate new tag:\n\n{tag.model_dump_json(indent=4)}")
        concept_category = ConceptCategoryDataTable(
            id=tag.id,
            parent_id=tag.parent_id,
            concept_category_tag=tag,
            category_tags=categories,
            last_updated=datetime.now(),
        )
        database_session.add(concept_category)

    database_session.commit()


@cache
def get_concept_category_tag_by_id(
    concept_category_id: str,
) -> ConceptCategoryDataTable:
    return (
        database_session.query(ConceptCategoryDataTable)
        .filter(ConceptCategoryDataTable.id == concept_category_id)
        .first()
    )


@cache
def get_concept_by_id(concept_id: str) -> ConceptDataTable:
    return (
        database_session.query(ConceptDataTable)
        .filter(ConceptDataTable.id == concept_id)
        .first()
    )


def get_db_concepts(
    id: str = None,
    source: str = None,
    categories: List[str] = None,
    taxonomy: ConceptTaxonomy = None,
) -> Union[ConceptDataTable, List[ConceptDataTable]]:
    if id is not None:
        return (
            database_session.query(ConceptDataTable)
            .filter(ConceptDataTable.id == id)
            .first()
        )

    concepts = database_session.query(ConceptDataTable).all()

    if source is not None:
        concepts = [concept for concept in concepts if source in concept.sources]
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


def delete_db_concept(concept_id: str, commit: bool = True):
    instance = (
        database_session.query(ConceptDataTable)
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
    database_session.delete(instance)
    if commit:
        database_session.commit()


def update_db_source_rag_concepts(
    source: str,
    categories: List[str],
    texts: List[str],
    contents: SourceContents,
    concepts: List[ConceptData] = None,
    filetype="txt",
):
    existing_source = (
        database_session.query(SourceDataTable)
        .filter(SourceDataTable.source == source)
        .first()
    )

    if existing_source is None:
        raise ValueError(f"Source {source} not found in the database.")

    category_id = "-".join(categories)
    defined_concept_ids = (
        [(category_id + "-" + concept.id) for concept in concepts] if concepts else []
    )

    existing_concepts = (
        database_session.query(ConceptDataTable)
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

    source_rag_chunks, source_rag_ids, source_rag_metadatas = get_source_rag_chunks(
        texts, source, categories, contents, filetype
    )

    rag_chunks.extend(source_rag_chunks)
    rag_ids.extend(source_rag_ids)
    rag_metadatas.extend(source_rag_metadatas)

    concepts_by_id = {}
    new_ids = []
    old_ids = [concept.id for concept in existing_concepts]
    if concepts is not None:
        resp: List[tuple[ConceptData, List, List, List]] = get_concept_rag_chunks(
            category_id=category_id, concepts=concepts
        )
        for concept, concept_chunks, concept_ids, concept_metadatas in resp:
            concept.id = category_id + "-" + concept.id
            if concept.parent_id is not None:
                concept.parent_id = category_id + "-" + concept.parent_id

            rag_chunks.extend(concept_chunks)
            rag_ids.extend(concept_ids)
            rag_metadatas.extend(concept_metadatas)
            concepts_by_id[str(concept.id)] = {"concept": concept, "rag_ids": rag_ids}
            if str(concept.id) not in old_ids:
                new_ids.append(str(concept.id))

    # print(f"\n\n\n{source=}\n\n{rag_chunks=}\n\n{rag_ids=}\n\n{rag_metadatas=}\n\n")

    existing_chroma_ids = (
        existing_source.chroma_ids if existing_source is not None else []
    )
    existing_chroma_collections = (
        existing_source.chroma_collections if existing_source is not None else []
    )

    handled_concepts = []

    if existing_concepts is not None:
        for concept in existing_concepts:
            existing_chroma_ids.extend(concept.chroma_ids)
            existing_chroma_collections.extend(concept.chroma_collections)

            if concept.id in concepts_by_id:
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
                concept.chroma_collections = ["rag_" + cat for cat in categories]
                concept.category_tags = categories
                concept.last_updated = datetime.now()
                handled_concepts.append(concept.id)

        # filter unique from existing chroma ids and collections
        existing_chroma_ids = list(set(existing_chroma_ids))
        existing_chroma_collections = list(set(existing_chroma_collections))
    if concepts is not None:
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
                    all_concept_ids = get_existing_concept_ids(True)
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

                new_concept = ConceptDataTable(
                    id=new_id,
                    parent_id=parent_id,
                    concept_contents=concept,
                    taxonomy=concept.taxonomy,
                    category_tags=categories,
                    chroma_ids=concepts_by_id[str(concept.id)]["rag_ids"],
                    chroma_collections=["rag_" + cat for cat in categories],
                    last_updated=datetime.now(),
                )
                database_session.add(new_concept)

    existing_chroma_ids = None if len(existing_chroma_ids) == 0 else existing_chroma_ids
    existing_chroma_collections = (
        None if len(existing_chroma_collections) == 0 else existing_chroma_collections
    )

    update_rag(
        categories,
        rag_ids,
        rag_chunks,
        rag_metadatas,
        existing_chroma_ids,
        existing_chroma_collections,
    )

    existing_source.texts = texts  # Update the text field with the new content
    existing_source.source_contents = contents
    existing_source.category_tags = categories
    existing_source.chroma_ids = rag_ids
    existing_source.chroma_collections = ["rag_" + cat for cat in categories]
    existing_source.last_updated = datetime.now()
    existing_source.source_concepts = defined_concept_ids
    database_session.commit()


def get_db_journey(
    journey_name: str = None, chroma_collections=None, reset=False
) -> Dict[str, JourneyModel]:
    db_journey: Dict[str, JourneyModel] = None
    if "db_journey" not in st.session_state or reset:
        if journey_name is None:
            journey = database_session.query(JourneyDataTable).all()
        else:
            journey = (
                database_session.query(JourneyDataTable)
                .filter(JourneyDataTable.journeyname == journey_name)
                .all()
            )
        db_journey = {}

        for step in journey:
            db_journey[step.journeyname] = JourneyModel(**step.__dict__)

        st.session_state.db_journey = db_journey
    else:
        db_journey = st.session_state.db_journey

    if isinstance(chroma_collections, str):
        chroma_collections = [chroma_collections]
    if chroma_collections:
        new_db_journeys = {}
        for cat in chroma_collections:
            new_db_journeys.update(
                {k: v for k, v in db_journey.items() if cat in v.chroma_collections}
            )
        db_journey = new_db_journeys

    return db_journey


collections = {}


def get_chroma_collections(
    name, update=False, path=SETTINGS.chroma_path, embedding_id=None
) -> chromadb.Collection:
    global collections

    if name in collections and not update:
        return collections[name]

    global chroma_client
    chroma_client = chroma_client or chromadb.PersistentClient(
        path=path, settings=ChromaSettings(anonymized_telemetry=False)
    )

    if update:
        chroma_client.delete_collection(name=name)

    embedding_function = None
    if embedding_id is not None:
        embedding_function = create_langchain_embedding(get_embeddings(embedding_id))
    else:
        embedding_function = create_langchain_embedding(get_embeddings("base"))

    collection = chroma_client.get_or_create_collection(
        name, embedding_function=embedding_function
    )
    collections[name] = collection
    return collection


vectorstores = {}


def get_vectorstore(
    id, embedding_id="base", update_vectorstores=False, path=SETTINGS.chroma_path
) -> Chroma:
    global chroma_client
    chroma_client = chroma_client or chromadb.PersistentClient(
        path=path, settings=ChromaSettings(anonymized_telemetry=False)
    )

    global vectorstores

    if id in vectorstores and not update_vectorstores:
        return vectorstores[id]

    print(f"\n\n\nInit vectorstore {id=} {embedding_id=}\n\n\n")
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=id,
        embedding_function=get_embeddings(embedding_id),
        client_settings=ChromaSettings(anonymized_telemetry=False),
    )

    vectorstores[id] = vectorstore
    return vectorstore


def get_vectorstore_as_retriever(store_id, embedding_id="base", amount_of_documents=5):
    vectorstore = get_vectorstore(store_id, embedding_id)
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": amount_of_documents, "score_threshold": 0.15},
    )
