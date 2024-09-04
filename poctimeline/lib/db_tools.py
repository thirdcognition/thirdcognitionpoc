from datetime import datetime
import io
import os
from typing import Dict, List
import sqlalchemy as sqla
from sqlalchemy.orm import sessionmaker
import chromadb

from langchain_chroma import Chroma
from chromadb.utils.embedding_functions import create_langchain_embedding
from chromadb.config import Settings as ChromaSettings
import streamlit as st
from lib.chains.init import get_embeddings
from lib.document_tools import get_rag_chunks
from lib.load_env import SETTINGS
from lib.models.journey import JourneyModel
from lib.models.sqlite_tables import Base, SourceContents, SourceData, SourceDataTable, JourneyDataTable, SourceType

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


def get_db_sources(reset=False, source=None, categories=None) -> Dict[str, SourceData]:
    db_sources: Dict[str, SourceData] = None
    if (
        "db_sources" not in st.session_state
        or reset
        or source != None
        and source not in st.session_state.db_sources.keys()
    ):
        if source == None:
            sources = database_session.query(SourceDataTable).all()
        else:
            sources = database_session.query(SourceDataTable).filter(
                SourceDataTable.source == source
            )

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
                {k: v for k, v in db_sources.items() if cat in v.category_tag}
            )
        db_sources = new_db_sources
    return db_sources


def delete_db_file(filename: str):
    instance = (
        database_session.query(SourceDataTable)
        .where(SourceDataTable.source == filename)
        .first()
    )
    database_session.delete(instance)
    database_session.commit()
    get_db_sources(reset=True)

def db_file_exists(filename: str) -> bool:
    return database_session.query(
        sqla.exists().where(SourceDataTable.source == filename)
    ).scalar()

def save_db_file(filename, texts:List[str], category:List[str]=None, collections:List[str]=None, uploaded_file:io.BytesIO=None):
    if db_file_exists(filename):
        # If the file exists, get the row and update its text field
        existing_file = (
            database_session.query(SourceDataTable)
            .filter(SourceDataTable.source == filename)
            .first()
        )

        existing_file.texts = (
            texts  # Update the text field with the new content
        )
        existing_file.category_tag = category or existing_file.category_tag
        existing_file.last_updated = datetime.now()
        existing_file.file_data = uploaded_file.getvalue() or existing_file.file_data
        existing_file.chroma_collection = collections or existing_file.chroma_collection
    else:
        # If the file does not exist, create a new row
        file = SourceDataTable(
            source=filename,
            type=SourceType.file,
            texts=texts,
            category_tag=category,
            chroma_collection=collections,
            last_updated=datetime.now(),
            file_data=uploaded_file.getvalue() if uploaded_file else None,
        )
        database_session.add(file)

    database_session.commit()

def update_rag(
    category:List[str],
    rag_ids:List[str],
    rag_split:List[str],
    rag_metadatas:List[str],
    existing_ids:List[str]=None,
    existing_collections:List[str]=None,
):
    if existing_ids is not None and existing_collections is not None and len(existing_ids) > 0:
        for collection in existing_collections:
            vectorstore = get_chroma_collection(collection)
            vectorstore.delete(existing_ids)

    collections = []

    for cat in category:
        collections.append("rag_" + cat)
        vectorstore = get_chroma_collection("rag_" + cat) #get_vectorstore("rag_" + cat)
        store_complete = False
        retries = 0
        while not store_complete and retries < 3:
            if (retries > 0):
                vectorstore.delete(rag_ids)
            retries += 1
            vectorstore.add(
                ids=rag_ids, documents=rag_split, metadatas=rag_metadatas
            )
            rag_items = vectorstore.get(
                    rag_ids,
                    include=["embeddings", "documents", "metadatas"],
                )
            store_complete = True

            for rag_id in rag_ids:
                if rag_id not in rag_items["ids"]:
                    store_complete = False
                    print(f"{rag_id} not in {rag_items["ids"]} - retrying...")
                    break



def update_db_file_and_rag(source:str, category:List[str], texts:List[str], contents:SourceContents, filetype="txt"):
    existing_source = (
        database_session.query(SourceDataTable)
        .filter(SourceDataTable.source == source)
        .first()
    )

    if existing_source is None:
        raise ValueError(f"Source {source} not found in the database.")

    rag_chunks, rag_ids, rag_metadatas = get_rag_chunks(texts, source, category, contents, filetype)

    # print(f"\n\n\n{source=}\n\n{rag_chunks=}\n\n{rag_ids=}\n\n{rag_metadatas=}\n\n")

    update_rag(category, rag_ids, rag_chunks, rag_metadatas, existing_source.chroma_ids if existing_source is not None else None, existing_source.chroma_collection if existing_source is not None else None)

    existing_source.texts = texts  # Update the text field with the new content
    existing_source.source_contents = contents
    existing_source.category_tag = category
    existing_source.chroma_ids = rag_ids
    existing_source.chroma_collection = ["rag_" + cat for cat in category]
    existing_source.last_updated = datetime.now()

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
                {k: v for k, v in db_journey.items() if cat in v.chroma_collection}
            )
        db_journey = new_db_journeys

    return db_journey


collections = {}


def get_chroma_collection(
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
        search_kwargs={"k": amount_of_documents, "score_threshold": 0.3},
    )
