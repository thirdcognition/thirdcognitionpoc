import streamlit as st
from typing import List
import chromadb

from langchain_chroma import Chroma
from chromadb.utils.embedding_functions import create_langchain_embedding
from chromadb.config import Settings as ChromaSettings
from chromadb.api import AsyncClientAPI, ClientAPI
from lib.chains.init import get_embeddings

from lib.helpers.shared import pretty_print
from lib.load_env import SETTINGS
from lib.models.user import get_user_chroma_path


def update_rag(
    categories: List[str],
    rag_ids: List[str],
    rag_split: List[str],
    rag_metadatas: List[str],
    existing_ids: List[str] = None,
    existing_collections: List[str] = None,
    type="",
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
        collections.append("rag_" + cat + ("_" + type if type else ""))
        vectorstore = get_chroma_collections(
            "rag_" + cat + ("_" + type if type else "")
        )  # get_vectorstore("rag_" + cat)
        store_complete = False
        retries = 0
        while not store_complete and retries < 3:
            if retries > 0:
                vectorstore.delete(rag_ids)
            retries += 1
            # pretty_print(rag_ids, "Adding ids to rag", force=True)
            # pretty_print(rag_split, "Adding split to rag", force=True)
            # pretty_print(rag_metadatas, "Adding metadata to rag", force=True)
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


def get_chroma_client() -> ClientAPI:
    path = get_user_chroma_path()
    if path is None:
        return None

    clients = st.session_state.get("chroma_clients", {})
    if path in clients:
        return clients[path]

    client = chromadb.PersistentClient(
        path=path, settings=ChromaSettings(anonymized_telemetry=False)
    )
    clients[path] = client
    st.session_state["chroma_clients"] = clients
    return client


def get_chroma_collections(
    name, update=False, embedding_id=None
) -> chromadb.Collection:

    collections = st.session_state.get("collections", {})

    if name in collections and not update:
        return collections[name]

    chroma_client = get_chroma_client()

    if update:
        chroma_client.delete_collection(name=name)

    embedding_function = create_langchain_embedding(
        get_embeddings(embedding_id if embedding_id is not None else "base")
    )

    collection = chroma_client.get_or_create_collection(
        name, embedding_function=embedding_function
    )
    collections[name] = collection
    st.session_state["collections"] = collections
    return collection


def get_vectorstore(id, embedding_id="base", update_vectorstores=False) -> Chroma:
    vectorstores = st.session_state.get("vectorstores", {})

    if id in vectorstores and not update_vectorstores:
        return vectorstores[id]

    chroma_client = get_chroma_client()
    print(f"\n\n\nInit vectorstore {id=} {embedding_id=}\n\n\n")
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=id,
        embedding_function=get_embeddings(embedding_id),
        client_settings=ChromaSettings(anonymized_telemetry=False),
    )

    vectorstores[id] = vectorstore
    st.session_state["vectorstores"] = vectorstores
    return vectorstore


def get_vectorstore_as_retriever(store_id, embedding_id="base", amount_of_documents=5):
    vectorstore = get_vectorstore(store_id, embedding_id)
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": amount_of_documents, "score_threshold": 0.15},
    )
