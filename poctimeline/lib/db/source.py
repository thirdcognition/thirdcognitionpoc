from datetime import datetime
import io
from typing import Dict, List
import sqlalchemy as sqla

import streamlit as st
from lib.db.rag import get_chroma_collections, update_rag
from lib.models.user import user_db_commit, user_db_get_session
from lib.document_tools import get_source_rag_chunks, get_topic_rag_chunks

from lib.models.source import (
    SourceContents,
    SourceDataTable,
    SourceType,
)
def get_db_sources(
    reset=False, source=None, categories: List[str] = None
) -> Dict[str, SourceDataTable]:
    db_sources: Dict[str, SourceDataTable] = None
    if (
        "db_sources" not in st.session_state
        or reset
        or source != None
        and source not in st.session_state.db_sources.keys()
    ):
        if source is not None:
            sources = (
                user_db_get_session()
                .query(SourceDataTable)
                .filter(SourceDataTable.source == source)
            )
        else:
            sources = list(user_db_get_session().query(SourceDataTable).all())

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
            db_sources[source.source] = source

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


def delete_db_source(filename: str, commit: bool = True):
    instance = (
        user_db_get_session()
        .query(SourceDataTable)
        .where(SourceDataTable.source == filename)
        .first()
    )
    if instance is None:
        return

    chroma_collections = instance.chroma_collections
    chroma_ids = instance.chroma_ids
    if len(chroma_ids) > 0:
        for collection in chroma_collections:
            try:
                vectorstore = get_chroma_collections(collection)
                vectorstore.delete(chroma_ids)
            except Exception as e:
                print(e)
    user_db_get_session().delete(instance)
    if commit:
        user_db_commit()


def db_source_exists(filename: str) -> bool:
    return (
        user_db_get_session()
        .query(sqla.exists().where(SourceDataTable.source == filename))
        .scalar()
    )


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
            user_db_get_session()
            .query(SourceDataTable)
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
        user_db_get_session().add(file)

    user_db_get_session().commit()


def update_db_source_rag(
    source: str,
    categories: List[str],
    texts: List[str],
    contents: SourceContents,
    filetype="txt",
):
    existing_source = (
        user_db_get_session()
        .query(SourceDataTable)
        .filter(SourceDataTable.source == source)
        .first()
    )

    if existing_source is None:
        raise ValueError(f"Source {source} not found in the database.")

    rag_chunks = []
    rag_ids = []
    rag_metadatas = []

    source_rag_chunks, source_rag_ids, source_rag_metadatas = get_source_rag_chunks(
        texts, source, categories, contents, filetype
    )

    existing_chroma_ids = (
        existing_source.chroma_ids if existing_source is not None else []
    )
    existing_chroma_collections = (
        existing_source.chroma_collections if existing_source is not None else []
    )

    existing_chroma_ids = None if len(existing_chroma_ids) == 0 else existing_chroma_ids
    existing_chroma_collections = (
        None if len(existing_chroma_collections) == 0 else existing_chroma_collections
    )

    rag_chunks.extend(source_rag_chunks)
    rag_ids.extend(source_rag_ids)
    rag_metadatas.extend(source_rag_metadatas)

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
    user_db_get_session().commit()


