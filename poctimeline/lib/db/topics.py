from datetime import datetime
import io
from typing import Dict, List
import sqlalchemy as sqla

import streamlit as st
from lib.db.rag import get_chroma_collections, update_rag
from lib.models.user import user_db_commit, user_db_get_session
from lib.document_tools import get_topic_rag_chunks

from lib.models.source import (
    SourceContents,
    SourceDataTable,
)
from lib.models.topics import TopicDataTable, dict_to_topic_data_table
from lib.models.reference import (
    Reference,
    ReferenceType,
    is_in_references,
    unique_references,
)


def get_db_topics(
    reset=False, id=None, categories: List[str] = None
) -> Dict[str, TopicDataTable]:
    db_topics: Dict[str, TopicDataTable] = None
    if (
        "db_topics" not in st.session_state
        or reset
        or id != None
        and id not in st.session_state.db_topics.keys()
    ):
        if id is not None:
            topics = (
                user_db_get_session()
                .query(TopicDataTable)
                .filter(TopicDataTable.id == id)
            )
        else:
            topics = list(user_db_get_session().query(TopicDataTable).all())

        if categories is not None:
            topics = [topic for topic in topics if topic.category_tags in categories]

        if "db_topics" not in st.session_state or reset:
            db_topics = {}
        else:
            db_topics = st.session_state.db_topics

        for topic in topics:
            db_topics[topic.id] = topic

        st.session_state.db_topics = db_topics
    else:
        db_topics = st.session_state.db_topics

    if isinstance(categories, str):
        categories = [categories]
    if categories:
        new_db_topics = {}
        for cat in categories:
            new_db_topics.update(
                {k: v for k, v in db_topics.items() if cat in v.category_tags}
            )
        db_topics = new_db_topics
    return db_topics


def get_topic_by_id(id: str) -> TopicDataTable:
    if "db_topics" in st.session_state and id in st.session_state.db_topics:
        return st.session_state.db_topics[id]
    else:
        topic = (
            user_db_get_session()
            .query(TopicDataTable)
            .filter(TopicDataTable.id == id)
            .first()
        )
        if "db_topics" not in st.session_state:
            st.session_state.db_topics = {}
        st.session_state.db_topics[id] = topic
        return topic


def delete_db_topic(id: str, commit: bool = True):
    instance = (
        user_db_get_session()
        .query(TopicDataTable)
        .where(TopicDataTable.id == id)
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


def db_topic_exists(id: str) -> bool:
    return (
        user_db_get_session()
        .query(sqla.exists().where(TopicDataTable.id == id))
        .scalar()
    )


def save_db_topic(
    id,
    page_content: str,
    page_number: int,
    topic_index: int,
    metadata: dict,
    topic: str,
    instruct: str,
    summary: str,
    category_tags: List[str] = None,
    collections: List[str] = None,
    chroma_ids: List[str] = None,
):
    if db_topic_exists(id):
        existing_topic = (
            user_db_get_session()
            .query(TopicDataTable)
            .filter(TopicDataTable.id == id)
            .first()
        )
        existing_topic.page_content = page_content
        existing_topic.page_number = page_number
        existing_topic.topic_index = topic_index
        existing_topic.doc_metadata = metadata
        existing_topic.topic = topic
        existing_topic.instruct = instruct
        existing_topic.summary = summary
        existing_topic.category_tags = category_tags or existing_topic.category_tags
        existing_topic.chroma_collections = (
            collections or existing_topic.chroma_collections
        )
        existing_topic.chroma_ids = chroma_ids or existing_topic.chroma_ids
    else:
        new_topic = TopicDataTable(
            id=id,
            page_content=page_content,
            page_number=page_number,
            topic_index=topic_index,
            doc_metadata=metadata,
            topic=topic,
            instruct=instruct,
            summary=summary,
            category_tags=category_tags,
            chroma_collections=collections,
            chroma_ids=chroma_ids,
        )
        user_db_get_session().add(new_topic)

    user_db_get_session().commit()


def update_db_topic_rag(
    source: str,
    categories: List[str],
    topics: List[TopicDataTable] = None,
):
    topics = (
        [
            dict_to_topic_data_table(topic) if not isinstance(topic, TopicDataTable) else topic
            for topic in topics
        ]
        if topics
        else []
    )
    defined_topic_ids = [topic.id for topic in topics] if topics else []

    existing_topics = (
        user_db_get_session()
        .query(TopicDataTable)
        .filter(
            TopicDataTable.id.in_(defined_topic_ids),
        )
        .distinct()
        .all()
    )
    existing_topic_ids = [topic.id for topic in existing_topics]
    existing_topics = [
        topic
        for topic in existing_topics
        if set(categories).issubset(topic.category_tags)
    ]

    rag_chunks = []
    rag_ids = []
    rag_metadatas = []

    topics_by_id = {}
    new_ids = []
    old_ids = [topic.id for topic in existing_topics]
    if topics is not None:
        resp: List[tuple[TopicDataTable, List, List, List]] = get_topic_rag_chunks(
            categories=categories, topics=topics, source=source
        )
        for topic, topic_chunks, topic_ids, topic_metadatas in resp:
            rag_chunks.extend(topic_chunks)
            rag_ids.extend(topic_ids)
            rag_metadatas.extend(topic_metadatas)
            topics_by_id[str(topic.id)] = {"topic": topic, "rag_ids": rag_ids}
            if str(topic.id) not in old_ids:
                new_ids.append(str(topic.id))

    existing_chroma_ids = []
    existing_chroma_collections = []
    handled_topics = []

    topic_ids = []

    if existing_topics is not None:
        for topic in existing_topics:
            existing_chroma_ids.extend(
                id for id in topic.chroma_ids if id not in existing_chroma_ids
            )
            existing_chroma_collections.extend(
                collection
                for collection in topic.chroma_collections
                if collection not in existing_chroma_collections
            )

            if topic.id in topics_by_id:
                topic_ids.append(topic.id)
                new_topic: TopicDataTable = topics_by_id[str(topic.id)]["topic"]
                topic.page_content = new_topic.page_content
                topic.page_number = new_topic.page_number
                topic.topic_index = new_topic.topic_index
                topic.doc_metadata = new_topic.doc_metadata
                topic.topic = new_topic.topic
                topic.instruct = new_topic.instruct
                topic.summary = new_topic.summary
                topic.category_tags = categories
                topic.chroma_ids = topics_by_id[str(topic.id)]["rag_ids"]
                topic.chroma_collections = [
                    "rag_" + cat + "_topic" for cat in categories
                ]
                new_references = (
                    new_topic.references
                    if isinstance(new_topic.references, list)
                    else [new_topic.references]
                )
                if not is_in_references(source, ReferenceType.source, new_references):
                    new_references.append(
                        Reference(id=source, type=ReferenceType.source, index=0)
                    )
                topic.references = unique_references(topic.references + new_references)
                handled_topics.append(topic.id)

    if topics is not None and len(topics) > 0:
        for topic in topics:
            if topic.id not in handled_topics:
                new_id = topic.id
                if db_topic_exists(topic.id):
                    matching_ids = sorted(
                        set(
                            [
                                topic_id
                                for topic_id in existing_topic_ids
                                if topic.id in topic_id
                            ]
                        )
                    )
                    new_id = topic.id + "-" + str(len(matching_ids))
                    print(
                        f"Topic with id {topic.id} already exists. Changing id to {new_id}"
                    )
                    defined_topic_ids.remove(topic.id)
                    defined_topic_ids.append(new_id)
                topic_ids.append(new_id)
                new_references = (
                    topic.references
                    if isinstance(topic.references, list)
                    else [topic.references]
                )
                if not is_in_references(source, ReferenceType.source, new_references):
                    new_references.append(
                        Reference(id=source, type=ReferenceType.source, index=0)
                    )
                new_topic = TopicDataTable(
                    id=new_id,
                    page_content=topic.page_content,
                    page_number=topic.page_number,
                    topic_index=topic.topic_index,
                    doc_metadata=topic.doc_metadata,
                    topic=topic.topic,
                    instruct=topic.instruct,
                    summary=topic.summary,
                    references=new_references,
                    category_tags=categories,
                    chroma_ids=topics_by_id[str(topic.id)]["rag_ids"],
                    chroma_collections=["rag_" + cat + "_topic" for cat in categories],
                )
                user_db_get_session().add(new_topic)

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
            type="topic",
        )

    existing_source = (
        user_db_get_session()
        .query(SourceDataTable)
        .filter(SourceDataTable.source == source)
        .first()
    )

    if existing_source is not None:
        existing_source.source_topics = list(
            set(list(existing_source.source_topics) + topic_ids)
        )

    user_db_get_session().commit()
