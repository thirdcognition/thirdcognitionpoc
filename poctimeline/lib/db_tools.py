import os
import sqlalchemy as sqla
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.mutable import MutableList
import chromadb

from langchain_chroma import Chroma
from chromadb.utils.embedding_functions import create_langchain_embedding
import streamlit as st
from lib.chain import (
    get_embeddings,
    init_llms
)
from lib.load_env import CHROMA_PATH, FILE_TABLENAME, JOURNEY_TABLENAME, SQLITE_DB

chroma_client = None

# Create a base class for SQLAlchemy's declarative extension
Base = declarative_base()

# Define a new class for FileDataTable with filename as primary key
class FileDataTable(Base):
    __tablename__ = FILE_TABLENAME

    # id = Column(Integer, primary_key=True)
    filename = sqla.Column(sqla.String, primary_key=True)
    texts = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    formatted_text = sqla.Column(sqla.Text)
    category_tag = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    last_updated = sqla.Column(sqla.DateTime)
    chroma_collection = sqla.Column(
        MutableList.as_mutable(sqla.PickleType), default=[]
    )
    chroma_ids = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    disabled = sqla.Column(sqla.Boolean, default=False)
    title = sqla.Column(sqla.String)
    summary = sqla.Column(sqla.Text)
    edited_content = sqla.Column(sqla.Text)
    file_data = sqla.Column(sqla.BLOB)

# Define a new class for JourneyDataTable with list_name as primary key
class JourneyDataTable(Base):
    __tablename__ = JOURNEY_TABLENAME

    # id = Column(Integer, primary_key=True)
    journeyname = sqla.Column(sqla.String, primary_key=True)
    files = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    subjects = sqla.Column(sqla.PickleType)
    chroma_collection = sqla.Column(
        MutableList.as_mutable(sqla.PickleType), default=[]
    )
    disabled = sqla.Column(sqla.Boolean, default=False)
    title = sqla.Column(sqla.String)
    summary = sqla.Column(sqla.Text)
    instructions = sqla.Column(sqla.Text)
    last_updated = sqla.Column(sqla.DateTime)

database_session = None

def init_db():
    global database_session
    if database_session is None:
        engine = sqla.create_engine("sqlite:///{}".format(SQLITE_DB))
        Base.metadata.create_all(engine)
        DatabaseSession = sessionmaker(bind=engine)
        database_session = DatabaseSession()

    return database_session

def get_db_files(reset=False, filename=None):
    if "db_files" not in st.session_state or reset or filename != None and filename not in st.session_state.db_files.keys():
        if filename == None:
           files = database_session.query(FileDataTable).all()
        else:
            files = database_session.query(FileDataTable).filter(FileDataTable.filename == filename)

        if "db_files" not in st.session_state:
            db_files = {}
        else:
            db_files = st.session_state.db_files

        for file in files:
            db_files[file.filename] = file.__dict__

        st.session_state.db_files = db_files
    else:
        db_files = st.session_state.db_files

    return db_files


def get_db_journey(journey_name:str = None, reset=False):
    if "db_journey" not in st.session_state or reset:
        if journey_name is None:
            journey = database_session.query(JourneyDataTable).all()
        else:
            journey = database_session.query(JourneyDataTable).filter(JourneyDataTable.journeyname == journey_name).all()
        db_journey = {}

        for step in journey:
            db_journey[step.journeyname] = step.__dict__

        st.session_state.db_journey = db_journey
    else:
        db_journey = st.session_state.db_journey

    return db_journey


collections = {}

def get_chroma_collection(name, update=False, path=CHROMA_PATH, embedding_id = None) -> chromadb.Collection:
    global collections

    if name in collections and not update:
        return collections[name]

    global chroma_client
    chroma_client = chroma_client or chromadb.PersistentClient(path=path)

    if update:
        chroma_client.delete_collection(name=name)

    init_llms()

    embedding_function = None
    if embedding_id is not None:
        embedding_function = create_langchain_embedding(get_embeddings(embedding_id))
    else:
        embedding_function = create_langchain_embedding(get_embeddings("base"))

    collection = chroma_client.get_or_create_collection(name, embedding_function=embedding_function)
    collections[name] = collection
    return collection


vectorstores = {}

def get_vectorstore(id, embedding_id="base", update_vectorstores = False) -> Chroma:
    init_llms()
    global vectorstores

    if id in vectorstores and not update_vectorstores:
        return vectorstores[id]

    print(f"\n\n\nInit vectorstore {id=} {embedding_id=}\n\n\n")
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=id,
        embedding_function=get_embeddings(embedding_id),
    )

    vectorstores[id] = vectorstore
    return vectorstore