from typing import Dict, List, Optional
from groq import BaseModel
from pydantic import Field
import sqlalchemy as sqla
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.mutable import MutableList
import chromadb

from langchain_chroma import Chroma
from chromadb.utils.embedding_functions import create_langchain_embedding
from chromadb.config import Settings as ChromaSettings
import streamlit as st
from chains.init import get_embeddings
from chains.prompt_generator import CustomPrompt, CustomPromptContainer
from lib.load_env import SETTINGS
from chains.prompts import (
    SubjectStructure,
    journey_steps,
    journey_step_content,
    # journey_step_content_redo,
    journey_step_intro,
    journey_step_actions,
    journey_step_action_details,
)

chroma_client = None

# Create a base class for SQLAlchemy's declarative extension
Base = declarative_base()


class StepModel(BaseModel):
    title: str = Field(default=None)
    subject: str = Field(default=None)
    content: str = Field(default=None)
    intro: str = Field(default=None)
    actions: str = Field(default=None)
    structured: SubjectStructure = Field(
        default=SubjectStructure(title="", subject="", intro="", content="", actions=[])
    )


class JourneyPrompts(BaseModel):
    steps: CustomPrompt = Field(
        default=CustomPrompt(system=journey_steps.system, user=journey_steps.user)
    )
    step_content: CustomPrompt = Field(
        default=CustomPrompt(
            system=journey_step_content.system, user=journey_step_content.user
        )
    )
    step_intro: CustomPrompt = Field(
        default=CustomPrompt(
            system=journey_step_intro.system, user=journey_step_intro.user
        )
    )
    step_actions: CustomPrompt = Field(
        default=CustomPrompt(
            system=journey_step_actions.system, user=journey_step_actions.user
        )
    )
    step_action_details: CustomPrompt = Field(
        default=CustomPrompt(
            system=journey_step_action_details.system,
            user=journey_step_action_details.user,
        )
    )
    # step_content_redo: CustomPrompt = Field(
    #     default=CustomPrompt(
    #         system=journey_step_content_redo.system,
    #         user=journey_step_content_redo.user,
    #     )
    # )

def convert_to_journey_prompts(container: CustomPromptContainer) -> JourneyPrompts:
    return JourneyPrompts(
        steps=container.steps,
        step_content=container.step_content,
        step_intro=container.step_intro,
        step_actions=container.step_actions,
        step_action_details=container.step_action_details
    )



class SubjectModel(BaseModel):
    title: str = Field(default=None)
    summary: str = Field(default=None)
    steps: List[StepModel] = Field(default_factory=list)
    prompts: JourneyPrompts = Field(default=JourneyPrompts())
    instructions: str = Field(default=None)
    step_amount: int = Field(default=None)
    action_amount: int = Field(default=None)
    files: List[str] = Field(default_factory=list)
    db_files: Dict[str, Dict] = Field(default_factory=dict)


class JourneyModel(BaseModel):
    journeyname: Optional[str] = None
    files: List[str] = Field(default_factory=list)
    db_files: Dict[str, Dict] = Field(default_factory=dict)
    subjects: List[SubjectModel] = Field(default_factory=list)
    chroma_collection: List[str] = Field(default_factory=list)
    disabled: Optional[bool] = False
    title: Optional[str] = None
    summary: Optional[str] = None
    instructions: Optional[str] = None
    complete: bool = False


# Define a new class for FileDataTable with filename as primary key
class FileDataTable(Base):
    __tablename__ = SETTINGS.file_tablename

    # id = Column(Integer, primary_key=True)
    filename = sqla.Column(sqla.String, primary_key=True)
    texts = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    formatted_text = sqla.Column(sqla.Text)
    category_tag = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    last_updated = sqla.Column(sqla.DateTime)
    chroma_collection = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    chroma_ids = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    disabled = sqla.Column(sqla.Boolean, default=False)
    title = sqla.Column(sqla.String)
    summary = sqla.Column(sqla.Text)
    edited_content = sqla.Column(sqla.Text)
    file_data = sqla.Column(sqla.BLOB)


# Define a new class for JourneyDataTable with list_name as primary key
class JourneyDataTable(Base):
    __tablename__ = SETTINGS.journey_tablename

    # id = Column(Integer, primary_key=True)
    journeyname = sqla.Column(sqla.String, primary_key=True)
    files = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    subjects = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    chroma_collection = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    disabled = sqla.Column(sqla.Boolean, default=False)
    title = sqla.Column(sqla.String, default="")
    summary = sqla.Column(sqla.Text, default="")
    instructions = sqla.Column(sqla.Text, default="")
    last_updated = sqla.Column(sqla.DateTime, default=None)


database_session = None


def init_db():
    global database_session
    if database_session is None:
        engine = sqla.create_engine("sqlite:///{}".format(SETTINGS.sqlite_db))
        Base.metadata.create_all(engine)
        DatabaseSession = sessionmaker(bind=engine)
        database_session = DatabaseSession()
    # current_directory = os.getcwd()
    # print("Current working directory:", current_directory)
    # print("sqlite:///{}".format(SQLITE_DB))

    return database_session


def get_db_files(reset=False, filename=None, categories=None):
    if (
        "db_files" not in st.session_state
        or reset
        or filename != None
        and filename not in st.session_state.db_files.keys()
    ):
        if filename == None:
            files = database_session.query(FileDataTable).all()
        else:
            files = database_session.query(FileDataTable).filter(
                FileDataTable.filename == filename
            )

        if "db_files" not in st.session_state or reset:
            db_files = {}
        else:
            db_files = st.session_state.db_files

        for file in files:
            db_files[file.filename] = file.__dict__

        st.session_state.db_files = db_files
    else:
        db_files = st.session_state.db_files

    if isinstance(categories, str):
        categories = [categories]
    if categories:
        new_db_files = {}
        for cat in categories:
            new_db_files.update(
                {k: v for k, v in db_files.items() if cat in v["category_tag"]}
            )
        db_files = new_db_files
    return db_files


def delete_db_file(filename: str):
    instance = (
        database_session.query(FileDataTable)
        .where(FileDataTable.filename == filename)
        .first()
    )
    database_session.delete(instance)
    database_session.commit()
    get_db_files(reset=True)


def get_db_journey(journey_name: str = None, chroma_collections = None, reset=False) -> Dict[str, JourneyModel]:
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
            print(f"{cat=} {[f"{name}: {journey.chroma_collection=}" for name, journey in db_journey.items()]}")
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
        search_kwargs={"k": amount_of_documents, "score_threshold": 0.15},
    )
