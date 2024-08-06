import sqlalchemy as sqla
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.mutable import MutableList
import streamlit as st
from chain import (
    SQLITE_DB
)

FILE_TABLENAME = "files"
JOURNEY_TABLENAME = "journey"


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
        MutableList.as_mutable(sqla.PickleType), default=["rag_all"]
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
    days = sqla.Column(sqla.PickleType)
    chroma_collection = sqla.Column(sqla.String, default="rag_all")
    disabled = sqla.Column(sqla.Boolean, default=False)
    title = sqla.Column(sqla.String)
    summary = sqla.Column(sqla.Text)
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


def get_db_journey(reset=False):
    if "db_journey" not in st.session_state or reset:
        journey = database_session.query(JourneyDataTable).all()

        db_journey = {}

        for step in journey:
            print(f"{ step = }")
            db_journey[step.journeyname] = step.__dict__

        st.session_state.db_journey = db_journey
    else:
        db_journey = st.session_state.db_journey

    return db_journey