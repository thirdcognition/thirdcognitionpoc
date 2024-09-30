from typing import Dict
import sqlalchemy as sqla
from sqlalchemy.orm import sessionmaker, Session, declarative_base

from lib.load_env import SETTINGS

Base = declarative_base()

chroma_client = None
database_session:Dict[str, Session] = {}

def init_system_db() -> Session:
    db_file = SETTINGS.system_db_filename
    return init_db(db_file)

def init_db(db_file=SETTINGS.sqlite_db) -> Session:
    global database_session
    if db_file not in database_session:
        engine = sqla.create_engine("sqlite:///{}".format(db_file))
        Base.metadata.create_all(engine)
        DatabaseSession = sessionmaker(bind=engine, autoflush=False)
        database_session[db_file] = DatabaseSession()

    return database_session[db_file]

def db_session(db_file=SETTINGS.sqlite_db) -> Session:
    return init_db(db_file)


def db_commit(db_file=SETTINGS.sqlite_db):
    database_session[db_file].commit()

