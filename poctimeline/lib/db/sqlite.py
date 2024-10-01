import os
from typing import Dict
import sqlalchemy as sqla
from sqlalchemy.orm import sessionmaker, Session, declarative_base

from lib.load_env import SETTINGS

Base = declarative_base()

database_session:Dict[str, Session] = {}

def init_system_db() -> Session:
    db_file = SETTINGS.system_db_filename
    return init_db(db_file)

def init_db(db_file=SETTINGS.sqlite_db) -> Session:
    db_folder = os.path.dirname(os.path.join(SETTINGS.db_path, db_file, "sqlite.db"))
    if not os.path.exists(db_folder):
        os.makedirs(db_folder)
    db_system_file = os.path.join(SETTINGS.db_path, db_file, "sqlite.db")

    global database_session
    if db_file not in database_session:
        print("Open db: {}".format(db_system_file))
        engine = sqla.create_engine("sqlite:///{}".format(db_system_file))
        Base.metadata.create_all(engine)
        DatabaseSession = sessionmaker(bind=engine, autoflush=False)
        database_session[db_file] = DatabaseSession()

    return database_session[db_file]

def db_session(db_file=SETTINGS.sqlite_db) -> Session:
    return init_db(db_file)


def db_commit(db_file=SETTINGS.sqlite_db):
    if db_file in database_session:
        database_session[db_file].commit()

