import sqlalchemy as sqla
from sqlalchemy.orm import sessionmaker

from lib.load_env import SETTINGS
from lib.models.sqlite_tables import (
    Base,
)

chroma_client = None
database_session = None


def init_db():
    global database_session
    if database_session is None:
        engine = sqla.create_engine("sqlite:///{}".format(SETTINGS.sqlite_db))
        Base.metadata.create_all(engine)
        DatabaseSession = sessionmaker(bind=engine, autoflush=False)
        database_session = DatabaseSession()

    return database_session


def db_session():
    global database_session
    return database_session


def db_commit():
    database_session.commit()
