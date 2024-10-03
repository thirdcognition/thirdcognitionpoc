import time
from typing import Dict

import streamlit as st
from lib.models.user import user_db_get_session

from lib.models.journey import (
    JourneyDataTable,
    JourneyDataTable,
)


def get_db_journey(
    journey_name: str = None, chroma_collections=None, reset=False
) -> Dict[str, JourneyDataTable]:
    db_journey: Dict[str, JourneyDataTable] = None
    if "db_journey" not in st.session_state or reset:
        if journey_name is None:
            journey = user_db_get_session().query(JourneyDataTable).all()
        else:
            journey = (
                user_db_get_session()
                .query(JourneyDataTable)
                .filter(JourneyDataTable.journey_name == journey_name)
                .all()
            )
        db_journey = {}

        for subsubject in journey:
            db_journey[subsubject.journey_name] = subsubject

        st.session_state["db_journey"] = db_journey
    else:
        db_journey = st.session_state["db_journey"]

    if isinstance(chroma_collections, str):
        chroma_collections = [chroma_collections]
    if chroma_collections:
        new_db_journeys = {}
        for cat in chroma_collections:
            new_db_journeys.update(
                {k: v for k, v in db_journey.items() if cat in v.chroma_collections}
            )
        db_journey = new_db_journeys

    return db_journey


def save_journey(journey_name, journey: JourneyDataTable) -> bool:
    print(f"Save journey {journey_name}")
    # st.write(f"Save journey {journey_name}")
    # st.write(journey)
    database_session = user_db_get_session()
    journey_db = (
        database_session.query(JourneyDataTable)
        .filter(JourneyDataTable.journey_name == journey_name)
        .first()
    )

    if journey_db is not None:
        print("Remove old Journey")
        database_session.delete(journey_db)

    try:
        database_session.add(journey)
        database_session.commit()
    except Exception as e:
        print(f"Error saving journey.\n\n{e}")
        return False

    get_db_journey(reset=True)
    return True


def delete_journey(journey_name):
    database_session = user_db_get_session()
    journey_db = (
        database_session.query(JourneyDataTable)
        .filter(JourneyDataTable.journey_name == journey_name)
        .first()
    )

    if journey_db is not None:
        database_session.delete(journey_db)
        database_session.commit()
        get_db_journey(reset=True)
