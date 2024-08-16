from datetime import datetime
import time
import streamlit as st
from typing import Any, Dict, List
from lib.db_tools import (
    JourneyDataTable,
    get_db_journey,
    init_db,
)
from lib.prompts import JourneyStructure

def save_journey(journey_name, journey:Dict):
    database_session = init_db()
    journey_db = (
        database_session.query(JourneyDataTable)
        .filter(JourneyDataTable.journeyname == journey_name)
        .first()
    )

    if journey_db is not None:
        print("Remove old Journey")
        database_session.delete(journey_db)

    print("Create journey")
    journey_db = JourneyDataTable(
        journeyname=journey_name,
        files=journey.get("files", None),
        subjects=journey.get("subjects", None),
        title=journey.get("title", None),
        summary=journey.get("summary", None),
        last_updated=datetime.now(),
        chroma_collection = journey.get("chroma_collection", ["rag_" + cat for cat in journey.get("category", [])])
    )
    database_session.add(journey_db)
    database_session.commit()
    get_db_journey(reset=True)
    st.session_state.editing_journey = None
    st.session_state.editing_journey_details = None
    st.session_state.editing_journey_subject = None
    st.session_state.editing_journey_step = None
    time.sleep(0.1)
    st.rerun()

def delete_journey(journey_name):
    database_session = init_db()
    journey_db = (
        database_session.query(JourneyDataTable)
        .filter(JourneyDataTable.journeyname == journey_name)
        .first()
    )

    if journey_db is not None:
        database_session.delete(journey_db)
        database_session.commit()
        get_db_journey(reset=True)
        st.success(f"{journey_name} has been deleted successfully.")
        time.sleep(0.1)
        st.rerun()
    else:
        st.warning(f"{journey_name} does not exist in the database.")