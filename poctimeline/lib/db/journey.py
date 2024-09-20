from typing import Dict

import streamlit as st
from lib.db.sqlite import db_session

from lib.models.journey import (
    JourneyModel,
    JourneyDataTable,
)


def get_db_journey(
    journey_name: str = None, chroma_collections=None, reset=False
) -> Dict[str, JourneyModel]:
    db_journey: Dict[str, JourneyModel] = None
    if "db_journey" not in st.session_state or reset:
        if journey_name is None:
            journey = db_session().query(JourneyDataTable).all()
        else:
            journey = (
                db_session()
                .query(JourneyDataTable)
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
            new_db_journeys.update(
                {k: v for k, v in db_journey.items() if cat in v.chroma_collections}
            )
        db_journey = new_db_journeys

    return db_journey
