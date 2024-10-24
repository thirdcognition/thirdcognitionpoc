import time
from typing import List

import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.grid import grid

import os
import sys

from admin.global_styles import get_theme
from admin.sidebar import get_image, init_sidebar

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))


from lib.models.user import AuthStatus, UserLevel
from lib.streamlit.journey import assign_journey
from lib.models.journey import (
    JourneyItem,
    get_all_journeys_from_db,
)


st.set_page_config(
    page_title="TC POC: Admin",
    page_icon="static/icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": """# ThirdCognition PoC
[ThirdCognition](https://thirdcognition.com)
This is an *extremely* cool admin tool!
        """
    },
)


async def main():
    st.title("Modify Journey")

    if init_sidebar(UserLevel.org_admin) != AuthStatus.LOGGED_IN:
        st.switch_page("login.py")
        return

    st.markdown(" ")
    db_journey_items = get_all_journeys_from_db()

    journeys = [
        JourneyItem.get(journey_item=db_journey) for db_journey in db_journey_items
    ]

    journey: JourneyItem

    if len(journeys) > 1:
        journey_titles = [journey.title for journey in journeys]
        journey_title = st.selectbox("Choose journey", options=journey_titles)
        journey = journeys[journey_titles.index(journey_title)]
    else:
        journey = journeys[0]

    if journey is not None:
        assign_journey(journey.id)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
