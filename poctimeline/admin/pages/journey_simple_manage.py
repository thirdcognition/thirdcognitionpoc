import time
from typing import List

import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.grid import grid

import os
import sys

from admin.sidebar import get_image, init_sidebar
from lib.streamlit.journey import build_journey_cards

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))

from lib.models.journey import (
    JourneyItem,
    JourneyItemType,
    get_all_journeys_from_db,
)
from lib.chains.init import get_chain
from lib.helpers.journey import load_journey_template, match_title_to_cat_and_id
from lib.helpers.shared import pretty_print
from lib.models.user import AuthStatus, UserLevel

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

## Symbols from https://www.w3schools.com/charsets/


async def main():
    st.title("List of existing Journeys")

    if init_sidebar(UserLevel.org_admin) != AuthStatus.LOGGED_IN:
        st.switch_page("login.py")
        return

    st.markdown(" ")
    try:
        db_journey_items = get_all_journeys_from_db()

        journeys = [
            JourneyItem.get(journey_item=db_journey) for db_journey in db_journey_items
        ]

        journey: JourneyItem
        build_journey_cards(journeys, key_start="manage_")
    except Exception as e:
        print(e)
        # st.write("No journeys available")
        build_journey_cards(
            [
                {
                    "title": "Create a new journey",
                    "icon": "logo_4",
                    "page": "journey_simple_create",
                }
            ],
            key_start="new_",
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
