import time
import streamlit as st
import os
import sys
from streamlit_extras.grid import grid
from streamlit_extras.stylable_container import stylable_container

from admin.global_styles import get_theme
from lib.helpers.shared import pretty_print
from lib.models.journey_progress import JourneyItemProgress, JourneyProgressDataTable
from lib.streamlit.journey import build_journey_cards

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))

from admin.sidebar import init_sidebar
from lib.models.journey import JourneyItem, JourneyItemType, get_all_journeys_from_db
from lib.models.user import AuthStatus, get_db_user

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


def main():
    theme = get_theme()
    # if theme is None:
    #     st.write("Couldn't load theme")
    #     return

    hello = st.empty()
    login = st.empty()

    auth_valid = init_sidebar(login_container=login)
    # if auth_valid is None:
    #     st.write("Loading...")
    #     time.sleep(1)
    #     st.rerun()

    #     st.write("# Welcome! 👋")

    #     st.markdown(
    #         """
    # Select 👈 a the section from sidebar to edit the content!
    #     """
    #     )

    if (
        auth_valid == AuthStatus.NO_LOGIN
        or auth_valid is None
        or st.session_state["username"] is None
    ):
        with hello.container():
            st.write("# Welcome! 👋")
            st.write("Please log in for more functionality.")
            return
    else:
         st.switch_page("pages/my_journeys.py")


if __name__ == "__main__":
    main()
