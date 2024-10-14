import streamlit as st
import os
import sys
from streamlit_extras.grid import grid
from streamlit_extras.stylable_container import stylable_container

from lib.streamlit.journey import build_journey_cards, get_journey

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))

from admin.sidebar import init_sidebar
from lib.models.journey import JourneyItem, JourneyItemType, get_all_journeys_from_db
from lib.models.user import AuthStatus, get_db_user

st.set_page_config(
    page_title="TC POC: Admin",
    page_icon="static/icon.png",
    layout="centered",
    menu_items={
        "About": """# ThirdCognition PoC
[ThirdCognition](https://thirdcognition.com)
This is an *extremely* cool admin tool!
        """
    },
)


def main():

    auth_valid = init_sidebar()

    #     st.write("# Welcome! 👋")

    #     st.markdown(
    #         """
    # Select 👈 a the section from sidebar to edit the content!
    #     """
    #     )

    if auth_valid == AuthStatus.NO_LOGIN:
        st.write("# Welcome! 👋")
        st.write("Please log in for more functionality.")
        return

    db_journey_items = get_all_journeys_from_db()
    # db_journey = db_journey_items[0]
    journeys = [get_journey(journey_item=db_journey) for db_journey in db_journey_items]

    user = get_db_user(st.session_state["username"])
    # org = get_user_org(st.session_state["username"])
    st.header(f"Hello {user.name.split(" ")[0]}")
    st.subheader("Here’s what we are learning today!")

    ##---- Search Bar ----
    search_journey = st.text_input(
        "", placeholder="Search Onboarding Journey", label_visibility="hidden"
    )
    if search_journey:
        matches = []
        for journey in journeys:
            journey_matches = journey.search_children_with_token(
                    search_journey, item_type=JourneyItemType.MODULE
            )
            if len(journey_matches) > 0:
                matches.append((journey, journey_matches))
        if len(matches) > 0:
            st.subheader("Found matches:")
            journey: JourneyItem
            for journey, matches in matches:
                st.write(journey.title)
                build_journey_cards(
                    [journey.get_child_by_id(module_id) for module_id in matches], journey
                )
        else:
            st.write("No matches found")

    st.markdown("")

    # Number of cards on page
    # try:

    journey = journeys[0]
    all_children = journey.all_children_by_id()
    row_len = 3

    # Create two rows
    # for db_journey in db_journey_items:


    sections: list = journey.flatten(type_filter=JourneyItemType.SECTION)

    # for section_id in sections:
    # section_id = sections[0]
    # section = all_children[section_id]
    sections: list[JourneyItem] = [all_children[section_id] for section_id in sections]
    section_titles = [
        section.get_index(journey) + " - " + section.title for section in sections
    ]
    section_title = st.selectbox("Choose section", options=section_titles)

    section = sections[section_titles.index(section_title)]

    # st.subheader(section.get_index(journey) + " - " + section.title)
    modules = section.flatten(type_filter=JourneyItemType.MODULE)
    # journey_rows = [row_len for _ in range(len(modules)//row_len + 1)]
    # journey_grid = grid(*journey_rows, vertical_align="center")
    # for module_id in modules:
    #     with journey_grid.container(border=False):
    #         module = all_children[module_id]
    # with stylable_container(key=f"journey_{db_journey.id}_card", css_styles=""):
    build_journey_cards([all_children[module_id] for module_id in modules], journey)

    # st.markdown("")
    # st.markdown("**Time to complete module(s):** 5 Days - You are on track!")

    # except Exception as e:
    #     print(e)
    #     st.write("No journeys available")


if __name__ == "__main__":
    main()
