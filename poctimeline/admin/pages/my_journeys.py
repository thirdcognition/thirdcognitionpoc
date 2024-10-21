import time
import streamlit as st
import os
import sys
from streamlit_extras.grid import grid
from streamlit_extras.stylable_container import stylable_container

from admin.global_styles import get_theme
from lib.helpers.shared import pretty_print
from lib.models.journey_progress import JourneyItemProgress, JourneyItemProgressState, JourneyProgressDataTable
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
    # theme = get_theme()
    # if theme is None:
    #     st.write("Couldn't load theme")
    #     return

    hello = st.empty()
    login = st.empty()

    auth_valid = init_sidebar(login_container=login)

    if (
        auth_valid == AuthStatus.NO_LOGIN
        or auth_valid is None
        or st.session_state["username"] is None
    ):
        with hello.container():
            st.write("# Welcome! 👋")
            st.write("Please log in for more functionality.")
            return

    user = get_db_user(st.session_state["username"])

    my_journey_progress = JourneyProgressDataTable.load_all_from_db(
        user_id=user.id, item_type=JourneyItemType.JOURNEY
    )
    my_journey_progress_by_journey_id = {
        item.journey_item_id: item for item in my_journey_progress
    }

    # pretty_print([item.__dict__ for item in my_journey_progress],  "Assigned journeys",  force=True)

    my_active_modules = JourneyProgressDataTable.load_all_from_db(
        user_id=user.id, item_type=JourneyItemType.MODULE, started=True, completed=False
    )

    # my_journeys = []

    # pretty_print([module.__dict__ for module in my_active_modules], "Active modules", force=True)

    # org = get_user_org(st.session_state["username"])
    st.header(f"Hello {user.name.split(" ")[0] if user.name else "user"}")

    try:
        db_journey_items = get_all_journeys_from_db(
            ids=[
                journey_progress.journey_item_id
                for journey_progress in my_journey_progress
            ]
        )
        st.subheader("Here’s what we are learning today!")
    except ValueError as e:
        st.write("No journeys have been assigned to you yet")
        return

    # db_journey = db_journey_items[0]
    journeys = [
        JourneyItem.get(journey_item=db_journey) for db_journey in db_journey_items
    ]
    journeys_by_id = {journey.id: journey for journey in journeys}

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
                    [journey.get_child_by_id(module_id) for module_id in matches],
                    journey,
                )
        else:
            st.write("No matches found")

    st.divider()

    if len(my_active_modules) > 0:
        st.subheader("Active modules")

        cards = {}
        for module_progress in my_active_modules:
            journey_id = module_progress.journey_id
            journey_progress_item = JourneyItemProgress.from_db(item=module_progress)
            if journey_id not in cards:
                cards[journey_id] = []
            if journey_progress_item.get_progress() < 1:
                cards[journey_id].append(module_progress)

        first = True
        for journey_id in cards.keys():
            journey = journeys_by_id[journey_id]
            journey_progress = JourneyItemProgress.from_db(
                my_journey_progress_by_journey_id[journey_id]
            )

            all_children = journey.all_children_by_id()
            if first:
                container = st.container()
                container.write("#### " + journey.title)
            else:
                container = st.expander()

            with container:
                card_items = [
                    all_children[item.journey_item_id] for item in cards[journey_id]
                ]
                # print(len(card_items))
                build_journey_cards(card_items, journey, journey_progress)

        st.divider()

    # Number of cards on page
    # try:

    # journey = st.session_state.get("active_journey", journeys[0])
    if len(my_journey_progress) > 0:
        st.subheader(
            "Next modules" if len(my_active_modules) > 0 else "Choose your first module"
        )
        if len(journeys) > 1:
            journey_titles = [journey.title for journey in journeys]
            journey_title = st.selectbox("Choose journey", options=journey_titles)
            journey = journeys[journey_titles.index(journey_title)]
        else:
            journey = journeys[0]

        all_children = journey.all_children_by_id()

        sections: list = journey.flatten(type_filter=JourneyItemType.SECTION)
        journey_progress_data = next(
            (
                progress_item
                for progress_item in my_journey_progress
                if progress_item.journey_item_id == journey.id
            ),
            None,
        )

        journey_progress = JourneyItemProgress.from_db(journey_progress_data)

        sections: list[JourneyItem] = [
            all_children[section_id] for section_id in sections
        ]
        section_titles = [
            section.get_index(journey) + " - " + section.title for section in sections
        ]
        section_titles.insert(0, "Next modules in queue")
        section_title = st.selectbox("Choose section", options=section_titles)

        chosen_index = section_titles.index(section_title)
        if chosen_index > 0:
            section = sections[chosen_index - 1]
            modules = section.flatten(type_filter=JourneyItemType.MODULE)
        else:
            next_modules = journey_progress.get_next()
            modules = [
                all_children[progress.journey_item_id].id for progress in next_modules
            ]
            # pretty_print(next_modules, "Modules", force=True)

        build_journey_cards(
            [all_children[module_id] for module_id in modules],
            journey,
            journey_progress,
        )

    else:
        st.subheader("You've not been assigned any journeys yet")


if __name__ == "__main__":
    main()
