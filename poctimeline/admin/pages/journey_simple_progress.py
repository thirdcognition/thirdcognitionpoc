import time
from typing import List

import streamlit as st
from streamlit_extras.stylable_container import stylable_container

import os
import sys

from admin.global_styles import get_theme
from admin.sidebar import get_image, init_sidebar
from lib.models.journey_progress import (
    JourneyItemProgress,
    JourneyItemProgressState,
    JourneyProgressDataTable,
)
from lib.streamlit.journey import ChildPosition

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))

from lib.models.journey import (
    JourneyItem,
    JourneyItemType,
    get_all_journeys_from_db,
)
from lib.helpers.journey import (
    ActionSymbol,
)
from lib.models.user import (
    AuthStatus,
    UserDataTable,
    UserLevel,
    check_auth_level,
    get_db_user,
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


container_level = ["journey", "section", "module", "action"]

level_step = 1


@st.dialog("Feedback", width="large")
def see_feedback(journey_item: JourneyItem, journey: JourneyItem, feedback: dict):
    st.subheader(journey_item.get_index(journey) + " " + journey_item.title)
    all_children = journey_item.all_children_by_id()
    for key, item in feedback.items():
        _, col, _ = st.columns([0.01, 0.9, 0.03])
        col.write(all_children[key].get_index(journey) + " " + all_children[key].title)
        col.container(border=True).write(item)


def write_section_module(
    item_progress: JourneyItemProgress,
    journey_progress: JourneyItemProgress,
    parent_item_id: str,
    position=ChildPosition.MIDDLE,
):
    item_state: dict = st.session_state["journey_item_progress_state"].get(
        parent_item_id, {}
    )
    item_id = item_progress.id

    journey: JourneyItem = JourneyItem.get(journey_id=journey_progress.journey_item_id)
    all_items = journey.all_children_by_id()
    journey_item: JourneyItem = all_items[item_progress.journey_item_id]

    if (
        item_state["open"]
        and JourneyItemType.MODULE == journey_item.item_type
        and journey_item.icon
    ):
        content_container = stylable_container(
            key="journey_process_container_" + item_id,
            css_styles=[
                f"""
                > div {{
                    gap: 0;
                    margin: 0;
                    {("margin-top: 1rem; padding-bottom: 0;" if position == ChildPosition.FIRST else "")}
                    {("margin-bottom: 1.5rem;" if position == ChildPosition.LAST else "")}
                }}
                """,
                """
                .stButton {
                    text-align: center;
                }
                """,
            ],
        )

        with content_container:
            _, icon_container, _, content_col = st.columns(
                [0.025, 0.075, 0.05, 0.9], vertical_alignment="center"
            )

        with icon_container.container():
            st.image(
                get_image(journey_item.icon, "icon_files"),
                use_column_width=True,
            )

        title, progress, but = content_col.columns(
            [0.4, 0.3, 0.2], vertical_alignment="center"
        )
        title.write(journey_item.get_index(journey) + " " + journey_item.title)
        state = item_progress.get_state()
        if JourneyItemProgressState.COMPLETED == state:
            progress.write("Done.")
        elif JourneyItemProgressState.NOT_STARTED == state:
            progress.write("Not started")
        else:
            progress.progress(item_progress.get_progress())

        feedback = {}
        for child in item_progress.children:
            if (
                child.extras is not None
                and isinstance(child.extras, dict)
                and "completion_feedback" in child.extras
            ):
                feedback[child.journey_item_id] = child.extras["completion_feedback"]

        if feedback and but.button("See feedback", key="view_feedback_" + item_id):
            see_feedback(journey_item, journey, feedback)


@st.fragment
def write_progress(
    item: JourneyItemProgress,
    journey_progress: JourneyItemProgress = None,
    position=ChildPosition.MIDDLE,
):
    item_id = item.id
    level_multiplier = max(
        (container_level.index(item.item_type.value) - 1) * level_step, 0.01
    )

    # main_container = st.empty()

    theme = get_theme()
    if theme is not None:
        if theme["base"] == "light":
            color = theme["borderColorLight"]
        else:
            color = theme["borderColor"]
    else:
        color = "gray"

    button_styles = [
        """
        .stColumn:last-child {
            padding-right: 1rem;
            text-align: center;
        }
        """,
    ]
    item_state: dict = st.session_state["journey_item_progress_state"].get(item_id, {})
    st.session_state["journey_item_progress_state"][item_id] = item_state
    item_state["open"] = item_state.get("open", False)
    # with main_container:
    container = stylable_container(
        key=f"item_container_{item_id}",
        css_styles=[
            f"""
            {{
                border: 1px solid {color};
                {f"border-radius: 0.5rem 0.5rem 0 0;" if position == ChildPosition.FIRST and item.parent_id == journey_progress.id else ""}
                {f"border-radius: 0 0 0.5rem 0.5rem;" if position == ChildPosition.LAST and item.parent_id == journey_progress.id and not item_state["open"] else ""}
                margin-top: -1px;
                {f"margin-bottom: 1rem;" if position == ChildPosition.LAST and item.parent_id == journey_progress.id and not item_state["open"] else ""}
                padding: 0;
                padding-top: 1.5rem;
                padding-bottom: 0.5rem;
                padding-left: {level_multiplier}rem;
                gap: 0;
            }}"""
        ]
        + button_styles,
    )
    if (
        JourneyItemType.SECTION == item.item_type
        or JourneyItemType.MODULE == item.item_type
    ):
        col1, col2 = container.columns([0.7, 0.2], vertical_alignment="center")
        journey = JourneyItem.get(journey_id=journey_progress.journey_item_id)
        all_children = journey.all_children_by_id()
        journey_item = all_children[
            item.journey_item_id
        ]  # JourneyItem.get(journey_id=item.journey_item_id)
        if col1.button(
            (
                ActionSymbol.open.value
                if item_state["open"]
                else ActionSymbol.closed.value
            )
            + "\n\n"
            + journey_item.get_index(journey).replace(".", r"\.")
            + "\n\n"
            + journey_item.title,
            key=f"open_button_{item_id}",
            type="primary",
        ):
            item_state["open"] = not item_state["open"]
            st.rerun(scope="fragment")
        progress = item.get_progress()
        state = item.get_state()
        if JourneyItemProgressState.COMPLETED == state:
            col2.write("Done")
        elif JourneyItemProgressState.NOT_STARTED == state:
            col2.write("Not started")
        else:
            col2.progress(progress, str(progress * 100 // 1) + "% ")

    if item_state.get("open", False):
        _, col1 = st.columns([0.03, 0.9], gap="small")
        for i, child in enumerate(item.children):
            position = (
                ChildPosition.FIRST
                if i == 0
                else (
                    ChildPosition.LAST
                    if i == len(item.children) - 1
                    else ChildPosition.MIDDLE
                )
            )
            # with col1:
            write_section_module(child, journey_progress, item_id, position)


def journey_progress(journey: JourneyItem, user_id=None):
    journey_id = journey.id

    # Display current assignments and user progress

    # Load all progress items for the current journey
    progress_items = JourneyProgressDataTable.load_all_from_db(
        journey_item_id=journey_id, user_id=user_id
    )

    st.session_state["journey_item_progress_state"] = st.session_state.get(
        "journey_item_progress_state", {}
    )

    # theme = get_theme()

    # border_color = "gray"
    # if theme != None and theme["base"] == "dark":
    #     border_color = theme["borderColor"]
    # elif theme != None:
    #     border_color = theme["borderColorLight"]

    st.write(" ")
    st.subheader("Individual user progress" if user_id is None else "My progress")
    styles = stylable_container(
        key="progress_item_container_" + journey_id,
        css_styles=[
            """
        .stButton > button[kind=primary] {
            border-radius: 0;
            border: none;
            background: transparent;
            color: inherit;
            white-space: nowrap;
            overflow: hidden;
            max-width: 100%;
            justify-content: flex-start;
        }""",
            """
        .stButton > button[kind=primary] p {
            display: inline-block;
            margin-right: 0.2rem;
            vertical-align: middle;
        }""",
            """
        .stButton > button[kind=primary] p:first-child {
            font-size: 1rem;
            width: 1.5rem;
        }""",
        ],
    )

    if len(progress_items) > 0:
        # Display current assignments and user progress
        count = 0
        for i, progress_item in enumerate(progress_items):
            user = get_db_user(id=progress_item.user_id)
            if user is None:
                user = UserDataTable(id="none", email="N/A", name="Removed user")

            item_id = progress_item.id
            item_state: dict = st.session_state["journey_item_progress_state"].get(
                item_id, {}
            )
            st.session_state["journey_item_progress_state"][item_id] = item_state
            item_state["open"] = item_state.get("open", (user_id != None))
            with styles:
                container = st.container(border=True)
            col1, col2 = container.columns(
                [0.3 if user_id is None else 0.5, 0.7 if user_id is None else 0.5],
                vertical_alignment="center",
            )

            if user_id == None and col1.button(
                (
                    ActionSymbol.open.value
                    if item_state["open"]
                    else ActionSymbol.closed.value
                )
                + f"\n\n **{user.name or user.email}**",
                key=f"open_button_{item_id}",
                type="primary",
            ):
                item_state["open"] = not item_state["open"]
                st.rerun()
            elif user_id != None:
                col1.write("#### " + journey.title)
            if user:
                count += 1

                if item_state["open"]:
                    with container:
                        st.write(" ")
                        container2 = stylable_container(
                            key="journey_creation_section_container",
                            css_styles=[
                                f"""
                                {{
                                    gap: 0;
                                }}
                                """,
                                """
                                div[data-testid=stHeadingWithActionElements] {
                                    margin-bottom: 1rem;
                                }
                                """,
                            ],
                        )

                journey_progress = JourneyItemProgress.from_db(progress_item)
                if JourneyItemProgressState.NOT_STARTED > journey_progress.get_state():
                    next_modules_progress = journey_progress.get_next(reset=True)
                    # all_sections = journey_progress.flatten(type_filter=JourneyItemType.SECTION)
                    # pretty_print(next_modules_progress, force=True)
                    all_children = journey.all_children_by_id()
                    next_module = all_children[next_modules_progress[0].journey_item_id]
                    # journey_item = JourneyItem.get(journey_id = journey_progress.item_id)
                    # st.write(f"Journey: {journey_item.title}")
                    col2.progress(
                        journey_progress.get_progress(),
                        f"Next module: {next_module}",
                    )

                    if item_state["open"]:
                        with container2:
                            for index, child in enumerate(journey_progress.children):
                                position = (
                                    ChildPosition.FIRST
                                    if index == 0
                                    else (
                                        ChildPosition.LAST
                                        if index == len(journey_progress.children) - 1
                                        else ChildPosition.MIDDLE
                                    )
                                )

                                write_progress(
                                    child, journey_progress, position=position
                                )
                else:
                    col2.write("Not yet started")

                    # if JourneyItemProgressState.NOT_STARTED > journey_progress.get_state() and col3.button(f"View feedback", key=f"remove_{journey_progress.id}"):
                    #     print("no-op")
                if (
                    i == len(progress_items) - 1
                    and len(progress_items) > 1
                    and item_state["open"]
                ):
                    st.divider()

            if count == 0:
                st.write("No assigned users have yet signed up and started the journey")
    else:
        st.write(" ")
        st.write("Journey hasn't been assigned to anyone yet")


async def main():
    st.title("Journey progress")

    if init_sidebar(UserLevel.user) != AuthStatus.LOGGED_IN:
        st.switch_page("login.py")
        return

    st.markdown(" ")
    # try:
    user_level = check_auth_level()
    user_id = None

    if UserLevel.user == user_level:
        user = get_db_user(st.session_state["username"])
        user_id = user.id
        my_journey_progress = JourneyProgressDataTable.load_all_from_db(
            user_id=user.id, item_type=JourneyItemType.JOURNEY
        )

        try:
            db_journey_items = get_all_journeys_from_db(
                ids=[
                    journey_progress.journey_item_id
                    for journey_progress in my_journey_progress
                ]
            )
        except ValueError as e:
            st.write("No journeys have been assigned to you yet.")
            return
        journeys = [
            JourneyItem.get(journey_item=db_journey) for db_journey in db_journey_items
        ]
    else:
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
        journey_progress(journey, user_id)

        # for i, journey in enumerate(journeys):
        #     with st.expander(journey.title, expanded=i==0):
        #         journey_progress(journey)

    # except Exception as e:
    #     print(e)
    #     st.write("No journeys available")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
