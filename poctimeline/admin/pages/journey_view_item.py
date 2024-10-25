import time
from typing import List

import streamlit as st

import os
import sys

from admin.sidebar import get_image, init_sidebar
from lib.models.user import get_db_user
from lib.streamlit.journey import write_action_ui

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))

from lib.models.journey_progress import JourneyItemProgress, JourneyItemProgressState

from lib.models.journey import (
    ActionItemType,
    JourneyItem,
    JourneyItemType,
)

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

selected_symbol = "&#9673;"
unselected_symbol = "&#9678;"
up_down_symbol = "&#8597;"
below_symbol = "&#8615;"
image_symbol = "&#9968;"
edit_symbol = "&#9881;"

container_level = ["journey", "section", "module", "action"]


def get_stylable_container_selector(id):
    return f'div[data-testid="stVerticalBlock"]:has(> div.element-container > div.stMarkdown > div[data-testid="stMarkdownContainer"] > p > span.{id})'


def view_journey_item(
    journey_item: JourneyItem,
    journey: JourneyItem,
    journey_progress: JourneyItemProgress = None,
    content_container=None,
    icon_container=None,
    task_container=None,
):
    if (
        JourneyItemType.JOURNEY == journey_item.item_type
        or JourneyItemType.MODULE == journey_item.item_type
    ):
        if icon_container == None:
            _, icon_container, _ = st.columns([0.3, 0.4, 0.3])

        icon_container.image(
            str(get_image(journey_item.icon, "icon_files")),
            use_column_width=True,
        )

    if content_container is None:
        content_container = st.container()

    with content_container:
        if JourneyItemType.MODULE == journey_item.item_type:
            all_items = journey.all_children_by_id()
            parent_item = all_items[journey_item.parent_id]
            st.header(parent_item.get_index(journey) + " " + parent_item.title)

            st.subheader(f"{journey_item.get_index(journey)} {journey_item.title}")
            # st.write(journey_item.description)

    journey_item_progress = (
        JourneyItemProgress.get(
            journey_item_id=journey_item.id,
            user_id=get_db_user(email=st.session_state.get("username")).id,
        )
        if journey_progress is None
        else journey_progress.get_by_journey_item(journey_item)
    )

    if JourneyItemType.ACTION == journey_item.item_type:
        write_action_ui(
            journey_item,
            journey,
            journey_item_progress,
            content_container,
            task_container,
        )


def view_modules_items(
    item: JourneyItem,
    journey: JourneyItem,
    journey_progress: JourneyItemProgress = None,
    show_children=False,
):
    all_children = journey.all_children_by_id()
    # relations = journey.get_relations()
    # ancestry = item.get_ancestry(journey)
    items_filtered = {
        "section": [
            (item_id, all_children[item_id].parent_id)
            for item_id in journey.flatten(JourneyItemType.SECTION)
        ],
        "module": [
            (item_id, all_children[item_id].parent_id)
            for item_id in journey.flatten(JourneyItemType.MODULE)
        ],
        "action": [
            (item_id, all_children[item_id].parent_id)
            for item_id in journey.flatten(JourneyItemType.ACTION)
        ],
    }

    id_str = ""
    # form = st.form(key="journey_item_form_" + item.id, border=False)
    # Loop through ancestry and add fields for each item

    changes = []

    icon_col, content_col, _ = st.columns([0.15, 0.8, 0.05], gap="small")

    # changes +=
    view_journey_item(item, journey, journey_progress, content_col, icon_col)

    if show_children:
        task_container = st.container()
        _, task_header_container, _ = task_container.columns(
            [0.15, 0.7, 0.15], vertical_alignment="top"
        )
        task_header_container.subheader(
            "Action:" if len(item.children) == 1 else "Actions:"
        )
        for child in item.children:
            journey_item_progress = (
                JourneyItemProgress.get(
                    journey_item_id=child.id,
                    user_id=get_db_user(email=st.session_state.get("username")).id,
                )
                if journey_progress is None
                else journey_progress.get_by_journey_item(child)
            )
            if journey_item_progress:
                # changes +=
                view_journey_item(
                    child, journey, journey_progress, content_col, icon_col, task_container
                )

    if len(changes) > 0:
        for change in changes:
            st.toast(f"For {change[0]} update: {change[1]}")

    st.write(" ")

    margin, status, back, complete = st.columns([0.15, 0.6, 0.1, 0.15])
    # with status:
    #     st.markdown("**Time to complete module:** 5 Days - You are on track!")

    if back.button("Back", key="back_button_" + id_str, use_container_width=True):
        st.switch_page("pages/my_journeys.py")

    if complete.button(
        "Done",
        use_container_width=True,
        type="primary",
        key="save_button_" + id_str,
    ):
        journey_item_progress = (
            JourneyItemProgress.get(
                journey_item_id=item.id,
                user_id=get_db_user(email=st.session_state.get("username")).id,
            )
            if journey_progress is None
            else journey_progress.get_by_journey_item(item)
        )
        if journey_item_progress.get_state() == JourneyItemProgressState.COMPLETED:
            journey_item_progress.complete(solo=True)
        if changes:
            item.save_to_db()
        # st.session_state.vote = {"item": item, "reason": feedback}
        del st.session_state["journey_view_id"]
        del st.session_state["journey_view_item_id"]
        del st.session_state["journey_item_show_children"]
        st.switch_page("pages/my_journeys.py")


def module_view():
    journey_id = st.query_params.get("journey") or st.session_state.get(
        "journey_view_id"
    )
    journey_item_id = st.query_params.get("item") or st.session_state.get(
        "journey_view_item_id"
    )
    journey_progress_id = st.query_params.get("item") or st.session_state.get(
        "journey_view_progress_id"
    )
    journey_item_show_children = st.query_params.get(
        "with_children"
    ) or st.session_state.get("journey_item_show_children")

    if journey_item_id is not None:
        journey = JourneyItem.get(journey_id=journey_id)
        item = journey.get_child_by_id(journey_item_id)
        journey_progress = None
        if journey_progress_id:
            journey_progress = JourneyItemProgress.get(progress_id=journey_progress_id)

        # st.title(f"{item.get_index(journey)} {item.title}")

        view_modules_items(item, journey, journey_progress, journey_item_show_children)

    else:
        st.error("No data to show. Please go back and recreate the journey.")


async def main():
    if init_sidebar(UserLevel.user) != AuthStatus.LOGGED_IN:
        st.switch_page("login.py")
        return

    module_view()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
