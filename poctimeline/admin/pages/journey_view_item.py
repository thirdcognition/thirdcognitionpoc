import time
from typing import List

import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.grid import grid

import os
import sys

from admin.sidebar import get_image, init_sidebar

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))

from lib.models.journey_progress import JourneyItemProgress, JourneyItemProgressState
from lib.streamlit.journey import (
    open_logo_dialog,
)
from lib.models.journey import (
    JourneyItem,
    JourneyItemType,
)
from lib.chains.init import get_chain
from lib.helpers.journey import (
    ActionSymbol,
    load_journey_template,
    match_title_to_cat_and_id,
)
from lib.helpers.shared import pretty_print
from lib.models.user import AuthStatus, UserLevel

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


@st.dialog("Confirm completion")
def complete_journey_item(
    journey_item: JourneyItem,
    journey: JourneyItem,
    journey_item_progress: JourneyItemProgress,
):
    st.write(f"#### {journey_item.get_index(journey)} - {journey_item.title}")

    feedback = st.text_area(
        "Leave feedback",
        key=f"leave_feedback_{journey.id}_{journey_item.id}",
        placeholder="Please add some details",
    )

    left, _, right = st.columns([0.4, 0.2, 0.4])
    left.button(
        "Cancel",
        type="secondary",
        use_container_width=True,
        key=f"cancel_completion_{journey.id}_{journey_item.id}",
    )
    if right.button(
        "Complete",
        type="primary",
        use_container_width=True,
        key=f"completion_complete_{journey.id}_{journey_item.id}",
    ):
        journey_item_progress.complete(feedback=feedback if feedback else None)
        st.switch_page("pages/my_journeys.py")


@st.fragment
def view_journey_item(
    journey_item: JourneyItem,
    journey: JourneyItem,
    id_str: str,
    all_children,
    relations,
    items_filtered,
    use_container=False,
    as_children=False,
):
    if (
        JourneyItemType.JOURNEY == journey_item.item_type
        or JourneyItemType.MODULE == journey_item.item_type
    ):
        _, image_col, _ = st.columns([0.3, 0.4, 0.3])

        image_col.image(
            get_image(journey_item.icon, "icon_files"),
            use_column_width=True,
        )

    st.subheader(f"{journey_item.get_index(journey)} - {journey_item.title}")
    st.write(journey_item.description)

    journey_item_progress = JourneyItemProgress.get(journey_item_id=journey_item.id)

    if JourneyItemType.ACTION == journey_item.item_type:
        complete = st.checkbox(
            journey_item.action,
            key="action_" + journey_item.id,
            value=st.session_state.get(
                f"journey_item_complete_{journey_item.id}",
                JourneyItemProgressState.COMPLETED == journey_item_progress.get_state(),
            ),
        )
        print(f"{complete=}")
        if (
            complete
            and JourneyItemProgressState.COMPLETED != journey_item_progress.get_state()
        ):
            complete_journey_item(journey_item, journey, journey_item_progress)

    return False  # changes


@st.fragment
def view_item(item: JourneyItem, journey: JourneyItem, show_children=False):
    all_children = journey.all_children_by_id()
    relations = journey.get_relations()
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

    # changes +=
    view_journey_item(
        item, journey, item.id, all_children, relations, items_filtered, False
    )

    if show_children:
        for child in item.children:
            # changes +=
            view_journey_item(
                child,
                journey,
                item.id + "_" + child.id,
                all_children,
                relations,
                items_filtered,
                True,
                True,
            )

    _, save_col = st.columns([0.85, 0.15])

    if len(changes) > 0:
        for change in changes:
            st.toast(f"For {change[0]} update: {change[1]}")

        # st.rerun()

    if save_col.button(
        "Done",
        use_container_width=True,
        type="primary",
        key="save_button_" + id_str,
    ):
        if changes:
            item.save_to_db()
        # st.session_state.vote = {"item": item, "reason": feedback}
        del st.session_state["journey_view_id"]
        del st.session_state["journey_view_item_id"]
        del st.session_state["journey_item_show_children"]
        st.switch_page("pages/my_journeys.py")


def journey_view():
    journey_id = st.query_params.get("journey") or st.session_state.get(
        "journey_view_id"
    )
    journey_item_id = st.query_params.get("item") or st.session_state.get(
        "journey_view_item_id"
    )
    journey_item_show_children = st.query_params.get(
        "with_children"
    ) or st.session_state.get("journey_item_show_children")

    if journey_item_id is not None:
        journey = JourneyItem.get(journey_id=journey_id)
        item = journey.get_child_by_id(journey_item_id)

        # st.title(f"{item.get_index(journey)} {item.title}")

        view_item(item, journey, journey_item_show_children)

    else:
        st.error("No data to show. Please go back and recreate the journey.")


async def main():
    if init_sidebar(UserLevel.org_admin) != AuthStatus.LOGGED_IN:
        st.switch_page("login.py")
        return

    journey_view()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
