import time
from typing import List

import streamlit as st
import streamlit_shadcn_ui as ui
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.grid import grid

import os
import sys

from admin.sidebar import init_sidebar

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))

from lib.streamlit.journey import (
    ChildPosition,
    get_journey,
    get_journey_item_cache,
)
from lib.models.journey import (
    JourneyItem,
    JourneyItemType,
)
from lib.chains.init import get_chain
from lib.helpers.journey import load_journey_template, match_title_to_cat_and_id
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


@st.fragment
def edit_journey_item(
    journey_item: JourneyItem,
    journey: JourneyItem,
    id_str: str,
    all_children,
    relations,
    items_filtered,
    use_container = False,
    as_children = False
):
    print(f"step {id_str}")

    if use_container:
        container = st.expander(
            f"{journey_item.get_index(journey)}: {journey_item.title}"
            # f"{journey_item.item_type.value.capitalize()} Details:",
            # expanded=JourneyItemType.ACTION == journey_item.item_type,
        )
    else:
        container = st.container(border=True)

    changes = []

    with container:
        if not use_container:
            st.write(journey_item.title)

        if (
            JourneyItemType.JOURNEY != journey_item.item_type
            and JourneyItemType.SECTION != journey_item.item_type
            and not as_children
        ):
            if JourneyItemType.MODULE == journey_item.item_type:
                sel_options = items_filtered["section"]
            else:
                sel_options = items_filtered["module"]

            options = [
                id[0]
                for id in sel_options
                if id[1] == all_children[journey_item.parent_id].parent_id
                and journey_item.id != id[0]
            ]
            items = [all_children[id] for id in options]  # section_items
            titles = [
                f"{all_children[id].title}{'' if len(all_children[id].children) > 0 else ' (empty)'}"
                for id in options
            ]  # section_items

            journey_item_parent_title = st.selectbox(
                "Parent",
                options=titles,
                index=(
                    options.index(journey_item.parent_id)
                    if journey_item.parent_id in options
                    else None
                ),
                key="parent_" + id_str,
            )

            # try:
            new_id = items[titles.index(journey_item_parent_title)].id
            if new_id != journey_item.parent_id:
                # journey_item.parent_id = new_id
                changes.append(
                    (
                        journey_item.title,
                        f"Set parent as: {all_children[new_id].title}",
                        journey_item.parent_id,
                        new_id,
                    )
                )
                journey_item.move(all_children[new_id], journey)
                journey.save_to_db()
                st.rerun(scope="fragment")
                # st.rerun()
            # except Exception as e:
            #     st.error("Failed to set:" + e)
            # journey_item.parent_id = None

        if JourneyItemType.JOURNEY != journey_item.item_type:
            if JourneyItemType.SECTION == journey_item.item_type:
                sel_options = items_filtered["section"]
            elif JourneyItemType.MODULE == journey_item.item_type:
                sel_options = items_filtered["module"]
            else:
                sel_options = items_filtered["action"]

            options = [
                id[0]
                for id in sel_options
                if id[1] == journey_item.parent_id and journey_item.id != id[0]
            ]
            items = [all_children[id] for id in options]  # section_items
            titles = [all_children[id].title for id in options]  # section_items

            if (
                journey_item.id in relations.keys()
                and len(
                    all_children[relations[journey_item.id]].children
                    if relations[journey_item.id] in all_children
                    else journey.children
                )
                > 1
            ):
                journey_item_after_title = st.selectbox(
                    "After",
                    options=["[as first item]"] + titles,
                    index=(
                        options.index(journey_item.after_id) + 1
                        if journey_item.after_id in options and journey_item.after_id is not None
                        else 0
                    ),
                    key="after_" + id_str,
                )

                if journey_item_after_title == "[as first item]":
                    if journey_item.after_id is not None:
                        changes.append(
                            (journey_item.title, "Move to first", all_children[journey_item.after_id].title, None, titles)
                        )
                        journey_item.move(None, journey)
                        journey.save_to_db()
                        st.rerun(scope="fragment")
                else:
                    # try:
                    new_id = items[titles.index(journey_item_after_title)].id
                    if new_id != journey_item.after_id:
                        changes.append(
                            (
                                journey_item.title,
                                "Move after: " + all_children[new_id].title,
                                journey_item.after_id,
                                new_id,
                            )
                        )
                        journey_item.move(all_children[new_id], journey)
                        journey.save_to_db()
                        st.rerun(scope="fragment")
                        # st.rerun(scope="fragment")
                # except Exception as e:
                #     st.error("Failed to set:" + repr(e))

        new_title = st.text_input("Title", value=journey_item.title)
        if new_title.strip() != journey_item.title.strip():
            changes.append(
                (
                    journey_item.title,
                    "Change title to: " + new_title,
                    journey_item.title,
                    new_title,
                )
            )
            journey_item.title = new_title.strip()
            journey_item.save_to_db()

        if JourneyItemType.ACTION == journey_item.item_type:

            # # journey_item.icon = st.text_input("Icon", value=journey_item.icon, key = "icon_"+id_str)
            # # journey_item.intro = st.text_area("Introduction", value=journey_item.intro, key = "intro_"+id_str)
            # # journey_item.summary = st.text_area("Summary", value=journey_item.summary, key = "summary_"+id_str)

            description = st.text_area(
                "Description",
                value=journey_item.description,
                key="description_" + id_str,
                height=300,
            )
            if description.strip() != journey_item.description.strip():
                changes.append(
                    (
                        journey_item.title,
                        "Change description to: " + description,
                        journey_item.description,
                        description,
                    )
                )
                journey_item.description = description.strip()
                journey_item.save_to_db()
                st.rerun(scope="fragment")
            # journey_item.test = st.text_area("Test", value=journey_item.test, key = "test_"+id_str)
            action = st.text_input(
                "Action", value=journey_item.action, key="action_" + id_str
            )
            if action.strip() != journey_item.action.strip():
                changes.append(
                    (
                        journey_item.title,
                        "Change action to: " + action,
                        journey_item.action,
                        action,
                    )
                )
                journey_item.action = action
                journey_item.save_to_db()
                st.rerun(scope="fragment")

        if JourneyItemType.ACTION == journey_item.item_type:
            new_eod = st.number_input(
                "Done by end of day #",
                value=journey_item.end_of_day,
                key="eod_" + id_str,
            )
            if new_eod != journey_item.end_of_day:
                changes.append(
                    (
                        journey_item.title,
                        "Change eod to: " + new_eod,
                        journey_item.end_of_day,
                        new_eod,
                    )
                )
                journey_item.end_of_day = new_eod
                journey.update_eod()
                journey.save_to_db()
                st.rerun(scope="fragment")

        # ancestor.item_type = st.selectbox(
        #     "Item Type", options=[e.value for e in JourneyItemType]
        # )

    return changes


@st.fragment
def edit_item(item: JourneyItem, journey: JourneyItem, show_children=False):
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
    # toasting = st.toast("...")

    # for ancestor_id in ancestry:
    #     ancestor = all_children[ancestor_id] if ancestor_id in all_children else journey
    #     id_str = f"{id_str}_{ancestor.id}" if id_str else ancestor.id
    #     if (
    #         JourneyItemType.ACTION == item.item_type
    #         and JourneyItemType.ACTION != ancestor.item_type
    #     ):
    #         continue

    changes += edit_journey_item(
        item, journey, item.id, all_children, relations, items_filtered, False
    )

    if show_children:
        for child in item.children:
            changes += edit_journey_item(
                child, journey, item.id + "_" + child.id, all_children, relations, items_filtered, True, True
            )

    _, save_col = st.columns([0.85, 0.15])

    if len(changes) > 0:
        print("changes", changes)

        for change in changes:
            st.toast(f"For {change[0]} update: {change[1]}")

        # st.rerun()

    if save_col.button(
        "Done",
        use_container_width=True,
        type="primary",
        key="save_button_" + id_str,
        # disabled=changes == False,
    ):
        if changes:
            item.save_to_db()
        # st.session_state.vote = {"item": item, "reason": feedback}
        del st.session_state["journey_edit_item_id"]
        del st.session_state["journey_edit_item_show_children"]
        st.switch_page("pages/journey_edit.py")


def journey_edit():
    journey_id = st.query_params.get("journey") or st.session_state.get(
        "journey_edit_id"
    )
    journey_item_id = st.query_params.get("item") or st.session_state.get(
        "journey_edit_item_id"
    )
    journey_item_show_children = st.query_params.get("with_children") or st.session_state.get(
        "journey_edit_item_show_children"
    )

    if journey_item_id is not None:
        journey = get_journey(journey_id=journey_id)
        item = journey.get_child_by_id(journey_item_id)

        st.title(f"Modify {item.item_type.name.capitalize()} {item.get_index(journey)}")

        edit_item(item, journey, journey_item_show_children)

    else:
        st.error("No data to show. Please go back and recreate the journey.")


async def main():
    if init_sidebar(UserLevel.org_admin) != AuthStatus.LOGGED_IN:
        st.switch_page("Admin_Home.py")
        return

    journey_edit()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
