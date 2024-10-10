import time
from typing import List

import streamlit as st
import streamlit_shadcn_ui as ui
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.grid import grid

import os
import sys

from admin.sidebar import get_image, get_theme, init_sidebar

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))

from lib.streamlit.journey import (
    ChildPosition,
    get_journey_item_cache,
)
from lib.models.journey import (
    JourneyItem,
    JourneyItemType,
    add_journey_to_cache,
    get_available_journeys,
    get_journey_from_cache,
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
def edit_item_ancestor(
    item: JourneyItem,
    journey: JourneyItem,
    id_str: str,
    ancestor: JourneyItem,
    all_children,
    relations,
    items_filtered,
):
    print(f"step {id_str}")

    if ancestor.item_type != item.item_type:
        container = st.expander(
            f"{ancestor.item_type.value.capitalize()} Details:",
            expanded=JourneyItemType.ACTION == ancestor.item_type,
        )
    else:
        container = st.container(border=True)

    changes = []

    with container:
        st.write(ancestor.title)

        if (
            JourneyItemType.JOURNEY != ancestor.item_type
            and JourneyItemType.SECTION != ancestor.item_type
        ):
            if JourneyItemType.MODULE == ancestor.item_type:
                sel_options = items_filtered["section"]
            else:
                sel_options = items_filtered["module"]

            options = [
                id[0]
                for id in sel_options
                if id[1] == all_children[ancestor.parent_id].parent_id
                and ancestor.id != id[0]
            ]
            items = [all_children[id] for id in options]  # section_items
            titles = [
                f"{all_children[id].title}{'' if len(all_children[id].children) > 0 else ' (empty)'}"
                for id in options
            ]  # section_items

            ancestor_parent_title = st.selectbox(
                "Parent",
                options=titles,
                index=(
                    options.index(ancestor.parent_id)
                    if ancestor.parent_id in options
                    else None
                ),
                key="parent_" + id_str,
            )

            # try:
            new_id = items[titles.index(ancestor_parent_title)].id
            if new_id != ancestor.parent_id:
                # ancestor.parent_id = new_id
                changes.append(
                    (
                        ancestor.title,
                        f"Set parent as: {all_children[new_id].title}",
                        ancestor.parent_id,
                        new_id,
                    )
                )
                ancestor.move(all_children[new_id], journey)
                journey.save_to_db()
                # st.rerun()
            # except Exception as e:
            #     st.error("Failed to set:" + e)
            # ancestor.parent_id = None

        if JourneyItemType.JOURNEY != ancestor.item_type:
            if JourneyItemType.SECTION == ancestor.item_type:
                sel_options = items_filtered["section"]
            elif JourneyItemType.MODULE == ancestor.item_type:
                sel_options = items_filtered["module"]
            else:
                sel_options = items_filtered["action"]

            options = [
                id[0]
                for id in sel_options
                if id[1] == ancestor.parent_id and ancestor.id != id[0]
            ]
            items = [all_children[id] for id in options]  # section_items
            titles = [all_children[id].title for id in options]  # section_items

            if (
                ancestor.id in relations.keys()
                and len(
                    all_children[relations[ancestor.id]].children
                    if relations[ancestor.id] in all_children
                    else journey.children
                )
                > 1
            ):
                ancestor_after_title = st.selectbox(
                    "After",
                    options=["[as first item]"] + titles,
                    index=(
                        options.index(ancestor.after_id) + 1
                        if ancestor.after_id in options and ancestor.after_id is not None
                        else 0
                    ),
                    key="after_" + id_str,
                )

                if ancestor_after_title == "[as first item]":
                    if ancestor.after_id is not None:
                        changes.append(
                            (ancestor.title, "Move to first", all_children[ancestor.after_id].title, None, titles)
                        )
                        ancestor.move(None, journey)
                        journey.save_to_db()
                        # st.rerun(scope="fragment")
                else:
                    # try:
                    new_id = items[titles.index(ancestor_after_title)].id
                    if new_id != ancestor.after_id:
                        changes.append(
                            (
                                ancestor.title,
                                "Move after: " + all_children[new_id].title,
                                ancestor.after_id,
                                new_id,
                            )
                        )
                        ancestor.move(all_children[new_id], journey)
                        journey.save_to_db()
                        # st.rerun(scope="fragment")
                # except Exception as e:
                #     st.error("Failed to set:" + repr(e))

        new_title = st.text_input("Title", value=ancestor.title)
        st.write(new_title, ancestor.title.strip())
        if new_title.strip() != ancestor.title.strip():
            print("Changes")
            changes.append(
                (
                    ancestor.title,
                    "Change title to: " + new_title,
                    ancestor.title,
                    new_title,
                )
            )
            ancestor.title = ancestor.title.strip()
            print("changes 2")
            ancestor.save_to_db()
            print("Changes 3")

        if JourneyItemType.ACTION == ancestor.item_type:

            # # ancestor.icon = st.text_input("Icon", value=ancestor.icon, key = "icon_"+id_str)
            # # ancestor.intro = st.text_area("Introduction", value=ancestor.intro, key = "intro_"+id_str)
            # # ancestor.summary = st.text_area("Summary", value=ancestor.summary, key = "summary_"+id_str)

            description = st.text_area(
                "Description",
                value=ancestor.description,
                key="description_" + id_str,
                height=300,
            )
            if description.strip() != ancestor.description.strip():
                changes.append(
                    (
                        ancestor.title,
                        "Change description to: " + description,
                        ancestor.description,
                        description,
                    )
                )
                ancestor.description = description.strip()
                ancestor.save_to_db()
            # ancestor.test = st.text_area("Test", value=ancestor.test, key = "test_"+id_str)
            action = st.text_input(
                "Action", value=ancestor.action, key="action_" + id_str
            )
            if action.strip() != ancestor.action.strip():
                changes.append(
                    (
                        ancestor.title,
                        "Change action to: " + action,
                        ancestor.action,
                        action,
                    )
                )
                ancestor.action = action
                ancestor.save_to_db()

        if JourneyItemType.ACTION == ancestor.item_type:
            new_eod = st.number_input(
                "Done by end of day #",
                value=ancestor.end_of_day,
                key="eod_" + id_str,
            )
            if new_eod != ancestor.end_of_day:
                changes.append(
                    (
                        ancestor.title,
                        "Change eod to: " + new_eod,
                        ancestor.end_of_day,
                        new_eod,
                    )
                )
                ancestor.end_of_day = new_eod
                journey.update_eod()
                journey.save_to_db()

        # ancestor.item_type = st.selectbox(
        #     "Item Type", options=[e.value for e in JourneyItemType]
        # )

    return changes


@st.fragment
def edit_item(item: JourneyItem, journey: JourneyItem):
    all_children = journey.all_children_by_id()
    relations = journey.get_relations()
    ancestry = item.get_ancestry(journey)
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

    for ancestor_id in ancestry:
        ancestor = all_children[ancestor_id] if ancestor_id in all_children else journey
        id_str = f"{id_str}_{ancestor.id}" if id_str else ancestor.id
        if (
            JourneyItemType.ACTION == item.item_type
            and JourneyItemType.ACTION != ancestor.item_type
        ):
            continue

        changes += edit_item_ancestor(
            item, journey, id_str, ancestor, all_children, relations, items_filtered
        )

    cancel_col, _, save_col = st.columns([0.15, 0.7, 0.15])

    if len(changes) > 0:
        print("changes", changes)
        toasting = st.toast("Update db...")
        for change in changes:
            toasting.toast(f"For {change[0]} update: {change[1]}")

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
        st.switch_page("pages/journey_edit.py")

    # if cancel_col.button(
    #     "Cancel",
    #     use_container_width=True,
    #     type="secondary",
    #     key="cancel_button_" + id_str,
    # ):
    #     # st.session_state.vote = {"item": item, "reason": feedback}
    #     del st.session_state["journey_edit_item_id"]
    #     st.switch_page("pages/journey_edit.py")


def journey_edit():
    journey_id = st.query_params.get("journey") or st.session_state.get(
        "journey_edit_id"
    )
    journey_item_id = st.query_params.get("item") or st.session_state.get(
        "journey_edit_item_id"
    )
    if journey_item_id is not None:
        if journey_id not in get_journey_item_cache().keys():
            journey: JourneyItem = JourneyItem.load_from_db(journey_id)
            get_journey_item_cache()[journey_id] = journey
        else:
            journey = get_journey_item_cache()[journey_id]

        item = journey.get_child_by_id(journey_item_id)

        st.title("Modify " + item.title)

        edit_item(item, journey)

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
