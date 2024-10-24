import time
from typing import List

import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.grid import grid

import os
import sys

# from admin.pages.journey_edit import assign_journey
from admin.sidebar import get_image, init_sidebar

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))

from lib.streamlit.journey import (
    assign_journey,
    open_item,
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


def edit_journey_item(
    journey_item: JourneyItem,
    journey: JourneyItem,
    id_str: str,
    all_children,
    relations,
    items_filtered,
    use_container=False,
    as_children=False,
):
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
            st.subheader(((journey_item.get_index(journey) + " ") if JourneyItemType.JOURNEY != journey_item.item_type else "") + journey_item.title)


        if (
            JourneyItemType.JOURNEY == journey_item.item_type
            or JourneyItemType.MODULE == journey_item.item_type
        ):
            image_col, container = st.columns([0.25,0.75])

            image_col.image(
                get_image(journey_item.icon, "icon_files"),
                use_column_width=True,
            )
            if image_col.button(
                ActionSymbol.image.value,
                use_container_width=True,
                key=f"journey_process_change_item_image_{journey_item.id}",
            ):
                open_logo_dialog(journey_item, journey)
        else:
            container = st.container()

        with container:
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
                    f"{all_children[id].get_index(journey)} {all_children[id].title}{'' if len(all_children[id].children) > 0 else ' (empty)'}"
                    for id in options
                ]  # section_items

                journey_item_parent_title = st.selectbox(
                    JourneyItemType.previous(journey_item.item_type).value.capitalize(),  #"Parent",
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
                            f"Set {JourneyItemType.previous(journey_item.item_type).value} as: {all_children[new_id].title}",
                            journey_item.parent_id,
                            new_id,
                        )
                    )
                    journey_item.move(all_children[new_id], journey)
                    journey.save_to_db()
                    # st.rerun(scope="fragment")
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
                titles = [
                    f"{all_children[id].get_index(journey)} {all_children[id].title}"
                    for id in options
                ]  # section_items

                if (
                    journey_item.id in relations.keys()
                    and len(
                        all_children[relations[journey_item.id]].children
                        if relations[journey_item.id] in all_children
                        else journey.children
                    )
                    > 1
                ):
                    col1, col2 = st.columns([0.85, 0.15], vertical_alignment="bottom")
                    # print(journey_item.after_id in options, journey_item.id in options, journey_item.after_id.split('-')[-1] if journey_item.after_id else None, journey_item.id.split('-')[-1])
                    # if journey_item.after_id not in options and journey_item.id in options:
                    #     after_index = options.index(journey_item.id) - 1
                    #     journey_item.after_id = options[after_index] if after_index > 0 else None
                    #     print("Index", after_index)

                    journey_item_after_title = col1.selectbox(
                        "Located after",
                        options=["[as first item]"] + titles,
                        index=(
                            options.index(journey_item.after_id) + 1
                            if journey_item.after_id in options
                            and journey_item.after_id is not None
                            else 0
                        ),
                        key="after_" + id_str,
                    )

                    if col2.button("Move", key="move_after_"+id_str ,use_container_width=True):
                        if journey_item_after_title == "[as first item]":
                            if journey_item.after_id is not None:
                                changes.append(
                                    (
                                        journey_item.title,
                                        "Move to first",
                                        all_children[journey_item.after_id].title if journey_item.after_id in all_children.keys() else "None",
                                        None,
                                        titles,
                                    )
                                )
                                journey_item.move(None, journey)
                                journey.save_to_db()
                                st.rerun()
                            else:
                                print("No after_id found for "+journey_item.get_ident(with_index=False))
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
                                st.rerun()
                                    # st.rerun(scope="fragment")
                                    # st.rerun(scope="fragment")
                        # except Exception as e:
                    #     st.error("Failed to set:" + repr(e))

            new_title = st.text_input("Title", value=journey_item.title, key="title_"+id_str)
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

            # # journey_item.icon = st.text_input("Icon", value=journey_item.icon, key = "icon_"+id_str)
            # # journey_item.intro = st.text_area("Introduction", value=journey_item.intro, key = "intro_"+id_str)
            # # journey_item.summary = st.text_area("Summary", value=journey_item.summary, key = "summary_"+id_str)

            description = st.text_area(
                "Description",
                value=journey_item.description,
                key="description_" + id_str,
                height=300 if JourneyItemType.ACTION == journey_item.item_type else 30,
            )
            if (description or "").strip() != (journey_item.description or "").strip():
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
                # st.rerun(scope="fragment")
            # journey_item.test = st.text_area("Test", value=journey_item.test, key = "test_"+id_str)
            if JourneyItemType.ACTION == journey_item.item_type:
                action = st.text_input(
                    "Action", value=journey_item.action, key="action_" + id_str
                )
                if (action or "").strip() != (journey_item.action or "").strip():
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
                    # st.rerun(scope="fragment")

            # if JourneyItemType.ACTION == journey_item.item_type:
            #     new_eod = st.number_input(
            #         "Done by end of day #",
            #         help="Will only affect new journeys",
            #         value=journey_item.end_of_day,
            #         key="eod_" + id_str,
            #     )
            #     if new_eod != journey_item.end_of_day:
            #         changes.append(
            #             (
            #                 journey_item.title,
            #                 "Change eod to: " + str(new_eod),
            #                 journey_item.end_of_day,
            #                 new_eod,
            #             )
            #         )
            #         journey_item.end_of_day = new_eod
            #         journey.update_eod()
            #         journey.save_to_db()
            #         st.rerun(scope="fragment")

            edit_col, _, remove_col = st.columns([0.3, 0.4, 0.3])
            if as_children and journey_item.item_type not in [JourneyItemType.ACTION] and edit_col.button(
                f"Edit "+JourneyItemType.next(journey_item.item_type).value + "s",
                key=f"open_button_{journey_item.id}",
                use_container_width=True,
            ):
                open_item(
                    journey_item,
                    journey,
                    JourneyItemType.MODULE == journey_item.item_type,
                )

            with remove_col.popover("Remove "+journey_item.item_type.value, use_container_width=True):
                if st.button(
                    f"Are you sure you want to remove:\n\n{journey_item.title}?",
                    key=f"delete_button_{journey_item.id}",
                    use_container_width=True,
                ):
                    parent:JourneyItem = all_children[journey_item.parent_id]
                    parent.remove_child(journey_item)
                    parent.save_to_db()
                    journey.reset_cache()
                    st.rerun()

            # ancestor.item_type = st.selectbox(
            #     "Item Type", options=[e.value for e in JourneyItemType]
            # )

    return changes


@st.fragment
def edit_item(item: JourneyItem, journey: JourneyItem, show_children=False):
    all_children = journey.all_children_by_id(reset=True)
    relations = journey.get_relations()
    # ancestry = item.get_ancestry(journey)
    items_filtered = {
        "section": [
            (item_id, all_children[item_id].parent_id)
            for item_id in journey.flatten(JourneyItemType.SECTION, reset=True)
        ],
        "module": [
            (item_id, all_children[item_id].parent_id)
            for item_id in journey.flatten(JourneyItemType.MODULE, reset=True)
        ],
        "action": [
            (item_id, all_children[item_id].parent_id)
            for item_id in journey.flatten(JourneyItemType.ACTION, reset=True)
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

    if item.item_type != JourneyItemType.JOURNEY and item.parent_id:
        if st.button("Edit " + JourneyItemType.previous(item.item_type).value.capitalize(), key="edit_parent_"+item.parent_id):
            open_item(item.parent_id, journey)
    if item.item_type == JourneyItemType.JOURNEY and not st.session_state.get("journey_simple_edit", False):
        if st.button("Edit journey", key="edit_parent_"+journey.id):
            st.session_state["journey_edit_id"] = item.id
            del st.session_state["journey_edit_item_id"]
            del st.session_state["journey_edit_item_show_children"]
            st.switch_page("pages/journey_edit.py")

    changes += edit_journey_item(
        item, journey, item.id, all_children, relations, items_filtered, False
    )

    if item.children:
        for child in item.children:
            changes += edit_journey_item(
                child,
                journey,
                item.id + "_" + child.id,
                all_children,
                relations,
                items_filtered,
                not show_children,
                True,
            )

    with st.container(border=True):
        st.write("Add " + JourneyItemType.next(item.item_type).value + "s")

        options = [child.id for child in item.children] if item.children else []
        titles = [
            f"{all_children[id].get_index(journey)} - {all_children[id].title}"
            for id in options
            if id != item.id
        ]  # section_items

        add_after_title = st.selectbox(
            "After",
            options=["[as first item]"] + titles,
            index=(
                options.index(item.after_id) + 1
                if item.after_id in options and item.after_id is not None
                else 0
            ),
            key="after_" + id_str,
        )

        item_type = JourneyItemType.ACTION
        if JourneyItemType.JOURNEY == item.item_type:
            item_type = JourneyItemType.SECTION
        elif JourneyItemType.SECTION == item.item_type:
            item_type = JourneyItemType.MODULE

        if st.button("Add " + JourneyItemType.next(item.item_type).value, key="add_child_" + item.id):
            new_item = JourneyItem.create_new(
                {
                    "parent_id": item.id,
                    "icon": item.icon,
                    "template_id": item.template_id,
                    "end_of_day": item.end_of_day,
                    "item_type": item_type,
                    "content_instructions": item.content_instructions.model_copy(),
                }
            )
            item.add_child(
                new_item,
                (
                    0
                    if add_after_title == "[as first item]"
                    else titles.index(add_after_title)
                ),
            )
            journey.save_to_db()
            journey.reset_cache()
            st.rerun()

    _, save_col = st.columns([0.8, 0.2])

    if len(changes) > 0:
        for change in changes:
            st.toast(f"For {change[0]} update: {change[1]}")

        st.rerun()

    if save_col.button(
        "Done",
        use_container_width=True,
        type="primary",
        key="save_button_" + id_str,
        # disabled=changes == False,
    ):
        if changes:
            item.save_to_db()
            journey.reset_cache()
        # st.session_state.vote = {"item": item, "reason": feedback}
        simple_journey_edit = st.session_state.get("journey_simple_edit", False)
        if "journey_simple_edit" in st.session_state and simple_journey_edit:
            del st.session_state["journey_simple_edit"]
            del st.session_state["journey_edit_id"]

        del st.session_state["journey_edit_item_id"]
        del st.session_state["journey_edit_item_show_children"]
        if not simple_journey_edit:
            st.switch_page("pages/journey_edit.py")
        else:
            st.switch_page("pages/journey_simple_manage.py")


def journey_edit():
    journey_id = st.query_params.get("journey") or st.session_state.get(
        "journey_edit_id"
    )
    journey_item_id = st.query_params.get("item") or st.session_state.get(
        "journey_edit_item_id"
    )
    journey_item_show_children = st.query_params.get(
        "with_children"
    ) or st.session_state.get("journey_edit_item_show_children")

    if journey_item_id is not None:
        journey = JourneyItem.get(journey_id=journey_id)
        item = journey.get_child_by_id(journey_item_id)

        if item is None:
            st.error("Unable to find item")
            return

        st.title(f"Modify {item.item_type.name.capitalize()} {item.get_index(journey)}")

        edit_item(item, journey, journey_item_show_children)

    else:
        st.error("No data to show. Please go back and recreate the journey.")


async def main():
    if init_sidebar(UserLevel.org_admin) != AuthStatus.LOGGED_IN:
        st.switch_page("login.py")
        return

    simple_journey_edit = st.session_state.get("journey_simple_edit", False)
    if simple_journey_edit:
        tab1, tab2 = st.tabs(["Modify journey", "Assign to individual(s)"])
    else:
        tab1 = st.empty()

    with tab1:
        journey_edit()

    if simple_journey_edit:
        with tab2:
            journey_id = st.query_params.get("journey") or st.session_state.get(
                "journey_edit_id"
            )
            if journey_id:
                assign_journey(journey_id)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
