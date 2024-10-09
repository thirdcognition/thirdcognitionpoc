from enum import Enum
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

container_level = ["journey", "section", "module", "action"]


def get_stylable_container_selector(id):
    return f'div[data-testid="stVerticalBlock"]:has(> div.element-container > div.stMarkdown > div[data-testid="stMarkdownContainer"] > p > span.{id})'


@st.dialog("Change logo to...", width="large")
def open_logo_dialog(item: JourneyItem, journey: JourneyItem):
    id_str = journey.id + "_" + item.id

    container = stylable_container(
        key=f"item_container_{id_str}",
        css_styles=[
            f"""
            div {{
                margin: 0;
                padding: 0;
                justify-content: center;
            }}""",
            """
            label {{
                margin: 0;
                padding: 0;
                justify-content: center;
            }}
            """,
        ],
    ).container()
    logo_list = {}
    with container:
        image_grid = grid(10, 10, 10, 10, 10, vertical_align="center")
        for i in range(1, 51):
            with image_grid.container():
                logo_id = "logo_" + str(i)
                st.image(get_image(logo_id, path="icon_files"))
                if st.button(
                    selected_symbol if item.icon == logo_id else unselected_symbol,
                    key="image_" + id_str + "_" + logo_id,
                    disabled=item.icon == logo_id,
                    use_container_width=True,
                ):
                    print("Change icon", item.icon, logo_id)
                    if item.icon != logo_id:
                        item.icon = logo_id
                        st.rerun(scope="fragment")
                    else:
                        item.icon = None

                # st.checkbox(label=str(i), label_visibility="collapsed", value=(logo_id) == ancestor.icon, key="image_"+id_str+"_"+logo_id, on_change=change_icon)

    # Selector with 1-50 images that when selected updates ancestor.icon to the id
    # Image id's are logo_1 to logo_50 and the image url can be fetched via get_image(image_id)
    # The existing icon should be automatically selected.
    # for logo_id, selected in logo_list.items():
    #     print(logo_id, selected)
    print("logo", item.icon)


@st.dialog(f"Change contents of..", width="large")
def open_action(item: JourneyItem, journey: JourneyItem):
    # st.write(f"Let's get started with action {item.title}")
    journey_relations = journey.get_relations()

    ancestry = item.get_ancestry(journey)

    # ancestry: list[JourneyItem] = [parent]
    # while parent:
    #     # add parent to the beginning of the list and check its parent
    #     if parent.id and parent.id in journey_relations.keys():
    #         parent = journey_relations[parent.id]
    #     elif parent.id is not None:
    #         parent = None
    #     if parent and parent not in ancestry:
    #         ancestry.insert(0, parent)

    # ancestry.append(item)

    section_items = journey.flatten(JourneyItemType.SECTION)
    section_options = [item.id for item in section_items]
    # Get all MODULE type items from journey children and children's children
    module_items = journey.flatten(JourneyItemType.MODULE)
    module_options = [item.id for item in module_items]
    # Get all ACTION type items from journey children and children's children
    action_items = journey.flatten(JourneyItemType.ACTION)
    action_options = [item.id for item in action_items]

    id_str = ""
    # form = st.form(key="journey_item_form_" + item.id, border=False)
    # Loop through ancestry and add fields for each item
    for ancestor in ancestry:
        id_str = f"{id_str}_{ancestor.id}" if id_str else ancestor.id
        with st.expander(
            f"{ancestor.item_type.value.capitalize()} Details:",
            expanded=ancestor.item_type == JourneyItemType.ACTION,
        ):
            st.write(ancestor.title)
            # Create input fields for each attribute of the JourneyItem object
            # ancestor.id = st.text_input('Unique Identifier', value=ancestor.id)
            # ancestor.template_id = st.text_input('Template ID', value=ancestor.template_id)

            if (
                ancestor.item_type != JourneyItemType.JOURNEY
                and ancestor.item_type != JourneyItemType.SECTION
            ):
                if ancestor.item_type == JourneyItemType.MODULE:
                    items = section_items
                    options = section_options
                else:
                    items = module_items
                    options = module_options

                ancestor.parent_id = st.selectbox(
                    "Parent",
                    options=items,
                    index=(
                        options.index(ancestor.parent_id)
                        if ancestor.parent_id in options
                        else None
                    ),
                    key="parent_" + id_str,
                )

            if ancestor.item_type != JourneyItemType.JOURNEY:
                if ancestor.item_type == JourneyItemType.SECTION:
                    items = section_items
                    options = section_options
                elif ancestor.item_type == JourneyItemType.MODULE:
                    items = module_items
                    options = module_options
                else:
                    items = action_items
                    options = action_options
                if (
                    ancestor.id in journey_relations.keys()
                    and len(journey_relations[ancestor.id].children) > 1
                ):
                    ancestor.after_id = st.selectbox(
                        "After",
                        options=items,
                        index=(
                            options.index(ancestor.after_id)
                            if ancestor.after_id in options
                            else None
                        ),
                        key="after_" + id_str,
                    )

            ancestor.title = st.text_input("Title", value=ancestor.title)
            if ancestor.item_type == JourneyItemType.ACTION:

                # # ancestor.icon = st.text_input("Icon", value=ancestor.icon, key = "icon_"+id_str)
                # # ancestor.intro = st.text_area("Introduction", value=ancestor.intro, key = "intro_"+id_str)
                # # ancestor.summary = st.text_area("Summary", value=ancestor.summary, key = "summary_"+id_str)

                ancestor.description = st.text_area(
                    "Description",
                    value=ancestor.description,
                    key="description_" + id_str,
                    height=300,
                )
                # ancestor.test = st.text_area("Test", value=ancestor.test, key = "test_"+id_str)
                ancestor.action = st.text_input(
                    "Action", value=ancestor.action, key="action_" + id_str
                )

            ancestor.end_of_day = st.number_input(
                "Done by end of day #", value=ancestor.end_of_day, key="eod_" + id_str
            )
            # ancestor.item_type = st.selectbox(
            #     "Item Type", options=[e.value for e in JourneyItemType]
            # )

    if st.button("Save", use_container_width=True, type="primary"):
        # st.session_state.vote = {"item": item, "reason": feedback}
        st.rerun()


level_step = 1


class ChildPosition(Enum):
    FIRST = 1
    MIDDLE = 2
    LAST = 3


@st.fragment
def edit_journey(journey: JourneyItem):
    theme = get_theme()
    # with st.container(border=True):
    with stylable_container(
        key="journey_creation_section_container",
        css_styles=[
            f"""
            {{
                border: 1px solid {theme["borderColor"] if theme != None and theme["base"] == "dark" else theme["borderColorLight"]};
                border-radius: 0.5rem;
                overflow: hidden;
                padding: 0.25rem 0.75rem;
                padding-top: 1.5rem;
                gap: 0;
            }}
            """,
            """
            > div[data-testid=stVerticalBlockBorderWrapper]:last-child > div > div[data-testid=stVerticalBlock] {
                gap: 0;
            }
            """,
            """
            div[data-testid=stHeadingWithActionElements] {
                margin-bottom: 1rem;
            }
            """,
        ],
    ):
        # st.write(st.session_state["journey_creation_data"])

        write_item(journey, journey=journey)


def write_item(
    item: JourneyItem,
    id_chain="",
    journey: JourneyItem = None,
    position=ChildPosition.MIDDLE,
):
    item_id = item.id if id_chain == "" else f"{id_chain}_{item.id}"
    move_item: JourneyItem = st.session_state.get("journey_item_move")
    in_move = move_item is not None
    st.session_state["journey_item_state"] = st.session_state.get(
        "journey_item_state", {}
    )
    level_multiplier = max(
        (container_level.index(item.item_type.value) - 1) * level_step, 0.01
    )

    if (
        item.item_type not in [JourneyItemType.JOURNEY, JourneyItemType.ACTION]
        and in_move
        and item.id != move_item.id
    ):
        select_column, main_container = st.columns([0.05, 0.95])
        ancestry = move_item.get_ancestry(journey)
        ancestry_ids = [ancestor.id for ancestor in ancestry]

        if item.id in ancestry_ids:
            # print(f"{item.id} is in:")
            # pretty_print(ancestry_ids, force=True)
            select_column.button(
                selected_symbol, key=f"move_item_{item_id}_parent", disabled=True
            )
        else:
            if select_column.button(unselected_symbol, key=f"move_item_{item_id}_in"):
                move_item.move(item, journey)
                st.session_state["journey_item_move"] = None
                # ancestry = move_item.get_ancestry(journey, reset=True)
                st.rerun()
    else:
        main_container = st.empty()

    if (
        item.item_type.value != container_level[-1]
        and item.item_type.value in container_level
    ):
        # print(
        #     container_level,
        #     item.item_type.value,
        #     container_level.index(item.item_type.value),
        #     type(item.item_type),
        #     f"\n{item.item_type == JourneyItemType.JOURNEY=}",
        #     f"\n{item.item_type == JourneyItemType.SECTION=}",
        #     f"\n{item.item_type == JourneyItemType.MODULE=}",
        #     f"\n{item.item_type == JourneyItemType.ACTION=}\n\n"
        # )
        item_state: dict = st.session_state["journey_item_state"].get(item_id, {})
        item_state["open"] = item_state.get("open", False)
        st.session_state["journey_item_state"][item_id] = item_state
        if item.item_type == JourneyItemType.JOURNEY:
            container = main_container
        else:
            theme = get_theme()
            if theme["base"] == "light":
                color = theme["borderColorLight"]
            else:
                color = theme["borderColor"]

            key = f"item_container_{item_id}"
            with main_container:
                container = stylable_container(
                    key=key,
                    css_styles=[
                        f"""
                        {{
                            border: 1px solid {color};
                            {f"border-radius: 0.5rem 0.5rem 0 0;" if position == ChildPosition.FIRST and item.parent_id == journey.id else ""}
                            {f"border-radius: 0 0 0.5rem 0.5rem;" if position == ChildPosition.LAST and item.parent_id == journey.id and not item_state["open"] else ""}
                            margin-top: -1px;
                            {f"margin-bottom: 1rem;" if position == ChildPosition.LAST and item.parent_id == journey.id and not item_state["open"] else ""}
                            padding: 0;
                            padding-top: 1rem;
                            padding-left: {level_multiplier}rem;
                            gap: 0;
                        }}""",
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
                        """"
                        .stButton > button[kind=primary] p:last-child {
                            font-size: 1.1rem;
                            max-width: 100%;
                            text-overflow: ellipsis;
                        }
                """,
                    ],
                )
        if item.item_type == JourneyItemType.JOURNEY:
            st.subheader(item.title)
        elif (
            item.item_type == JourneyItemType.SECTION
            or item.item_type == JourneyItemType.MODULE
        ):

            col1, col3 = container.columns([0.9, 0.1], vertical_alignment="center")
            with col1:
                if st.button(
                    ("&#9660;" if item_state["open"] else "&#9654;")
                    + "\n\n"
                    + item.title,
                    key=f"open_button_{item_id}",
                    type="primary",
                ):
                    item_state["open"] = not item_state["open"]
                    st.rerun(scope="fragment")
        else:
            st.markdown(
                "####"
                + ("#" * container_level.index(item.item_type.value))
                + " "
                + item.title
            )

        if (
            item.item_type == JourneyItemType.SECTION
            or item.item_type == JourneyItemType.MODULE
        ):
            if item_state["open"]:
                # with container:
                with stylable_container(
                    key="journey_process_container_" + item_id,
                    css_styles=[
                        """
                        {
                            gap: 0;
                        }
                        """, """
                        > div[data-testid=stVerticalBlockBorderWrapper]:nth-child(2) {
                            margin-top:1rem;
                        }
                        """
                    ],
                ):
                    if item.item_type == JourneyItemType.MODULE and item.icon:
                        with stylable_container(
                            key="journey_process_button_container_" + item_id,
                            css_styles=[
                                """
                                .stColumn:last-child {
                                    justify-content: flex-end;
                                    display: flex;
                                }
                                """,
                                """
                                .stColumn:last-child > div {
                                    width: 4rem;
                                }
                                """,
                                """
                                button {
                                    margin-top: 1rem;
                                    gap: 0;
                                    width: 4rem;
                                }
                                """,
                                """
                                button p {
                                    font-weight: bold;
                                    font-size: 1.33rem;
                                }
                                """,
                            ],
                        ):
                            _, image_col, edit_col = st.columns([0.2, 0.6, 0.2])

                            image_col.image(
                                get_image(item.icon, "icon_files"),
                                use_column_width=True,
                            )

                            if edit_col.button(
                                "&#9968;",
                                use_container_width=True,
                                key=f"journey_process_change_item_image_{item_id}",
                            ):
                                open_logo_dialog(item, journey)

                        # pretty_print(item, force=True)
                    for index, child in enumerate(item.children):
                        position = (
                            ChildPosition.FIRST
                            if index == 0
                            else (
                                ChildPosition.LAST
                                if index == len(item.children) - 1
                                else ChildPosition.MIDDLE
                            )
                        )
                        write_item(child, journey=journey, position=position)
        elif item.children:
            for index, child in enumerate(item.children):
                position = (
                    ChildPosition.FIRST
                    if index == 0
                    else (
                        ChildPosition.LAST
                        if index == len(item.children) - 1
                        else ChildPosition.MIDDLE
                    )
                )

                write_item(child, journey=journey, position=position)
    elif item.item_type == JourneyItemType.ACTION:
        _, subcol1, subcol2 = st.columns([0.025, 0.8, 0.1])
        with subcol1:
            with stylable_container(
                key=f"content_container_{item_id}",
                css_styles="""
                {
                    margin: 1rem 0;
                    padding-top: 1rem;
                    gap: 0.5rem;
                }
                """,
            ):
                # header_row1, _, header_row2 = st.columns(
                #     [0.15, 0.05, 0.8], vertical_alignment="center"
                # )
                # header_row1.image(
                #     get_image(item.icon, "icon_files"), use_column_width=True
                # )
                st.markdown("##### " + item.title)
                st.markdown(item.description)
        with subcol2:
            key = f"button_container_{item_id}"
            with stylable_container(
                key=key,
                css_styles=[
                    f"""
                    {{
                        margin-top: 2.2rem;
                        gap: 0.5rem;
                        width: 4rem;
                        display: flex;
                        justify-content: flex-end;
                    }}
                """,
                    """
                    button p {
                        font-weight: bold;
                        font-size: 1.33rem;
                    }
                """,
                ],
            ):
                if in_move:
                    if move_item.id == item.id and st.button(
                        selected_symbol,
                        key=f"move_button_{item_id}",
                        type="secondary",
                        use_container_width=True,
                    ):
                        move_item.move(item, journey)
                        st.session_state["journey_item_move"] = None
                        st.rerun()
                    elif move_item.id != item.id and st.button(
                        below_symbol,
                        key=f"move_button_{item_id}",
                        type="secondary",
                        use_container_width=True,
                        disabled=move_item.after_id == item.id,
                    ):
                        move_item.move(item, journey)
                        st.session_state["journey_item_move"] = None
                        st.rerun()
                else:
                    with st.popover("\+", use_container_width=True):
                        if st.button(
                            "Add new before",
                            key=f"add_before_button_{item_id}",
                            type="secondary",
                            use_container_width=True,
                        ):
                            print("add before")
                        if st.button(
                            "Add new after",
                            key=f"add_after_button_{item_id}",
                            type="secondary",
                            use_container_width=True,
                        ):
                            print("add after")
                    if st.button(
                        up_down_symbol,
                        key=f"move_button_{item_id}",
                        type="secondary",
                        use_container_width=True,
                    ):
                        st.session_state["journey_item_move"] = item
                        st.rerun()
                    if st.button(
                        "&#9881;",
                        key=f"edit_button_{item_id}",
                        type="secondary",
                        use_container_width=True,
                    ):
                        open_action(item, journey)
                    with st.popover("&#9747;", use_container_width=True):
                        if st.button(
                            f"Are you sure you want to remove:\n\n{item.title}?",
                            key=f"delete_button_{item_id}",
                            use_container_width=True,
                        ):
                            print("remove")


def journey_edit():
    journey_id = (
        st.query_params.get("journey", None)
        or ("journey_creation_id" in st.session_state and st.session_state.journey_creation_id)
        or None
    )
    if (
        journey_id is not None
    ):

        journey: JourneyItem = JourneyItem.load_from_db(journey_id)

        print("Journey ID", journey_id)
        print(f"Journey keys: {get_available_journeys()}")
        edit_journey(journey)

    else:
        st.error("No data to show. Please go back and recreate the journey.")

    st.subheader("Assign Journey")

    with st.container(border=True):
        # st.write(
        #     JourneyItem.from_json(st.session_state["journey_creation_data"])
        #     if "journey_creation_data" in st.session_state
        #     else "No data"
        # )
        assign_to = st.multiselect(
            "Assign Journey to",
            [
                "Individual(s)",
                "Team(s)",
                "Department(s)",
                "Location(s)",
                "Subsidiary",
            ],
            ["Individual(s)"],
        )
        st.divider()

        st.subheader("Assign to Individual(s)")
        emp_id = st.text_input(
            "Employee Details (Name / ID)",
            placeholder="eg. John Smith / XH12345",
            key=1,
        )
        st.button("Add Employee", type="primary")
        st.markdown(" ")

    col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
    with col1:
        # st.page_link("pages/create_journey_2.py", label="Back")
        if st.button("Back", use_container_width=True):
            st.session_state["journey_creation_state"] = "init"
            st.rerun()
    with col3:
        if st.button("Continue", use_container_width=True):
            print("donothing")
        # st.page_link("main.py", label="Continue")


async def main():
    st.title("Modify Journey")

    if init_sidebar(UserLevel.org_admin) != AuthStatus.LOGGED_IN:
        return

    st.markdown(" ")
    journey_edit()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
