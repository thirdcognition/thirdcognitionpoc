import time
from typing import List

import streamlit as st
import streamlit_shadcn_ui as ui
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.grid import grid

import os
import sys

from admin.global_styles import get_theme
from admin.sidebar import get_image, init_sidebar

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))

from lib.streamlit.journey import ChildPosition
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
from lib.models.user import AuthStatus, UserLevel, get_all_users, get_user_org

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


container_level = ["journey", "section", "module", "action"]

level_step = 1


def get_stylable_container_selector(id):
    return f'div[data-testid="stVerticalBlock"]:has(> div.element-container > div.stMarkdown > div[data-testid="stMarkdownContainer"] > p > span.{id})'




                # st.checkbox(label=str(i), label_visibility="collapsed", value=(logo_id) == ancestor.icon, key="image_"+id_str+"_"+logo_id, on_change=change_icon)

    # Selector with 1-50 images that when selected updates ancestor.icon to the id
    # Image id's are logo_1 to logo_50 and the image url can be fetched via get_image(image_id)
    # The existing icon should be automatically selected.
    # for logo_id, selected in logo_list.items():
    #     print(logo_id, selected)
    print("logo", item.icon)


# @st.dialog(f"Change contents of...", width="large")
def open_item(item: JourneyItem, journey: JourneyItem, show_children=False):
    st.session_state["journey_edit_item_id"] = item.id
    st.session_state["journey_edit_journey"] = journey.id
    st.session_state["journey_edit_item_show_children"] = show_children
    st.switch_page("pages/journey_edit_item.py")

    # edit_item(item, journey)


@st.fragment
def write_section_module(item: JourneyItem, journey: JourneyItem, item_id: str):
    item_state: dict = st.session_state["journey_item_state"].get(item_id, {})

    if item_state["open"]:

        # with container:
        with stylable_container(
            key="journey_process_container_" + item_id,
            css_styles=[
                """
                {
                    gap: 0;
                }
                """,
                # """
                # > div[data-testid=stVerticalBlockBorderWrapper]:nth-child(2) {
                #     margin-top:1rem;
                # }
                # """,
            ],
        ):
            if JourneyItemType.MODULE == item.item_type and item.icon:
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
                        .stElementContainer:first-child button {
                            margin-top: 1rem;
                        }
                        """,
                        """
                        .stVerticalBlock {
                            gap: 0.5rem;
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
                    _, image_col, edit_col = st.columns([0.175, 0.3, 0.225])

                    image_col.image(
                        get_image(item.icon, "icon_files"),
                        use_column_width=True,
                    )


                    if edit_col.button(
                        ActionSymbol.edit.value,
                        key=f"edit_button_{item_id}",
                        type="secondary",
                        use_container_width=True,
                        disabled=item_state.get("edit", False),
                    ):
                        # item_state["edit"] = True
                        # st.rerun(scope="fragment")
                        open_item(item, journey, True)

            # if item_state.get("edit"):
            #     with st.container(border=True):
            #         title_con, cancel_con = st.columns([0.9, 0.1])
            #         title_con.subheader("Edit: " + item.title)
            #         if cancel_con.button(
            #             "X", key=f"cancel_edit_" + item.id, use_container_width=True
            #         ):
            #             del item_state["edit"]
            #             st.rerun(scope="fragment")
            #         edit_item(item, journey)

            # pretty_print(item, force=True)
            # else:
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


@st.fragment
def write_action(item: JourneyItem, journey: JourneyItem, item_id: str):
    # item_state: dict = st.session_state["journey_item_state"].get(item_id, {})

    # if item_state.get("edit"):
    #     with st.container(border=True):
    #         title_con, cancel_con = st.columns([0.9, 0.1])
    #         title_con.subheader("Edit: " + item.title)
    #         if cancel_con.button(
    #             "X", key=f"cancel_edit_" + item.id, use_container_width=True
    #         ):
    #             del item_state["edit"]
    #             st.rerun(scope="fragment")
    #         edit_item(item, journey)
    #     return

    move_item: JourneyItem = st.session_state.get("journey_item_move")
    in_move = move_item is not None
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
                    display: flex;
                    justify-content: flex-end;
                    text-align: right;
                }}
            """,
             """
                button {
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
            if in_move:
                if move_item.id == item.id and st.button(
                    ActionSymbol.selected.value,
                    key=f"move_button_{item_id}",
                    type="secondary",
                ):
                    move_item.move(item, journey)
                    st.session_state["journey_item_move"] = None
                    st.rerun()
                elif move_item.id != item.id and st.button(
                    ActionSymbol.below.value,
                    key=f"move_button_{item_id}",
                    type="secondary",
                    disabled=move_item.after_id == item.id,
                ):
                    move_item.move(item, journey)
                    st.session_state["journey_item_move"] = None
                    st.rerun()
            else:
                all_children = journey.all_children_by_id()
                with st.popover(ActionSymbol.add.value, use_container_width=True):
                    if st.button(
                        "Add new before",
                        key=f"add_before_button_{item_id}",
                        type="secondary",
                        use_container_width=True
                    ):
                        parent = all_children[item.parent_id]
                        new_item = JourneyItem.create_new(
                            {
                                "parent_id": parent.id,
                                "after_id": item.after_id,
                                "icon": item.icon,
                                "template_id": item.template_id,
                                "end_of_day": item.end_of_day,
                                "item_type": JourneyItemType.ACTION,
                                "content_instructions": item.content_instructions.model_copy(),
                            }
                        )
                        # parent.children.insert(parent.children.index(item), new_item)
                        parent.add_child(new_item, parent.children.index(item))
                        # item.after_id = new_item.id
                        # item.move(new_item, journey)
                        journey.save_to_db()
                        open_item(new_item, journey)
                        # st.rerun()
                        # print("add before")
                    if st.button(
                        "Add new after",
                        key=f"add_after_button_{item_id}",
                        type="secondary",
                        use_container_width=True
                    ):
                        parent = all_children[item.parent_id]
                        new_item = JourneyItem.create_new(
                            {
                                "parent_id": parent.id,
                                "icon": item.icon,
                                "after_id": item.id,
                                "template_id": item.template_id,
                                "end_of_day": item.end_of_day,
                                "item_type": JourneyItemType.ACTION,
                                "content_instructions": item.content_instructions.model_copy(),
                            }
                        )
                        parent.add_child(new_item, parent.children.index(item) + 1)
                        # item_index = parent.children.index(item)
                        # if len(parent.children) > (item_index + 1):
                        #     parent.children[item_index + 1].after_id = new_item.id

                        # parent.children.insert(item_index + 1, new_item)
                        # item.move(new_item, journey)
                        journey.save_to_db()
                        open_item(new_item, journey)
                        # print("add after")
                if st.button(
                    ActionSymbol.up_down.value,
                    key=f"move_button_{item_id}",
                    type="secondary",
                ):
                    st.session_state["journey_item_move"] = item
                    st.rerun()
                # if st.button(
                #     ActionSymbol.edit.value,
                #     key=f"edit_button_{item_id}",
                #     type="secondary",
                #     use_container_width=True,
                #     disabled=item_state.get("edit", False),
                # ):
                #     # item_state["edit"] = True
                #     # st.rerun(scope="fragment")
                #     open_item(item, journey)
                with st.popover(ActionSymbol.remove.value):
                    if st.button(
                        f"Are you sure you want to remove:\n\n{item.title}?",
                        key=f"delete_button_{item_id}",
                        use_container_width=True,
                    ):
                        parent = all_children[item.parent_id]
                        parent.remove_child(item)
                        journey.save_to_db()
                        st.rerun()
                        print("remove")


@st.fragment
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
    item_state: dict = st.session_state["journey_item_state"].get(item_id, {})
    st.session_state["journey_item_state"][item_id] = item_state
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

        if item.id in ancestry:
            select_column.button(
                ActionSymbol.selected.value,
                key=f"move_item_{item_id}_parent",
                disabled=True,
            )
        else:
            if select_column.button(
                ActionSymbol.unselected.value, key=f"move_item_{item_id}_in"
            ):
                move_item.move(item, journey)
                st.session_state["journey_item_move"] = None
                st.rerun()
    else:
        main_container = st.empty()

    if (
        item.item_type.value != container_level[-1]
        and item.item_type.value in container_level
    ):

        button_styles = [
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
        ]

        item_state["open"] = item_state.get("open", False)
        if JourneyItemType.JOURNEY == item.item_type:
            # container = main_container
            container = stylable_container(
                key=f"item_container_{item_id}", css_styles=button_styles
            )
        else:
            theme = get_theme()
            if theme is not None:
                if theme["base"] == "light":
                    color = theme["borderColorLight"]
                else:
                    color = theme["borderColor"]
            else:
                color = "gray"

            with main_container:
                container = stylable_container(
                    key=f"item_container_{item_id}",
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
                        }}"""
                    ]
                    + button_styles,
                )
        if JourneyItemType.JOURNEY == item.item_type:
            col1, col3 = container.columns([0.9, 0.1], vertical_alignment="center")
            col1.subheader(item.title)
            with col3:
                if st.button(
                    ActionSymbol.edit.value,
                    key="edit_parent_button_" + item_id,
                    type="primary",
                ):
                    open_item(item, journey)
        elif (
            JourneyItemType.SECTION == item.item_type
            or JourneyItemType.MODULE == item.item_type
        ):
            col1, col3 = container.columns([0.9, 0.1], vertical_alignment="center")
            with col1:
                if st.button(
                    (
                        ActionSymbol.open.value
                        if item_state["open"]
                        else ActionSymbol.closed.value
                    )
                    + "\n\n"
                    + item.get_index(journey)
                    + "\n\n"
                    + item.title,
                    key=f"open_button_{item_id}",
                    type="primary",
                ):
                    item_state["open"] = not item_state["open"]
                    st.rerun(scope="fragment")
            with col3:
                if st.button(
                    ActionSymbol.edit.value,
                    key="edit_parent_button_" + item_id,
                    type="primary",
                ):
                    open_item(item, journey)
        else:
            st.markdown(
                "####"
                + ("#" * container_level.index(item.item_type.value))
                + " "
                + item.title
            )

        if (
            JourneyItemType.SECTION == item.item_type
            or JourneyItemType.MODULE == item.item_type
        ):
            write_section_module(item, journey, item_id)

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
    elif JourneyItemType.ACTION == item.item_type:
        write_action(item, journey, item_id)


@st.fragment
def edit_journey(journey: JourneyItem):
    theme = get_theme()
    # with st.container(border=True):
    border_color = "gray"
    if theme != None and theme["base"] == "dark":
        border_color = theme["borderColor"]
    elif theme != None:
        border_color = theme["borderColorLight"]

    with stylable_container(
        key="journey_creation_section_container",
        css_styles=[
            f"""
            {{
                border: 1px solid {border_color};
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


def journey_edit():
    tab1, tab2 = st.tabs(["Modify journey", "Assign to individual(s)"])
    with tab1:
        journey_id = st.query_params.get("journey") or st.session_state.get(
            "journey_edit_id"
        )

        if journey_id is not None:
            journey = JourneyItem.get(journey_id=journey_id)

            # print("Journey ID", journey_id)
            # print(f"Journey keys: {get_available_journeys()}")
            edit_journey(journey)

        else:
            st.error("No data to show. Redirecting to journey management.")
            time.sleep(2)
            st.session_state["journey_edit"] = None
            st.switch_page("pages/journey_simple_manage.py")

        # st.subheader("Assign Journey")

    with tab2.container(border=True):
        # st.write(
        #     JourneyItem.from_json(st.session_state["journey_creation_data"])
        #     if "journey_creation_data" in st.session_state
        #     else "No data"
        # )
        # assign_to = st.multiselect(
        #     "Assign Journey to",
        #     [
        #         "Individual(s)",
        #         "Team(s)",
        #         "Department(s)",
        #         "Location(s)",
        #         "Subsidiary",
        #     ],
        #     ["Individual(s)"],
        # )
        # st.divider()

        st.subheader("Connected users")

        st.divider()
        st.subheader("Assign to Individual(s)")

        org_id = get_user_org(st.session_state["username"]).organization_id
        print("Organization id", org_id)
        users = get_all_users(org_id, reset=True)
        user_titles = [
                f"{user.email if user.name is None else ''}{f'{user.name} ({user.email})' if user.name is not None else ''}" for user in users
            ]
        assign_to = st.multiselect(
            "Assign Journey to",
            user_titles,
            None
        )
        st.button("Assign", type="primary")
        st.divider()

        st.subheader("Create new account and assign to journey")
        emp_id = st.text_input(
            "Employee email",
            placeholder="john@acmeorg.com",
            key="add_new_user",
        )
        st.button("Add Employee", type="primary")
        st.markdown(" ")

    col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
    # with col1:
    #     # st.page_link("pages/create_journey_2.py", label="Back")
    #     if st.button("Back", use_container_width=True):
    #         st.session_state["journey_creation_state"] = "init"
    #         st.rerun()
    with col3:
        if st.button("Done", use_container_width=True, type="primary"):
            # print("donothing")
            st.session_state["journey_edit"] = None
            st.switch_page("pages/journey_simple_manage.py")


async def main():
    st.title("Modify Journey")

    if init_sidebar(UserLevel.org_admin) != AuthStatus.LOGGED_IN:
        st.switch_page("Admin_Home.py")
        return

    st.markdown(" ")
    journey_edit()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
