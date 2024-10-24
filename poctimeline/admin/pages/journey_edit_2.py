import time
from typing import List

import streamlit as st
from streamlit_extras.stylable_container import stylable_container

import os
import sys

from admin.global_styles import get_theme
from admin.sidebar import get_image, init_sidebar

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))

from lib.streamlit.journey import ChildPosition, open_item
from lib.models.journey import (
    JourneyItem,
    JourneyItemType,
)
from lib.helpers.journey import (
    ActionSymbol,
)
from lib.models.user import (
    AuthStatus,
    UserLevel,
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
                """
                > :first-child {
                    margin-top: 1rem;
                }
                """,
                """
                img {
                    margin-top: 1rem;
                    max-width: 10rem;
                }
                """,
            ],
        ):
            action_content_col = None
            if JourneyItemType.MODULE == item.item_type and item.icon:
                image_col, action_content_col, _ = st.columns([0.2, 0.9, 0.01])
                with image_col:
                    st.image(
                        get_image(item.icon, "icon_files"),
                        use_column_width=True,
                    )

            if item.children:
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
                    # with action_content_col:
                    if action_content_col is not None:
                        with action_content_col:
                            write_item(child, journey=journey, position=position)
                    else:
                        write_item(child, journey=journey, position=position)
            elif action_content_col is not None:
                with action_content_col:
                    st.write(" ")
                    st.write("No actions added yet")
                    if st.button("Add first action", key="add_first_child_" + item_id):
                        new_item = JourneyItem.create_new(
                            {
                                "parent_id": item.id,
                                "icon": item.icon,
                                "template_id": item.template_id,
                                "end_of_day": item.end_of_day,
                                "item_type": JourneyItemType.ACTION,
                                "content_instructions": item.content_instructions.model_copy(),
                            }
                        )
                        item.add_child(new_item, 0)
                        item.save_to_db()
                        journey.reset_cache()
                        open_item(
                            new_item,
                            journey,
                            JourneyItemType.MODULE == new_item.item_type,
                        )
        if JourneyItemType.MODULE > item.item_type and not item.children:
            col1, col2 = st.columns([0.01, 0.9])
            with col2:
                st.write(f"{item.title} has no modules yet, add first?")
                if st.button("Add first module", key="add_first_child_" + item_id):
                    new_item = JourneyItem.create_new(
                        {
                            "parent_id": item.id,
                            "icon": item.icon,
                            "template_id": item.template_id,
                            "end_of_day": item.end_of_day,
                            "item_type": JourneyItemType.MODULE,
                            "content_instructions": item.content_instructions.model_copy(),
                        }
                    )
                    item.add_child(new_item, 0)
                    item.save_to_db()
                    journey.reset_cache()
                    open_item(
                        new_item,
                        journey,
                        JourneyItemType.MODULE == new_item.item_type,
                    )
                st.write(" ")


def write_action(item: JourneyItem, journey: JourneyItem, item_id: str):

    subcol1, subcol2 = st.columns([0.8, 0.1])

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
            st.markdown("##### " + item.title)
            st.markdown(item.description)


stat_button_styles = [
    """
    :is(.stButton, .stPopover) :is(button[kind=primary], button[data-testid=stPopoverButton]) {
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
    :is(.stButton, .stPopover) :is(button[kind=primary], button[data-testid=stPopoverButton]) p {
        display: inline-block;
        margin-right: 0.2rem;
        vertical-align: middle;
    }""",
    """
    :is(.stButton, .stPopover) :is(button[kind=primary], button[data-testid=stPopoverButton]) p:first-child {
        font-size: 1rem;
        width: 1.5rem;
    }""",
    """
    .stColumn:not(:first-child) {
        max-width: 3.5rem;
    }
    """,
    """
    .stColumn:not(:first-child) .stPopover button[data-testid=stPopoverButton] > :last-child {
        display: none;
    }
    """,
    """
    .stColumn:not(:first-child) :is(.stButton, .stPopover) :is(button[kind=primary], button[data-testid=stPopoverButton]) p {
        font-size: 1.5rem;
        text-align: center;
    }
    """,
    """
    .stColumn:not(:first-child) :is(.stButton, .stPopover) :is(button[kind=primary], button[data-testid=stPopoverButton]) p {
        border: 1px solid rgba(255, 255, 255, 0);
        border-radius: 0.5rem;
        width: 2rem;
        display: inline-block;
        line-height: 2.2rem;
        padding-bottom: 0.2rem;
    }
    """,
]


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

    theme = get_theme()
    if theme is not None:
        if theme["base"] == "light":
            color = theme["borderColorLight"]
        else:
            color = theme["borderColor"]
    else:
        color = "gray"

    if (
        item.item_type.value != container_level[-1]
        and item.item_type.value in container_level
    ):
        button_styles = stat_button_styles + [
            f"""
            .stColumn:not(:first-child) :is(.stButton, .stPopover) :is(button[kind=primary], button[data-testid=stPopoverButton]):hover p {{
                border: 1px solid {color};
            }}
            """,
        ]

        item_state["open"] = item_state.get("open", False)

        container = stylable_container(
            key=f"item_container_{item_id}",
            css_styles=button_styles
            + (
                [
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
                > div[data-testid="stHorizontalBlock"] {
                    gap: 0;
                }
                """,
                ]
                if JourneyItemType.JOURNEY != item.item_type
                else []
            ),
        )
        col1, col3 = container.columns([0.9, 0.1], vertical_alignment="center")
        if JourneyItemType.JOURNEY == item.item_type:
            col1.subheader(item.title)
            if col3.button(
                ActionSymbol.edit.value,
                key="edit_parent_button_" + item_id,
                type="primary",
            ):
                open_item(item, journey, JourneyItemType.MODULE == item.item_type)
        elif (
            JourneyItemType.SECTION == item.item_type
            or JourneyItemType.MODULE == item.item_type
        ):
            if col1.button(
                (
                    ActionSymbol.open.value
                    if item_state["open"]
                    else ActionSymbol.closed.value
                )
                + "\n\n "
                + item.get_index(journey).replace(".", r"\.")
                + "\n\n"
                + item.title,
                key=f"open_button_{item_id}",
                type="primary",
            ):
                item_state["open"] = not item_state["open"]
                st.rerun(scope="fragment")  #

            if col3.button(
                ActionSymbol.edit.value,
                key="edit_parent_button_" + item_id,
                type="primary",
            ):
                open_item(item, journey, JourneyItemType.MODULE == item.item_type)
        elif JourneyItemType.ACTION == item.item_type:
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

        write_item(journey, journey=journey)


def journey_edit():
    journey_id = st.query_params.get("journey") or st.session_state.get(
        "journey_edit_id"
    )
    journey: JourneyItem = None
    if journey_id is not None:
        journey = JourneyItem.get(journey_id=journey_id)
    if journey:
        edit_journey(journey)
    else:
        st.error("No data to show. Redirecting to journey management.")
        time.sleep(2)
        st.session_state["journey_edit"] = None
        st.switch_page("pages/journey_simple_manage.py")

    col1, col2, col3 = st.columns([0.15, 0.7, 0.15])

    with col3:
        if st.button("Done", use_container_width=True, type="primary"):
            # print("donothing")
            st.session_state["journey_edit"] = None
            st.switch_page("pages/journey_simple_manage.py")


async def main():
    st.title("Modify Journey")

    if init_sidebar(UserLevel.org_admin) != AuthStatus.LOGGED_IN:
        st.switch_page("login.py")
        return

    st.markdown(" ")
    journey_edit()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
