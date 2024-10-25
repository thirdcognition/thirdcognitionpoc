from enum import Enum
import random
from pydantic import BaseModel
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.grid import grid
from admin.global_styles import get_theme
from admin.sidebar import get_image
from lib.helpers.journey import ActionSymbol
from lib.helpers.shared import pretty_print
from lib.models.journey import ActionItemType, JourneyDataTable, JourneyItem, JourneyItemType
from lib.models.journey_progress import (
    JourneyItemProgress,
    JourneyItemProgressState,
    JourneyProgressDataTable,
)
from lib.models.user import add_user, get_all_users, get_db_user, get_user_org


class ChildPosition(Enum):
    FIRST = 1
    MIDDLE = 2
    LAST = 3


def open_item(item: JourneyItem, journey: JourneyItem, show_children=False):
    st.session_state["journey_edit_id"] = (
        journey.id if isinstance(journey, JourneyItem) else journey
    )
    st.session_state["journey_edit_item_id"] = (
        item.id if isinstance(item, JourneyItem) else item
    )
    st.session_state["journey_edit_item_show_children"] = show_children
    st.switch_page("pages/journey_edit_item.py")

    # edit_item(item, journey)


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
    with container:
        image_grid = grid(10, 10, 10, 10, 10, vertical_align="center")
        for i in range(1, 51):
            with image_grid.container():
                logo_id = "logo_" + str(i)
                st.image(get_image(logo_id, path="icon_files"))
                if st.button(
                    (
                        ActionSymbol.selected.value
                        if item.icon == logo_id
                        else ActionSymbol.unselected.value
                    ),
                    key="image_" + id_str + "_" + logo_id,
                    disabled=item.icon == logo_id,
                    use_container_width=True,
                ):
                    print("Change icon", item.icon, logo_id)
                    if item.icon != logo_id:
                        item.icon = logo_id
                        item.save_to_db()
                        st.rerun()
                    else:
                        item.icon = None





def build_journey_cards(
    items: list,
    journey: JourneyItem = None,
    journey_progress: JourneyItemProgress = None,
    row_len=3,
    key_start="",
):
    theme = get_theme()
    css_styles = [
        """
        {
            margin-top: 1rem;
        }
        """,
        f"""
        .stColumn > div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] > div[data-testid=stVerticalBlockBorderWrapper] {{
            border: 2px solid {"rgba(256, 256, 256, 0.2)" if theme["base"] == "dark" else "rgba(0,0,0, 0.2)"};
            border-radius: 3rem;
            padding: 2rem;
            background-color: {"#2e2e2e" if theme["base"] == "dark" else "#eee"};
        }}
        """,
        """
        .stColumn > div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] > div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] {
            min-height: 26rem;
            justify-content: space-between;
        }
        """,
        """
        .stColumn > div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] > div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] > div[data-testid=stHorizontalBlock]:last-child:has(.stColumn) {
            max-height: 4rem;
        }
        """,
        """
        .card-title {
            display: flex;
        }
        """,
        """
        .index {
           font-size: 1.3rem;
           font-weight: 600;
           margin-bottom: 1.2rem;
           margin-right: 0.75rem;
           margin-top: 0;
           padding: 0;
           line-height: 1.5rem;
        }
        """,
        """
        .title {
            font-size:1.1rem;
            font-weight: 600;
            margin-bottom: 20px;
            margin-top: 0px;
            height: 40px;
            line-height: 22px;
            padding: 0px;
        }
        """,
        """
        .title-center {
            font-size:1.3rem;
            font-weight: 600;
            margin-bottom: 20px;
            margin-top: 0.4rem;
            height: 40px;
            line-height: 22px;
            width: 100%;
            text-align: center;
            padding: 0px;
        }
        """,
        """
        .description {
            border: 0;
            padding-left: 0.6rem;
            padding-right: 0.6rem;
            margin-bottom: 1rem;
            font-size: 0.8rem;
            font-weight: 100;
            border-radius: 0;
            text-align: left;
            height: 3.5rem;
        }
        """,
        """
        div[data-testid=stFullScreenFrame] {
            justify-content: center;
            display:flex;
        }
        """,
        """
        .stImage {
            max-width: 70%;
        }
        """,
        """
        button {
            background-color: green;
            border-color: green;
            color: white;
            border-radius: 1.3rem;
        }""",
        """
        button:hover {
            background-color: red;
            border-color: red;
            color: white;
            border-radius: 1.3rem;
        }
        """,
        f"""
        button[kind=secondary] {{
            background-color: transparent;
            border-color: transparent;
            color: {"white" if theme["base"] == "dark" else "black"};
            border-radius: 1.3rem;
        }}""",
        """
        .stPopover button[kind=secondary] * {
            font-size: 2rem;
        }""",
    ]

    key = (
        key_start
        + f"container_with_border_{hash(",".join([item.__getattribute__("id") if isinstance(item, BaseModel) else item["title"] for item in items]))}"
    )
    styled_container = stylable_container(key=key, css_styles=css_styles)

    with styled_container.container(border=False):
        card_rows = [row_len for _ in range(len(items) // row_len + 1)]
        card_grid = grid(*card_rows, vertical_align="center")
        for i, item in enumerate(items):
            with card_grid.container(border=False):
                if isinstance(item, BaseModel):
                    if JourneyItemType.JOURNEY != item.item_type:
                        render_journey_item(
                            item, journey, journey_progress, key=key_start
                        )
                    else:
                        render_journey_card(item)
                elif isinstance(item, dict):
                    render_generic_card(item)


def render_journey_item(
    item: JourneyItem,
    journey: JourneyItem,
    journey_progress: JourneyItemProgress,
    key="",
):
    st.markdown(
        f"""
            <div class="card-title">
            <span class="index">{item.get_index(journey)}</span>
            <span class="title">{item.title}</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.image(get_image(item.icon, path="icon_files"))

    st.markdown(
        f"""
    <div class="description">
        {item.description:.90}{"..." if len(item.description) > 90 else ""}
    </div>
    """,
        unsafe_allow_html=True,
    )
    progress_item = None
    if journey_progress is not None:
        progress_item = journey_progress.get_by_journey_item(item)

    if (
        progress_item is None
        or JourneyItemProgressState.NOT_STARTED == progress_item.get_state()
    ):
        if st.button(
            "Start" if item.item_type == JourneyItemType.MODULE else "Open",
            key=key
            + ("start" if progress_item else "open")
            + "_journey_"
            + journey.id
            + "_"
            + item.id
            + ("_" + progress_item.id if progress_item else ""),
            type="primary",
            use_container_width=True,
        ):
            # print(
            #     f"{progress_item.item_type=} {progress_item.id=} {journey_progress.item_type=} {journey_progress.id=}"
            # )
            progress_item.start()
            st.session_state["journey_view_id"] = journey.id
            st.session_state["journey_view_item_id"] = item.id
            st.session_state["journey_item_show_children"] = True
            if journey_progress is not None:
                st.session_state["journey_view_progress_id"] = journey_progress.id
            st.switch_page("pages/journey_view_item.py")
    else:
        if st.button(
            "Open" if progress_item.get_state() != JourneyItemProgressState.COMPLETED else "Completed",
            key=key
            + "continue_journey_"
            + journey.id
            + "_"
            + item.id
            + "_"
            + progress_item.id,
            type="primary" if progress_item.get_state() != JourneyItemProgressState.COMPLETED else "secondary",
            use_container_width=True,
        ):
            st.session_state["journey_view_id"] = journey.id
            st.session_state["journey_view_item_id"] = item.id
            st.session_state["journey_item_show_children"] = True
            if journey_progress is not None:
                st.session_state["journey_view_progress_id"] = journey_progress.id
            st.switch_page("pages/journey_view_item.py")


def render_journey_card(item: JourneyItem):
    st.markdown(
        f"""
            <div class="card-title">
            <span class="title-center">{item.title}</span>
            </div>
        """,
        unsafe_allow_html=True,
    )
    if item.icon:
        st.image(get_image(item.icon, path="icon_files"))

    if st.button(
        "Open journey", key=f"journey_{item.id}_open", type="primary", use_container_width=True
    ):
        st.session_state["journey_edit_id"] = item.id
        st.switch_page("pages/journey_edit.py")
    but1, but2 = st.columns([1, 1])
    with but1.popover(
        ActionSymbol.duplicate.value,
        use_container_width=True,
    ):
        st.write(f"Do you want to duplicate {item.title}?")
        if st.button(
            "Yes",
            key=f"journey_{item.id}_duplicate",
            type="primary",
            use_container_width=True,
        ):
            JourneyDataTable.duplicate_with_children(
                item.id, overwrite={"title": item.title + " (copy)"}
            )
            st.rerun()
            # print("no-op")
    with but2.popover(ActionSymbol.delete.value, use_container_width=True):
        st.write(f"Are you sure you want to remove {item.title}")
        if st.button(
            "Remove",
            key=f"journey_{item.id}_delete",
            use_container_width=True,
        ):
            item.remove()
            st.rerun()
        st.write("*Warning: This is permanent and cannot be undone.*")


def render_generic_card(item: dict):
    title = item.get("title", "No Title")
    icon = item.get("icon", None)
    description = item.get("description", "")
    page = item.get("page")

    st.markdown(
        f"""
            <div class="card-title">
            <span class="title-center">{title}</span>
            </div>
        """,
        unsafe_allow_html=True,
    )
    if icon:
        st.image(get_image(icon, path="icon_files"))

    if description:
        st.markdown(
            f"""
        <div class="description">
            {description:.90}{"..." if len(description) > 90 else ""}
        </div>
        """,
            unsafe_allow_html=True,
        )

    if page is not None and st.button(
        "Create", key=f"generic_{title}_open", type="primary", use_container_width=True
    ):
        st.switch_page(f"pages/{page}.py")
        # st.write(f"Opening {title}")

def run_complete(journey_item_progress: JourneyItemProgress, feedback):
    journey_item_progress.complete(feedback=feedback if feedback else None)
    # parent_progress = journey_item_progress.get(
    #     progress_id=journey_item_progress.parent_id
    # )
    st.rerun()
    # if parent_progress.get_progress() == 1.0:
    #     st.switch_page("pages/my_journeys.py")


@st.dialog("Confirm completion")
def complete_journey_item(
    journey_item: JourneyItem,
    journey: JourneyItem,
    journey_item_progress: JourneyItemProgress,
) -> bool:
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
        run_complete(journey_item_progress, feedback=feedback if feedback else None)


def write_action_ui(journey_item: JourneyItem, journey:JourneyItem=None, journey_item_progress:JourneyItemProgress=None, container=None, task_container=None):
    if container is None:
        container = st.container()
    with container:
        st.write("##### " +journey_item.get_index(journey) + " " + journey_item.title)
        # Only display the description if it exists
        if journey_item.description:
            st.write(" - " + journey_item.description)

        media_types = [ActionItemType.VIDEO, ActionItemType.IMAGE, ActionItemType.AUDIO]

        if not journey_item_progress and journey_item.action.action_type in media_types:
            action_container = st.expander(journey_item.action.title)
        else:
            if not journey_item_progress:
                action_container = st.empty()
            else:
                _, action_container, _ = st.columns([0.025, 0.8, 0.025])


        # Render media items within the contentaction_container
        with action_container:
            if journey_item.action.action_type in media_types:
                if journey_item.action.action_type == ActionItemType.VIDEO and journey_item.action.extras:
                    st.video(journey_item.action.extras.get("url"))

                elif journey_item.action.action_type == ActionItemType.IMAGE and journey_item.action.extras:
                    st.image(journey_item.action.extras.get("url"))

                elif journey_item.action.action_type == ActionItemType.AUDIO and journey_item.action.extras:
                    st.audio(journey_item.action.extras.get("url"))
            elif not journey_item_progress:
                text = "\n\n##### Actions"
                if journey_item.action.action_type == ActionItemType.LINK and journey_item.action.extras and journey_item.action.extras.get("url"):
                    link_title = journey_item.action.extras.get("url_title", "Link")
                    link_url = journey_item.action.extras.get("url")
                    if link_url:
                        text += f"\n\n * {journey_item.action.title}\n\n\t[{link_title}]({link_url})"

                elif journey_item.action.action_type == ActionItemType.CALENDAR and journey_item.action.extras and journey_item.action.extras.get("url"):
                    link_url = journey_item.action.extras.get("url")
                    if link_url:
                        text += f"\n\n - {journey_item.action.title}: [Select a slot]({link_url})"
                else:
                    text += f"\n\n - {journey_item.action.title}"
                            # st.write(f"Calendar: {calendar_info}")
                st.write(text)

        # elif journey_item.action.action_type == ActionItemType.REFERENCE:
        #     # Process and display references if any
        #     if journey_item.action.references:
        #         for ref in journey_item.action.references:
        #             st.write(f"Reference: {ref.title} - {ref.description}")

    if task_container is None:
        task_container = st.container()


    if journey_item_progress:
        _, act_col1, act_col2 = task_container.columns(
            [0.15, 0.7, 0.15], vertical_alignment="top"
        )

        with act_col1:
            if journey_item.action.action_type == ActionItemType.LINK and journey_item.action.extras and journey_item.action.extras.get("url"):
                link_title = journey_item.action.extras.get("url_title", "Link")
                link_url = journey_item.action.extras.get("url")
                if link_url:
                    st.write(f" * {journey_item.action.title}\n\n\t[{link_title}]({link_url})")

            elif journey_item.action.action_type == ActionItemType.CALENDAR and journey_item.action.extras and journey_item.action.extras.get("url"):
                link_url = journey_item.action.extras.get("url")
                if link_url:
                    st.write(f" - {journey_item.action.title}: [Select a slot]({link_url})")
            else:
                st.write(f" - {journey_item.action.title}")
                    # st.write(f"Calendar: {calendar_info}")

        with act_col2:
            complete = st.checkbox(
                "Done",
                key="action_" + journey_item.id,
                value=st.session_state.get(
                    f"journey_item_complete_{journey_item.id}",
                    JourneyItemProgressState.COMPLETED == journey_item_progress.get_state(),
                ),
            )
            if (
                complete
                and JourneyItemProgressState.COMPLETED != journey_item_progress.get_state()
            ):
                complete_journey_item(journey_item, journey, journey_item_progress)

