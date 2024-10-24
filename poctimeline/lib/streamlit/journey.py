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
from lib.models.journey import JourneyDataTable, JourneyItem, JourneyItemType
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


@st.fragment
def assign_journey(journey_id: str):
    journey = JourneyItem.get(journey_id=journey_id)
    org_id = get_user_org(st.session_state["username"]).id
    users = get_all_users(org_id, reset=True)

    # Display current assignments and user progress

    # Load all progress items for the current journey
    progress_items = JourneyProgressDataTable.load_all_from_db(
        journey_item_id=journey_id
    )

    if len(progress_items) > 0:
        st.subheader("Connected Users")
        # Display current assignments and user progress
        count = 0
        for progress_item in progress_items:
            # pretty_print(progress_item.__dict__, force=True)
            # try:
            # Get the user for the current progress item
            user = get_db_user(id=progress_item.user_id)

            container = st.container(border=True)
            col1, col2, col3 = container.columns(
                [0.2, 0.7, 0.15], vertical_alignment="center"
            )

            if user:
                col1.markdown(f"**{user.name or user.email}**")
                count += 1

                journey_progress = JourneyItemProgress.from_db(progress_item)
                if JourneyItemProgressState.NOT_STARTED > journey_progress.get_state():
                    next_modules_progress = journey_progress.get_next(reset=True)
                    # pretty_print(next_modules_progress, force=True)
                    all_children = journey.all_children_by_id()
                    next_module = all_children[next_modules_progress[0].journey_item_id]
                    # journey_item = JourneyItem.get(journey_id = journey_progress.item_id)
                    # st.write(f"Journey: {journey_item.title}")
                    col2.progress(
                        journey_progress.get_progress(), f"Next module: {next_module}"
                    )
                else:
                    col2.write("Not yet started")

                with col3.popover("Manage"):
                    if st.button(
                        f"Remove",
                        key=f"remove_{journey_progress.id}",
                        use_container_width=True,
                    ):
                        journey_progress.removed = True
                        journey_progress.save_to_db()
                        st.rerun()

        if count == 0:
            st.write("No assigned users have yet signed up and started the journey")

        st.divider()

    st.subheader("Assign to Users")
    assign_users = st.multiselect(
        "Select users to assign journeys",
        users,
        format_func=lambda u: f"{u.name} ({u.email})" if u.name else u.email,
    )

    if assign_users and st.button(
        "Assign to User(s)", key="assing_journey_users_" + journey_id
    ):
        journey_id = st.session_state.get("journey_edit_id")
        for user in assign_users:
            user_id = user.id
            journey_item = JourneyItem.load_from_db(journey_id)
            JourneyProgressDataTable.from_journey_item(journey_item, user_id=user_id)

        journey_item.reset_cache()
        st.success("Journeys assigned successfully.")
        st.rerun(scope="fragment")

    st.divider()

    st.subheader("Assign to New User")
    new_user_email = st.text_input(
        "New User Email", placeholder="e.g., newuser@example.com"
    )

    if new_user_email and st.button(
        "Assign to New User", key="assing_journey_new_user_" + journey_id
    ):
        journey_id = st.session_state.get("journey_edit_id")
        new_user = add_user(email=new_user_email, org_id=org_id)
        journey_item = JourneyItem.load_from_db(journey_id)
        JourneyProgressDataTable.from_journey_item(journey_item, user_id=new_user.id)
        journey_item.reset_cache()

        st.success("New user created and journey assigned.")
        st.rerun(scope="fragment")


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

    st.write("Open journey")
    col1, col2, col3 = st.columns([1, 1, 1])
    if col1.button(
        "#1", key=f"journey_{item.id}_open", type="primary", use_container_width=True
    ):
        st.session_state["journey_edit_id"] = item.id
        st.switch_page("pages/journey_edit.py")
    if col2.button(
        "#2", key=f"journey_{item.id}_open2", type="primary", use_container_width=True
    ):
        st.session_state["journey_edit_id"] = item.id
        st.switch_page("pages/journey_edit_2.py")
    if col3.button(
        "#3", key=f"journey_{item.id}_open3", type="primary", use_container_width=True
    ):
        st.session_state["journey_simple_edit"] = True
        open_item(item, item, False)
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
