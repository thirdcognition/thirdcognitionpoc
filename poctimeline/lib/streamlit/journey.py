from enum import Enum
import random
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.grid import grid
from admin.global_styles import get_theme
from admin.sidebar import get_image
from lib.helpers.journey import ActionSymbol
from lib.models.journey import JourneyDataTable, JourneyItem, JourneyItemType


class ChildPosition(Enum):
    FIRST = 1
    MIDDLE = 2
    LAST = 3


@st.cache_resource
def get_journey_item_cache() -> dict[str, JourneyItem]:
    return {}


def get_journey(
    journey_item: JourneyDataTable = None, journey_id: str = None, reset=False
) -> JourneyItem:
    if reset:
        del get_journey_item_cache()[journey_id]

    if journey_item is not None:
        if journey_item.id not in get_journey_item_cache().keys():
            get_journey_item_cache()[journey_item.id] = journey_item.to_journey_item()
        return get_journey_item_cache()[journey_item.id]
    elif journey_id is not None:
        if journey_id not in get_journey_item_cache().keys():
            journey: JourneyItem = JourneyItem.load_from_db(journey_id)
            get_journey_item_cache()[journey_id] = journey
        return get_journey_item_cache()[journey_id]

    raise ValueError("Either journey_item or journey_id must be defined")

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

def build_journey_cards(items: list[JourneyItem], journey: JourneyItem=None, row_len=3):
    theme = get_theme()
    styled_container = stylable_container(
        key="container_with_border",
        css_styles=[
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
            """
            button[kind=secondary] {
                background-color: transparent;
                border-color: transparent;
                color: white;
                border-radius: 1.3rem;
            }""",
            """
            button[kind=secondary] * {
                font-size: 2rem;
            }""",
        ],
    )

    with styled_container.container(border=False):
        card_rows = [row_len for _ in range(len(items)//row_len + 1)]
        card_grid = grid(*card_rows, vertical_align="center")
        for i, item in enumerate(items):
            with card_grid.container(border=False):
                if JourneyItemType.JOURNEY != item.item_type:
                    st.markdown(
                        f"""
                            <div class="card-title">
                            <span class="index">{item.get_index(journey)}</span>
                            <span class="title">{item.title}</span>
                            </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # _, image, __ = st.columns([0.1, 0.8, 0.1])
                    # with image:
                        # Invert image & Fix Contrast
                    st.image(get_image(item.icon, path="icon_files"))

                    st.markdown(
                        f"""
                    <div style="border: 0px solid #fff; padding-left: 10px; padding-right: 10px; font-size: 14px; font-weight: 100; border-radius: 0px; text-align: left; height: 90px;">
                        {journey.title}.
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                    st.button(
                        "Start" if item.item_type == JourneyItemType.MODULE else "Open",
                        key="open_journey_" + journey.id + "_" + item.id + "_" + str(random.randint(1, 100)),
                        type="primary",
                        use_container_width=True,
                    )
                else:
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

                    if st.button("Open", key=f"journey_{item.id}_open", type="primary", use_container_width=True):
                        st.session_state["journey_edit_id"] = item.id
                        st.switch_page("pages/journey_edit.py")
                    but1, but2 = st.columns([1, 1])
                    if but1.button(ActionSymbol.duplicate.value, key=f"journey_{item.id}_duplicate", type="secondary", use_container_width=True):
                        print("no-op")
                    with but2.popover(ActionSymbol.delete.value, use_container_width=True):
                        if st.button(f"Are you sure you want to remove {item.title}", key=f"journey_{item.id}_delete", use_container_width=True):
                            item.remove()
                            st.rerun()
                    # _, button, __ = st.columns([0.1, 0.8, 0.1])
                    # with button:

