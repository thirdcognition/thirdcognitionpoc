import time
from typing import List

import streamlit as st
import streamlit_shadcn_ui as ui
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.grid import grid

import os
import sys

from admin.sidebar import get_image, get_theme, init_sidebar
from lib.models.journey import JourneyItem, JourneyItemType

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))

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

container_level = ["journey", "section", "module", "action"]

def get_stylable_container_selector(id):
    return f'div[data-testid="stVerticalBlock"]:has(> div.element-container > div.stMarkdown > div[data-testid="stMarkdownContainer"] > p > span.{id})'

@st.dialog(f"Change contents of..", width="large")
def open_action(item: JourneyItem, journey: JourneyItem):
    # st.write(f"Let's get started with action {item.title}")
    journey_relations = journey.get_relations()

    if item.id:
        parent: JourneyItem = journey_relations[item.id]

    ancestry:list[JourneyItem] = [parent]
    while parent:
        # add parent to the beginning of the list and check its parent
        if parent.id and parent.id in journey_relations.keys():
            parent = journey_relations[parent.id]
        elif parent.id is not None:
            parent = None
        if parent and parent not in ancestry:
            ancestry.insert(0, parent)

    ancestry.append(item)

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
        with st.expander(f"{ancestor.item_type.value.capitalize()} Details:", expanded=ancestor.item_type==JourneyItemType.ACTION):
            st.write(ancestor.title)
            # Create input fields for each attribute of the JourneyItem object
            # ancestor.id = st.text_input('Unique Identifier', value=ancestor.id)
            # ancestor.template_id = st.text_input('Template ID', value=ancestor.template_id)

            if ancestor.item_type != JourneyItemType.JOURNEY and ancestor.item_type != JourneyItemType.SECTION:
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
                    key = "parent_"+id_str
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
                if ancestor.id in journey_relations.keys() and len(journey_relations[ancestor.id].children) > 1:
                    ancestor.after_id = st.selectbox(
                        "After",
                        options=items,
                        index=(
                            options.index(ancestor.after_id)
                            if ancestor.after_id in options
                            else None
                        ),
                        key = "after_"+id_str
                    )

            ancestor.title = st.text_input("Title", value=ancestor.title)
            if ancestor.item_type == JourneyItemType.ACTION:
                # container = stylable_container(
                #     key=f"item_container_{id_str}",
                #     css_styles=f"""
                # div {{
                #     margin: 0;
                #     padding: 0;
                #     justify-content: center;
                # }}

                # {get_stylable_container_selector(f"item_container_{id_str}")} label {{
                #     margin: 0;
                #     padding: 0;
                #     justify-content: center;
                # }}
                # """,
                # ).container()
                # logo_list = {}
                # with container:
                #     image_grid = grid(10, 10, 10, 10, 10, vertical_align="center")
                #     for i in range(1, 51):
                #         with image_grid.container():
                #             logo_id = "logo_"+str(i)
                #             st.image(get_image(logo_id, path="icon_files"))
                #             if st.button("use", key="image_"+id_str+"_"+logo_id):
                #                 print("Change icon", logo_id)
                #                 if ancestor.icon != logo_id:
                #                     ancestor.icon = logo_id
                #                     st.rerun()
                #                 else:
                #                     ancestor.icon = None

                #             # st.checkbox(label=str(i), label_visibility="collapsed", value=(logo_id) == ancestor.icon, key="image_"+id_str+"_"+logo_id, on_change=change_icon)

                # # Selector with 1-50 images that when selected updates ancestor.icon to the id
                # # Image id's are logo_1 to logo_50 and the image url can be fetched via get_image(image_id)
                # # The existing icon should be automatically selected.
                # # for logo_id, selected in logo_list.items():
                # #     print(logo_id, selected)
                # print("logo", ancestor.icon)

                # # ancestor.icon = st.text_input("Icon", value=ancestor.icon, key = "icon_"+id_str)
                # # ancestor.intro = st.text_area("Introduction", value=ancestor.intro, key = "intro_"+id_str)
                # # ancestor.summary = st.text_area("Summary", value=ancestor.summary, key = "summary_"+id_str)

                ancestor.description = st.text_area(
                    "Description", value=ancestor.description, key = "description_"+id_str, height = 300
                )
                # ancestor.test = st.text_area("Test", value=ancestor.test, key = "test_"+id_str)
                ancestor.action = st.text_input("Action", value=ancestor.action, key = "action_"+id_str)

            ancestor.end_of_day = st.number_input(
                "Done by end of day #", value=ancestor.end_of_day, key = "eod_"+id_str
            )
            # ancestor.item_type = st.selectbox(
            #     "Item Type", options=[e.value for e in JourneyItemType]
            # )

    if st.button("Save", use_container_width=True, type="primary"):
        # st.session_state.vote = {"item": item, "reason": feedback}
        st.rerun()



level_step = 1


@st.fragment
def write_item(item: JourneyItem, id_chain="", journey: JourneyItem = None):
    item_id = item.id if id_chain == "" else f"{id_chain}_{item.id}"
    st.session_state["journey_item_state"] = st.session_state.get(
        "journey_item_state", {}
    )
    level_multiplier = max(
        (container_level.index(item.item_type.value) - 1) * level_step, 0.01
    )
    if (
        item.item_type.value != container_level[-1]
        and item.item_type.value in container_level
    ):
        item_state: dict = st.session_state["journey_item_state"].get(item_id, {})
        item_state["open"] = item_state.get("open", False)
        st.session_state["journey_item_state"][item_id] = item_state
        if item.item_type == item.item_type == JourneyItemType.JOURNEY:
            container = st
        else:
            theme = get_theme()
            if theme["base"] == "light":
                color = "rgba(0,0,0,0.33)"
            else:
                color = "rgba(255,255,255,0.33)"

            container = stylable_container(
                key=f"item_container_{item_id}",
                css_styles=f"""
            {{
                border: 1px solid {color};
                margin-top: -1px;
                padding: 0;
                padding-top: 1rem;
                padding-left: {level_multiplier}rem;
            }}
            """,
            )
        # con_col1, con_col2 = container.columns(
        #     [
        #         level_multiplier,
        #         1 - level_multiplier,
        #     ], vertical_alignment="center"
        # )
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

            # container = st.container(border=False)
            # container = st.expander(label=item.title)
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
                for child in item.children:
                    write_item(child, journey=journey)
        elif item.children:
            for child in item.children:
                write_item(child, journey=journey)
    elif item.item_type == JourneyItemType.ACTION:
        # container = st.container(border=False)
        # col1, col2 = container.columns(
        #     [
        #         level_multiplier - level_step,
        #         0.8 - level_multiplier + level_step,
        #     ], vertical_alignment= "center"
        # )
        # with col2:
        # with st.container(border=True):
        _, subcol1, subcol2 = st.columns([0.025, 0.8, 0.15])
        with subcol1:
            with stylable_container(
                key=f"content_container_{item_id}",
                css_styles="""
                {
                    margin: 1rem 0;
                    gap: 0.5rem;
                }
                """,
            ):
                header_row1, _, header_row2 = st.columns(
                    [0.15, 0.05, 0.8], vertical_alignment="center"
                )
                header_row1.image(
                    get_image(item.icon, "icon_files"), use_column_width=True
                )
                header_row2.markdown("##### " + item.title)
                st.markdown(item.description)
        with subcol2:
            with stylable_container(
                key=f"button_container_{item_id}",
                css_styles="""
                    {
                        margin-top: 1.5rem;
                        gap: 0.5rem;
                    }
                """,
            ):
                with st.popover(":eight_spoked_asterisk:", use_container_width=True):
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
                with st.popover(":arrow_up_down:", use_container_width=True):
                    if st.button(
                        "Move to first",
                        key=f"move_first_button_{item_id}",
                        type="secondary",
                        use_container_width=True,
                    ):
                        print("move to first")
                    value = st.selectbox(
                        "Move after",
                        ["Last"],
                        index=None,
                        label_visibility="hidden",
                        placeholder="Move after",
                        key=f"move_after_button_{item_id}",
                    )
                    if value is not None:
                        print(f"Move to {value}")
                        # st.rerun(scope="fragment")
                if st.button(
                    "Edit",
                    key=f"edit_button_{item_id}",
                    type="secondary",
                    use_container_width=True,
                ):
                    open_action(item, journey)
                with st.popover(":x:", use_container_width=True):
                    if st.button(
                        f"Are you sure you want to remove:\n\n{item.title}?",
                        key=f"delete_button_{item_id}",
                        use_container_width=True,
                    ):
                        print("remove")

            # if st.button("Remove", key=f"remove_button_{item.id}",  type="secondary", use_container_width=True):
            #     open_action(item)


async def journey_creation():
    if (
        "journey_creation_state" not in st.session_state
        or st.session_state["journey_creation_state"] == "init"
    ):
        st.subheader("Journey Details")
        available_options = [
            "New Hire Onboarding",
            "Project Onboarding",
            "Skill Development",
            "Customer Onboarding",
            "Other",
        ]
        with st.container(border=True):
            target = st.selectbox(
                "Journey Purpose",
                available_options,
                index=0,
            )

            # location = st.selectbox(
            #     "Location",
            #     ("Helsinki", "Stockholm", "Berlin", "Tokyo", "Other"),
            #     index=None,
            #     placeholder="Choose location",
            # )
        st.markdown(" ")

        if available_options.index(target) == 0:
            st.subheader("Role Details")
            container = st.container(border=True)
            with container:
                role_title = st.text_input(
                    "Job Title", placeholder="Account Manager", key=1
                )
                role_description = st.text_area(
                    "Role Description",
                    placeholder="Describe role in detail or paste a job description...",
                    key=2,
                    height=200,
                )

                # col1, col2 = st.columns([0.4,0.6])
                # with col1:
                #     job_description = st.text_input("Link to job description", placeholder="www.linkedin.com", key=3)
                # with col2:
                #     uploaded_files = st.file_uploader("Upload job descritpion (optional)", accept_multiple_files=True)
                st.markdown(" ")
            col1, col2 = st.columns([0.85, 0.15])
            with col2:
                button_placeholder = st.empty()
                # st.page_link("pages/create_journey_2.py", label="Continue")
                if button_placeholder.button("Continue", use_container_width=True):
                    st.session_state["journey_creation_state"] = "assignto"
                    # button_placeholder.empty()
                    with button_placeholder.container(border=True, height=40):
                        st.markdown(
                            """<style>
                                div[data-testid=column]:last-child div[data-testid=stVerticalBlockBorderWrapper] {
                                overflow: hidden;
                                }
                                .element-container > .stSpinner {
                                margin-top:-22px;
                                margin-left:20px
                                }
                                </style>
                            """,
                            unsafe_allow_html=True,
                        )
                        with st.spinner(""):
                            result = await get_chain(
                                "journey_template_selector"
                            ).ainvoke(
                                {
                                    "job_description": f"Title:\n{role_title.strip()}\n\nDescription:\n{role_description.strip()}"
                                }
                            )

                            print("Result", result)
                            matching_ids = match_title_to_cat_and_id(result)
                            print("Matching IDs", matching_ids)
                            template = load_journey_template(matching_ids[1])
                            pretty_print(template, force=True)
                            st.session_state["journey_creation_data"] = template

                            st.rerun()
        else:
            st.subheader("Work in progress")
            with st.container(border=True):
                st.markdown("Coming soon...")

    # elif st.session_state["journey_creation_state"] == "common_data":
    #     st.subheader("Include Common Data")
    #     container = st.container(border=True)
    #     with container:
    #         st.markdown("Mandatory")
    #         culture = st.checkbox("Introduction to Culture & Values", value=True, disabled=True, key=1)
    #         conduct = st.checkbox("Introduction to Code of Conduct", value=True, disabled=True, key=2)
    #         st.divider()
    #         st.markdown("Optional")
    #         quality = st.checkbox("Introduction to Quality Policy", value=True, key=3)
    #         health = st.checkbox("Introduction to Healt & Safety", value=True, key=4)
    #         human = st.checkbox("Introduction to Human Rights Policy", value=True, key=5)
    #         environment = st.checkbox("Introduction to Environmental Policy", value=True, key=6)
    #         whistle = st.checkbox("Introduction to Whistleblowing", value=True, key=7)

    #     col1, col2, col3 = st.columns([0.15, 0.7,0.15])
    #     with col1:
    #         # st.page_link("pages/create_journey_1.py", label="Back")
    #         if st.button("Back"):
    #             st.session_state["journey_creation_state"] = "init"
    #             st.rerun()
    #     with col3:
    #         # st.page_link("pages/create_journey_3.py", label="Continue")
    #         if st.button("Continue"):
    #             st.session_state["journey_creation_state"] = "assignto"
    #             st.rerun()
    elif st.session_state["journey_creation_state"] == "assignto":
        # div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] > div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] {
        #     gap: 0;
        # }
        # div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] > div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] div[data-testid=stHorizontalBlock] {
        #     gap: 0;
        #     column-gap: 0.2rem;
        # }
        # div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] > div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] > div[data-testid=stVerticalBlockBorderWrapper]:first-child {
        #     margin-top: 0;
        # }
        # div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] > div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] > div[data-testid=stVerticalBlockBorderWrapper] {
        #     padding: 0.1rem 0;
        #     margin: 0;
        #     border-radius: 0;
        # }
        st.markdown(
            """<style>
                div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock],
                div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] div[data-testid=stHorizontalBlock] {
                    gap: 0;
                }
                div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] > .element-container:first-child:not(:only-child) {
                    margin-bottom: 1rem;
                }
                div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] > div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] > div[data-testid=stVerticalBlockBorderWrapper] .stButton > button[kind=primary] {
                    border-radius: 0;
                    border: none;
                    background: transparent;
                    color: inherit;
                    white-space: nowrap;
                    overflow: hidden;
                    max-width: 100%;
                    justify-content: flex-start;
                }
                div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] > div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] > div[data-testid=stVerticalBlockBorderWrapper] .stButton > button[kind=primary] p {
                    display: inline-block;
                    margin-right: 0.2rem;
                    vertical-align: middle;
                }
                div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] > div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] > div[data-testid=stVerticalBlockBorderWrapper] .stButton > button[kind=primary] p:first-child {
                    font-size: 1rem;
                    width: 1.5rem;
                }
                div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] > div[data-testid=stVerticalBlockBorderWrapper] > div > div[data-testid=stVerticalBlock] > div[data-testid=stVerticalBlockBorderWrapper] .stButton > button[kind=primary] p:last-child {
                    font-size: 1.1rem;
                    max-width: 100%;
                    text-overflow: ellipsis;
                }
                </style>
            """,
            unsafe_allow_html=True,
        )
        if (
            "journey_creation_data" in st.session_state
            and st.session_state["journey_creation_data"] is not None
        ):
            with st.container(border=True):
                # st.write(st.session_state["journey_creation_data"])
                journey = JourneyItem.from_json(
                    st.session_state["journey_creation_data"]
                )
                write_item(journey, journey=journey)
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
    st.title("Create new Journey")

    if init_sidebar(UserLevel.org_admin) != AuthStatus.LOGGED_IN:
        return

    st.markdown(" ")
    await journey_creation()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
