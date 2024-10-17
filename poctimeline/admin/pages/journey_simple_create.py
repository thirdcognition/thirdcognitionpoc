import time
from typing import List

import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.grid import grid

import os
import sys

from admin.global_styles import get_theme
from admin.sidebar import init_sidebar

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))

from lib.models.journey import (
    JourneyItem,
    get_journey_item_cache,
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


async def journey_creation():
    if (
        "journey_creation_state" not in st.session_state
        or st.session_state["journey_creation_state"] == "init"
    ):
        # st.subheader("Journey Details")
        # available_options = [
        #     "New Hire Onboarding",
        #     "Project Onboarding",
        #     "Skill Development",
        #     "Customer Onboarding",
        #     "Other",
        # ]
        # with st.container(border=True):
        #     target = st.selectbox(
        #         "Journey Purpose",
        #         available_options,
        #         index=0,
        #     )

            # location = st.selectbox(
            #     "Location",
            #     ("Helsinki", "Stockholm", "Berlin", "Tokyo", "Other"),
            #     index=None,
            #     placeholder="Choose location",
            # )
        st.markdown(" ")

        if True: #available_options.index(target) == 0:
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
                theme = get_theme()
                if button_placeholder.button("Continue", use_container_width=True, disabled=not (bool(role_title) and bool(role_description))):
                    st.session_state["journey_creation_state"] = "name"
                    # button_placeholder.empty()
                    with button_placeholder:
                        with stylable_container(
                            key="journey_process_container",
                            css_styles=[
                                """
                                {
                                    gap: 0;
                                }
                                """,
                                f"""
                                .element-container:first-child {{
                                    border: 1px solid {theme["borderColor"] if theme["base"] == "dark" else theme["borderColorLight"]};
                                    border-radius: 0.5rem;
                                    overflow: hidden;
                                    padding: 0.25rem 0.75rem;
                                    height: 2.5rem;
                                }}""",
                                """
                                .element-container:last-child {
                                    top: -0.9rem;
                                    display: flex;
                                    justify-content: center;
                                }
                                """,
                                """
                                .element-container > .stSpinner {
                                    width: 1.5rem;
                                }""",
                            ],
                        ):
                            with st.spinner(""):
                                result = await get_chain(
                                    "journey_template_selector"
                                ).ainvoke(
                                    {
                                        "job_description": f"Title:\n{role_title.strip()}\n\nDescription:\n{role_description.strip()}"
                                    }
                                )
                                # result = "Software Engineer"

                                # print("Result", result)
                                matching_ids = match_title_to_cat_and_id(result)
                                # print("Matching IDs", matching_ids)
                                template = load_journey_template(matching_ids[1])
                                # pretty_print(template, force=True)
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
    elif st.session_state["journey_creation_state"] == "name":
        if (
            "journey_creation_data" in st.session_state
            and st.session_state["journey_creation_data"] is not None
        ):
            # pretty_print(get_journey_item_cache() , force=True)
            journey_id = st.session_state.get("journey_creation_id")
            # print("Journey id "+repr(journey_id))
            if journey_id == None or journey_id not in get_journey_item_cache().keys():
                # print("recreate journey")
                journey = JourneyItem.from_json(
                    st.session_state["journey_creation_data"], from_template=True
                )
                get_journey_item_cache()[journey.id] = journey
            else:
                journey = get_journey_item_cache()[journey_id]

            st.session_state["journey_creation_id"] = journey.id
            journey_name = journey.title
            st.subheader("Name Journey")
            with st.container(border=True):
                st.subheader(journey.title)
                journey_name = st.text_input(
                    "Journey Name",
                    value=journey_name,
                    placeholder="eg. Onboarding for Software Engineer",
                    key="journey_create_name_" + journey.id,
                )

            col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
            with col1:
                # st.page_link("pages/create_journey_2.py", label="Back")
                if st.button(
                    "Back",
                    use_container_width=True,
                    key="journey_create_cancel",
                ):
                    st.session_state["journey_creation_state"] = "init"
                    st.rerun()
            with col3:
                # print(f"journey_create_{journey.id}")
                if st.button(
                    "Continue",
                    use_container_width=True,
                    key=f"journey_create_{journey.id}"
                ):
                    # print("Create journey " + journey.id)
                    journey.title = journey_name
                    journey.save_to_db()
                    del get_journey_item_cache()[journey.id]
                    st.session_state["journey_edit_id"] = journey.id
                    st.session_state["journey_creation_id"] = None
                    st.session_state["journey_creation_data"] = None
                    st.session_state["journey_creation_state"] = None
                    st.session_state["journey_creation_state"] = "init"
                    st.switch_page("pages/journey_edit.py")
                # st.page_link("main.py", label="Continue")

        # st.subheader("Assign Journey")

        # with st.container(border=True):
        #     # st.write(
        #     #     JourneyItem.from_json(st.session_state["journey_creation_data"])
        #     #     if "journey_creation_data" in st.session_state
        #     #     else "No data"
        #     # )
        #     assign_to = st.multiselect(
        #         "Assign Journey to",
        #         [
        #             "Individual(s)",
        #             "Team(s)",
        #             "Department(s)",
        #             "Location(s)",
        #             "Subsidiary",
        #         ],
        #         ["Individual(s)"],
        #     )
        #     st.divider()

        #     st.subheader("Assign to Individual(s)")
        #     emp_id = st.text_input(
        #         "Employee Details (Name / ID)",
        #         placeholder="eg. John Smith / XH12345",
        #         key=1,
        #     )
        #     st.button("Add Employee", type="primary")
        #     st.markdown(" ")

        # col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
        # with col1:
        #     # st.page_link("pages/create_journey_2.py", label="Back")
        #     if st.button("Back", use_container_width=True):
        #         st.session_state["journey_creation_state"] = "init"
        #         st.rerun()
        # with col3:
        #     if st.button("Continue", use_container_width=True):
        #         print("donothing")
        #     # st.page_link("main.py", label="Continue")


async def main():
    st.title("Create new Journey")

    if init_sidebar(UserLevel.org_admin) != AuthStatus.LOGGED_IN:
        st.switch_page("login.py")
        return

    st.markdown(" ")
    await journey_creation()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
