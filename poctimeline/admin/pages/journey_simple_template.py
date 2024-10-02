import time
import streamlit as st
import os
import sys

from admin.sidebar import init_sidebar

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

container_level = ["journey", "subject", "subsubject", "module"]


@st.dialog(f"Get familiar with...", width="large")
def open_module(item):
    st.write(f"Let's get started with module {item['title']}")
    feedback = st.text_input("Provide feedback...")
    if st.button("Submit"):
        st.session_state.vote = {"item": item, "reason": feedback}
        st.rerun()


def write_item(item):
    level_multiplier = max((container_level.index(item["type"])-2) * 0.025, 0.01)
    print("multiplier", level_multiplier)
    if item["type"] != container_level[-1] and item["type"] in container_level:
        container = st
        if item["type"] == "journey":
            st.subheader(item["title"])
        elif item["type"] == "subject":
            container = st.expander(label=item["title"])
        else:
        # with st.container(border=True):
            col1, col2 = container.columns(
                [
                    level_multiplier,
                    1 - level_multiplier,
                ]
            )
            col2.markdown(
                "####" + ("#" * container_level.index(item["type"])) + " " + item["title"]
            )

        if item["type"] == "subject":
            with container:
                for child in item["children"]:
                                write_item(child)
        elif item["children"]:
            for child in item["children"]:
                write_item(child)
    elif item["type"] == "module":
        col1, col2 = st.columns(
            [
                level_multiplier,
                0.8 - level_multiplier,
            ]
        )
        with col2:
            with st.container(border=True):
                subcol1, subcol2 = st.columns([0.8, 0.15])
                with subcol1:
                    st.markdown(item["title"])
                with subcol2:
                    if st.button("Open", key=f"open_button_{item['id']}"):
                        open_module(item)


async def journey_creation():
    if (
        "journey_creation_state" not in st.session_state
        or st.session_state["journey_creation_state"] == "init"
    ):
        st.subheader("Journey Details")
        container = st.container(border=True)
        available_options = [
            "New Hire Onboarding",
            "Project Onboarding",
            "Skill Development",
            "Customer Onboarding",
            "Other",
        ]
        with container:
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
                            "<style> div[data-testid=column]:last-child div[data-testid=stVerticalBlockBorderWrapper] { overflow: hidden; } .element-container > .stSpinner { margin-top:-22px; margin-left:20px } </style>",
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
            container = st.container(border=True)
            with container:
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

        if "journey_creation_data" in st.session_state and st.session_state["journey_creation_data"] is not None:
            write_item(st.session_state["journey_creation_data"])
        else:
            st.error("No data to show. Please go back and recreate the journey.")

        st.subheader("Assign Journey")

        container = st.container(border=True)
        with container:
            st.write(
                st.session_state["journey_creation_data"]
                if "journey_creation_data" in st.session_state
                else "No data"
            )
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
