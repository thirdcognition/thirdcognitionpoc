import asyncio
import os
import sys
import time
from typing import List
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../../lib"))

from lib.models.journey import SubjectModel
from lib.prompts.journey import JourneyPrompts
from lib.journey_shared import (
    create_subject_prompt_editor,
    gen_subject,
    get_files_for_journey,
    save_journey,
    llm_gen_title_summary,
)
from lib.db_tools import (
    JourneyModel,
    get_db_journey,
    init_db,
)

from lib.streamlit_tools import check_auth, get_all_categories

st.set_page_config(
    page_title="TC POC: Create Journey",
    page_icon="static/icon.png",
    layout="centered",
    menu_items={
        "About": """# ThirdCognition PoC
[ThirdCognition](https://thirdcognition.com)
This is an *extremely* cool admin tool!
        """
    },
)


def create_subject(
    journey_name, subject: SubjectModel, subject_index, chroma_collection
) -> None:
    st.subheader(f"Subject {subject_index+1}")

    subject.prompts = create_subject_prompt_editor(
        f"{journey_name}_subject_{subject_index + 1}", subject
    )

    subject.instructions = st.text_area(
        "Instructions for subject",
        key=f"journey_gen_instructions_{journey_name}_{subject_index}",
        value=subject.instructions,
    )

    col1, col2 = st.columns([4, 1], vertical_alignment="bottom")
    with col1:
        subject.db_sources = get_files_for_journey(
            chroma_collection, journey_name, subject_index, subject.db_sources
        )

    subject.step_amount = col2.number_input(
        "(approx) Plan steps",
        min_value=1,
        max_value=20,
        value=3,
        # value=subject.step_amount or 5,
        key=f"journey_gen_step_amount_{journey_name}_{subject_index}",
    )

    subject.task_amount = col2.number_input(
        "(approx) Modules",
        min_value=1,
        max_value=20,
        value=3,
        # value=subject.step_amount or 5,
        key=f"journey_gen_task_amount_{journey_name}_{subject_index}",
    )

    return subject


@st.fragment
async def get_journey_gen(journey_name):
    st.subheader("Journey generator")
    if (
        "journey_get_details" not in st.session_state
        or st.session_state.journey_get_details == None
    ):
        st.session_state.journey_get_details = {}
        st.session_state.journey_create = False
        st.session_state.journey_generator_running = False
        st.session_state.journey_generator_generate_resources = False

    file_categories = get_all_categories()
    default_category = []
    journey_details: JourneyModel = None
    if (
        journey_name not in st.session_state.journey_get_details
        or st.session_state.journey_get_details[journey_name] == None
    ):
        journey_details = JourneyModel(
            journeyname=journey_name,
            chroma_collection=["rag_" + category for category in default_category],
        )
        # {"journeyname": journey_name, "category": default_category}
        st.session_state.journey_get_details[journey_name] = journey_details
    else:
        journey_details = st.session_state.journey_get_details[journey_name]
        default_category = [cat[4:] for cat in journey_details.chroma_collection]

    default_category = [
        st.selectbox(
            "Select category for journey",
            file_categories,
            key=f"journey_gen_category_{journey_name}",
            index=None,
        )
    ]
    journey_details.chroma_collection = (
        ["rag_" + category for category in default_category]
        if default_category[0] != None
        else []
    )

    but_col1 = None
    but_col2 = None

    if len(default_category) > 0 and default_category[0] != None:
        # col1, col2 = st.columns([5, 1], vertical_alignment="bottom")
        generate_start = False
        if not st.session_state.journey_generator_running:
            journey_details.instructions = (
                st.text_area(
                    "Journey Instructions",
                    height=10,
                    key=f"journey_gen_instructions_{journey_name}",
                    value=journey_details.instructions or "",
                )
                or ""
            )
            # journey_details.instructions = journey_instructions
            if len(journey_details.subjects) == 0:
                journey_details.subjects = [SubjectModel()]

            if len(journey_details.subjects) > 0:
                for i, subject in enumerate(journey_details.subjects):
                    journey_details.subjects[i] = create_subject(
                        journey_name, subject, i, default_category
                    )
            but_col1, but_col2 = st.columns([4, 1], vertical_alignment="bottom")

            but_subcol1, but_subcol2 = but_col1.columns(
                [1, 3], vertical_alignment="center"
            )

            but_subcol2.write("_will copy prompts from previous subject_")
            if but_subcol1.button("Add subject", use_container_width=True):
                i = len(journey_details.subjects)
                prompts = JourneyPrompts()
                if i > 0 and journey_details.subjects[-1].prompts is not None:
                    prompts = JourneyPrompts(
                        **journey_details.subjects[-1].prompts.model_dump()
                    )

                journey_details.subjects.append(
                    create_subject(
                        journey_name,
                        SubjectModel(prompts=prompts),
                        i,
                        default_category,
                    )
                )
                st.session_state.journey_get_details[journey_name] = journey_details
                st.rerun(scope="fragment")

    # Check that all subjects have files defined
    def check_subject_files(subjects: List[SubjectModel]):
        for subject in subjects:
            if len(subject.db_sources) < 1:
                return False
        return True

    if (
        but_col2 is not None
        and len(journey_details.subjects or []) > 0
        and but_col2.button(
            "Generate",
            key=f"generate_journey_{journey_name}",
            disabled=not check_subject_files(journey_details.subjects),
            use_container_width=True,
        )
    ):
        st.session_state.journey_generator_running = True

        for i, subject in enumerate(journey_details.subjects):
            # print(f"Generating subject {i+1} for journey {journey_name}")
            subject = await gen_subject(
                journey_details, subject, subject_index=i
            )

            if subject is not None:
                journey_details.subjects[i] = subject

        st.session_state.journey_get_details[journey_name] = journey_details
        st.session_state.journey_generator_running = False

        with st.spinner("Generating journey titles and summaries"):
            files = []
            for i, subject in enumerate(journey_details.subjects):
                title, summary = await llm_gen_title_summary(subject.plan)
                journey_details.subjects[i].title = title
                journey_details.subjects[i].summary = summary

                files.extend(subject.files)

            title, summary = await llm_gen_title_summary(journey_details.subjects)

            st.success("Generating journey titles and summaries done.")

            journey_details.title = title
            journey_details.summary = summary
            journey_details.db_sources = files

        if save_journey(journey_name, journey_details):
            st.success("Journey saved.")
            journey_details.complete = True
            st.session_state.journey_get_details[journey_name] = journey_details
            st.session_state.journey_get_details = {}
            st.session_state.journey_create = False
            st.session_state.journey_generator_running = False
            time.sleep(0.1)
            # get_db_journey(reset=True)
            st.rerun(scope="fragment")
        else:
            st.error("Journey save failed.")
    return journey_details


# Streamlit app
async def main():

    init_db()
    st.title("Create Journey")

    if not check_auth():
        return

    db_journey = get_db_journey()

    with st.container(border=True):
        journey_create: JourneyModel = None
        if "journey_create_data" in st.session_state:
            journey_create = st.session_state.journey_create_data

        if "creating_journey" not in st.session_state:
            st.header("Create new journey")
            col1, col2 = st.columns([2, 1], vertical_alignment="bottom")

            journey_name = col1.text_input("Unique name for the journey", value="test")
            if col2.button("Create", disabled=journey_name in db_journey.keys()):
                st.session_state.creating_journey = journey_name
                time.sleep(0.01)
                st.rerun()

        if "creating_journey" in st.session_state and (
            journey_create == None or journey_create.complete == False
        ):
            journey_name = st.session_state.creating_journey
            st.header(f"Create new journey: {journey_name}")
            st.session_state.journey_create_data = await get_journey_gen(journey_name)
            if st.session_state.journey_create_data.complete:
                st.session_state.journey_create_data = JourneyModel()
                st.session_state.journey_get_details = {}
                st.session_state.journey_create = False
                st.session_state.journey_generator_running = False
                time.sleep(0.01)
                st.rerun()


if __name__ == "__main__":
    asyncio.run(main())
