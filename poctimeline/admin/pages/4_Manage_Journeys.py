import streamlit as st

from lib.journey_shared import delete_journey, save_journey
from lib.db_tools import (
    JourneyModel,
    StepModel,
    get_db_journey,
    init_db,
)

from lib.load_env import (
    CLIENT_HOST,
)

from lib.prompts import JourneyStructure

from lib.streamlit_tools import check_auth

st.set_page_config(
    page_title="TC POC: Create Journey",
    page_icon="static/icon.png",
    layout="centered",
    menu_items={
        'About': """# ThirdCognition PoC
[ThirdCognition](https://thirdcognition.com)
This is an *extremely* cool admin tool!
        """
    }
)

# def edit_journey(journey_name, journey:JourneyModel):
#     st.header(f"Edit journey: {journey_name}")

#     journey = edit_journey_details(journey_name, journey)

#     if journey.subjects:
#         st.subheader("Journey subjects")
#         for i, subject in enumerate(journey.subjects):
#             st.write(f"### Subject {i}: {subject['title']}")
#             for j, step in enumerate(subject.steps):
#                 with st.expander(f"Step {j}: {step.title}"):
#                     journey.subjects[i].steps[j] = edit_journey_subject_step(journey_name, journey.subjects[i].steps[j], i, j)

#     if st.button("Save into database"):
#         save_journey(journey_name, journey)


def edit_journey_details(journey_name, journey:JourneyModel):
    # print(journey_name, journey)
    if journey.title:
        journey.title = st.text_input(
            f"Journey title", value=journey.title, key=f"journey_title_{journey_name}"
        )
    if journey.summary:
        journey.summary = st.text_area(
            f"Journey summary", value=journey.summary, key=f"journey_summary_{journey_name}", height=200
        )
    if journey.chroma_collection:
        journey.chroma_collection = st.text_input(
            f"Chroma databases", value=",".join(journey.chroma_collection), key=f"journey_chroma_collections_{journey_name}"
        ).split(",")

    return journey

def edit_journey_subject_step(journey_name, step:StepModel, subject_index, step_index, show_structured=True, show_actions=True):
    st.write(f"##### Step {step_index+1}:")
    if show_structured:
        try:
            step_json:JourneyStructure = step.structured
            step_json.title = st.text_input(
                "Title", value=step_json.title, key=f"journey_step_title_{journey_name}_{subject_index}_{step_index}"
            )

            step_json.intro = st.text_area(
                "Intro",
                value=step_json.intro,
                key=f"journey_step_intro_{journey_name}_{subject_index}_{step_index}",
                height=200
            )

            step_json.content = st.text_area(
                "Content",
                value=step_json.content,
                key=f"journey_step_content_{journey_name}_{subject_index}_{step_index}",
                height=400
            )

            if show_actions:
                for j, action in enumerate(step_json.actions):
                    st.write(f"##### Teaching action {j+1}:")

                    action.title = st.text_input("Title", value=action.title, key=f"journey_step_action_title_{journey_name}_{subject_index}_{step_index}_{j}")
                    action.description = st.text_area("Description", value=action.description, key=f"journey_step_action_description_{journey_name}_{subject_index}_{step_index}_{j}")
                    for k, resource in enumerate(action.resources):
                        action.resources[k] = st.text_input(f"Resource {k+1}", value=resource, key=f"journey_step_action_resource_{journey_name}_{subject_index}_{step_index}_{j}_{k}")
                    action.test = st.text_area("Test", value=action.test, key=f"journey_step_action_test_{journey_name}_{subject_index}_{step_index}_{j}")
        except Exception as e:
            st.write(f"Failed to use JSON: {e}")
    else:
        step.title = st.text_input(
            "Title", value=step.title, key=f"journey_step_title_{journey_name}_{subject_index}_{step_index}"
        )

        step.intro = st.text_area(
            "Intro",
            value=step.intro,
            key=f"journey_step_intro_{journey_name}_{subject_index}_{step_index}",
        )

        step.content = st.text_area(
            "Content",
            value=step.content,
            key=f"journey_step_content_{journey_name}_{subject_index}_{step_index}",
        )
        if show_actions:
            step.actions = st.text_area(
                "actions",
                value=step.actions,
                key=f"journey_step_Actions_{journey_name}_{subject_index}_{step_index}",
            )

    return step

def main():
    init_db()
    st.title("Manage Journeys")

    if not check_auth():
        return

    db_journey = get_db_journey()

    if db_journey is not None and len(db_journey.keys()) > 0:
        st.header("Journey database")
    else:
        st.header("No journeys created yet.")

    for journey_name in db_journey.keys():
        # st.write(db_journey[journey_name])
        journey:JourneyModel = db_journey[journey_name]
        col1, col2, col3 = st.columns([2, 12, 3], vertical_alignment="center")
        with col1.popover(":x:"):
            if st.button(f"Are you sure you want to remove {journey_name}?", key=f"delete_button_{journey_name}", use_container_width=True):
                delete_journey(journey_name)
        col2.subheader(f"&nbsp;{journey_name}", divider=True)
        col3.link_button(":paperclip:&nbsp;&nbsp;Link", f"{CLIENT_HOST}?journey={journey_name}", use_container_width=True)

        if "editing_journey" not in st.session_state:
            st.session_state.editing_journey = None
            st.session_state.editing_journey_details = None
            st.session_state.editing_journey_subject = None
            st.session_state.editing_journey_step = None

        col1, col2 = st.columns([1, 4], vertical_alignment="top")
        if (
            col1.button("Edit details", key=f"edit_button_{journey_name}", disabled=st.session_state.editing_journey == journey_name
            and st.session_state.editing_journey_details is True)
            or st.session_state.editing_journey == journey_name
            and st.session_state.editing_journey_details
        ):
            st.session_state.editing_journey = journey_name
            st.session_state.editing_journey_details = True
            # with col2:
            journey_edit = edit_journey_details(journey_name, journey)
            subcol1, subcol2, _ = st.columns([1, 1, 5])
            if subcol1.button("Save", key=f"save_button_{journey_name}", use_container_width=True):
                save_journey(journey_name, journey_edit)
            if subcol2.button("Cancel", key=f"cancel_button_{journey_name}", use_container_width=True):
                st.session_state.editing_journey = None
                st.session_state.editing_journey_details = None
                st.session_state.editing_journey_subject = None
                st.session_state.editing_journey_step = None
                st.rerun()
        else:
            col2.write(f'#### {journey.title.strip()}')
            col2.write(journey.summary)

        if journey.chroma_collection and len(journey.chroma_collection) > 0:
            col2.write("Connected Chroma databases:")
            col2.write("* " + "\n* ".join(journey.chroma_collection))

        if journey.files and len(journey.files) > 0:
            col2.write("Files used:")
            col2.write("* " + "\n* ".join(journey.files))

        if col1.toggle("Extend", key=f"show_toggle_subjects_{journey_name}") or (st.session_state.editing_journey_subject is not None or st.session_state.editing_journey_step is not None):
            journey_edit = journey
            for i, subject in enumerate(journey.subjects):
                with st.container(border=True):
                    col1, col2 = st.columns([1, 3])
                    col1.write(f"#### Subject {i+1}:")
                    if (
                        col1.button(
                            "Edit subject", key=f"edit_button_{journey_name}_{i}"
                        )
                        or "editing_journey" in st.session_state
                        and st.session_state.editing_journey == journey_name
                        and "editing_journey_step" in st.session_state
                        and st.session_state.editing_journey_subject == i
                        and st.session_state.editing_journey_step == None
                    ):

                        st.session_state.editing_journey = journey_name
                        st.session_state.editing_journey_subject = i
                        st.session_state.editing_journey_step = None
                        # col1, col2 = st.columns([1, 3])
                        journey_edit.subjects[i].title = st.text_input(
                            f"Title", value=subject.title, key=f"subject_title_{journey_name}_{i}"
                        )
                        journey_edit.subjects[i].summary = st.text_area(
                            f"Summary", value=subject.summary, key=f"subject_summary_{journey_name}_{i}", height=200
                        )
                        subcol1, subcol2, _ = st.columns([1, 1, 5])
                        if subcol1.button("Save", key=f"save_button_{journey_name}_{i}", use_container_width=True):
                            save_journey(journey_name, journey_edit)
                        if subcol2.button("Cancel", key=f"cancel_button_{journey_name}_{i}", use_container_width=True):
                            st.session_state.editing_journey = None
                            st.session_state.editing_journey_details = None
                            st.session_state.editing_journey_subject = None
                            st.session_state.editing_journey_step = None
                            st.rerun()
                    else:
                        if col1.button("Delete", key=f"subject_delete_{journey_name}_{i}"):
                            print("delete")
                            new_subjects = journey_edit.subjects
                            st.rerun()
                        else:
                            col2.write(subject.title + ":")
                            col2.write(subject.summary)
                            if subject.files and len(subject.files) > 0:
                                col2.write("Files used:")
                                col2.write("* " + "\n* ".join(subject.files))

                    if (col1.toggle("Extend", key=f"show_toggle_steps_{journey_name}_{i}") or (st.session_state.editing_journey_subject is not None or st.session_state.editing_journey_step is not None)):
                        for j, step in enumerate(subject.steps):
                            with st.expander(f'##### Step {j+1}: {step.title}'):
                                col1, col2 = st.columns([1, 2])
                                show_structured = col1.toggle("Show structured", key=f"journey_step_info_structured_{journey_name}_{i}_{j}", value=True)
                                show_actions = col2.toggle("Show actions", key=f"journey_step_info_actions_{journey_name}_{i}_{j}")
                                if (
                                    st.button(
                                        "Edit step", key=f"edit_button_{journey_name}_{i}_{j}"
                                    )
                                    or "editing_journey" in st.session_state
                                    and st.session_state.editing_journey == journey_name
                                    and "editing_journey_step" in st.session_state
                                    and st.session_state.editing_journey_subject == i
                                    and st.session_state.editing_journey_step == j
                                ):
                                    st.session_state.editing_journey = journey_name
                                    st.session_state.editing_journey_subject = i
                                    st.session_state.editing_journey_step = j
                                    journey_edit.subjects[i].steps[j] = edit_journey_subject_step(journey_name, journey.subjects[i].steps[j], i, j, show_structured=show_structured, show_actions=show_actions)
                                    col1, col2, _ = st.columns([1, 1, 5])
                                    if col1.button("Save", key=f"save_button_{journey_name}_{i}_{j}", use_container_width=True):
                                        save_journey(journey_name, journey_edit)
                                    if col2.button("Cancel", key=f"cancel_button_{journey_name}_{i}_{j}", use_container_width=True):
                                        st.session_state.editing_journey = None
                                        st.session_state.editing_journey_details = None
                                        st.session_state.editing_journey_subject = None
                                        st.session_state.editing_journey_step = None
                                        st.rerun()

                                else:
                                    if step.structured and show_structured:
                                        structured:JourneyStructure = step.structured
                                        col1, col2 = st.columns([1, 5])
                                        col1.write("##### Intro:")
                                        col2.write(structured.intro)

                                        col1, col2 = st.columns([1, 5])
                                        col1.write("##### Content:")
                                        col2.write(structured.content)

                                        # st.write("##### Actions:")
                                        if show_actions:
                                            for k, action in enumerate(structured.actions):
                                                col1, col2 = st.columns([1, 5])
                                                col1.write(f"###### Action {k+1}:")
                                                col2.write(f"**Title:** {action.title}")
                                                col2.write(f"**Description:** {action.description}")
                                                col2.write(f"**Resources:**")
                                                for l, resource in enumerate(action.resources):
                                                    col2.write(f"  - {resource}")
                                                col2.write(f"**Test:** {action.test}")

                                        # col1, col2 = st.columns([1, 5])
                                        # col1.write("##### Priority:")
                                        # col2.write(structured.priority)
                                    else:
                                        col1, col2 = st.columns([1, 5])
                                        col1.write("##### Intro:")
                                        col2.write(step.intro)

                                        col1, col2 = st.columns([1, 5])
                                        col1.write("##### Content:")
                                        col2.write(step.content)

                                        if show_actions:
                                            col1, col2 = st.columns([1, 5])
                                            col1.write("##### Actions:")
                                            col2.write(step.actions)


if __name__ == "__main__":

    main()
