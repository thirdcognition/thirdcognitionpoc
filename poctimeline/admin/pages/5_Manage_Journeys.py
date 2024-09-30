import os
import sys
import time
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from langchain_core.messages import BaseMessage

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../../lib"))

from admin.sidebar import init_sidebar
from lib.models.journey import (
    TaskStructure,
    JourneyModel,
    StepModel,
    SubjectModel,
    StepStructure,
)
from lib.chains.init import get_chain
from lib.journey_shared import (
    create_subject_prompt_editor,
    delete_journey,
    gen_subject,
    get_files_for_journey,
    llm_gen_json_step,
    llm_gen_resource,
    llm_gen_update_tasks,
    save_journey,
)
from lib.load_env import SETTINGS
from lib.models.user import AuthStatus, UserLevel
from lib.streamlit.user import check_auth

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


def save_journey_command(journey_name, id, journey: JourneyModel, scope="fragment"):
    if save_journey(journey_name, journey):
        st.session_state.editing_journey = None
        st.session_state.editing_journey_details = None
        st.session_state.editing_subject = None
        st.session_state.editing_step = None
        st.session_state.edit_mode[id] = False
        time.sleep(0.1)
        st.rerun(scope=scope)
    else:
        st.error("Error saving journey")


def save_journey_ui(id, journey_name, journey: JourneyModel):
    subcol1, subcol2, _ = st.columns([1, 1, 5])
    if subcol1.button("Save", key=f"save_button_{id}", use_container_width=True):
        save_journey_command(journey_name, id, journey, scope="app")
    if subcol2.button("Cancel", key=f"cancel_button_{id}", use_container_width=True):
        st.session_state.editing_journey = None
        st.session_state.editing_journey_details = None
        st.session_state.editing_subject = None
        st.session_state.editing_step = None
        st.session_state.editing_journey_tasks = False
        st.session_state.edit_mode[id] = False
        time.sleep(0.1)
        st.rerun()


def journey_details_ui(
    journey_name: str, journey: JourneyModel, edit_mode: bool = False
):
    if edit_mode:
        journey.instructions = st.text_area(
            f"(gen) Instructions",
            value=journey.instructions,
            key=f"journey_instructions_{journey_name}",
        )
        journey.title = st.text_input(
            f"Journey title",
            value=journey.title,
            key=f"journey_title_{journey_name}",
        )
        journey.summary = st.text_area(
            f"Journey summary",
            value=journey.summary,
            key=f"journey_summary_{journey_name}",
            height=200,
        )
        journey.chroma_collections = st.text_input(
            f"Chroma databases",
            value=",".join(journey.chroma_collections),
            key=f"journey_chroma_collections_{journey_name}",
        ).split(",")

    else:
        st.write(f"#### {journey.title.strip()}")
        st.write(journey.summary)

        if journey.chroma_collections and len(journey.chroma_collections) > 0:
            st.write("Connected Chroma databases:")
            st.write("* " + "\n* ".join(journey.chroma_collections))


def subject_details_ui(
    journey_name: str,
    subject_index: int,
    journey: JourneyModel,
    edit_mode: bool = False,
):
    subject = journey.subjects[subject_index]

    subject.prompts = create_subject_prompt_editor(
        f"{journey_name}_subject_{subject_index + 1}", subject, edit_mode
    )

    if edit_mode:
        journey.subjects[subject_index].instructions = st.text_area(
            f"(gen) Instructions",
            value=subject.instructions,
            key=f"subject_instructions_{journey_name}_{subject_index}",
        )
        journey.subjects[subject_index].step_amount = st.number_input(
            f"(gen) (approx) Plan steps",
            min_value=1,
            max_value=20,
            value=subject.step_amount or 3,
            key=f"step_amount_{journey_name}_{subject_index}",
        )
        journey.subjects[subject_index].task_amount = st.number_input(
            f"(gen) (approx) Modules",
            min_value=1,
            max_value=20,
            value=subject.task_amount or 3,
            key=f"subject_task_amount_{journey_name}_{subject_index}",
        )
        journey.subjects[subject_index].title = st.text_input(
            f"Title",
            value=subject.title,
            key=f"subject_title_{journey_name}_{subject_index}",
        )
        journey.subjects[subject_index].summary = st.text_area(
            f"Summary",
            value=subject.summary,
            key=f"subject_summary_{journey_name}_{subject_index}",
            height=200,
        )
        journey.subjects[subject_index].db_sources = get_files_for_journey(
            journey.chroma_collections[0],
            journey_name,
            subject_index,
            subject.db_sources,
        )
        # if st.button("Regenerate", key=f"regenerate_button_{journey_name}_{subject_index}"):
        #     journey.subjects[subject_index] = await gen_subject((journey, subject)
        # save_journey_ui(f"{journey_name}_{subject_index}", journey_name, journey)
    else:
        if subject.instructions:
            st.write("### Instructions:")
            st.write(subject.instructions)
        st.write(f"### {subject.title}:")
        st.write(subject.summary)
        if subject.files and len(subject.files) > 0:
            st.write("Files used:")
            st.write("* " + "\n* ".join(subject.files))


def step_ui(
    journey_name: str,
    subject_index: int,
    step_index: int,
    journey: JourneyModel,
    edit_mode: bool = False,
    container1: DeltaGenerator = None,
):
    step: StepModel = journey.subjects[subject_index].plan[step_index]
    step_structured: StepStructure = step.structured
    if container1 is None:
        container1 = st.container()

    with container1:
        if edit_mode:
            if step_structured is None:
                step.title = st.text_input(
                    "Title",
                    value=step.title,
                    key=f"step_title_simple_{journey_name}_{subject_index}_{step_index}",
                )

                step.subject = st.text_input(
                    "Subject",
                    value=step.subject,
                    key=f"step_subject_simple_{journey_name}_{subject_index}_{step_index}",
                )

                step.intro = st.text_area(
                    "Intro",
                    value=step.intro,
                    key=f"step_intro_simple_{journey_name}_{subject_index}_{step_index}",
                    height=200,
                )

                step.content = st.text_area(
                    "Content",
                    value=step.content,
                    key=f"step_content_simple_{journey_name}_{subject_index}_{step_index}",
                    height=400,
                )
            else:
                step_structured.title = st.text_input(
                    "Title",
                    value=step_structured.title,
                    key=f"step_title_structured_{journey_name}_{subject_index}_{step_index}",
                )
                step.title = step_structured.title

                step_structured.subject = st.text_input(
                    "Subject",
                    value=step_structured.subject,
                    key=f"step_subject_structured_{journey_name}_{subject_index}_{step_index}",
                )
                step.subject = step_structured.subject

                step_structured.intro = st.text_area(
                    "Intro",
                    value=step_structured.intro,
                    key=f"step_intro_structured_{journey_name}_{subject_index}_{step_index}",
                    height=200,
                )
                step.intro = step_structured.intro

                step_structured.content = st.text_area(
                    "Content",
                    value=step_structured.content,
                    key=f"step_content_structured_{journey_name}_{subject_index}_{step_index}",
                    height=400,
                )
                step.content = step_structured.content

            # save_journey_ui(f"{journey_name}_{subject_index}_{step_index}", journey_name, journey)
        else:
            if step_structured is None:
                st.write(f"#### {step.title}")
                st.write(f"{step.subject}")

                st.write("##### Intro:")
                st.write(step.intro)

                st.write("##### Content:")
                st.write(step.content)
            else:
                st.write(f"#### {step_structured.title}")
                st.write(f"{step_structured.subject}")

                st.write("##### Intro:")
                st.write(step_structured.intro)

                st.write("##### Content:")
                st.write(step_structured.content)


def step_tasks_ui(
    journey_name: str,
    subject_index: int,
    step_index: int,
    journey: JourneyModel,
    edit_mode: bool = False,
    container1: DeltaGenerator = None,
    container2: DeltaGenerator = None,
):
    subject = journey.subjects[subject_index]
    step: StepModel = subject.plan[step_index]
    if container1 is None or container2 is None:
        container1, container2 = st.tabs(["Simple", "Structured"])
    task_index = 0
    task: TaskStructure = None

    col1, col2 = container2.columns([1, 3])
    with container2:
        if step.structured is not None:
            if 0 < len(step.structured.tasks) > 0:
                task_index = (
                    col1.radio(
                        "Module",
                        [i + 1 for i, task in enumerate(step.structured.tasks)],
                        key=f"step_task_index_{journey_name}_{subject_index}_{step_index}",
                        captions=[
                            task.title
                            for i, task in enumerate(step.structured.tasks)
                        ],
                        index=0,
                    )
                    - 1
                )
                task = step.structured.tasks[task_index]
            if edit_mode:
                col1.button(
                    "Add module",
                    key=f"add_task_{journey_name}_{subject_index}_{step_index}",
                    on_click=add_task,
                    args=(journey_name, journey, subject_index, step_index),
                )
    if edit_mode:
        with container1:
            with st.popover(":sparkle: Generate new modules", use_container_width=True):
                if st.button(
                    f"Are you sure you want to generate {journey_name}: {step.title} modules?",
                    key=f"generate_tasks_button_{journey_name}",
                    use_container_width=True,
                ):
                    new_tasks = get_chain("step_tasks")(
                        (
                            subject.prompts.step_tasks.system,
                            subject.prompts.step_tasks.user,
                        )
                    ).invoke(
                        {
                            "context": step.content,
                            "journey_instructions": journey.instructions,
                            "subject_instructions": subject.instructions,
                            "step_instructions": step.instructions,
                            "subject": f"Title: {step.title}\nSubject: {step.subject}",
                            "amount": (
                                len(step.structured.tasks)
                                if step.structured is not None
                                else 10
                            ),
                        }
                    )
                    if isinstance(new_tasks, tuple) and len(new_tasks) == 2:
                        new_tasks, _ = new_tasks
                    if isinstance(new_tasks, BaseMessage):
                        new_tasks = new_tasks.content

                    step.tasks = new_tasks.strip()
                    new_structured = llm_gen_json_step(step)
                    step.structured.tasks = new_structured.tasks
                    step = llm_gen_update_tasks(
                        journey,
                        subject,
                        step,
                        new_structured,
                    )
                    journey.subjects[subject_index].plan[step_index] = step

                    st.rerun(scope="fragment")

                st.write(
                    "This will regenerate the modules for this step. This may take a few minutes.\n\nYou'll need to manually save after regeneration."
                )
            step.tasks = st.text_area(
                "Tasks",
                value=step.tasks,
                key=f"step_Tasks_{journey_name}_{subject_index}_{step_index}",
                height=400,
            )
            with st.popover(
                ":sparkle: Regenerate structured modules", use_container_width=True
            ):
                if st.button(
                    f"Are you sure you want regenerate {journey_name}: {journey.subjects[subject_index].plan[step_index].title} modules?",
                    key=f"generate_structured_tasks_button_{journey_name}",
                    use_container_width=True,
                ):
                    new_structured = llm_gen_json_step(step)
                    step.structured.tasks = new_structured.tasks

                    step = llm_gen_update_tasks(
                        journey,
                        subject,
                        step,
                        new_structured,
                    )
                    journey.subjects[subject_index].plan[step_index] = step
                    st.rerun(scope="fragment")

                st.write(
                    "This will regenerate the structured modules for this step. This may take a few minutes.\n\nYou'll need to manually save after regeneration."
                )
        with container2:
            if step.structured is not None:
                if task is not None:
                    with col2:
                        edit_task_ui(
                            journey_name,
                            journey,
                            subject_index,
                            step_index,
                            task_index,
                        )

    else:
        with container1:
            st.write(step.tasks.replace("\n", "\n\n"))
        with container2:
            if step.structured is not None and task is not None:
                col2.write("##### Title:")
                col2.write(f"{task.title}")
                col2.write("##### Description:")
                col2.write(f"{task.description}")
                if len(task.resources) > 0:
                    col2.write("##### Resources:")
                    for resource in task.resources:
                        with col2.expander(resource.title):
                            # col2.write(f"###### {resource.title}")
                            st.write(f"{resource.summary}")
                            st.write(f"_Ref: {resource.reference}_")
                # col2.write(
                #     "\n".join([f"  - {resource}" for resource in task.resources])
                # )
                col2.write("##### Test:")
                col2.write(f"{task.test}")


def edit_task_ui(
    journey_name: str,
    journey: JourneyModel,
    subject_index: int,
    step_index: int,
    task_index: int,
):
    task = (
        journey.subjects[subject_index]
        .plan[step_index]
        .structured.tasks[task_index]
    )
    st.button(
        "Remove module",
        key=f"remove_task_{journey_name}_{subject_index}_{step_index}_{task_index}",
        on_click=remove_task,
        args=(
            journey_name,
            journey,
            subject_index,
            step_index,
            task_index,
        ),
    )
    task.title = st.text_input(
        "Title",
        value=task.title,
        key=f"step_task_title_{journey_name}_{subject_index}_{step_index}_{task_index}",
    )
    task.description = st.text_area(
        "Description",
        value=task.description,
        key=f"step_task_description_{journey_name}_{subject_index}_{step_index}_{task_index}",
    )
    st.button(
        "Add Resource",
        key=f"step_task_resource_add_{journey_name}_{subject_index}_{step_index}_{task_index}",
        on_click=add_resource,
        args=(
            journey_name,
            journey,
            subject_index,
            step_index,
            task_index,
        ),
    )
    for resource_index, resource in enumerate(task.resources):
        resource = task.resources[resource_index]
        resource.title = st.text_input(
            f"Resource {resource_index+1} title",
            value=resource.title,
            key=f"step_task_resource_title_{journey_name}_{subject_index}_{step_index}_{task_index}_{resource_index}",
        )
        resource.summary = st.text_area(
            f"Resource {resource_index+1} summary",
            value=resource.summary,
            key=f"step_task_resource_summary_{journey_name}_{subject_index}_{step_index}_{task_index}_{resource_index}",
            height=200,
        )
        if st.button(
            "Regenerate resource summary",
            key=f"step_task_resource_generate_{journey_name}_{subject_index}_{step_index}_{task_index}_{resource_index}",
            on_click=generate_resource,
            args=(
                journey_name,
                journey,
                subject_index,
                step_index,
                task_index,
                resource_index,
            ),
        ):
            st.rerun(scope="fragment")
        resource.reference = st.text_area(
            f"Resource {resource_index+1} reference",
            value=resource.reference,
            key=f"step_task_resource_reference_{journey_name}_{subject_index}_{step_index}_{task_index}_{resource_index}",
        )
        # = st.text_area(
        #     f"Resource {resource_index+1}",
        #     value=resource,
        #     key=f"step_task_resource_{journey_name}_{subject_index}_{step_index}_{task_index}_{resource_index}",
        # )

        st.button(
            "Remove Resource",
            key=f"step_task_resource_remove_{journey_name}_{subject_index}_{step_index}_{task_index}_{resource_index}",
            on_click=remove_resource,
            args=(
                journey_name,
                journey,
                subject_index,
                step_index,
                task_index,
                resource_index,
            ),
        )
    task.test = st.text_area(
        "Test",
        value=task.test,
        key=f"step_task_test_{journey_name}_{subject_index}_{step_index}_{task_index}",
    )
    journey.subjects[subject_index].plan[step_index].structured.tasks[
        task_index
    ] = task


def add_subject(journey_name: str, journey: JourneyModel):
    journey.subjects.append(
        SubjectModel(
            title="New subject",
            summary="",
            plan=[StepModel(title="New step", tasks="")],
            prompts=journey.subjects[-1].prompts.model_copy(),
        )
    )


def remove_subject(journey_name: str, journey: JourneyModel, subject_index: int):
    journey.subjects.pop(subject_index)


def add_step(journey_name: str, journey: JourneyModel, subject_index: int):
    journey.subjects[subject_index].plan.append(
        StepModel(title="New step", tasks="")
    )


def remove_step(
    journey_name: str, journey: JourneyModel, subject_index: int, step_index: int
):
    journey.subjects[subject_index].plan.pop(step_index)


def add_task(
    journey_name: str, journey: JourneyModel, subject_index: int, step_index: int
):
    journey.subjects[subject_index].plan[step_index].structured.tasks.append(
        TaskStructure(title="", description="", resources=[], test="")
    )


def remove_task(
    journey_name: str,
    journey: JourneyModel,
    subject_index: int,
    step_index: int,
    task_index: int,
):
    journey.subjects[subject_index].plan[step_index].structured.tasks.pop(
        task_index
    )


def add_resource(
    journey_name: str,
    journey: JourneyModel,
    subject_index: int,
    step_index: int,
    task_index: int,
):
    journey.subjects[subject_index].plan[step_index].structured.tasks[
        task_index
    ].resources.append("")


def generate_resource(
    journey_name: str,
    journey: JourneyModel,
    subject_index: int,
    step_index: int,
    task_index: int,
    resource_index: int,
):

    journey.subjects[subject_index].plan[step_index].structured.tasks[
        task_index
    ].resources[resource_index].summary = llm_gen_resource(
        journey,
        journey.subjects[subject_index],
        journey.subjects[subject_index].plan[step_index],
        journey.subjects[subject_index]
        .plan[step_index]
        .structured.tasks[task_index],
        journey.subjects[subject_index]
        .plan[step_index]
        .structured.tasks[task_index]
        .resources[resource_index],
    )


def remove_resource(
    journey_name: str,
    journey: JourneyModel,
    subject_index: int,
    step_index: int,
    task_index: int,
    resource_index: int,
):
    journey.subjects[subject_index].plan[step_index].structured.tasks[
        task_index
    ].resources.pop(resource_index)


@st.fragment
async def subjects_ui(
    journey_name: str, journey_index: int, journey: JourneyModel
):
    col1, col2 = st.columns([1, 1], vertical_alignment="center")
    subject_titles = [subject.title for subject in journey.subjects]
    subject_title = col1.selectbox(
        "Subject",
        options=subject_titles,
        key=f"subject_select_{journey_name}_subject",
        index=(
            min(
                int(
                    st.session_state.subject_index
                    if "subject_index" in st.session_state
                    else 0
                ),
                len(subject_titles),
            )
        ),
    )
    st.session_state.subject_index = subject_titles.index(subject_title)
    subject = journey.subjects[st.session_state.subject_index]
    if st.session_state.edit_mode[journey_index]:
        mod_col1, mod_col2, mod_col3 = col1.columns([1, 1, 1])
        if mod_col1.button(
            "**+**",
            key=f"add_subject_button_{journey_name}",
            on_click=add_subject,
            args=(journey_name, journey),
            use_container_width=True,
        ):
            st.session_state.subject_index = len(journey.subjects) - 1
            st.rerun(scope="fragment")

        with mod_col2.popover(":sparkle:", use_container_width=True):
            if st.button(
                f"Are you sure you want regenerate {journey_name}: {subject_title}?",
                key=f"generate_subject_button_{journey_name}",
                use_container_width=True,
            ):
                journey.subjects[st.session_state.subject_index] = (
                    await gen_subject(
                        journey, subject, st.session_state.subject_index
                    )
                )
                save_journey_command(journey_name, journey_index, journey)

        with mod_col3.popover(":x:", use_container_width=True):
            st.button(
                f"Are you sure you want to remove {journey_name}: {subject_title}?",
                key=f"delete_subject_button_{journey_name}",
                use_container_width=True,
                on_click=remove_subject,
                args=(journey_name, journey, st.session_state.subject_index),
            )
    step_titles = [step.title for step in subject.plan]
    step_title = col2.selectbox(
        "Step",
        options=step_titles,
        key=f"step_select_{journey_name}_step",
        index=0,
    )
    step_index = None
    if step_title is not None:
        step_index = step_titles.index(step_title) if 0 < len(step_titles) else []
    # step = subject.plan[step_index]
    if st.session_state.edit_mode[journey_index]:
        mod_col1, mod_col2, mod_col3 = col2.columns([1, 1, 1])
        mod_col1.button(
            "**+**",
            key=f"add_step_button_{journey_name}",
            on_click=add_step,
            args=(journey_name, journey, st.session_state.subject_index),
            use_container_width=True,
        )
        with mod_col2.popover(":sparkle:", use_container_width=True):
            if st.button(
                f"Are you sure you want regenerate {journey_name}: {subject_title} - {step_title}?",
                key=f"generate_step_button_{journey_name}",
                use_container_width=True,
            ):
                journey.subjects[st.session_state.subject_index] = (
                    await gen_subject(
                        journey,
                        subject,
                        step_index=step_index,
                    )
                )
                save_journey_command(journey_name, journey_index, journey)
        with mod_col3.popover(":x:", use_container_width=True):
            st.button(
                f"Are you sure you want to remove {journey_name}: {subject_title} - {step_title}?",
                key=f"delete_step_button_{journey_name}",
                use_container_width=True,
                on_click=remove_step,
                args=(
                    journey_name,
                    journey,
                    st.session_state.subject_index,
                    step_index,
                ),
            )

    # for st.session_state.subject_index, subject in enumerate(journey.subjects):
    tab1, tab2, tab4, tab5 = st.tabs(
        [
            "Subject details",
            "Step details",
            "Modules as text",
            "Modules structured",
        ]
    )

    with tab1:
        subject_details_ui(
            journey_name,
            st.session_state.subject_index,
            journey,
            st.session_state.edit_mode[journey_index],
        )

    if 0 < len(step_titles) and step_index is not None:
        step_ui(
            journey_name,
            st.session_state.subject_index,
            step_index,
            journey,
            st.session_state.edit_mode[journey_index],
            tab2,
        )
        step_tasks_ui(
            journey_name,
            st.session_state.subject_index,
            step_index,
            journey,
            st.session_state.edit_mode[journey_index],
            tab4,
            tab5,
        )


def main():
    st.title("Manage Journeys")

    if init_sidebar(UserLevel.org_admin) != AuthStatus.LOGGED_IN:
        return

    st.write("NA, upgrade in progress")

    # file_categories = get_all_categories()
    # categories = st.multiselect("Categories", file_categories)

    # if not categories:
    #     st.header("First, choose a categories.")
    #     return

    # db_journey = get_db_journey(
    #     chroma_collections=["rag_" + categories for categories in categories]
    # )
    # if len(db_journey.keys()) == 0:
    #     db_journey = get_db_journey(
    #         chroma_collections=["rag_" + categories for categories in categories],
    #         reset=True,
    #     )

    # if db_journey is not None and len(db_journey.keys()) > 0:
    #     st.header("Journey database")
    # else:
    #     st.header("No journeys created yet.")

    # if "edit_mode" not in st.session_state or len(st.session_state.edit_mode) == 0:
    #     st.session_state.edit_mode = [False for _ in db_journey.keys()]
    # for journey_index, journey_name in enumerate(db_journey.keys()):
    #     journey: JourneyModel = db_journey[journey_name]
    #     col_edit, col_delete, col2, col3 = st.columns(
    #         [2, 2, 12, 3], vertical_alignment="center"
    #     )
    #     edit_mode = st.session_state.edit_mode[journey_index]
    #     st.session_state.edit_mode[journey_index] = col_edit.checkbox(
    #         ":pencil:",
    #         key=f"edit_button_{journey_name}",
    #         value=edit_mode,
    #     )
    #     with col_delete.popover(":x:"):
    #         if st.button(
    #             f"Are you sure you want to remove {journey_name}?",
    #             key=f"delete_button_{journey_name}",
    #             use_container_width=True,
    #         ):
    #             delete_journey(journey_name)
    #     col2.subheader(f"&nbsp;{journey_name}", divider=True)
    #     col3.link_button(
    #         ":paperclip:&nbsp;&nbsp;Link",
    #         f"{SETTINGS.client_host}?journey={journey_name}",
    #         use_container_width=True,
    #     )

    #     tab1, tab2 = st.tabs(["Details", "Subjects"])
    #     with tab1:
    #         journey_details_ui(
    #             journey_name, journey, st.session_state.edit_mode[journey_index]
    #         )
    #     with tab2:
    #         subjects_ui(journey_name, journey_index, journey)

    #     if st.session_state.edit_mode[journey_index]:
    #         save_journey_ui(journey_index, journey_name, journey)


if __name__ == "__main__":

    main()
