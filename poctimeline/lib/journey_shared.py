from datetime import datetime
import re
import textwrap
import time
import streamlit as st
from typing import Any, Callable, Dict, List, Union
from langchain_core.messages import BaseMessage, AIMessage
import yaml

from lib.chains.init import get_base_chain, get_chain
from lib.chains.rag_chain import get_rag_chain
from lib.db_tools import (
    get_db_sources,
    get_db_journey,
    init_db,
)
from lib.document_tools import markdown_to_text
from lib.document_tools import create_document_lists
from lib.load_env import SETTINGS
from lib.streamlit_tools import llm_edit
from lib.models.journey import (
    TaskStructure,
    JourneyModel,
    ResourceStructure,
    StepModel,
    SubjectModel,
    StepStructure,
)
from lib.models.prompts import CustomPrompt
from lib.models.sqlite_tables import SourceContents, SourceData, JourneyDataTable
from lib.prompts.journey import Step, Plan, convert_to_journey_prompts


def save_journey(journey_name, journey: JourneyModel) -> bool:
    print(f"Save journey {journey_name}")
    # st.write(f"Save journey {journey_name}")
    # st.write(journey)
    database_session = init_db()
    journey_db = (
        database_session.query(JourneyDataTable)
        .filter(JourneyDataTable.journeyname == journey_name)
        .first()
    )

    if journey_db is not None:
        print("Remove old Journey")
        database_session.delete(journey_db)

    print("Create journey")
    journey_db = JourneyDataTable(
        journeyname=journey_name,
        files=journey.files,
        subjects=journey.subjects,
        title=journey.title,
        summary=journey.summary,
        last_updated=datetime.now(),
        chroma_collection=journey.chroma_collection,
    )
    try:
        database_session.add(journey_db)
        database_session.commit()
    except Exception as e:
        print(f"Error saving journey.\n\n{e}")
        return False

    get_db_journey(reset=True)
    return True


def delete_journey(journey_name):
    database_session = init_db()
    journey_db = (
        database_session.query(JourneyDataTable)
        .filter(JourneyDataTable.journeyname == journey_name)
        .first()
    )

    if journey_db is not None:
        database_session.delete(journey_db)
        database_session.commit()
        get_db_journey(reset=True)
        st.success(f"{journey_name} has been deleted successfully.")
        time.sleep(0.1)
        st.rerun()
    else:
        st.warning(f"{journey_name} does not exist in the database.")


async def llm_gen_title_summary(
    plan: List[Union[StepModel, SubjectModel]]
) -> tuple[str, str]:
    if isinstance(plan[0], SubjectModel):
        context = "\n".join(
            [
                f"Title: {subject.title}\n Summary:\n{subject.summary}\n"
                for subject in plan
            ]
        )
    else:
        context = "\n".join(
            [
                textwrap.dedent(
                   f"""
                    Title: {step.title}
                    Subject:
                    {step.subject.replace("\n", " ")}
                    Introduction:
                    {(step.intro or "").replace("\n", " ")}
                    """
                )
                for step in plan
            ]
        )
    title = get_chain("task").invoke(
        {
            "context": context,
            "task": "Summarize context with 10 words or less to a title",
        }
    )

    summary = await llm_edit(
        [context],
        "Summarize the following list of titles and intros into a summary description.",
        force=True,
    )

    return title.strip(), summary.strip()


def llm_gen_plan(
    content, journey: JourneyModel, subject: SubjectModel
) -> Plan:
    return get_base_chain("plan")(
        (subject.prompts.plan.system, subject.prompts.plan.user)
    ).invoke(
        {
            "context": content,
            "amount": subject.step_amount,
            "journey_instructions": journey.instructions,
            "subject_instructions": subject.instructions,
        }
    )


def llm_gen_step(
    content,
    journey: JourneyModel,
    subject: SubjectModel,
    step: Union[Step | StepModel],
    existing_plan: List[Union[Step | StepModel]] = None,
    task_amount=10,
    progress_cb: Callable[[float, str], None] = None,
) -> StepModel:
    class_content = None
    subject_string = (
        f"Title: {step.title}\nSubject: {step.description}"
        if isinstance(step, Step)
        else f"Title: {step.title}\nSubject: {step.subject}"
    )
    previous_class_subjects = []
    previous_class_intros = []
    previous_class_tasks = []
    if existing_plan != None and len(existing_plan) > 0:
        for prev_step in existing_plan:
            if isinstance(prev_step, Step):
                previous_class_subjects.append(
                    AIMessage(
                        f"Previous title and subject: {prev_step.title}: {prev_step.description.replace('\n', ' ')}"
                    )
                )
            else:
                previous_class_subjects.append(
                    AIMessage(
                        f"Previous title and subject: {prev_step.title}: {prev_step.subject.replace('\n', ' ')}"
                    )
                )
                previous_class_intros.append(
                    AIMessage(f"Previous intro: {prev_step.intro.replace('\n', ' ')}")
                )
                previous_class_tasks.append(
                    AIMessage(
                        f"Previous tasks: {', '.join([f'Title: {task.title}' for task in prev_step.structured.tasks])}"
                    )
                )

    if progress_cb is not None:
        progress_cb(
            0, "Generating step `" + step.title + "` preliminary content - "
        )
    doc_chain = get_rag_chain(
        journey.chroma_collection, "hyde_document", amount_of_documents=10
    )
    class_content = get_base_chain("step_content")(
        (subject.prompts.step_content.system, subject.prompts.step_content.user)
    ).invoke(
        {
            "context": doc_chain.invoke(
                {"question": subject_string, "context": content}
            )["answer"],
            "journey_instructions": journey.instructions,
            "subject_instructions": subject.instructions,
            "step_instructions": step.instructions,
            "subject": subject_string,
            "chat_history": previous_class_subjects,
        }
    )

    if isinstance(class_content, tuple) and len(class_content) == 2:
        class_content, _ = class_content

    if progress_cb is not None:
        progress_cb(0.1, "Generating step `" + step.title + "` modules - ")

    class_tasks = get_base_chain("step_tasks")(
        (subject.prompts.step_tasks.system, subject.prompts.step_tasks.user)
    ).invoke(
        {
            "context": class_content,
            "journey_instructions": journey.instructions,
            "subject_instructions": subject.instructions,
            "step_instructions": step.instructions,
            "subject": subject_string,
            "amount": task_amount,
            "chat_history": previous_class_subjects + previous_class_tasks,
        }
    )
    if isinstance(class_tasks, tuple) and len(class_tasks) == 2:
        class_tasks, _ = class_tasks
    gen_step: StepModel = None

    if isinstance(step, Step):
        gen_step = StepModel(
            title=step.title.strip(),
            subject=step.subject.strip(),
            content=(
                class_content.strip()
                if isinstance(class_content, str)
                else class_content.content.strip()
            ),
            intro="",  # class_intro.strip() if isinstance(class_intro, str) else class_intro.content.strip(),
            tasks=(
                class_tasks.strip()
                if isinstance(class_tasks, str)
                else class_tasks.content.strip()
            ),
        )
    else:
        gen_step = StepModel(
            title=step.title.strip(),
            subject=step.subject.strip(),
            content=class_content.strip(),
            intro="",  # class_intro.strip(),
            tasks=class_tasks.strip(),
        )

    if progress_cb is not None:
        progress_cb(
            0.2, "Generating step `" + step.title + "` structured format - "
        )

    json_step = llm_gen_json_step(gen_step)

    gen_step = llm_gen_update_tasks(journey, subject, gen_step, json_step)

    class_content = llm_gen_step_content(
        journey,
        subject,
        gen_step,
        progress_cb=progress_cb,
        progress_start=0.3,
        progress_end=0.85,
    )

    if isinstance(class_content, tuple) and len(class_content) == 2:
        class_content, _ = class_content

    gen_step.content = (
        class_content.strip()
        if isinstance(class_content, str)
        else class_content.content.strip()
    )
    gen_step.structured.content = gen_step.content

    if progress_cb is not None:
        progress_cb(0.9, "Generating step `" + step.title + "` intro - ")

    class_intro = get_base_chain("step_intro")(
        (subject.prompts.step_intro.system, subject.prompts.step_intro.user)
    ).invoke(
        {
            "context": gen_step.content,
            "journey_instructions": journey.instructions,
            "subject_instructions": subject.instructions,
            "step_instructions": step.instructions,
            "subject": subject_string,
            "chat_history": previous_class_intros,
        }
    )

    if isinstance(class_intro, tuple) and len(class_intro) == 2:
        class_intro, _ = class_intro

    gen_step.intro = (
        class_intro.strip()
        if isinstance(class_intro, str)
        else class_intro.content.strip()
    )
    gen_step.structured.intro = gen_step.intro

    return gen_step


def get_task_details_str(index: int, task: TaskStructure) -> str:
    return textwrap.dedent(
        f"""
        ## Task {index+1}: {task.title.strip()}:
        {task.description.strip()}
        """
        + (
            f"""
        ##### Resources:

        - {f"\n        - ".join([f"{resource.title}: {resource.summary.replace('\n', '')}" for resource in task.resources])}
        """
            if len(task.resources) > 0
            else ""
        )
        + f"""
        #### Test:
        {task.test.strip()}

        ---
        """
    )


def llm_gen_json_step(
    step: StepModel, instructions=""
) -> Union[StepStructure, None]:
    structured = get_chain("journey_structured").invoke(
        {
            "context": f"""
            Title:
            {step.title}
            Intro:
            {step.intro}
            Content:
            {step.content}
            Tasks:
            {step.tasks}
        """
            + (
                """
            Instructions:
            {instructions}
        """
                if instructions
                else ""
            ),
        }
    )

    return structured


def llm_gen_update_tasks(
    journey: JourneyModel,
    subject: SubjectModel,
    gen_step: StepModel,
    json_step: StepStructure,
    progress_cb: Callable[[float, str], None] = None,
    progress_start: float = 0,
    progress_end: float = 1,
) -> StepModel:
    if json_step is not None and isinstance(json_step, StepStructure):
        gen_step.structured = json_step
        json_step.title = gen_step.title
        json_step.subject = gen_step.subject
        json_step.intro = gen_step.intro
        json_step.content = gen_step.content

    return gen_step


def llm_gen_resource(
    journey: JourneyModel,
    subject: SubjectModel,
    step: StepModel,
    task: TaskStructure,
    resource: ResourceStructure,
) -> str:
    resource_text = f"{resource.title} (from: {resource.reference}) - {resource.summary.replace('\n', '')} for {task.title} in {step.title} of {subject.title} in {journey.title}"
    doc_chain = get_rag_chain(
        journey.chroma_collection, "hyde_document", amount_of_documents=5
    )

    class_task_details = get_base_chain("task_details")(
        (
            subject.prompts.task_details.system,
            subject.prompts.task_details.user,
        )
    ).invoke(
        {
            "context": doc_chain.invoke(
                {"question": resource_text, "context": step.content}
            )["answer"],
            "journey_instructions": journey.instructions,
            "subject_instructions": subject.instructions,
            "step_instructions": step.instructions,
            "resource": resource_text,
        }
    )
    if isinstance(class_task_details, tuple):
        class_task_details, _ = class_task_details
    if isinstance(class_task_details, BaseMessage):
        class_task_details = class_task_details.content

    return class_task_details


def llm_gen_step_content(
    journey: JourneyModel,
    subject: SubjectModel,
    step: Union[Step | StepModel],
    progress_cb: Callable[[float, str], None] = None,
    progress_start: float = 0,
    progress_end: float = 1,
) -> str:
    doc_chain = get_rag_chain(
        journey.chroma_collection, "hyde_document", amount_of_documents=5
    )
    subject_string = (
        f"Title: {step.title}\nSubject: {step.description}"
        if isinstance(step, Step)
        else f"Title: {step.title}\nSubject: {step.subject}"
    )
    content = f"{subject_string.strip()}\n\n{step.content.strip()}"
    if step.structured is not None and isinstance(step.structured, StepStructure):
        total_resources = len(step.structured.tasks)
        cur = 0
        for task in step.structured.tasks:
            if progress_cb is not None:
                progress_cb(
                    progress_start
                    + (cur / total_resources) * (progress_end - progress_start),
                    f"Generating step `{step.title}` task {cur+1} `{task.title}` subject content - ",
                )
            content += (
                "\n\n"
                + f"Subject: {task.title.strip()}\n\nSubject description: {task.description.strip()}"
            )
            content += (
                "\n\n"
                + "\n\nSubject content:"
                + doc_chain.invoke(
                    {
                        "question": f"Subject: {task.title.strip()}\n\nSubject description: {task.description.strip()}",
                        "context": step.content,
                    }
                )["answer"]
            )
            cur += 1
    else:
        content += "\n\n" + step.tasks.strip()

    if progress_cb is not None:
        progress_cb(
            progress_end, "Generating step `" + step.title + "` content - "
        )

    # TODO: add subject.instructions
    class_content = get_base_chain("step_content")(
        (subject.prompts.step_content.system, subject.prompts.step_content.user)
    ).invoke(
        {
            "context": content,
            "journey_instructions": journey.instructions,
            "subject_instructions": subject.instructions,
            "step_instructions": step.instructions,
            "subject": subject_string,
        }
    )
    if isinstance(class_content, tuple):
        class_content, _ = class_content
    if isinstance(class_content, BaseMessage):
        class_content = class_content.content

    return class_content


def llm_gen_journey_doc(list_of_strings=[]) -> tuple[str, str]:
    text = "\n".join(list_of_strings)

    # bar = st.progress(0, text="Compressing journey document")

    reduce = False

    reduce = len(text) > SETTINGS.default_llms.instruct.char_limit

    # print(f"{reduce = } ({len(text)})")

    if reduce:
        list_of_docs = create_document_lists(list_of_strings)

        chain = get_chain("stuff_documents")

        list_of_strings = []
        total = len(list_of_docs)
        for i, document in enumerate(list_of_docs):
            # bar.progress(i / total, text=f"Compressing page {i+1}/{total}")
            result = chain.invoke({"context": [document]})
            if isinstance(result, tuple):
                list_of_strings.append(result[0])
            else:
                list_of_strings.append(result)

        text = "\n".join(list_of_strings)

        reduce = len(text) > SETTINGS.default_llms.instruct.char_limit
        if reduce:
            # bar.progress(1 - 1/total, text="Result too long, 2nd pass")
            list_of_docs = create_document_lists(list_of_strings)
            text = chain.invoke(
                {
                    "context": list_of_docs,
                }
            )

        # bar.progress(1.0, text="Compression complete")

    # bar.empty()

    return text


def build_journey_doc_from_files(db_sources: Dict[str, SourceData]) -> str:
    list_of_strings = []
    for filename in db_sources:
        content = SourceContents(**db_sources[filename].source_contents.__dict__)
        if content.formatted_content is not None and content.formatted_content != "":
            list_of_strings.append(content.formatted_content)
        else:
            list_of_strings.append(
                markdown_to_text("\n".join(db_sources[filename].texts))
            )

    compressed = ""

    with st.spinner("Generating journey document"):
        compressed = llm_gen_journey_doc(list_of_strings)

    return compressed


def get_files_for_journey(
    default_category, journey_name, step, gen_from: Dict = None
) -> List[Any]:
    db_sources = get_db_sources()
    shown_files = {}

    if default_category is None or len(default_category) < 1:
        st.write("Select category tag(s) first to see available files.")
    else:
        for filename in db_sources.keys():
            category_tags = db_sources[filename].category_tag

            if len([i for i in category_tags if i in default_category]) > 0:
                shown_files[filename] = db_sources[filename]

    if gen_from is None:
        gen_from = {}

    # gen_from = st.session_state.gen_checklist_from

    if len(shown_files) > 0:
        st.write(f"Files:")
        for filename in shown_files.keys():
            file = shown_files[filename]

            select = st.checkbox(
                filename,
                value=filename in gen_from.keys(),
                key=f"select_for_journey_gen_{journey_name}_{step}_{filename}",
            )

            if select:
                gen_from[filename] = file

    return gen_from


def create_subject_prompt_editor(
    id: str, subject: SubjectModel, edit_mode: bool = True
) -> SubjectModel:
    def edit_prompt(i, prompt: CustomPrompt, container):
        with container:
            tab1, tab2 = st.tabs(["System", "User"])
            if edit_mode:
                tab1.write("System prompts don't take variables.")
                prompt.system = tab1.text_area(
                    "System prompt",
                    value=prompt.system.strip(),
                    height=300,
                    key=f"system_prompt_{i}_{id}",
                )
                tab2.write(
                    "User prompts take variables. Use {variable_name} to insert variables into the prompt."
                )
                prompt.user = tab2.text_area(
                    "User prompt",
                    value=prompt.user.strip(),
                    height=300,
                    key=f"user_prompt_{i}_{id}",
                )
            else:
                tab1.write(prompt.system.strip())
                tab2.write(prompt.user.strip())
        return prompt

    with st.expander(f"Prompts for {id}", expanded=False):

        tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "Prompt generator",
                "Plan steps",
                "Step\nIntro",
                "Step\n content",
                "Step\nModules",
                "Step\nModule Details",
                "Bulk",
            ]
        )
        try:
            with tab0:
                actor = st.text_input(
                    "Actor",
                    value="Teacher",
                    key=f"prompt_actor_{id}",
                    help="Actor string, e.g.: 'Teacher' or 'HR Representative'",
                )
                target = st.text_input(
                    "Use case",
                    value="Create a class curriculum for students.",
                    key=f"prompt_target_{id}",
                    help="Actor string, e.g.: 'Create a class curriculum for students.'",
                )

                if st.button("Generate new prompts", key=f"generate_prompts_{id}"):
                    with st.spinner("Generating prompts..."):
                        new_prompts = get_chain("journey_prompt_generator").invoke(
                            {"actor": actor, "target": target}
                        )
                        subject.prompts = convert_to_journey_prompts(new_prompts)

            subject.prompts.plan = edit_prompt(1, subject.prompts.plan, tab1)
            subject.prompts.step_intro = edit_prompt(
                2, subject.prompts.step_intro, tab2
            )
            subject.prompts.step_content = edit_prompt(
                3, subject.prompts.step_content, tab3
            )
            # subject.prompts.step_content_redo = edit_prompt(3, subject.prompts.step_content_redo, tab3)
            subject.prompts.step_tasks = edit_prompt(
                4, subject.prompts.step_tasks, tab4
            )
            subject.prompts.task_details = edit_prompt(
                5, subject.prompts.task_details, tab5
            )

            with tab6:
                if edit_mode:
                    yaml_prompts = yaml.dump(
                        subject.prompts.model_dump(), default_flow_style=False
                    )
                    yaml_prompts = st.text_area(
                        "Prompts as YAML",
                        value=yaml_prompts,
                        height=300,
                        key=f"yaml_prompts_{id}",
                    )

                    if st.button("Update prompts", key=f"update_prompts_{id}"):
                        update_subject_prompts(subject, yaml_prompts)
                        st.rerun()
        except Exception as e:
            st.error(f"An error occurred: {e}")

    return subject.prompts


# def separate_prompts(bulk_prompt) -> Dict[str, Dict[str, str]]:
#     prompts = {}
#     pattern = r'Prompt: (\w+)\s*--system--\s*(.*?)\s*--system--\s*--user--\s*(.*?)\s*--user--'
#     matches = re.findall(pattern, bulk_prompt, re.DOTALL)
#     for match in matches:
#         prompt_name = match[0]
#         system_prompt = match[1].strip()
#         user_prompt = match[2].strip()
#         prompts[prompt_name] = {'system': system_prompt, 'user': user_prompt}
#     return prompts


def update_subject_prompts(subject: SubjectModel, bulk: str):
    prompts = yaml.safe_load(bulk)
    subject.prompts.plan.system = prompts["plan"]["system"]
    subject.prompts.plan.user = prompts["plan"]["user"]
    subject.prompts.step_intro.system = prompts["step_intro"]["system"]
    subject.prompts.step_intro.user = prompts["step_intro"]["user"]
    subject.prompts.step_content.system = prompts["step_content"]["system"]
    subject.prompts.step_content.user = prompts["step_content"]["user"]
    subject.prompts.step_tasks.system = prompts["step_tasks"]["system"]
    subject.prompts.step_tasks.user = prompts["step_tasks"]["user"]
    subject.prompts.task_details.system = prompts["task_details"][
        "system"
    ]
    subject.prompts.task_details.user = prompts["task_details"]["user"]


async def gen_subject(
    journey: JourneyModel,
    subject: SubjectModel,
    subject_index: int = 0,
    step_index: int = None,
) -> SubjectModel:
    # journey:JourneyModel = st.session_state.journey_get_details[journey_name]
    # vectorstore = get_vectorstore("rag_"+ journey["category"][0], "hyde")
    with st.status(f"Building subject {subject_index+1} document"):
        compressed = build_journey_doc_from_files(subject.db_sources)
        st.success("Generating subject document done.")
    if step_index is None:
        with st.status(f"Building subject {subject_index+1}"):
            subject = await gen_subject(
                compressed,
                journey,
                subject,
                subject_index=subject_index,
            )
            st.success(f"Subject {subject_index+1} done.")
            subject.files = list(subject.db_sources.keys())
    else:
        with st.status(f"Building subject {subject_index+1} step {step_index+1}"):
            bar = st.progress(
                0,
                text=f"Generating subject {subject_index+1} step {step_index+1}",
            )
            start_time = time.time()  # Start time of the process
            average_time_per_portion = 0  # Running average of time per portion

            def progress_cb(progress: float, message: str):
                nonlocal average_time_per_portion
                # Update the running average
                if progress > 0:  # Avoid division by zero
                    average_time_per_portion = (
                        progress - 0.01
                    ) * average_time_per_portion + (time.time() - start_time) / progress
                else:
                    average_time_per_portion = time.time() - start_time

                # Calculate the estimated time to completion
                remaining_portions = 1 - progress
                estimated_time_to_completion = (
                    remaining_portions * average_time_per_portion
                )

                # Format the estimated time to completion as a string
                minutes, seconds = divmod(estimated_time_to_completion, 60)
                estimated_time_str = f"{int(minutes)}m {int(seconds)}s"
                bar.progress(progress, text=f"{message} - ETC {estimated_time_str}")

            existing_plan = subject.plan[:step_index] + subject.plan[step_index + 1 :]
            subject.plan[step_index] = llm_gen_step(
                compressed,
                journey,
                subject,
                subject.plan[step_index],
                existing_plan=existing_plan,
                progress_cb=progress_cb,
                task_amount=subject.task_amount or 5,
            )
            bar.progress(
                1.0, text=f"Subject {subject_index+1} step {step_index+1} done."
            )
            st.success(f"Subject {subject_index+1} step {step_index+1} done.")

    return subject


async def gen_subject(
    content, journey: JourneyModel, subject: SubjectModel, subject_index: int = None
) -> SubjectModel:
    bar = st.progress(0, text="Generating")

    bar.progress(0, text=f"Generating {subject_index+1} subject")
    step_items: Plan = llm_gen_plan(content, journey, subject)
    bar.progress(0.1, text="Generate plan...")

    plan: list[StepModel] = []
    start_time = time.time()  # Start time of the process
    average_time_per_portion = 0  # Running average of time per portion
    total_items = len(step_items.plan)
    for i, step in enumerate(step_items.plan):
        bar.progress(
            0.15 + (0.8 * i / total_items),
            text=f"Generating step {i+1} of {total_items}",
        )
        prog_start = 0.15 + (0.8 * i / total_items)
        prog_end = 0.15 + (0.8 * (i + 1) / total_items)
        prog_total = prog_end - prog_start

        def progress_cb(progress: float, message: str):
            nonlocal average_time_per_portion
            cur_progress = progress * prog_total + prog_start
            # Update the running average
            if progress > 0:  # Avoid division by zero
                average_time_per_portion = (
                    cur_progress - 0.01
                ) * average_time_per_portion + (time.time() - start_time) / cur_progress
            else:
                average_time_per_portion = time.time() - start_time

            # Calculate the estimated time to completion
            remaining_portions = 1 - cur_progress
            estimated_time_to_completion = remaining_portions * average_time_per_portion

            # Format the estimated time to completion as a string
            if estimated_time_to_completion > 5 or prog_end - cur_progress < 0.02:
                minutes, seconds = divmod(estimated_time_to_completion, 60)
                estimated_time_str = f"{int(minutes)}m {int(seconds)}s"
            else:
                estimated_time_str = "Calculating..."
            bar.progress(
                cur_progress,
                text=f"{message.replace('step', 'step '+str(i+1)+':')} ETC {estimated_time_str}",
            )

        existing_plan = plan + (
            step_items.plan[i + 1 :] if i < len(step_items.plan) - 1 else []
        )
        new_step = llm_gen_step(
            content,
            journey,
            subject,
            step,
            existing_plan=existing_plan,
            progress_cb=progress_cb,
            task_amount=subject.task_amount or 5,
        )
        plan.append(new_step)
    bar.progress(0.95, text="Generating title")

    title, summary = await llm_gen_title_summary(plan)
    bar.progress(1.0, text="Generation complete")

    bar.empty()
    # st.write(title)
    # st.write(summary)
    # st.write(instructions)
    subject.title = title
    subject.summary = summary
    subject.plan = plan
    return subject
