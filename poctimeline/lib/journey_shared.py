from datetime import datetime
import re
import textwrap
import time
import streamlit as st
from typing import Any, Callable, Dict, List, Union
from langchain_core.messages import BaseMessage, AIMessage

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
from lib.models.journey import ActionStructure, JourneyModel, ResourceStructure, StepModel, SubjectModel, SubjectStructure
from lib.models.prompts import CustomPrompt
from lib.models.sqlite_tables import SourceContents, SourceData, JourneyDataTable
from lib.prompts.journey import JourneyStep, JourneyStepList

def save_journey(journey_name, journey:JourneyModel) -> bool:
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
        chroma_collection = journey.chroma_collection
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

async def llm_gen_title_summary(steps: List[Union[StepModel, SubjectModel]]) -> tuple[str, str]:
    if isinstance(steps[0], SubjectModel):
        context = "\n".join([f"Title: {subject.title}\n Summary:\n{subject.summary}\n" for subject in steps])
    else:
        context = "\n".join([f"Title: {step.title}\n Subject:\n{step.subject}\nIntroduction:\n{step.intro or ""}" for step in steps])
    title = get_chain("action").invoke(
            {
                "context": context,
                "action": "Summarize context with 10 words or less to a title",
            }
    )

    summary = await llm_edit([context], "Summarize the following list of titles and intros into a summary description.", force=True)

    return title.strip(), summary.strip()

def llm_gen_steps(content, journey:JourneyModel, subject:SubjectModel) -> JourneyStepList:
    return get_base_chain("journey_steps")((subject.prompts.steps.system, subject.prompts.steps.user)).invoke(
        {
            "context": content,
            "amount": subject.step_amount,
            "journey_instructions": journey.instructions,
            "instructions": subject.instructions
        }
    )

def llm_gen_step(content, journey:JourneyModel, subject:SubjectModel, step:Union[JourneyStep|StepModel], prev_steps:List[Union[JourneyStep|StepModel]] = None, action_amount=10, progress_cb:Callable[[float, str], None]=None) -> StepModel:
    class_content = None
    subject_string = f"Title: {step.title}\nSubject: {step.description}" if isinstance(step, JourneyStep) else f"Title: {step.title}\nSubject: {step.subject}"
    previous_class_subjects = []
    previous_class_intros = []
    previous_class_actions = []
    if prev_steps != None and len(prev_steps) > 0:
        for prev_step in prev_steps:
            if isinstance(prev_step, JourneyStep):
                previous_class_subjects.append(AIMessage(f"Previous title and subject: {prev_step.title}: {prev_step.description.replace('\n', ' ')}"))
            else:
                previous_class_subjects.append(AIMessage(f"Previous title and subject: {prev_step.title}: {prev_step.subject.replace('\n', ' ')}"))
                previous_class_intros.append(AIMessage(f"Previous intro: {prev_step.intro.replace("\n", ' ')}"))
                previous_class_actions.append(AIMessage(f"Previous actions: {", ".join([f"Title: {action.title}" for action in prev_step.structured.actions])}"))

    if progress_cb is not None:
        progress_cb(0, "Generating subsection `"+step.title+"` preliminary content - ")
    doc_chain = get_rag_chain(journey.chroma_collection, "hyde_document", amount_of_documents=10)
    class_content = get_base_chain("journey_step_content")((subject.prompts.step_content.system, subject.prompts.step_content.user)).invoke(
            {
                "context": doc_chain.invoke({"question": subject_string, "context": content})["answer"],
                "journey_instructions": journey.instructions,
                "instructions": subject.instructions,
                "subject": subject_string,
                "chat_history": previous_class_subjects
            }
        )

    if isinstance(class_content, tuple) and len(class_content) == 2:
        class_content, _ = class_content

    if progress_cb is not None:
        progress_cb(0.1, "Generating subsection `"+step.title+"` modules - ")

    class_actions = get_base_chain("journey_step_actions")((subject.prompts.step_actions.system, subject.prompts.step_actions.user)).invoke(
        {
            "context": class_content,
            "journey_instructions": journey.instructions,
            "instructions": subject.instructions,
            "subject": subject_string,
            "amount": action_amount,
            "chat_history": previous_class_subjects + previous_class_actions
        }
    )
    if isinstance(class_actions, tuple) and len(class_actions) == 2:
        class_actions, _ = class_actions
    gen_step:StepModel = None

    if isinstance(step, JourneyStep):
        gen_step = StepModel(
            title = step.title.strip(),
            subject = step.description.strip(),
            content = class_content.strip() if isinstance(class_content, str) else class_content.content.strip(),
            intro = '', #class_intro.strip() if isinstance(class_intro, str) else class_intro.content.strip(),
            actions = class_actions.strip() if isinstance(class_actions, str) else class_actions.content.strip(),
        )
    else:
        gen_step = StepModel(
            title = step.title.strip(),
            subject = step.subject.strip(),
            content = class_content.strip(),
            intro = '', #class_intro.strip(),
            actions = class_actions.strip(),
        )

    if progress_cb is not None:
        progress_cb(0.2, "Generating subsection `"+step.title+"` structured format - ")

    json_step = llm_gen_json_step(gen_step)

    gen_step = llm_gen_update_actions(journey, subject, gen_step, json_step)

    class_content = llm_gen_step_content(journey, subject, gen_step, progress_cb=progress_cb, progress_start=0.3, progress_end=0.85)

    if isinstance(class_content, tuple) and len(class_content) == 2:
        class_content, _ = class_content

    gen_step.content = class_content.strip() if isinstance(class_content, str) else class_content.content.strip()
    gen_step.structured.content = gen_step.content

    if progress_cb is not None:
        progress_cb(0.9, "Generating subsection `"+step.title+"` intro - ")

    class_intro = get_base_chain("journey_step_intro")((subject.prompts.step_intro.system, subject.prompts.step_intro.user)).invoke(
        {
            "context": gen_step.content,
            "journey_instructions": journey.instructions,
            "instructions": subject.instructions,
            "subject":subject_string,
            "chat_history": previous_class_intros
        }
    )

    if isinstance(class_intro, tuple) and len(class_intro) == 2:
        class_intro, _ = class_intro

    gen_step.intro = class_intro.strip() if isinstance(class_intro, str) else class_intro.content.strip()
    gen_step.structured.intro = gen_step.intro

    return gen_step

def get_action_details_str(index:int, action:ActionStructure) -> str:
    return textwrap.dedent( f"""
        ## Action {index+1}: {action.title.strip()}:
        {action.description.strip()}
        """ + (f"""
        ##### Resources:

        - {f"\n        - ".join([f"{resource.title}: {resource.summary.replace('\n', '')}" for resource in action.resources])}
        """ if len(action.resources) > 0 else "") + f"""
        #### Test:
        {action.test.strip()}

        ---
        """)

def llm_gen_json_step(step: StepModel, instructions="") -> Union[SubjectStructure, None]:
    structured = get_chain("journey_structured").invoke({
        "context": f"""
            Title:
            {step.title}
            Intro:
            {step.intro}
            Content:
            {step.content}
            Actions:
            {step.actions}
        """ + ("""
            Instructions:
            {instructions}
        """ if instructions else ""),
    })

    return structured

def llm_gen_update_actions(journey:JourneyModel, subject: SubjectModel, gen_step:StepModel, json_step: SubjectStructure, progress_cb:Callable[[float, str], None]=None, progress_start: float=0, progress_end:float=1) -> StepModel:
    if json_step is not None and isinstance(json_step, SubjectStructure):
        gen_step.structured = json_step
        json_step.title = gen_step.title
        json_step.subject = gen_step.subject
        json_step.intro = gen_step.intro
        json_step.content = gen_step.content

    return gen_step

def llm_gen_resource(journey:JourneyModel, subject: SubjectModel, step:StepModel, action:ActionStructure, resource:ResourceStructure) -> str:
    resource_text = f"{resource.title} (from: {resource.reference}) - {resource.summary.replace('\n', '')} for {action.title} in {step.title} of {subject.title} in {journey.title}"
    doc_chain = get_rag_chain(journey.chroma_collection, "hyde_document", amount_of_documents=5)

    class_action_details = get_base_chain("journey_step_action_details")((subject.prompts.step_action_details.system, subject.prompts.step_action_details.user)).invoke(
        {
            "context": doc_chain.invoke({"question": resource_text, "context": step.content})["answer"],
            "journey_instructions": journey.instructions,
            "instructions": subject.instructions,
            "resource": resource_text,
        }
    )
    if isinstance(class_action_details, tuple):
        class_action_details, _ = class_action_details
    if isinstance(class_action_details, BaseMessage):
        class_action_details = class_action_details.content

    return class_action_details

def llm_gen_step_content(journey:JourneyModel, subject:SubjectModel, step:Union[JourneyStep|StepModel], progress_cb:Callable[[float, str], None]=None, progress_start: float=0, progress_end:float=1) -> str:
    doc_chain = get_rag_chain(journey.chroma_collection, "hyde_document", amount_of_documents=5)
    subject_string = f"Title: {step.title}\nSubject: {step.description}" if isinstance(step, JourneyStep) else f"Title: {step.title}\nSubject: {step.subject}"
    content = f"{subject_string.strip()}\n\n{step.content.strip()}"
    if step.structured is not None and isinstance(step.structured, SubjectStructure):
        total_resources = len(step.structured.actions)
        cur = 0
        for action in step.structured.actions:
            if progress_cb is not None:
                progress_cb(progress_start + (cur / total_resources) * (progress_end - progress_start), f"Generating subsection `{step.title}` action {cur+1} `{action.title}` section content - ")
            content += "\n\n" + f"Section: {action.title.strip()}\n\nSection description: {action.description.strip()}"
            content += "\n\n" + "\n\nSection content:" + doc_chain.invoke({"question": f"Section: {action.title.strip()}\n\nSection description: {action.description.strip()}", "context": step.content})["answer"]
            cur += 1
    else:
        content += "\n\n" + step.actions.strip()

    if progress_cb is not None:
        progress_cb(progress_end, "Generating subsection `"+step.title+"` content - ")

    # TODO: add subject.instructions
    class_content = get_base_chain("journey_step_content_redo")((subject.prompts.step_content_redo.system, subject.prompts.step_content_redo.user)).invoke(
        {
                "context": content,
                "journey_instructions": journey.instructions,
                "instructions": subject.instructions,
                "subject": subject_string
        }
    )
    if isinstance(class_content, tuple):
        class_content, _ = class_content
    if isinstance(class_content, BaseMessage):
        class_content = class_content.content

    return class_content

def llm_gen_journey_doc(list_of_strings = []) -> tuple[str, str]:
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
        if (
            content.formatted_content is not None
            and content.formatted_content != ""
        ):
            list_of_strings.append(content.formatted_content)
        else:
            list_of_strings.append(
                markdown_to_text("\n".join(db_sources[filename].texts))
            )

    compressed = ''

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

def create_subject_prompt_editor(id:str, subject: SubjectModel, edit_mode: bool = True) -> SubjectModel:
    def edit_prompt(i, prompt:CustomPrompt, container):
        with container:
            tab1, tab2 = st.tabs(["System", "User"])
            if edit_mode:
                tab1.write("System prompts don't take variables.")
                prompt.system = tab1.text_area("System prompt", value=prompt.system.strip(), height=300, key=f"system_prompt_{i}_{id}")
                tab2.write("User prompts take variables. Use {variable_name} to insert variables into the prompt.")
                prompt.user = tab2.text_area("User prompt", value=prompt.user.strip(), height=300, key=f"user_prompt_{i}_{id}")
            else:
                tab1.write(prompt.system.strip())
                tab2.write(prompt.user.strip())
        return prompt

    with st.expander(f"Prompts for {id}", expanded=False):
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Subsections", "Subsection\nIntro", "Subsection\nPrelim content", "Subsection\nFinal content", "Subsection\nModules", "Subsection\nModule Details", "Bulk"])
        try:
            subject.prompts.steps = edit_prompt(1, subject.prompts.steps, tab1)
            subject.prompts.step_content = edit_prompt(2, subject.prompts.step_content, tab2)
            subject.prompts.step_content_redo = edit_prompt(3, subject.prompts.step_content_redo, tab3)
            subject.prompts.step_intro = edit_prompt(4, subject.prompts.step_intro, tab4)
            subject.prompts.step_actions = edit_prompt(5, subject.prompts.step_actions, tab5)
            subject.prompts.step_action_details = edit_prompt(6, subject.prompts.step_action_details, tab6)

            with tab7:
                if edit_mode:
                    bulk_prompt = (
                        "Prompt: steps\n"
                        "--system--\n"
                        f"{subject.prompts.steps.system}\n"
                        "--system--\n"
                        "--user--\n"
                        f"{subject.prompts.steps.user}\n"
                        "--user--\n\n"

                        "Prompt: step_intro\n"
                        "--system--\n"
                        f"{subject.prompts.step_intro.system}\n"
                        "--system--\n"
                        "--user--\n"
                        f"{subject.prompts.step_intro.user}\n"
                        "--user--\n\n"

                        "Prompt: step_content\n"
                        "--system--\n"
                        f"{subject.prompts.step_content.system}\n"
                        "--system--\n"
                        "--user--\n"
                        f"{subject.prompts.step_content.user}\n"
                        "--user--\n\n"

                        "Prompt: step_content_redo\n"
                        "--system--\n"
                        f"{subject.prompts.step_content_redo.system}\n"
                        "--system--\n"
                        "--user--\n"
                        f"{subject.prompts.step_content_redo.user}\n"
                        "--user--\n\n"

                        "Prompt: step_actions\n"
                        "--system--\n"
                        f"{subject.prompts.step_actions.system}\n"
                        "--system--\n"
                        "--user--\n"
                        f"{subject.prompts.step_actions.user}\n"
                        "--user--\n\n"

                        "Prompt: step_action_details\n"
                        "--system--\n"
                        f"{subject.prompts.step_action_details.system}\n"
                        "--system--\n"
                        "--user--\n"
                        f"{subject.prompts.step_action_details.user}\n"
                        "--user--"
                    )
                    bulk = st.text_area("Bulk prompt", value=bulk_prompt, height=300, key=f"bulk_prompt_{id}")

                    if st.button("Update prompts", key=f"update_prompts_{id}"):
                        update_subject_prompts(subject, bulk)
                        st.rerun()
        except Exception as e:
            st.error(f"An error occurred: {e}")

    return subject.prompts

def separate_prompts(bulk_prompt) -> Dict[str, Dict[str, str]]:
    prompts = {}
    pattern = r'Prompt: (\w+)\s*--system--\s*(.*?)\s*--system--\s*--user--\s*(.*?)\s*--user--'
    matches = re.findall(pattern, bulk_prompt, re.DOTALL)
    for match in matches:
        prompt_name = match[0]
        system_prompt = match[1].strip()
        user_prompt = match[2].strip()
        prompts[prompt_name] = {'system': system_prompt, 'user': user_prompt}
    return prompts

def update_subject_prompts(subject: SubjectModel, bulk: str):
    prompts = separate_prompts(bulk)
    subject.prompts.steps.system = prompts['steps']['system']
    subject.prompts.steps.user = prompts['steps']['user']
    subject.prompts.step_intro.system = prompts['step_intro']['system']
    subject.prompts.step_intro.user = prompts['step_intro']['user']
    subject.prompts.step_content.system = prompts['step_content']['system']
    subject.prompts.step_content.user = prompts['step_content']['user']
    subject.prompts.step_actions.system = prompts['step_actions']['system']
    subject.prompts.step_actions.user = prompts['step_actions']['user']
    subject.prompts.step_action_details.system = prompts['step_action_details']['system']
    subject.prompts.step_action_details.user = prompts['step_action_details']['user']

async def gen_journey_subject(journey: JourneyModel, subject: SubjectModel, subject_index: int = 0, step_index: int = None) -> SubjectModel:
    # journey:JourneyModel = st.session_state.journey_get_details[journey_name]
    # vectorstore = get_vectorstore("rag_"+ journey["category"][0], "hyde")
    with st.status(f"Building section {subject_index+1} document"):
        compressed = build_journey_doc_from_files(subject.db_sources)
        st.success("Generating section document done.")
    if step_index is None:
        with st.status(f"Building section {subject_index+1}"):
            subject = await gen_subject(
                compressed,
                journey,
                subject,
                subject_index=subject_index,            )
            st.success(f"Section {subject_index+1} done.")
            subject.files = list(subject.db_sources.keys())
    else:
        with st.status(f"Building section {subject_index+1} subsection {step_index+1}"):
            bar = st.progress(0, text=f"Generating section {subject_index+1} subsection {step_index+1}")
            start_time = time.time()  # Start time of the process
            average_time_per_portion = 0  # Running average of time per portion
            def progress_cb(progress: float, message:str):
                nonlocal average_time_per_portion
                # Update the running average
                if progress > 0:  # Avoid division by zero
                    average_time_per_portion = ((progress - 0.01) * average_time_per_portion + (time.time() - start_time) / progress)
                else:
                    average_time_per_portion = (time.time() - start_time)

                # Calculate the estimated time to completion
                remaining_portions = 1 - progress
                estimated_time_to_completion = remaining_portions * average_time_per_portion

                # Format the estimated time to completion as a string
                minutes, seconds = divmod(estimated_time_to_completion, 60)
                estimated_time_str = f"{int(minutes)}m {int(seconds)}s"
                bar.progress(progress , text=f"{message} - ETC {estimated_time_str}")
            prev_steps = subject.steps[:step_index] + subject.steps[step_index+1:]
            subject.steps[step_index] = llm_gen_step(
                compressed, journey, subject, subject.steps[step_index], prev_steps=prev_steps, progress_cb=progress_cb, action_amount=subject.action_amount or 5
            )
            bar.progress(1.0, text=f"Section {subject_index+1} subsection {step_index+1} done.")
            st.success(f"Section {subject_index+1} subsection {step_index+1} done.")

    return subject

async def gen_subject(
    content,
    journey:JourneyModel,
    subject:SubjectModel,
    subject_index: int = None
) -> SubjectModel:
    bar = st.progress(0, text="Generating")

    bar.progress(0, text=f"Generating {subject_index+1} section")
    step_items: JourneyStepList = llm_gen_steps(
        content, journey, subject
    )
    bar.progress(0.1, text="Generate subsections...")

    steps: list[StepModel] = []
    start_time = time.time()  # Start time of the process
    average_time_per_portion = 0  # Running average of time per portion
    total_items = len(step_items.steps)
    for i, step in enumerate(step_items.steps):
        bar.progress(
            0.15 + (0.8 * i / total_items),
            text=f"Generating subsection {i+1} of {total_items}",
        )
        prog_start = 0.15 + (0.8 * i / total_items)
        prog_end = 0.15 + (0.8 * (i+1) / total_items)
        prog_total = prog_end - prog_start

        def progress_cb(progress: float, message:str):
            nonlocal average_time_per_portion
            cur_progress = progress * prog_total + prog_start
            # Update the running average
            if progress > 0:  # Avoid division by zero
                average_time_per_portion = ((cur_progress - 0.01) * average_time_per_portion + (time.time() - start_time) / cur_progress)
            else:
                average_time_per_portion = (time.time() - start_time)

            # Calculate the estimated time to completion
            remaining_portions = 1 - cur_progress
            estimated_time_to_completion = remaining_portions * average_time_per_portion

            # Format the estimated time to completion as a string
            if estimated_time_to_completion > 5 or prog_end - cur_progress < 0.02:
                minutes, seconds = divmod(estimated_time_to_completion, 60)
                estimated_time_str = f"{int(minutes)}m {int(seconds)}s"
            else:
                estimated_time_str = "Calculating..."
            bar.progress(cur_progress, text=f"{message.replace('step', 'step '+str(i+1)+':')} ETC {estimated_time_str}")

        prev_steps = steps + (step_items.steps[i+1:] if i < len(step_items.steps) - 1 else [])
        new_step = llm_gen_step(
            content, journey, subject, step, prev_steps=prev_steps, progress_cb=progress_cb, action_amount=subject.action_amount or 5
        )
        steps.append(new_step)
    bar.progress(0.95, text="Generating title")

    title, summary = await llm_gen_title_summary(steps)
    bar.progress(1.0, text="Generation complete")

    bar.empty()
    # st.write(title)
    # st.write(summary)
    # st.write(instructions)
    subject.title = title
    subject.summary = summary
    subject.steps = steps
    return subject
