from datetime import datetime
import textwrap
import time
import streamlit as st
from typing import Any, Dict, List, Union
from langchain_core.runnables.base import RunnableSequence


from lib.chain import get_chain
from lib.db_tools import (
    CustomPrompt,
    JourneyDataTable,
    get_db_files,
    get_db_journey,
    init_db,
    JourneyModel,
    StepModel,
    SubjectModel,
)
from lib.document_parse import markdown_to_text
from lib.document_tools import create_document_lists, rag_chain
from lib.load_env import INSTRUCT_CHAR_LIMIT
from lib.prompts import JourneyStep, JourneyStepList, JourneyStructure
from lib.streamlit_tools import llm_edit

def save_journey(journey_name, journey:JourneyModel):
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
    database_session.add(journey_db)
    database_session.commit()
    get_db_journey(reset=True)


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

def llm_gen_title_summary(steps: List[Union[StepModel, SubjectModel]]) -> tuple[str, str]:
    if isinstance(steps[0], SubjectModel):
        context = "\n".join([f"Title: {subject.title}\n Summary:\n{subject.summary}\n" for subject in steps])
    else:
        context = "\n".join([f"Title: {step.title}\n Subject:\n{step.subject}\nIntroduction:\n{step.intro or ""}" for step in steps])
    title, title_thoughts = get_chain("action")().invoke(
            {
                "context": context,
                "action": "Summarize context with 10 words or less to a title",
            }
    )

    summary, summary_thoughts = llm_edit("summary", [context], "Summarize the following list of titles and intros into a summary description.", force=True)

    return title.strip(), summary.strip()

def llm_gen_steps(content, journey:JourneyModel, subject:SubjectModel) -> JourneyStepList:
    return get_chain("journey_steps")((subject.prompts.steps.system, subject.prompts.steps.user)).invoke(
        {
            "context": content,
            "amount": subject.step_amount,
            "journey_instructions": journey.instructions,
            "instructions": subject.instructions
        }
    )

def llm_gen_step(rag_chain:RunnableSequence, content, journey:JourneyModel, subject:SubjectModel, step:Union[JourneyStep|StepModel], amount=10) -> StepModel:
    class_content = None
    subject_string =  f"Title: {step.title}\nSubject: {step.description}" if isinstance(step, JourneyStep) else f"Title: {step.title}\nSubject: {step.subject}"
    retry_lambda = lambda: get_chain("journey_step_details")((subject.prompts.step_detail.system, subject.prompts.step_detail.user)).invoke(
            {
                "context": rag_chain.invoke({"question": subject_string, "context": content}),
                "journey_instructions": journey.instructions,
                "instructions": subject.instructions,
                "subject": subject_string,
            }
        )
    try:
        class_content, _ = retry_lambda()
    except Exception as e:
        print(f"Error generating class content: {e}")
        print("Retrying once")
        try:
            class_content, _ = retry_lambda()
        except Exception as e:
            print(f"Error generating class content: {e}")
            class_content = "Error generating class content"

    class_intro, _ = get_chain("journey_step_intro")((subject.prompts.step_intro.system, subject.prompts.step_intro.user)).invoke(
        {
            "context": class_content,
            "journey_instructions": journey.instructions,
            "instructions": subject.instructions,
            "subject":subject_string,
        }
    )
    class_actions, _ = get_chain("journey_step_actions")((subject.prompts.step_actions.system, subject.prompts.step_actions.user)).invoke(
        {
            "context": class_content,
            "journey_instructions": journey.instructions,
            "instructions": subject.instructions,
            "subject": subject_string,
            "amount": amount
        }
    )
    gen_step:StepModel = None

    if isinstance(step, JourneyStep):
        gen_step = StepModel(
            title= step.title.strip(),
            subject= step.description.strip(),
            content= class_content.strip(),
            intro= class_intro.strip(),
            actions= class_actions.strip(),
        )
    else:
        gen_step = step
        gen_step.content = class_content.strip()
        gen_step.intro = class_intro.strip()
        gen_step.actions = class_actions.strip()

    gen_step.structured = llm_gen_json_step(gen_step)
    if gen_step.structured is not None:
        gen_step.title = gen_step.structured.title
        gen_step.subject = gen_step.structured.subject
        gen_step.intro = gen_step.structured.intro
        gen_step.content = gen_step.structured.content
        gen_step.actions = "\n".join([ textwrap.dedent(f"""
            ## Action {i+1}: {action.title}:
            {action.description}

            ##### Resources:

            - {f"\n - ".join(action.resources)}

            #### Test:
            {action.test}

            ---
            """) for i, action in enumerate(gen_step.structured.actions)])


    return gen_step

def llm_gen_json_step(step: StepModel, instructions="") -> Union[JourneyStructure, None]:
    retry_lambda = lambda: get_chain("journey_structured")().invoke({
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
    structured: JourneyStructure = None
    try:
        structured = retry_lambda()
    except Exception as e:
        try:
            structured = retry_lambda()
        except Exception as e:
            print(e)

    return structured

def llm_gen_journey_doc(list_of_strings = []) -> tuple[str, str]:
    text = "\n".join(list_of_strings)
    list_of_thoughts = []
    thoughts = ''

    # bar = st.progress(0, text="Compressing journey document")

    reduce = False

    reduce = len(text) > INSTRUCT_CHAR_LIMIT

    # print(f"{reduce = } ({len(text)})")

    if reduce:
        list_of_docs = create_document_lists(list_of_strings)

        chain = get_chain("reduce_journey_documents")

        list_of_strings = []
        list_of_thoughts = []
        total = len(list_of_docs)
        for i, document in enumerate(list_of_docs):
            # bar.progress(i / total, text=f"Compressing page {i+1}/{total}")
            result, thinking = chain().invoke({"context": [document]})
            list_of_strings.append(result[0])
            list_of_thoughts.append(result[1])

        text = "\n".join(list_of_strings)
        thoughts = "\n".join(list_of_thoughts)

        reduce = len(text) > INSTRUCT_CHAR_LIMIT
        if reduce:
            # bar.progress(1 - 1/total, text="Result too long, 2nd pass")
            list_of_docs = create_document_lists(list_of_strings, list_of_thoughts)
            text, thoughts = chain().invoke(
                    {
                        "context": list_of_docs,
                    }
                )
        # bar.progress(1.0, text="Compression complete")

    # bar.empty()

    return text, thoughts

def build_journey_doc_from_files(db_files: Dict[str, Any]) -> str:
    list_of_strings = []
    for filename in db_files:
        if (
            db_files[filename]["formatted_text"] is not None
            and db_files[filename]["formatted_text"] != ""
        ):
            list_of_strings.append(db_files[filename]["formatted_text"])
        else:
            list_of_strings.append(
                markdown_to_text("\n".join(db_files[filename]["texts"]))
            )

    compressed = ''

    with st.spinner("Generating journey document"):
        compressed, compress_thoughts = llm_gen_journey_doc(list_of_strings)

    return compressed

def get_files_for_journey(
    default_category, journey_name, step, gen_from: Dict = None
) -> List[Any]:
    db_files = get_db_files()
    shown_files = {}

    if default_category is None or len(default_category) < 1:
        st.write("Select category tag(s) first to see available files.")
    else:
        for filename in db_files.keys():
            category_tags = db_files[filename]["category_tag"]

            if len([i for i in category_tags if i in default_category]) > 0:
                shown_files[filename] = db_files[filename]

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
                tab2.write("User prompts take variables. Use {{variable_name}} to insert variables into the prompt.")
                prompt.user = tab2.text_area("User prompt", value=prompt.user.strip(), height=300, key=f"user_prompt_{i}_{id}")
            else:
                tab1.write(prompt.system.strip())
                tab2.write(prompt.user.strip())
        return prompt

    with st.expander(f"Prompts for {id}", expanded=True):
        tab1, tab2, tab3, tab4 = st.tabs(["Steps", "Step Intro", "Step Detail", "Step Actions"])
        subject.prompts.steps = edit_prompt(1, subject.prompts.steps, tab1)
        subject.prompts.step_detail = edit_prompt(2, subject.prompts.step_detail, tab2)
        subject.prompts.step_intro = edit_prompt(3, subject.prompts.step_intro, tab3)
        subject.prompts.step_actions = edit_prompt(4, subject.prompts.step_actions, tab4)

    return subject.prompts

def gen_journey_subject(journey: JourneyModel, subject: SubjectModel, step_index: int = None) -> SubjectModel:
    # journey:JourneyModel = st.session_state.journey_get_details[journey_name]
    # vectorstore = get_vectorstore("rag_"+ journey["category"][0], "hyde")
    doc_chain = rag_chain(journey.chroma_collection[0], "hyde")
    with st.status(f"Building subject document"):
        compressed = build_journey_doc_from_files(subject.db_files)
        st.success("Generating subject document done.")
    if step_index is None:
        with st.status(f"Building subject"):
            subject = gen_subject(
                doc_chain,
                compressed,
                journey,
                subject
            )
            st.success("Generating subject done.")
            subject.files = list(subject.db_files.keys())
    else:
        with st.status(f"Building subject step {step_index+1}"):
            subject.steps[step_index] = llm_gen_step(
                doc_chain, compressed, journey, subject, subject.steps[step_index]
            )
            st.success(f"Generating subject step {step_index+1} done.")

    return subject

def gen_subject(
    rag_chain: RunnableSequence,
    content,
    journey:JourneyModel,
    subject:SubjectModel,
) -> SubjectModel:
    bar = st.progress(0, text="Generating")

    bar.progress(0, text="Generating subject")
    step_items: JourneyStepList = llm_gen_steps(
        content, journey, subject
    )
    bar.progress(0.1, text="Generate subjects...")

    steps: list[StepModel] = []

    total_items = len(step_items.steps)
    for i, step in enumerate(step_items.steps):
        bar.progress(
            0.35 + (0.6 * i / total_items),
            text=f"Generating step {i+1} of {total_items}",
        )
        new_step = llm_gen_step(
            rag_chain, content, journey, subject, step
        )
        steps.append(new_step)
    bar.progress(0.95, text="Generating title")

    title, summary = llm_gen_title_summary(steps)
    bar.progress(1.0, text="Generation complete")

    bar.empty()
    # st.write(title)
    # st.write(summary)
    # st.write(instructions)
    subject.title = title
    subject.summary = summary
    subject.steps = steps
    return subject
