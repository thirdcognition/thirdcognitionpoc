from datetime import datetime
import time
from groq import BaseModel
from pydantic import Field
import streamlit as st
from typing import Any, Dict, List, Optional, Union
from langchain_core.runnables.base import RunnableSequence
from lib.chain import get_chain
from lib.db_tools import (
    JourneyDataTable,
    get_db_journey,
    init_db,
    JourneyModel,
    StepModel,
    SubjectModel,
)
from lib.document_parse import markdown_to_text
from lib.document_tools import create_document_lists
from lib.load_env import INSTRUCT_CHAR_LIMIT
from lib.prompts import JourneyStep, JourneyStepList, JourneyStructure
from lib.streamlit_tools import llm_edit




def save_journey(journey_name, journey:JourneyModel):
    print(f"Save journey {journey_name}")
    st.write(f"Save journey {journey_name}")
    st.write(journey)
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
    st.session_state.editing_journey = None
    st.session_state.editing_journey_details = None
    st.session_state.editing_journey_subject = None
    st.session_state.editing_journey_step = None
    time.sleep(0.1)
    st.rerun()

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
    title, title_thoughts = get_chain("action").invoke(
            {
                "context": context,
                "action": "Summarize context with 10 words or less to a title",
            }
    )

    summary, summary_thoughts = llm_edit("summary", [context], "Summarize the following list of titles and intros into a summary description.", force=True)

    return title.strip(), summary.strip()

def llm_gen_steps(content, amount=5, journey_instructions="", instructions="") -> JourneyStepList:
    return get_chain("journey_steps").invoke(
        {
            "context": content,
            "amount": amount,
            "journey_instructions": journey_instructions,
            "instructions": instructions
        }
    )

def llm_gen_step(step:JourneyStep, rag_chain:RunnableSequence, content, journey_instructions="", instructions="", amount=10) -> StepModel:
    class_content = None
    retry_lambda = lambda: get_chain("journey_step_details").invoke(
            {
                "context": rag_chain.invoke({"question": f"{step.title}\n{step.description}", "context": content}),
                "journey_instructions": journey_instructions,
                "instructions": instructions,
                "subject": f"Title: {step.title}\nSubject: {step.description}",
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

    class_intro, _ = get_chain("journey_step_intro").invoke(
        {
            "context": class_content,
            "journey_instructions": journey_instructions,
            "instructions": instructions,
            "subject": f"Title: {step.title}\nSubject: {step.description}",
        }
    )
    class_actions, _ = get_chain("journey_step_actions").invoke(
        {
            "context": class_content,
            "journey_instructions": journey_instructions,
            "instructions": instructions,
            "subject": f"Title: {step.title}\nSubject: {step.description}",
            "amount": amount
        }
    )
    gen_step = StepModel(
        title= step.title.strip(),
        subject= step.description.strip(),
        content= class_content.strip(),
        intro= class_intro.strip(),
        actions= class_actions.strip()
    )
    gen_step.structured = llm_gen_json_step(gen_step)

    return gen_step

def llm_gen_json_step(step: StepModel) -> Union[JourneyStructure, None]:
    retry_lambda = lambda: get_chain("journey_structured").invoke({
        "context": f"""
            Title:
            {step.title}
            Intro:
            {step.intro}
            Content:
            {step.content}
            Actions:
            {step.actions}
        """
    })
    structured = None
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
            result, thinking = chain.invoke({"context": [document]})
            list_of_strings.append(result[0])
            list_of_thoughts.append(result[1])

        text = "\n".join(list_of_strings)
        thoughts = "\n".join(list_of_thoughts)

        reduce = len(text) > INSTRUCT_CHAR_LIMIT
        if reduce:
            # bar.progress(1 - 1/total, text="Result too long, 2nd pass")
            list_of_docs = create_document_lists(list_of_strings, list_of_thoughts)
            text, thoughts = chain.invoke(
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