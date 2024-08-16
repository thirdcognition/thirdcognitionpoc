import time
from typing import Any, Dict, List

import streamlit as st
from langchain_core.runnables.base import RunnableSequence

from pages.shared.journey import save_journey # type: ignore
from lib.db_tools import (
    get_db_files,
    get_db_journey,
    init_db,
)

from lib.document_parse import markdown_to_text
from lib.load_env import (
    INSTRUCT_CHAR_LIMIT,
    CLIENT_HOST,
)

# from rapidocr_paddle import RapidOCR  # type: ignore
from lib.chain import (
    get_chain,
)
from lib.document_tools import create_document_lists, rag_chain
from lib.prompts import JourneyStepList

from lib.streamlit_tools import check_auth, get_all_categories, llm_edit

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

def gen_journey_doc(list_of_strings = []) -> tuple[str, str]:
    text = "\n".join(list_of_strings)
    list_of_thoughts = []
    thoughts = ''

    bar = st.progress(0, text="Compressing journey document")

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
            bar.progress(i / total, text=f"Compressing page {i+1}/{total}")
            result, thinking = chain.invoke({"context": [document]})
            list_of_strings.append(result[0])
            list_of_thoughts.append(result[1])

        text = "\n".join(list_of_strings)
        thoughts = "\n".join(list_of_thoughts)

        reduce = len(text) > INSTRUCT_CHAR_LIMIT
        if reduce:
            bar.progress(1 - 1/total, text="Result too long, 2nd pass")
            list_of_docs = create_document_lists(list_of_strings, list_of_thoughts)
            text, thoughts = chain.invoke(
                    {
                        "context": list_of_docs,
                    }
                )
        bar.progress(1.0, text="Compression complete")

    bar.empty()

    return text, thoughts

def gen_subject(rag_chain:RunnableSequence, content, journey_instructions="", instructions="", amount=10) -> Dict:
    bar = st.progress(0, text="Generating curriculum")

    bar.progress(0, text="Generate subjects for each day")
    step_items:JourneyStepList = get_chain("journey_steps").invoke(
        {
            "context": content,
            "amount": amount,
            "journey_instructions": journey_instructions,
            "instructions": instructions
        }
    )
    bar.progress(0.1, text="Generate subjects...")

    steps = []

    total_items = len(step_items.steps)
    for i, step in enumerate(step_items.steps):
        bar.progress(0.35 + (0.6 * i/total_items), text=f"Generating curriculum for step {i+1} of {total_items}")
        class_content, _ = get_chain("journey_step_details").invoke(
            {
                "context": rag_chain.invoke({"question": f"{step.title}\n{step.description}", "context": content}),
                "journey_instructions": journey_instructions,
                "subject": f"Title: {step.title}\nSubject: {step.description}",
            }
        )
        class_intro, _ = get_chain("journey_step_intro").invoke(
            {
                "context": class_content,
                "journey_instructions": journey_instructions,
                "subject": f"Title: {step.title}\nSubject: {step.description}",
            }
        )
        class_actions, _ = get_chain("journey_step_actions").invoke(
            {
                "context": class_content,
                "journey_instructions": journey_instructions,
                "subject": f"Title: {step.title}\nSubject: {step.description}",
            }
        )
        new_step = {
            "title": step.title.strip(),
            "subject": step.description.strip(),
            "content": class_content.strip(),
            "intro": class_intro.strip(),
            "actions": class_actions.strip()
        }
        steps.append(new_step)
    bar.progress(0.95, text="Generating title for the curriculum")

    title, summary = gen_title_summary(
        "\n".join([f"Title: {step["title"]}\n Subject:\n{step['subject']}\nIntroduction:\n{step["intro"]}" for step in steps]),
    )
    bar.progress(1.0, text="Curriculum generation complete.")

    bar.empty()
    return {
        "title": title,
        "summary": summary,
        "steps": steps,
        "instructions": instructions
    }

def gen_title_summary(context) -> tuple[str, str]:
    title, title_thoughts = get_chain("action").invoke(
            {
                "context": context,
                "action": "Summarize context with 10 words or less to a title",
            }
    )

    summary, summary_thoughts = llm_edit("summary", [context], "Summarize the following list of titles and intros into a summary description.", force=True)

    return title.strip(), summary.strip()

def get_files_for_journey(default_category, journey_name, step, gen_from: Dict = None) -> List[Any]:
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

def gen_journey_doc_from_files(files: List[Any]) -> str:
    list_of_strings = []
    for filename in files.keys():
        if (
            files[filename]["formatted_text"] is not None
            and files[filename]["formatted_text"] != ""
        ):
            list_of_strings.append(files[filename]["formatted_text"])
        else:
            list_of_strings.append(
                markdown_to_text("\n".join(files[filename]["texts"]))
            )

    compressed = ''

    with st.spinner("Generating journey document"):
        compressed, compress_thoughts = gen_journey_doc(list_of_strings)

    return compressed

def gen_journey_subject(journey_name, journey_instructions, files, instructions, subject_index, amount) -> None:
    journey = st.session_state.journey_get_details[journey_name]
    # vectorstore = get_vectorstore("rag_"+ journey["category"][0], "hyde")
    doc_chain = rag_chain("rag_"+ journey["category"][0], "hyde")
    subject = None
    with st.status(f"Building subject document"):
        compressed = gen_journey_doc_from_files(files)
        st.success("Generating subject document done.")
    with st.status(f"Building subject curriculum"):
        subject = gen_subject(doc_chain, compressed, journey_instructions, instructions, amount)
        st.success("Generating subject curriculum done.")
        subject["files"] = list(files.keys())

    with st.status(f"Generating JSON for journey"):
        bar = st.progress(0, "Generating JSON for journey")

        total_items = len(subject["steps"])
        for i, step in enumerate(subject["steps"]):
            bar.progress(i/total_items, f"Generating JSON for step {i+1}/{total_items} ")
            structured = get_chain("journey_structured").invoke({
                "context": f"""
                    Title:
                    {step["title"]}
                    Intro:
                    {step["intro"]}
                    Content:
                    {step["content"]}
                    Actions:
                    {step["actions"]}
                """
            })

            subject["steps"][i]["json"] = structured
        bar.empty()
        st.success("Generating JSON for journey done.")
    return subject

def create_subject(journey_name, subject, subject_index, default_category) -> None:
    st.subheader(f"Subject {subject_index+1}")

    subject["instructions"] = st.text_area("Instructions for subject", key=f"journey_gen_instructions_{journey_name}_{subject_index}", value=subject["instructions"] if "instructions" in subject else "")

    col1, col2 = st.columns([4, 1], vertical_alignment="bottom")
    with col1:
        subject["db_files"] = get_files_for_journey(default_category, journey_name, subject_index, subject["db_files"] if "db_files" in subject else None)

    subject["step_amount"] = col2.number_input(
        "(approx) Items", min_value=1, max_value=20, value=subject["step_amount"] if "step_amount" in subject else 5, key=f"journey_gen_step_amount_{journey_name}_{subject_index}"
    )

    return subject

def get_journey_gen(journey_name):
    st.subheader("Journey generator")
    if "journey_get_details" not in st.session_state or st.session_state.journey_get_details == None:
        st.session_state.journey_get_details = {}
        st.session_state.journey_create = False
        st.session_state.journey_generator_running = False

    file_categories = get_all_categories()
    default_category = None
    if journey_name not in st.session_state.journey_get_details or st.session_state.journey_get_details[journey_name] == None:
        journey_details = {"journeyname": journey_name, "category": default_category}
        st.session_state.journey_get_details[journey_name] = journey_details
    else:
        journey_details = st.session_state.journey_get_details[journey_name]
        default_category = journey_details["category"]

    default_category = [st.selectbox("Select category for journey", file_categories, key=f"journey_gen_category_{journey_name}", index=file_categories.index(default_category[0]) if default_category is not None and len(default_category) > 0 else 0)]
    journey_details["category"] = default_category

    but_col1 = None
    but_col2 = None

    if len(default_category) > 0:
        # col1, col2 = st.columns([5, 1], vertical_alignment="bottom")
        generate_start = False
        if not st.session_state.journey_generator_running:
            journey_instructions = st.text_area("Journey Instructions", height=10, key=f"journey_gen_instructions_{journey_name}", value=journey_details["instructions"] if "instructions" in journey_details else "")
            journey_details["instructions"] = journey_instructions
            if "subjects" not in journey_details:
                journey_details["subjects"] = [{}]

            if "subjects" in journey_details:
                for i, subject in enumerate(journey_details["subjects"]):
                    journey_details["subjects"][i] = create_subject(journey_name, subject, i, default_category)
            but_col1, but_col2 = st.columns([4, 1], vertical_alignment="bottom")

            if but_col1.button("Add subject"):
                i = len(journey_details["subjects"])
                journey_details["subjects"].append({})
                journey_details["subjects"][i] = create_subject(journey_name, journey_details["subjects"][i], i, default_category)
                st.session_state.journey_get_details[journey_name] = journey_details

    # Check that all subjects have files defined
    def check_subject_files(subjects):
        for subject in subjects:
            if "db_files" not in subject or len(subject["db_files"]) < 1:
                return False
        return True

    if but_col2 is not None and len(journey_details["subjects"] if "subjects" in journey_details else []) > 0 and but_col2.button("Generate", key=f"generate_journey_{journey_name}", disabled=not check_subject_files(journey_details["subjects"])):
        st.session_state.journey_generator_running = True

        for i, subject in enumerate(journey_details["subjects"]):
            # print(f"Generating subject {i+1} for journey {journey_name}")
            subject = gen_journey_subject(journey_name, journey_instructions, subject["db_files"], subject["instructions"], i, subject["step_amount"])

            if subject is not None:
                journey_details["subjects"][i] = subject


        st.session_state.journey_get_details[journey_name] = journey_details
        st.session_state.journey_generator_running = False

        with st.spinner("Generating journey titles and summaries"):
            files = []
            for i, subject in enumerate(journey_details["subjects"]):
                title, summary = gen_title_summary(
                     "\n".join([f"{step['title']} {step['intro']}" for i, step in enumerate(subject["steps"])])
                )
                journey_details["subjects"][i]["title"] = title
                journey_details["subjects"][i]["summary"] = summary

                files.extend(subject["files"])

            title, summary = gen_title_summary(
                    "\n".join([f"{subject['title']} {subject['summary']}" for i, subject in enumerate(journey_details["subjects"])])
            )

            st.success("Generating journey titles and summaries done.")

            journey_details["title"] = title
            journey_details["summary"] = summary
            journey_details["db_files"] = files


        save_journey(journey_name, journey_details)
        journey_details["__complete"] = True
        # st.session_state.journey_get_details[journey_name] = journey_details
        st.session_state.journey_get_details = {}
        st.session_state.journey_create = False
        st.session_state.journey_generator_running = False
        get_db_journey(reset=True)
        st.rerun()

    return journey_details

# Streamlit app
def main():

    init_db()
    st.title("Create Journey")

    if not check_auth():
        return

    db_journey = get_db_journey()

    with st.container(border=True):
        if "journey_create_data" in st.session_state:
            journey_create = st.session_state.journey_create_data
        else:
            journey_create = None

        if "creating_journey" not in st.session_state:
            st.header("Create new journey")
            col1, col2 = st.columns([2, 1], vertical_alignment="bottom")

            journey_name = col1.text_input(
                "Unique name for the journey", value="test"
            )
            if col2.button("Create", disabled=journey_name in db_journey.keys()):
                st.session_state.creating_journey = journey_name
                time.sleep(0.01)
                st.rerun()

        if "creating_journey" in st.session_state and (
            journey_create is None or "__complete" not in journey_create.keys()
        ):
            journey_name = st.session_state.creating_journey
            st.header(f"Create new journey: {journey_name}")
            st.session_state.journey_create_data = {}
            journey_create = get_journey_gen(journey_name)
            st.session_state.journey_create_data = journey_create
            if "__complete" in journey_create.keys():
                st.session_state.journey_create_data = {}
                st.session_state.journey_get_details = {}
                st.session_state.journey_create = False
                st.session_state.journey_generator_running = False
                time.sleep(0.01)
                st.rerun()


if __name__ == "__main__":

    main()
