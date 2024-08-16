from datetime import datetime
import time
from typing import Any, Dict, List

import streamlit as st
from langchain_core.runnables.base import RunnableSequence

from lib.db_tools import (
    JourneyDataTable,
    get_db_files,
    get_db_journey,
    init_db,
)

from lib.document_parse import markdown_to_text
from lib.load_env import (
    EMBEDDING_CHAR_LIMIT,
    EMBEDDING_OVERLAP,
    INSTRUCT_CHAR_LIMIT,
    CLIENT_HOST,
)

# from rapidocr_paddle import RapidOCR  # type: ignore
from lib.chain import (
    get_chain,
)
from lib.document_tools import create_document_lists, rag_chain
from lib.prompts import JourneyStepList, JourneyStructure

from lib.streamlit_tools import check_auth, get_all_categories, llm_edit

st.set_page_config(
    page_title="TC POC: Admin",
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

# def gen_journey_json(source):
#     result = exec_structured_chain(
#         "journey_structured",
#         {
#             "context": source
#         }
#     )
#     # print(result)

#     return result


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

def build_journey_subject(journey_name, journey_instructions, files, instructions, subject_index, amount) -> None:
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

def edit_subject(journey_name, subject, subject_index, default_category) -> None:
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
                    journey_details["subjects"][i] = edit_subject(journey_name, subject, i, default_category)
            but_col1, but_col2 = st.columns([4, 1], vertical_alignment="bottom")

            if but_col1.button("Add subject"):
                i = len(journey_details["subjects"])
                journey_details["subjects"].append({})
                journey_details["subjects"][i] = edit_subject(journey_name, journey_details["subjects"][i], i, default_category)
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
            subject = build_journey_subject(journey_name, journey_instructions, subject["db_files"], subject["instructions"], i, subject["step_amount"])

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


def edit_journey_subject_step(journey_name, step, subject_index, step_index):
    try:
        step_json:JourneyStructure = step["json"]
        st.write(f"##### Step {step_index+1}:")
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

        if st.toggle("Show actions", key=f"journey_step_actions_{journey_name}_{subject_index}_{step_index}"):
            for j, action in enumerate(step_json.actions):
                st.write(f"##### Teaching action {j+1}:")

                action.title = st.text_input("Title", value=action.title, key=f"journey_step_action_title_{journey_name}_{subject_index}_{step_index}_{j}")
                action.description = st.text_area("Description", value=action.description, key=f"journey_step_action_description_{journey_name}_{subject_index}_{step_index}_{j}")
                for k, resource in enumerate(action.resources):
                    action.resources[k] = st.text_input(f"Resource {k+1}", value=resource, key=f"journey_step_action_resource_{journey_name}_{subject_index}_{step_index}_{j}_{k}")
                action.test = st.text_area("Test", value=action.test, key=f"journey_step_action_test_{journey_name}_{subject_index}_{step_index}_{j}")

    except Exception as e:
        print(f"Failed to use JSON: {e}")
        st.write(f"##### Step {step_index+1}:")
        step["title"] = st.text_input(
            "Title", value=step["title"], key=f"journey_step_title_{journey_name}_{subject_index}_{step_index}"
        )

        step["intro"] = st.text_area(
            "Intro",
            value=step["intro"],
            key=f"journey_step_intro_{journey_name}_{subject_index}_{step_index}",
        )

        step["content"] = st.text_area(
            "Content",
            value=step["content"],
            key=f"journey_step_content_{journey_name}_{subject_index}_{step_index}",
        )

        step["Actions"] = st.text_area(
            "actions",
            value=step["actions"],
            key=f"journey_step_Actions_{journey_name}_{subject_index}_{step_index}",
        )

    return step


def edit_journey_details(journey_name, journey:Dict):
    # print(journey_name, journey)
    if "title" in journey.keys():
        journey["title"] = st.text_input(
            f"Journey title", value=journey["title"], key=f"journey_title_{journey_name}"
        )
    if "summary" in journey.keys():
        journey["summary"] = st.text_area(
            f"Journey summary", value=journey["summary"], key=f"journey_summary_{journey_name}", height=200
        )
    if "chroma_collection" in journey.keys():
        journey["chroma_collection"] = st.text_input(
            f"Chroma databases", value=",".join(journey["chroma_collection"]), key=f"journey_chroma_collections_{journey_name}"
        ).split(",")

    return journey

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



def save_journey(journey_name, journey:Dict):
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
        files=journey.get("files", None),
        subjects=journey.get("subjects", None),
        title=journey.get("title", None),
        summary=journey.get("summary", None),
        last_updated=datetime.now(),
        chroma_collection = journey.get("chroma_collection", ["rag_" + cat for cat in journey.get("category", [])])
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


def edit_journey(journey_name, journey:Dict):
    st.header(f"Edit journey: {journey_name}")

    journey = edit_journey_details(journey_name, journey)

    if "subjects" in journey.keys():
        st.subheader("Journey subjects")
        for i, subject in enumerate(journey["subjects"]):
            st.write(f"### Subject {i}: {subject['title']}")
            for j, step in enumerate(subject["steps"]):
                with st.expander(f"Step {j}: {step['title']}"):
                    journey["subjects"][i]["steps"][j] = edit_journey_subject_step(journey_name, journey["subjects"][i]["steps"][j], i, j)

    if st.button("Save into database"):
        save_journey(journey_name, journey)


# Streamlit app
def main():

    init_db()
    st.title("Admin interface for TC POC")

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

        if db_journey is not None and len(db_journey.keys()) > 0:
            st.header("Journey database")
        else:
            st.header("No journeys created yet.")

        for journey_name in db_journey.keys():
            journey = db_journey[journey_name]
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
                col2.write(f'#### {journey["title"].strip()}')
                col2.write(journey["summary"])

            if "chroma_collection" in journey.keys() and len(journey["files"]) > 0:
                col2.write("Connected Chroma databases:")
                col2.write("* " + "\n* ".join(journey["chroma_collection"]))

            if "files" in journey.keys() and len(journey["files"]) > 0:
                col2.write("Files used:")
                col2.write("* " + "\n* ".join(journey["files"]))

            if col1.toggle("Extend", key=f"show_toggle_subjects_{journey_name}") or (st.session_state.editing_journey_subject is not None or st.session_state.editing_journey_step is not None):
                journey_edit = journey
                for i, subject in enumerate(journey["subjects"]):
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
                            journey_edit["subjects"][i]["title"] = st.text_input(
                                f"Title", value=subject["title"], key=f"subject_title_{journey_name}_{i}"
                            )
                            journey_edit["subjects"][i]["summary"] = st.text_area(
                                f"Summary", value=subject["summary"], key=f"subject_summary_{journey_name}_{i}", height=200
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
                            col2.write(subject["title"] + ":")
                            col2.write(subject["summary"])
                            if "files" in subject.keys():
                                col2.write("Files used:")
                                col2.write("* " + "\n* ".join(subject["files"]))

                        if (col1.toggle("Extend", key=f"show_toggle_steps_{journey_name}_{i}") or (st.session_state.editing_journey_subject is not None or st.session_state.editing_journey_step is not None)):
                            for j, step in enumerate(subject["steps"]):
                                with st.expander(f'##### Step {j+1}: {step["title"]}'):
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
                                        journey_edit["subjects"][i]["steps"][j] = edit_journey_subject_step(journey_name, journey["subjects"][i]["steps"][j], i, j)
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
                                        if step.get("json", None):
                                            json:JourneyStructure = step["json"]
                                            col1, col2 = st.columns([1, 5])
                                            col1.write("##### Intro:")
                                            col2.write(json.intro)

                                            col1, col2 = st.columns([1, 5])
                                            col1.write("##### Content:")
                                            col2.write(json.content)

                                            # st.write("##### Actions:")
                                            if st.toggle("Show actions", key=f"journey_step_info_actions_{journey_name}_{i}_{j}"):
                                                for k, action in enumerate(json.actions):
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
                                            # col2.write(json.priority)
                                        else:
                                            col1, col2 = st.columns([1, 5])
                                            col1.write("##### Intro:")
                                            col2.write(step["intro"])

                                            col1, col2 = st.columns([1, 5])
                                            col1.write("##### Content:")
                                            col2.write(step["content"])

                                            col1, col2 = st.columns([1, 5])
                                            col1.write("##### Actions:")
                                            col2.write(step["actions"])


if __name__ == "__main__":

    main()
