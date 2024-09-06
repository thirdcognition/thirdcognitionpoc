import os
import time
import streamlit as st
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents.base import Document

from chains.rag_chain import get_rag_chain
from lib.db_tools import JourneyModel, get_db_files, get_db_journey, init_db
from lib.helpers import get_chain_with_history, get_session_history

DELIMITER = "±~"

def send_message(message, journey_name, chat_state):
    if journey_name:
        # chain_id = st.session_state.journey_chain_ids[journey_name]
        chain = st.session_state.chains[journey_name]
    else:
        chain = st.session_state.chains["ThirdCognition"]
    st.session_state["exec_query"] = message
    chain.invoke(
        {"question": message},
        config={
            "session_id": chat_state,
            "callbacks": [StreamingStdOutCallbackHandler()],
        },
    )
    st.session_state["exec_query"] = None
    # st.rerun()


def get_stream_data(message: str, chat_state: str):
    def stream_data():
        if message not in st.session_state.chat_history_seen[chat_state]:
            st.session_state.chat_history_seen[chat_state].append(message)

        for word in message.split(" "):
            yield word + " "
            time.sleep(0.01)

    return stream_data

def get_journey_chat(journey_name: str = "ThirdCognition", rag_collection: str = None):
    if "chains" not in st.session_state:
        init_journey_chat(journey_name, rag_collection)
    return st.session_state.chains[journey_name]

def init_journey_chat(journey_name: str = "ThirdCognition", rag_collection: str = None):
    init_db()

    if "journey_list" not in st.session_state:
        st.session_state.journey_list = get_db_journey(journey_name)

    if "chat_state" not in st.session_state:
        st.session_state.chat_state = "default"

    if "chains" not in st.session_state:
        # journey_chain_ids = {}
        chains = {}
        if journey_name and journey_name in st.session_state.journey_list.keys():
            collections = st.session_state.journey_list[journey_name].chroma_collection
            chains[journey_name] = get_chain_with_history(journey_name, get_rag_chain(collections, chat=True))
        else:
            collection = (
                "rag_ThirdCognition" if rag_collection is None else rag_collection
            )
            chains[journey_name] = get_chain_with_history(journey_name, get_rag_chain([collection], chat=True))

        # st.session_state.journey_chain_ids = journey_chain_ids
        st.session_state.chains = chains

    if journey_name not in st.session_state.journey_list.keys():
        return False

    return True

def chat_elements(chat_state, journey_name=None):
    st.session_state.chat_history_seen = (
        st.session_state.chat_history_seen
        if "chat_history_seen" in st.session_state
        else {}
    )
    if chat_state not in st.session_state.chat_history_seen:
        st.session_state.chat_history_seen[chat_state] = []

    journey = None
    subject_index = None
    step_index = None

    if "chat_journey" in st.session_state and st.session_state.chat_journey is not None and st.session_state.chat_journey != "":
        # print(f"{ chat_state = }")
        subject_index = int(chat_state.split(DELIMITER)[1])
        step_index = int(chat_state.split(DELIMITER)[2])
        journey: JourneyModel = st.session_state.journey_list[
            st.session_state.chat_journey
        ]
        # st.subheader(journey.subjects[subject_index].steps[step_index].title)
        st.write(journey.subjects[subject_index].steps[step_index].content)
    # print(f"chat state {st.session_state.chat_state} {chat_state}")
    # user_query = None
    # if "user_query" in st.session_state:
    #     user_query = st.session_state.user_query

    # if chat_state not in st.session_state.chat_history or st.session_state.chat_history[chat_state] is None:
    #     st.session_state.chat_history[chat_state] = []
    history = get_session_history(chat_state)
    for i, message in enumerate(history.messages):
        # print(f"\n\n Message: \n\n {message}\n\n\n")
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        elif i > 0:
            with st.chat_message("AI"):
                st.write(message.content)
                if journey_name:
                    references = dict[str, Document]({})
                    if (
                        message.response_metadata is not None
                        and "references" in message.response_metadata.keys()
                    ):
                        for reference in message.response_metadata["references"]:
                            if "file" in reference.metadata.keys():
                                references[
                                    reference.metadata["file"].replace("formatted_", "")
                                ] = reference

                    if len(references.keys()) > 0:
                        # st.write("###### _References:_")
                        with st.expander("_References_"):
                            # st.write(f'- {references[reference].metadata["file"].replace("formatted_", "")}\n' for reference in references.keys())
                            for file in references.keys():
                                filetype = os.path.basename(file).split(".")[-1]
                                # file = reference.metadata["file"].replace("formatted_", "")
                                reference = references[file]
                                # formatted = "formatted_" in reference.metadata["file"]

                                # meta_keys = reference.metadata.keys()
                                db_file = get_db_files(filename=file)[file]
                                [col1, col2] = st.columns([1, 3])
                                col1.write(file)
                                if filetype != "epub":
                                    col1.download_button(
                                        "Download",
                                        db_file["file_data"],
                                        file_name=file,
                                        key=file + "_" + chat_state + "_" + str(i),
                                    )

                                col2.container(height=150).write(
                                    reference.page_content
                                )  # db_file["summary"])
        else:
            # print(f"{message = }")
            if message.content not in st.session_state.chat_history_seen[chat_state]:
                with st.chat_message("AI"):
                    st.write_stream(get_stream_data(message.content, chat_state))
            else:
                with st.chat_message("AI"):
                    st.write(message.content)

    if (
        chat_state == "default"
        and st.session_state.chat_state == chat_state
        and len(history.messages) == 0
    ):
        if journey_name:
            journey = st.session_state.journey_list[journey_name]
            history.add_ai_message(
                f"""Welcome to ThirdCognition Virtual Buddy POC

This proof of concept has been provided for you to see how it is possible to generate learning content
with just the help of our tool. This is a work in progress and should not be considered a final product.
Also note that this is for internal use only and any results gained should not be shared
outside of your respective organization.

#### {journey.title}

{journey.summary}

You can now ask any questions you might have 👇. If you want to experience the preliminary learning journey,
you can do so by selecting any of the subjects provided for you from the menu on the left.
                    """,
            )
        else:
            history.add_ai_message(
                f"""Welcome to ThirdCognition Virtual Buddy
                    You can ask me anything you'd want to know about ThirdCognition
                    and I will do my best to answer you. :smile:
                    """,
            )
        st.rerun()


    # if journey is not None and len(history.messages) == 0:
    #     history.add_ai_message(
    #         AIMessage(journey.subjects[subject_index].steps[step_index].intro)
    #     )
    #     st.rerun()

    ai_state = st.empty()
    if len(history.messages) < 2:
        with ai_state.chat_message("AI"):
            st.write("I'm waiting for your questions...")

    user_state = st.empty()
    user_query = user_state.chat_input("Your message", key=f"chat_input_{chat_state}")

    ai_response = st.empty()

    if user_query != None and user_query != "":
        ai_state.empty()
        with user_state.chat_message("Human"):
            st.write(user_query)
        with ai_response.chat_message("AI"):
            st.write("Thinking...")

        send_message(user_query, journey_name, chat_state)
        st.rerun()
