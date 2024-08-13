# import required dependencies
from functools import cache
import os
import re
# import chromadb
import datetime
import time
from typing import Dict
import streamlit as st

from langchain_core.prompt_values import StringPromptValue
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.documents.base import Document
# from langchain.schema.runnable.config import RunnableConfig
# from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.globals import set_debug
import sqlalchemy as sqla
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.mutable import MutableList


from db_tables import (
    # Base,
    # FileDataTable,
    # JourneyDataTable,
    get_db_files,
    get_db_journey,
    init_db
)

from chain import (
    client_host,
    get_chroma_collection,
    get_llm,
    get_prompt,
    get_vectorstore,
    handle_thinking,
    rerank_documents,
)

# set_debug(True)

# from langchain.chains import RetrievalQA,RetrievalQAWithSourcesChain
# Set up RetrievelQA model

database_session = init_db()

chat_history = { "default": [] } #[[] for _ in range(11)]
query_history = { "default": [] }
chat_state = "default"

DELIMITER="±~"

def get_memory(combine=False, return_string=False, tail=-6):
    def run_memory(value=None):
        global chat_history
        global chat_state

        state = chat_state
        if "chat_state" in st.session_state:
            state = st.session_state.chat_state

        memory = chat_history[state]
        if "chat_history" in st.session_state:
            memory = st.session_state.chat_history[state]

        # print(f"received value {value} with memory {memory}")

        ret = None
        if return_string:
            ret = ""
        else:
            ret = []

        for mem in memory[tail:]:
            if return_string:
                ret = ret + f"\n{mem.type}:{mem.content}"
            else:
                ret = ret + [mem]

        if combine and value is not None:
            question = ""
            if isinstance(value, StringPromptValue):
                question = re.sub(r"[^{]*\{'question': '", "", value.text)
                question = question[0 : question.find("}")]
            else:
                question = value["question"]

            q_ret = f"Question: {question}"

            if return_string:
                if len(memory) > 0:
                    ret = f"\n\nHistory:{ret}\n\n" + q_ret
                else:
                    ret = q_ret
            else:
                if len(memory) > 0:
                    ret = ret + [HumanMessage(content=q_ret)]

        return ret

    return run_memory

cur_query = ''

def parse_retriever_input(get_mem, return_str=False):
    def get_retriever_input(params: Dict):
        print(f"\n\n{ params = }")
        prev_questions = get_mem()
        print(f"\n\n{prev_questions = }")
        global cur_query
        cur_query = params["question"]
        # if len(prev_questions) > 0:
        #     prev_questions = f"{prev_questions}"
        user_question= ''

        user_question= f'\nQuestion: \n{params["question"]}'
        return prev_questions + [user_question]

    return get_retriever_input

def parse_question_input(params: Dict):
    return params["question"]

def reformat_rag(params: list[Document]):
    global cur_query

    query = cur_query
    new_content = []
    new_content_strs = []
    def sort_by_references(document: Document):
        return len(document.metadata)

    params.sort(reverse=True, key=sort_by_references)
    for document in params:
        if document.page_content not in new_content_strs:
            new_content.append(document)
            new_content_strs.append(document.page_content)

    new_content = rerank_documents(new_content, query, 10)

    global query_history
    global chat_state
    # query_history = st.sesssion_state.query_history
    # chat_state = st.session_state.chat_state
    if chat_state not in query_history.keys():
        query_history[chat_state] = []

    query_history[chat_state].append(new_content)
    # st.session_state.query_history = query_history

    new_content_str = str(new_content).replace("Document", "\n\nDocument").replace('[', '').replace(']', '').replace('metadata', '\n\nmetadata').replace('page_content', '\npage_content')
    print(f'\n\nRAG reformat: \n{query = } \n\nresult = {new_content_str}\n\n')
    return new_content

def retrieval_qa_chain(llm, prompt, vectorstore):

    # retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    # print(f'test vectorstore: { retriever.invoke("What is antler") }')

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 40, "score_threshold": 0.3},
    )
    # retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "include_metadata": True})
    # retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, "score_threshold": 0.3})

    # print(f"{ llm = }")

    qa_no_context = (
        prompt
        | llm
        | StrOutputParser()
    )

    qa_chain = (
        RunnableParallel(
            {
                "context": RunnablePassthrough(hypothetical_document=qa_no_context) | parse_retriever_input(get_memory(tail=-8), True) | retriever | reformat_rag,
                "question": parse_question_input | RunnablePassthrough(),
                "history": RunnableLambda(get_memory()),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain

def now():
    return round(time.time() * 1000)

@cache
def qa_bot(id):
    print(f" set qa_bot for { id = }")
    vectorstore = get_vectorstore(id, "hyde")
    llm = get_llm("chat")
    prompt = get_prompt("question")

    qa = retrieval_qa_chain(llm, prompt, vectorstore)
    return qa


def send_message(message, journey_name, chat_state):
    print(f"\n({datetime.datetime.now()}) { message = }\n")
    chain_id = st.session_state.journey_chain_ids[journey_name]
    # print(f"\n({datetime.datetime.now()}) {chain_id = }")
    chain = st.session_state.chains[chain_id]
    memory = st.session_state.chat_history[chat_state]

    print(f"\n({datetime.datetime.now()}) {memory = }\n")
    print(f"\n({datetime.datetime.now()}) {chain_id = }")
    # print(f" {message = }")
    resp, _ = handle_thinking((lambda: chain.invoke({"question": message})))
    return resp

def init(journey_name:str):
    init_db()

    if "journey_list" not in st.session_state:
        st.session_state.journey_list = get_db_journey(journey_name)

    if journey_name not in st.session_state.journey_list.keys():
        return False

    if "chat_history" not in st.session_state:
        st.session_state.query_history = query_history
        st.session_state.chat_history = chat_history
        st.session_state.chat_state = chat_state


    if "chroma_collections" not in st.session_state:
        journey_chain_ids = {}
        chroma_collections = {}
        # chroma_collections = { "rag_all": get_chroma_collection(journey_chain_ids["default"]) }
        # journey_list = st.session_state.journey_list
        chains = {}
        # for journey_name in journey_list:
        journey = st.session_state.journey_list[journey_name]
        # print(f"{journey["chroma_collection"]}")
        collections = journey["chroma_collection"]
        collection_keys = list(chroma_collections.keys())

        journey_chain_ids[journey_name] = collections[0]
        if len(collections) > 0:
            for collection in collections:
                if collection not in collection_keys:
                    chroma_collections[collection] = get_chroma_collection(collection)
                    chains[collection] = qa_bot(collection)

        # print(f" { list(chroma_collections.keys()) = }")
        st.session_state.chroma_collections = chroma_collections
        st.session_state.journey_chain_ids = journey_chain_ids

        chroma_collections = st.session_state.chroma_collections
        # print(f" {collections=} { chroma_collections = }")

        st.session_state.chains = chains

    return True

def get_stream_data(message:str, chat_state:str):
    def stream_data():
        if message not in st.session_state.chat_history_seen[chat_state]:
            st.session_state.chat_history_seen[chat_state].append(message)

        for word in message.split(" "):
            yield word + " "
            time.sleep(0.01)

    return stream_data

def chat_elements(chat_state, journey_name):
    st.session_state.chat_history_seen = st.session_state.chat_history_seen if "chat_history_seen" in st.session_state else {}
    if chat_state not in st.session_state.chat_history_seen:
        st.session_state.chat_history_seen[chat_state] = []

    # print(f"chat state {st.session_state.chat_state} {chat_state}")
    user_query = None
    if "user_query" in st.session_state and st.session_state.user_query != None:
        user_query = st.session_state.user_query

    if chat_state not in st.session_state.chat_history or st.session_state.chat_history[chat_state] is None:
        st.session_state.chat_history[chat_state] = []

    for i, message in enumerate(st.session_state.chat_history[chat_state]):
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        elif i > 0:
            with st.chat_message("AI"):
                st.write(message.content)
                references = dict[str, Document]({})
                if message.response_metadata is not None and "references" in message.response_metadata.keys():
                    for reference in message.response_metadata["references"]:
                        if "file" in reference.metadata.keys():
                            references[reference.metadata["file"].replace("formatted_", "")] = reference

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
                                col1.download_button("Download", db_file["file_data"],  file_name=file, key=file+"_"+chat_state+"_"+str(i))

                            col2.container(height=150).write(reference.page_content) #db_file["summary"])
        else:
            if message.content not in st.session_state.chat_history_seen[chat_state]:
                with st.chat_message("AI"):
                    st.write_stream(get_stream_data(message.content, chat_state))
            else:
                with st.chat_message("AI"):
                    st.write(message.content)

    if chat_state == "default" and st.session_state.chat_state == chat_state and len(st.session_state.chat_history[chat_state]) == 0:
        journey = st.session_state.journey_list[journey_name]
        st.session_state.chat_history[chat_state].append(
            AIMessage(
                f"""Welcome to ThirdCognition Virtual Buddy POC

This proof of concept has been provided for you to see how it is possible to generate learning content
with just the help of our tool. This is a work in progress and should not be considered a final product.
Also note that this is for internal use only and any results gained should not be shared
outside of your respective organization.

#### {journey["title"]}

{journey["summary"]}

You can now ask any questions you might have 👇. If you want to experience the preliminary learning journey,
you can do so by selecting any of the subjects provided for you from the menu on the left.
                """,
            )
        )
        st.rerun()

    if (
        len(st.session_state.chat_history[chat_state]) == 0
        and "chat_journey" in st.session_state
    ):
        # print(f"{ chat_state = }")
        subject_index = int(chat_state.split(DELIMITER)[1])
        step_index = int(chat_state.split(DELIMITER)[2])
        journey = st.session_state.journey_list[st.session_state.chat_journey]
        # print(f"{subject_index=} {step_index=}")
        # print(f"{journey["subjects"][subject_index]["steps"]=}")
        st.subheader(journey["subjects"][subject_index]["steps"][step_index]["title"])
        # st.write(journey["subjects"][subject_index]["steps"][step_index]["description"])
        # st.write("##### Actions:")
        # st.write("* " + "\n* ".join(journey["subjects"][subject_index]["steps"][step_index]["actions"]))
        st.session_state.chat_history[chat_state].append(
            AIMessage(
                journey["subjects"][subject_index]["steps"][step_index]["intro"]
                # "## "
                # + journey["subject"][subject_index]["steps"][step_index]["title"]
                # + "\n"
                # + journey["subject"][subject_index]["steps"][step_index]["description"]
                # + "\n##### Actions:\n"
                # + "* "
                # + "\n* ".join(journey["steps"][chat_index]["actions"])
            )
        )
        st.rerun()

    print(f" {user_query = }")

    if user_query is not None and user_query != "":
        with st.chat_message("Human"):
            st.write(user_query)

        with st.chat_message("AI"):
            # print("Response:\n")
            # ai_response = send_message(user_query)
            # st.markdown(ai_response)
            # ai_response = st.write_stream(send_message(user_query, journey_name, chat_state))  # "I dont'know"
            ai_response = st.write("Thinking...")
            ai_response = send_message(user_query, journey_name, chat_state)

        # print(f"\n\n")

        human_message = HumanMessage(user_query)
        st.session_state.chat_history[chat_state].append(human_message)
        query_history = st.session_state.query_history
        if chat_state in query_history.keys() and len(query_history[chat_state]) > 0:
            references = query_history[chat_state][-1]
        else:
            references = []

        ai_response = AIMessage(ai_response, response_metadata={"references":references})
        st.session_state.chat_history[chat_state].append(ai_response)
        st.session_state.user_query = None
        st.rerun()

    else:
        if len(st.session_state.chat_history[chat_state]) == 1:
            with st.chat_message("AI"):
                st.write("I'm waiting for your questions...")

        if "user_query" not in st.session_state or st.session_state.user_query == None:
            user_query = st.chat_input("Your message", key=f"chat_input_{chat_state}")
            if user_query != None and user_query != "":
                st.session_state.user_query = user_query
                st.rerun()

def page_not_found():
    st.title("Journey not found")
    st.write("The Journey you are looking for does not exist.")
    st.write("Please check the URL and try again.")


def main():
    st.set_page_config(
        page_title="TC POC",
        page_icon="static/icon.png",
        layout="wide",
        menu_items={
            # 'Get Help': 'https://www.extremelycoolapp.com/help',
            # 'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': """# ThirdCognition PoC
[ThirdCognition](https://thirdcognition.com)
This is an *extremely* cool app!
            """
        }
    )

    if ("active_journey" not in st.session_state) and (st.query_params.get("journey", None) is None or st.query_params["journey"] == ""):
        page_not_found()
        return

    journey_name = st.query_params.get("journey", None) or st.session_state.active_journey
    chat_step = st.query_params.get("state", None) or ("active_step" in st.session_state and st.session_state.active_step) or ""
    journey_found = init(journey_name)

    if not journey_found:
        page_not_found()
        return

    if "active_journey" not in st.session_state:
        st.session_state.active_journey = journey_name

    if "active_step" not in st.session_state or chat_step != st.session_state.active_step:
        st.session_state.active_step = chat_step


    # st.header("ThirdCognition Proof of concept demostration")
    chat_state = st.session_state.chat_state

    journey_list = st.session_state.journey_list


    if chat_state != "default":
        if st.button(
            ":house: Return home", use_container_width=True, disabled=(0 == chat_state)
        ):
            chat_state = "default"
            st.session_state.chat_state = chat_state
            st.session_state.chat_journey = None
            st.rerun()

    journey = st.session_state.journey_list[journey_name]
    if chat_state == "default":
        st.subheader("ThirdCognition Virtual Buddy", divider=True)
    else:
        subject_index = int(chat_state.split(DELIMITER)[1])
        st.subheader(journey["subjects"][subject_index]["title"], divider=True)
    # st.subheader(journey["title"], divider=True)
    # st.write(journey["summary"])


    with st.sidebar:
        st.markdown("""<style>
        button[data-testid=baseButton-secondary] {
            display: block;
            text-align: left;
            color: inherit;
            text-decoration: none;
            background-color: unset;
            border: none;
            padding-top: 0;
            padding-bottom: 0;
        }
        button[data-testid=baseButton-secondary]:active {
            text-decoration: underline;
            background-color: unset;
            color: inherit;
        }
        button[data-testid=baseButton-secondary]:disabled {
            border: none;
            cursor: auto !important;
        }
        </style>""", unsafe_allow_html=True)
    # with st.container():
    #     with st.sidebar:


        for i, subject in enumerate(journey["subjects"]):
            with st.expander(f"{subject["title"]}", expanded=(f"{journey_name}_{i}" in chat_state)):
                for j, step in enumerate(subject["steps"]):
                    step_id = f'{journey_name}{DELIMITER}{i}{DELIMITER}{j}'
                    # if st.session_state.chat_state != i + 1:
                        # col1, col2 = st.columns([5, 1])

                    # st.write(f"{client_host}?journey={journey_name}&state={i}_{j}", label=step["title"], disabled=chat_state == step_id, use_container_width=True)
                    # url = f"{client_host}?journey={journey_name}&state={i}_{j}"


                        # st.write(f'#### {step["name"]}')
                    # if journey_name == st.session_state.active_journey and f"{i}_{j}" == st.session_state.active_step and chat_state != step_id:
                    if st.button(
                        step["title"],
                        use_container_width=True,
                        disabled=(step_id == chat_state),
                        key=f'step_{step_id}',
                    ):  # , on_click=set_chat_state, args=(i, task)
                        chat_state = step_id
                        st.session_state.chat_state = chat_state
                        st.session_state.chat_journey = journey_name
                        st.rerun()


    for journey_name in journey_list:
        if (
            "chat_journey" in st.session_state
            and st.session_state.chat_journey == journey_name
            and "chat_state" in st.session_state
        ):
            chat_elements(st.session_state.chat_state, journey_name)

    if chat_state == "default" and st.session_state.chat_state == chat_state:
        chat_elements("default", journey_name)

if __name__ == "__main__":
    main()
