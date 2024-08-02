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
    Base,
    FileDataTable,
    JourneyDataTable,
    get_db_files,
    get_db_journey,
    init_db
)

from chain import (
    SQLITE_DB,
    get_chroma_collection,
    get_llm_prompt,
    get_vectorstore,
    rerank_documents,
)

# set_debug(True)

# from langchain.chains import RetrievalQA,RetrievalQAWithSourcesChain
# Set up RetrievelQA model

database_session = init_db()

chat_history = { "default": [] } #[[] for _ in range(11)]
query_history = { "default": [] }
chat_state = "default"

# template = """<s> <<SYS>> You are a helpful startup coach from antler trying to answer questions thoroughly and exactly. Use the following pieces of retrieved context to answer the question. Use history if you don't undestand the question. If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise unless question requests for more. <</SYS>> </s>
# Terms:
# Antler is an early stage investment company
# PRE-IC Pre-IC stands for "Pre-Investment Committee."
# IC refers to Investment Committee
# POC refers to Proof of Concept
# MVP refers to Minimum viable product
# [INST] Context: {context} [/INST]
# History:
# {history}
# [INST] Question: {question} [/INST]
# Answer:
# """
# # template = """<s> [INST] You are a helpful startup coach . Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise.[/INST] </s>
# # [INST] Example: {question}
# # Context: {context}
# # Answer: [/INST]
# # """

# prompt = PromptTemplate(
#     template=template, input_variables=["context", "question", "history"]
# )  # ChatPromptTemplate.from_template(template)

# hyde_template = """<s> <<SYS>> You are a startup expert from Antler. Answer the question and use five sentences maximum and keep the answer concise. Use the terms if needed. If you don't know the answer, just repeat the question.<</SYS>> </s>
# Terms:
# Antler is an early stage investment company
# PRE-IC Pre-IC stands for "Pre-Investment Committee."
# IC refers to Investment Committee
# POC refers to Proof of Concept
# MVP refers to Minimum viable product
# [INST] Question: {question} [/INST]
# Answer:"""

# template = """<s> <<SYS>> Act as a helper trying to answer questions. Try to use the following pieces of retrieved context to answer the question. Use history if you don't undestand the question. Use five sentences maximum and keep the answer concise unless question requests for more. <</SYS>> </s>
# [INST] Context: {context} [/INST]
# History:
# {history}
# [INST] Question: {question} [/INST]
# Answer:
# """
# template = """<s> [INST] You are a helpful startup coach . Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise.[/INST] </s>
# [INST] Example: {question}
# Context: {context}
# Answer: [/INST]
# """

# prompt = PromptTemplate(
#     template=template, input_variables=["context", "question", "history"]
# )  # ChatPromptTemplate.from_template(template)

# hyde_template = """<s> <<SYS>> Act as a helper trying to answer questions. Answer the question and use five sentences maximum and keep the answer concise. If you don't know the answer, just repeat the question.<</SYS>> </s>
# [INST] Question: {question} [/INST]
# Answer:"""

# hyde_prompt = PromptTemplate(template=hyde_template, input_variables=["question"])

def get_memory(combine=False, llama=False, dolphin=True, only_questions=False, tail=-6):
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

        ret = ""
        for mem in memory[tail:]:
            if isinstance(mem, HumanMessage) and str(value) != str(mem.content):
                if only_questions:
                    ret = ret + f"\n{mem.content}"
                elif dolphin:
                    ret = ret + f"<|im_start|>user \n {mem.content} <|im_end|>"
                elif llama:
                    ret = ret + f"<|start_header_id|>user<|end_header_id|> \n\n {mem.content}<|eot_id|>"
                else:
                    ret = ret + f"[INST] User: {mem.content} [/INST]\n"
            elif str(value) != str(mem.content):# and not only_questions:
                if only_questions:
                    ret = ret + f"\n{mem.content}"
                elif dolphin:
                    ret = ret + f"<|im_start|>assistant \n {mem.content} <|im_end|>"
                elif llama:
                    ret = ret + f"<|start_header_id|>assistant<|end_header_id|> \n\n {mem.content}<|eot_id|>"
                else:
                    ret = ret + f"[INST] AI: {mem.content} [/INST]\n"

        if combine and value is not None:
            question = ""
            if isinstance(value, StringPromptValue):
                question = re.sub(r"[^{]*\{'question': '", "", value.text)
                question = question[0 : question.find("}")]
            else:
                question = value["question"]

            if dolphin:
                q_ret = f"<|im_start|>user\n {question} <|im_end|>"
            elif llama:
                q_ret = f"<|start_header_id|>user<|end_header_id|> \n\n {question}<|eot_id|>"
            else:
                q_ret = f"{question}"
            if len(memory) > 0:
                if dolphin:
                    ret = q_ret + ret
                elif llama:
                    ret = q_ret + ret
                else:
                    ret = q_ret + f"[/INST]\nHistory:{ret}[INST]"
            else:
                ret = q_ret

        return ret

    return run_memory

cur_query = ''

def parse_retriever_input(get_mem, llama=False, dolphin=True):
    def get_retriever_input(params: Dict):
        print(f"\n\n{ params = }")
        prev_questions = get_mem();
        print(f"\n\n{prev_questions = }")
        global cur_query
        cur_query = params["question"]
        # if len(prev_questions) > 0:
        #     prev_questions = f"{prev_questions}"
        user_question= ''

        if dolphin:
             user_question= f'<|im_start|>user \n{params["question"]}<|im_end|>'
        elif llama:
             user_question= f'<|start_header_id|>user<|end_header_id|> \n\n{params["question"]}<|eot_id|>'
        else:
             user_question= f'[INST]User: \n{params["question"]}[/INST]'
        return prev_questions + user_question

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

def retrieval_qa_chain(llm_setup, vectorstore):

    # retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    # print(f'test vectorstore: { retriever.invoke("What is antler") }')

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 40, "score_threshold": 0.3},
    )
    # retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "include_metadata": True})
    # retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, "score_threshold": 0.3})

    print(f"{ llm_setup = }")

    qa_no_context = (
        llm_setup["prompt"]
        | llm_setup["llm"]
        | StrOutputParser()
    )

    qa_chain = (
        RunnableParallel(
            { # only_questions=True,
              # |
                "context": RunnablePassthrough(hypothetical_document=qa_no_context) | parse_retriever_input(get_memory(tail=-8)) | retriever | reformat_rag,
                "question": parse_question_input | RunnablePassthrough(),
                "history": RunnableLambda(get_memory()),
            }
        )
        | llm_setup["prompt"]
        | llm_setup["llm"]
        | StrOutputParser()
    )

    # qa_chain = ConversationalRetrievalChain.from_llm(
    #     llm,
    #     prompt=prompt,
    #     retriever=retriever,
    #     memory=memory,
    #     return_source_documents=True,
    # )

    return qa_chain


# chroma_collection = "rag-cleaning"
# persistent_client = chromadb.PersistentClient(path="..\\databases\\chroma_db")

# async def load_tasklist():
#     journey_list = cl.TaskList()
#     journey_list.status = "Loaded."

#     json_f = open('ic-tasklist.json')
#     json_data = json.load(json_f)
#     json_f.close()

#     tasks = []

#     for task in json_data["journey_list"]:
#         task_item = cl.Task(title=task["name"]) # , status=cl.TaskStatus.RUNNING
#         # message_id = await cl.Message(content=get_task_helper(task["name"], task["description"])).send()
#         # task_item.forId = message_id
#         # task_item.status = "Waiting..."
#         await journey_list.add_task(task_item)

#     await journey_list.send()

#     cl.user_session.set("tasklist", journey_list)


def now():
    return round(time.time() * 1000)


@cache
def qa_bot(id): #collection
    # llm = load_llm()

    # conversation = (
    #     {"question": get_memory(session_state=st.session_state, combine=True)}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    # base_embeddings = GPT4AllEmbeddings()
    # embeddings = HypotheticalDocumentEmbedder.from_llm(
    #     conversation, base_embeddings, custom_prompt=hyde_prompt
    # )
    #embeddings = base_embeddings

    # vectorstore = Chroma(
    #     client=persistent_client,
    #     collection_name=chroma_collection,
    #     embedding_function=base_embeddings,
    # )

    print(f" set qa_bot for { id = }")
    vectorstore = get_vectorstore(id, "hyde")
    # vectorstore = get_vectorstore(id)
    llm_setup = get_llm_prompt("chat")

    qa = retrieval_qa_chain(llm_setup, vectorstore)
    return qa


def send_message(message, journey_name, chat_state):
    print(f"\n({datetime.datetime.now()}) { message = }\n")
    chain_id = st.session_state.journey_chain_ids[journey_name]
    chain = st.session_state.chains[chain_id]
    memory = st.session_state.chat_history[chat_state]

    print(f"\n({datetime.datetime.now()}) { memory = }\n")
    print(f"\n({datetime.datetime.now()}) {chain_id = }")
    # print(f" {message = }")
    return chain.stream({"question": message})


# def get_db_journey(reset=False):
#     if "db_journey" not in st.session_state or reset:
#         journey = database_session.query(JourneyDataTable).all()

#         db_journey = {}

#         for step in journey:
#             print(f"{ step = }")
#             db_journey[step.journeyname] = step.__dict__

#         st.session_state.db_journey = db_journey
#     else:
#         db_journey = st.session_state.db_journey

#     return db_journey


def init():
    init_db()

    if "journey_list" not in st.session_state:
        st.session_state.journey_list = get_db_journey()


    if "chat_history" not in st.session_state:
        st.session_state.query_history = query_history
        st.session_state.chat_history = chat_history
        st.session_state.chat_state = chat_state

    if "chroma_collections" not in st.session_state:
        journey_chain_ids = { "default": "rag_all" }
        chroma_collections = { "rag_all": get_chroma_collection(journey_chain_ids["default"]) }
        journey_list = st.session_state.journey_list
        chains = {"rag_all": qa_bot("rag_all")}
        for journey_name in journey_list:
            journey = st.session_state.journey_list[journey_name]
            collection = journey["chroma_collection"]
            collection_keys = list(chroma_collections.keys())
            if collection == "rag-all":
                collection = "rag_all"

            journey_chain_ids[journey_name] = collection
            if collection not in collection_keys:
                chroma_collections[collection] = get_chroma_collection(collection)
                chains[collection] = qa_bot(collection)

        st.session_state.chroma_collections = chroma_collections
        st.session_state.journey_chain_ids = journey_chain_ids

        chroma_collections = st.session_state.chroma_collections
        print(f" { chroma_collections = }")

        st.session_state.chains = chains


def chat_elements(chat_state, journey_name):
    print(f"chat state {st.session_state.chat_state} {chat_state}")
    user_query = None
    if "user_query" in st.session_state and st.session_state.user_query != None:
        user_query = st.session_state.user_query

    if chat_state == "default" and st.session_state.chat_state == chat_state:
        st.title("Virtual Buddy")
        # st.write("I'm here to help you with any of your questions!")
        st.write(
            """
            Welcome to the WIP of our POC provided FYI.

            This test version is briefly available during Antler Nordic 6 as a technology demonstration and should not be considered to be a final product. Please note that this is for internal use only and any results gained with this bot should not be shared outside of Antler Nordic 6.

            #### Feedback

            If you have any feedback or comments please reach out to any member of our team, i.e. [Markus Haverinen](https://antler-nordics6.slack.com/team/U06D681EE5S), [Kasra Aliyon](https://antler-nordics6.slack.com/team/U06DR34CYUW), [Niklas Slotte](https://antler-nordics6.slack.com/team/U06GRTTDP32).

            You can start by asking any question you want to know about Antler Nordic Cohort or running a Startup, or the Journeys by opening any of the lists below.

            ___NOTE__: Chatbot has a memory and you can switch between tasks, but if you refresh the browser all data will be reset._
        """
        )

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
            st.write(message.content)

    if (
        chat_state != "default"
        and len(st.session_state.chat_history[chat_state]) == 0
        and "chat_journey" in st.session_state
    ):
        # print(f"{ chat_state = }")
        chat_index = int(chat_state.split("_")[1])
        journey = st.session_state.journey_list[st.session_state.chat_journey]
        st.subheader(journey["steps"][chat_index]["name"])
        st.write(journey["steps"][chat_index]["description"])
        st.write("##### Actions:")
        st.write("* " + "\n* ".join(journey["steps"][chat_index]["actions"]))
        st.session_state.chat_history[chat_state].append(
            AIMessage(
                "## "
                + journey["steps"][chat_index]["name"]
                + "\n"
                + journey["steps"][chat_index]["description"]
                + "\n##### Actions:\n"
                + "* "
                + "\n* ".join(journey["steps"][chat_index]["actions"])
            )
        )

    print(f" {user_query = }")

    if user_query is not None and user_query != "":
        with st.chat_message("Human"):
            st.write(user_query)

        with st.chat_message("AI"):
            print("Response:\n")
            # ai_response = send_message(user_query)
            # st.markdown(ai_response)
            ai_response = st.write_stream(send_message(user_query, journey_name, chat_state))  # "I dont'know"

        print(f"\n\n")

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


def main():
    init()
    st.set_page_config(page_title="Antler Nordic 6 guide")

    chat_state = st.session_state.chat_state

    st.write(
        """<style>
        [data-testid="stHorizontalBlock"] > [data-testid="column"]{
            align-self: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # print(f"chat history: { st.session_state.chat_history}")

    # a = st.radio("Show chat", ["Show", "Hide"], 0)

    # if a == "Show":

    # else:
    # st.markdown('test for page 2')

    # def set_chat_state(i, task):
    #     chat_state = i+1
    #     st.session_state.chat_state = chat_state

    #     print(f"{i}: {task}")

    with st.container():
        with st.sidebar:
            if st.button(
                "Home", use_container_width=True, disabled=(0 == chat_state)
            ):  # , on_click=set_chat_state, args=(0, None)
                chat_state = "default"
                st.session_state.chat_state = chat_state
                st.session_state.chat_journey = None
                st.rerun()

    if chat_state == "default" and st.session_state.chat_state == chat_state:
        chat_elements("default", "default")

    if "journey_list" in st.session_state:
        journey_list = st.session_state.journey_list

        with st.sidebar:
            for journey_name in journey_list: #['pre-ic']:
                journey = st.session_state.journey_list[journey_name]
                # with st.expander(
                #     journey["title"],
                #     expanded="chat_journey" in st.session_state
                #     and st.session_state.chat_journey == journey_name,
                # ):
                st.subheader(journey["title"], divider=True)
                # st.write(journey["summary"] + "\n\n")
                for i, step in enumerate(journey["steps"]):
                    step_id = f'{journey_name}_{i}'
                    # if st.session_state.chat_state != i + 1:
                        # col1, col2 = st.columns([5, 1])

                        # st.write(f'#### {step["name"]}')
                    if st.button(
                        step["name"],
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
            ):
                chat_elements(chat_state, journey_name)


if __name__ == "__main__":
    main()
