from functools import cache
import pprint as pp
from typing import Dict, List
import streamlit as st
# from langchain_chroma import Chroma
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableBranch,
    RunnableWithMessageHistory,
    RunnableLambda
)
from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema.document import Document
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_huggingface.llms import HuggingFacePipeline
from chains.rag_chain import RagChain
from lib.db_tools import get_vectorstore, get_vectorstore_as_retriever
from lib.load_env import EMBEDDING_CHAR_LIMIT, INSTRUCT_CHAR_LIMIT
from chains.prompts import question_classifier
from chains.init_chains import  get_chain, get_embeddings, get_llm, init_llms, print_params

@cache
def get_text_splitter(chunk_size, chunk_overlap):
    if chunk_size < chunk_overlap:
        chunk_overlap = chunk_size / 2

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = {}

    if session_id not in st.session_state["chat_history"]:
        st.session_state["chat_history"][session_id] = ChatMessageHistory()
    return st.session_state["chat_history"][session_id]

def get_chain_with_history(chain_id: str, chain:RunnableSequence):
    if "history_chains" not in st.session_state:
        st.session_state["history_chains"] = {}
    if chain_id in st.session_state["history_chains"]:
        return st.session_state["history_chains"][chain_id]

    history_chain = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        output_messages_key="answer",
        history_messages_key="chat_history"
    )

    st.session_state["history_chains"][chain_id] = history_chain
    return history_chain

compressor = None

rag_chains = {}

def rag_chain(store_id:str, embedding_id="hyde", reset=False, with_history=False, with_chat=False, amount_of_documents=5) -> RunnableSequence:
    global rag_chains
    chat_chain = get_chain("chat")() | RunnableLambda(lambda x: x["answer"])

    chain_id = f"{store_id}-{embedding_id}-{"history" if with_history else "nohistory"}-{"chat" if with_chat else "nochat"}-#{amount_of_documents}"
    if chain_id in rag_chains and not reset:
        return rag_chains[chain_id]

    print(f"Initializing RAG chain: {chain_id} with {with_history=} and {with_chat=}")

    retriever = get_vectorstore_as_retriever(store_id, embedding_id, amount_of_documents)
    rag_chain = RagChain(retriever, llm=get_llm("chat"))()

    classification_chain = (
        RunnableLambda(lambda x: {"question": x["question"]}) |
        question_classifier.get_chat_prompt_template() |
        get_llm("tester") |
        # log_chain |
        RunnableLambda(lambda x: "yes" in str(x.content).lower())
    )

    def skip_search(params):
        return {
            "question": params["question"],
            # "context": params["context"] if "context" in params.keys() else [],
            "chat_history": params["chat_history"] if "chat_history" in params.keys() else []
        }

    qa_chain = (
        RunnableParallel({
            "is_question":
                RunnableBranch(
                    (lambda x: with_chat, classification_chain),
                    RunnableLambda(lambda x: True)
                ),
            "orig_params": RunnablePassthrough()
        }) |
        RunnableBranch(
            (lambda x: x["is_question"], rag_chain),
            RunnableLambda(skip_search) | chat_chain
        )
    )

    if with_history:
        qa_chain = get_chain_with_history(chain_id, qa_chain)

    rag_chains[chain_id] = qa_chain

    return qa_chain

def split_text(text, split=INSTRUCT_CHAR_LIMIT, overlap=100):
    text_len = len(text)
    split = text_len // (text_len / split)
    if (text_len - split) > overlap:
        splitter = get_text_splitter(chunk_size=split, chunk_overlap=overlap)
        return splitter.split_text(text)
    else:
        return [text]


def join_documents(texts, split=INSTRUCT_CHAR_LIMIT):
    joins = []
    text_join = ""

    total_len = 0
    for text in texts:
        _text = ""
        if isinstance(text, str):
            _text = text
        else:
            _text = text.page_content

        total_len += len(_text)

    chunks = total_len // split + 1
    chunk_length = total_len // chunks

    for text in texts:
        _text = ""
        if isinstance(text, str):
            _text = text
        else:
            _text = text.page_content

        if len(text_join) > 100 and (len(text_join) + len(_text)) > chunk_length:
            joins.append(text_join)
            text_join = _text
        else:
            text_join += _text + "\n\n"

    joins.append(text_join)

    return joins


def semantic_splitter(text, split=INSTRUCT_CHAR_LIMIT, progress_cb=None):
    init_llms()

    if len(text) > 1000:
        less_text = split_text(text, EMBEDDING_CHAR_LIMIT, 0)
    else:
        less_text = [text]

    semantic_splitter = SemanticChunker(
        get_embeddings("base"), breakpoint_threshold_type="percentile"
    )

    texts = []
    for i, txt in enumerate(less_text):
        texts = texts + (semantic_splitter.split_text(txt) if len(txt) > 100 else [txt.strip()])
        if progress_cb != None and callable(progress_cb):
            progress_cb(len(less_text), i)

    return join_documents(texts, split)


def split_markdown(text, split=INSTRUCT_CHAR_LIMIT):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )

    texts = markdown_splitter.split_text(text)

    return join_documents(texts, split)

def create_document_lists(
    list_of_strings: List[str], list_of_thoughts: List[str] = None, source="local", list_of_metadata: List[Dict[str, any]] = None
):
    doc_list = []

    for index, item in enumerate(list_of_strings):
        thinking = list_of_thoughts[index] if list_of_thoughts else None
        metadata = list_of_metadata[index] if list_of_metadata else None
        if metadata is None:
            metadata = {"source": source, "thought": thinking, "index": index} if thinking else {"source": source, "index": index}

        if len(item) > 3000:
            split_texts = split_text(item, split=3000, overlap=100)
            for split_item in split_texts:

                doc = Document(
                    page_content=split_item,
                    metadata=metadata,
                )
                doc_list.append(doc)
        else:
            doc = Document(
                page_content=item,
                metadata=metadata,
            )
            doc_list.append(doc)

    return doc_list
