from functools import cache
from typing import Dict, List
from langchain_chroma import Chroma
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableBranch,
    RunnableWithMessageHistory
)
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema.document import Document
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from lib.db_tools import get_vectorstore
from lib.load_env import EMBEDDING_CHAR_LIMIT, INSTRUCT_CHAR_LIMIT
from lib.chain import  get_embeddings, get_llm, get_prompt, init_llms

@cache
def get_text_splitter(chunk_size, chunk_overlap):
    if chunk_size < chunk_overlap:
        chunk_overlap = chunk_size / 2

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

chat_history_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    global chat_history_store
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    return chat_history_store[session_id]

history_chains = {}

def get_chain_with_history(chain_id: str, chain:RunnableSequence):
    global history_chains
    if chain_id in history_chains:
        return history_chains[chain_id]

    history_chain = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        output_messages_key="answer",
        history_messages_key="chat_history"
    )

    history_chains[chain_id] = history_chain
    return history_chain

compressor = None

def rerank_documents(list_of_documents: list[Document], query: str, amount=5):
    global compressor

    if len(list_of_documents) > 5:
        if compressor is None:
            compressor = FlashrankRerank(top_n=amount)

        ranked_documents = compressor.compress_documents(
            documents=list_of_documents, query=query
        )
        return ranked_documents

    return list_of_documents

def rerank_rag(params: Dict):
    query = params["question"]
    new_content = []
    new_content_strs = []
    def sort_by_references(document: Document):
        return len(document.metadata)

    documents: list[Document] = params["documents"]

    documents.sort(reverse=True, key=sort_by_references)
    for document in documents:
        if document.page_content not in new_content_strs:
            new_content.append(document)
            new_content_strs.append(document.page_content)

    new_content = rerank_documents(new_content, query, 10)

    return new_content

rag_chains = {}

def rag_chain(store_id:str, embedding_id="hyde", prompt_id = "question", llm_id = "chat", reset=False, with_history=False) -> RunnableSequence:
    global rag_chains
    chain_id = f"{store_id}-{embedding_id}-{prompt_id}-{llm_id}"
    if chain_id in rag_chains and not reset:
        return rag_chains[chain_id]

    vectorstore = get_vectorstore(store_id, embedding_id)
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 40, "score_threshold": 0.3},
    )

    llm = get_llm(llm_id)
    prompt = get_prompt(prompt_id)

    def format_params(params):
        question = params["question"]
        for key in question.keys():
            if key == "context":
                params["documents"].append(Document(page_content=question[key]))
            else:
                params[key] = question[key]

        return params

    def add_context(params):
        params["question"] = f"context:\n{params['context']}\n\nquery:\n{params['question']}"
        return params

    # def log_results(params):
    #     print(f"\n\n\nlog\n\n {params=}\n\n\n")
    #     return params

    documents:List = None
    def store_documents(params):
        documents = params["documents"]
        return params

    def set_metadata(params):
        if documents is None or len(documents) == 0:
            return params

        if isinstance(params, str):
            params = {
                "answer": params
            }
        if "metadata" not in params:
            params["metadata"] = {
                "references": documents
            }
        else:
            params["metadata"]["references"] = documents
        return params

    qa_chain = (
        RunnableParallel(
            {
                "context": RunnableParallel({
                    "documents": add_context | retriever, # | log_results,
                    "question": RunnablePassthrough(),
                }) | format_params | rerank_rag | store_documents,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
        | set_metadata
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
        texts = texts + semantic_splitter.split_text(txt)
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