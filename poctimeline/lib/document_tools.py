from functools import cache
from typing import Dict, List
# from langchain_chroma import Chroma
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

# from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema.document import Document

from lib.load_env import SETTINGS
from chains.init import  get_embeddings

@cache
def get_text_splitter(chunk_size, chunk_overlap):
    if chunk_size < chunk_overlap:
        chunk_overlap = chunk_size / 2

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

def split_text(text, split=SETTINGS.default_llms.instruct.char_limit, overlap=100):
    text_len = len(text)
    split = text_len // (text_len / split)
    if (text_len - split) > overlap:
        splitter = get_text_splitter(chunk_size=split, chunk_overlap=overlap)
        return splitter.split_text(text)
    else:
        return [text]


def join_documents(texts, split=SETTINGS.default_llms.instruct.char_limit):
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


def semantic_splitter(text, split=SETTINGS.default_llms.instruct.char_limit, progress_cb=None):
    if len(text) > 1000:
        less_text = split_text(text, SETTINGS.default_embeddings.default.char_limit, 0)
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


def split_markdown(text, split=SETTINGS.default_llms.instruct.char_limit):
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
