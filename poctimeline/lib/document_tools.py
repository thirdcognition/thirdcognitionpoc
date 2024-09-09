import concurrent.futures
from functools import cache
import re
from bs4 import BeautifulSoup
from markdown import markdown
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
from lib.chains.init import get_embeddings
from lib.models.sqlite_tables import SourceContents


@cache
def get_text_splitter(chunk_size, chunk_overlap):
    if chunk_size < chunk_overlap:
        chunk_overlap = chunk_size / 2

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


def markdown_to_text(markdown_string):
    """Converts a markdown string to plaintext"""

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    # remove code snippets
    html = re.sub(r"<pre>(.*?)</pre>", "\1", html)
    html = re.sub(r"<code>(.*?)</code>", "\1", html)
    html = html.replace("```markdown", "")
    html = html.replace("```", "")

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = "".join(soup.findAll(string=True))

    return text


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

        if (
            len(_text) > 100
            and (len(text_join) + len(_text)) > chunk_length
            and len(text_join) > 100
        ):
            joins.append(text_join)
            text_join = _text
        else:
            text_join += _text + "\n\n"

    joins.append(text_join)

    return joins


def semantic_splitter(
    text, split=SETTINGS.default_llms.instruct.char_limit, progress_cb=None
):
    if len(text) > 1000:
        less_text = split_text(text, SETTINGS.default_embeddings.default.char_limit, 0)
    else:
        less_text = [text]

    semantic_splitter = SemanticChunker(
        get_embeddings("base"), breakpoint_threshold_type="percentile"
    )

    texts = []
    for i, txt in enumerate(less_text):
        texts = texts + (
            semantic_splitter.split_text(txt)
            if len(txt.strip()) > 100
            else [txt.strip()]
        )
        if progress_cb != None and callable(progress_cb):
            progress_cb(len(less_text), i)

    return join_documents(texts, split)


def __split_text(semantic_splitter, txt):
    return (
        semantic_splitter.split_text(txt) if len(txt.strip()) > 100 else [txt.strip()]
    )


async def a_semantic_splitter(
    text: str, split=SETTINGS.default_llms.instruct.char_limit, progress_cb=None
):
    if len(text) > 1000:
        less_text = split_text(text, SETTINGS.default_embeddings.default.char_limit, 0)
    else:
        less_text = [text]

    semantic_splitter = SemanticChunker(
        get_embeddings("base"), breakpoint_threshold_type="percentile"
    )

    # async def split_text_async(txt):
    #     return semantic_splitter.split_text(txt) if len(txt.strip()) > 100 else [txt.strip()]

    # tasks = [split_text_async(txt) for txt in less_text]
    # texts = []
    # for i, task in enumerate(asyncio.as_completed(tasks)):
    #     texts.extend(await task)
    #     if progress_cb != None and callable(progress_cb):
    #         progress_cb(len(less_text), i)

    texts = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_text = {
            executor.submit(__split_text, semantic_splitter, txt): txt
            for txt in less_text
        }
        for future in concurrent.futures.as_completed(future_to_text):
            texts.extend(future.result())
            if progress_cb != None and callable(progress_cb):
                progress_cb(len(less_text), len(texts))

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

    md_texts = markdown_splitter.split_text(text)
    texts = [
        text for md_text in md_texts for text in split_text(md_text.page_content, split)
    ]
    # avg_len = sum(len(text) for text in texts) / len(texts)
    # min_len = min(len(text) for text in texts)
    # max_len = max(len(text) for text in texts)

    # print(f"Average length of each string in texts: {avg_len}, Min length: {min_len}, Max length: {max_len}, Total amount of strings: {len(texts)}")

    return join_documents(texts, split)


def create_document_lists(
    list_of_strings: List[str],
    source="local",
    list_of_metadata: List[Dict[str, any]] = None,
):
    doc_list = []

    for index, item in enumerate(list_of_strings):
        metadata = list_of_metadata[index] if list_of_metadata else None
        if metadata is None:
            metadata = {"source": source, "index": index}

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


def get_rag_chunks(
    texts: List[str],
    source: str,
    category: List[str],
    content: SourceContents,
    filetype: str = "txt",
) -> tuple[List[str], List[Dict[str, Dict]], List[Dict]]:
    if texts is not None and filetype != "md":
        rag_split = split_text(
            "\n".join(texts),
            SETTINGS.default_embeddings.default.char_limit,
            SETTINGS.default_embeddings.default.overlap,
        )
    elif filetype == "md":
        rag_split = split_text(
            markdown_to_text("\n".join(texts)),
            SETTINGS.default_embeddings.default.char_limit,
            SETTINGS.default_embeddings.default.overlap,
        )

    rag_ids = [source + "_" + str(i) for i in range(len(rag_split))]
    rag_metadatas = [
        {
            "source": source,
            "category": ", ".join(category),
            "filetype": filetype,
            "split": i,
        }
        for i in range(len(rag_split))
    ]

    if content.formatted_content is not None and content.formatted_content:
        if (
            len(content.formatted_content)
            > SETTINGS.default_embeddings.default.char_limit
        ):
            formatted_split = split_text(
                content.formatted_content,
                SETTINGS.default_embeddings.default.char_limit,
                SETTINGS.default_embeddings.default.overlap,
            )
        else:
            formatted_split = [content.formatted_content]

        rag_split = rag_split + formatted_split
        rag_ids = rag_ids + [
            source + "_formatted_" + str(i) for i in range(len(formatted_split))
        ]
        rag_metadatas = rag_metadatas + [
            {
                "source": "formatted_" + source,
                "category": ", ".join(category),
                "filetype": filetype,
                "split": i,
            }
            for i in range(len(formatted_split))
        ]

    if content.concepts is not None and len(content.concepts) > 0:
        concept_split = []
        concept_ids = []
        concept_metadatas = []
        for i, concept in enumerate(content.concepts):
            cur_concept_split = split_text(
                concept.content,
                SETTINGS.default_embeddings.default.char_limit,
                SETTINGS.default_embeddings.default.overlap,
            )
            concept_split += cur_concept_split
            concept_ids += [
                (
                    source + "_concept_" + concept.id + "_" + str(i) + "_" + str(j)
                ).replace(" ", "_")
                for j in range(len(cur_concept_split))
            ]

            concept_metadatas += [
                {
                    "concept_id": concept.id,
                    "concept_category": ", ".join(
                        [category.tag for category in concept.category]
                    ),
                    "split": str(i) + "_" + str(j),
                    "source": concept.reference.source,
                    "page": concept.reference.page_number or -1,
                }
                for j in range(len(cur_concept_split))
            ]

        rag_split = rag_split + concept_split
        rag_ids = rag_ids + concept_ids
        rag_metadatas = rag_metadatas + concept_metadatas

    return rag_split, rag_ids, rag_metadatas
