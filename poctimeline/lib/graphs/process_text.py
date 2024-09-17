import asyncio
import io
import os
import re
import operator
from typing import Annotated, Dict, List, Literal, Set, TypedDict, Union
from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langchain_community.document_loaders.url_playwright import PlaywrightURLLoader

from lib.chains.base_parser import get_text_from_completion
from lib.chains.init import get_chain
from lib.db_tools import (
    db_source_exists,
    get_db_sources,
    update_db_source_rag,
)
from lib.document_parse import SourceType, process_source_contents
from lib.document_tools import (
    a_semantic_splitter,
    markdown_to_text,
)
from lib.load_env import SETTINGS
from lib.models.sqlite_tables import (
    SourceContentPage,
    SourceContents,
)


def clean_dividers(text: str) -> str:
    text = re.sub(r"[pP]age [0-9]+", "", text)
    text = re.sub(r"[iI]mage [0-9]+", "", text)
    return text


# take text(s) and (join and) split them into max_text_len chunks using semantic splitter
# for each text chunk in parallel reformat text chunk with llm
# join all reformat text chunks into one text and run semantic split with max_text_len
# using ideas/concepts structure using previously found ideas/concepts iterate through all chunks to mine for ideas/concepts in a structured manner
# for each idea/concept generate combination document for all found content
# for each idea/concept generate summary
# return a structure with 1. reformat content, 2. ideas/concepts, 3. general summary using ideas/concepts


def length_function(documents: List[Union[Document, str]]) -> int:
    """Get number of tokens for input contents."""
    return sum(
        len(doc.page_content if isinstance(doc, Document) else doc) for doc in documents
    )


# This will be the overall state of the main graph.
# It will contain the input document contents, corresponding
# summaries, and a final summary.
class ProcessTextState(TypedDict):
    # Notice here we use the operator.add
    # This is because we want combine all the summaries we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    categories: List[str]
    file: io.BytesIO
    filename: str
    url: str
    source: str
    # semantic_contents: List[Document]
    contents: List[Union[Document, str]]  # Annotated[list[Document], operator.add]
    final_contents: List[Document]
    # reformat_contents: List[Document]  # Annotated[list[Document], operator.add]
    source_content_topics: List[SourceContentPage]
    reformatted_txt: Annotated[list, operator.add]
    summary: str
    topics: Set[str]
    source_contents: SourceContents
    instructions: str
    get_source_complete: bool = False
    split_complete: bool = False
    reformat_complete: bool = False
    collapse_complete: bool = False
    finalize_complete: bool = False
    rag_complete: bool = False


class ProcessTextConfig(TypedDict):
    instructions: str = None
    overwrite_sources: bool = False
    update_rag: bool = False
    summarize: bool = True
    rewrite_text: bool = True
    run_split_for_source: bool = True


async def should_rewrite_content(state: ProcessTextState, config: RunnableConfig):
    source = state["filename"] if "filename" in state else state["url"]
    if config["configurable"]["rewrite_text"] or not db_source_exists(source):
        return "split_content"
    else:
        return "get_source_content"


# Here we generate a summary, given a document
async def split_content(state: ProcessTextState, config: RunnableConfig):
    text = ""
    if config["configurable"]["update_rag"] and (
        "categories" not in state.keys()
        or state["categories"] is None
        or state["categories"] == []
        or state["categories"] == ""
    ):
        raise ValueError("Category must be provided if update_rag is set True")
    if "filename" in state.keys() and state["filename"] is not None:
        if state["file"] is None or len(state["file"].getvalue()) == 0:
            raise ValueError("File must be provided")

        filename = state["filename"]
        filetype = os.path.basename(filename).split(".")[-1]
        texts = await process_source_contents(
            filename,
            uploaded_file=state["file"],
            categories=state["categories"],
            overwrite=config["configurable"]["overwrite_sources"],
        )
        text = "\n\n".join(texts)
        if filetype == "md":
            text = markdown_to_text(text)
    elif "url" in state.keys() and state["url"] is not None:
        # TODO: Figure out image parsing.
        if (
            db_source_exists(state["url"])
            and not config["configurable"]["overwrite_sources"]
        ):
            text = "\n\n".join(get_db_sources(state["url"])[state["url"]].texts)
        else:
            loader = PlaywrightURLLoader(
                urls=[state["url"]],
                remove_selectors=["header", "footer"],
            )
            content = await loader.aload()
            texts = await process_source_contents(
                state["url"],
                type=SourceType.URL,
                text_content="\n\n".join([doc.page_content for doc in content]),
                categories=state["categories"],
                overwrite=config["configurable"]["overwrite_sources"],
            )
            text = "\n\n".join(texts)
    else:
        if config["configurable"]["update_rag"] and (
            "source" not in state.keys()
            or state["source"] is None
            or state["source"] == ""
        ):
            raise ValueError("Source must be provided if update_rag is set True")
        if isinstance(state["contents"], List):
            text = "\n".join(
                [
                    (
                        doc.page_content.strip()
                        if isinstance(doc, Document)
                        else doc.strip()
                    )
                    for doc in state["contents"]
                ]
            )
        else:
            text = state["contents"]

    split = await a_semantic_splitter(text)
    split = [
        Document(page_content=clean_dividers(doc), metadata={"snippet": i})
        for i, doc in enumerate(split)
    ]
    response = split_list_of_docs(
        split, length_function, SETTINGS.default_llms.instruct_detailed.char_limit
    )

    return {
        "split_complete": True,
        # "semantic_contents": split,
        "contents": response,
        "instructions": (
            config["configurable"]["instructions"]
            if "instructions" in config["configurable"]
            else None
        ),
    }


async def get_source_content(state: Dict, config: RunnableConfig):
    source = state["filename"] if state["filename"] is not None else state["url"]
    sources = get_db_sources(source)
    source_data = sources[source]
    if source_data is None:
        raise ValueError(f"Source {source} not found")

    if config["configurable"].get("run_split_for_source", False):
        split = await a_semantic_splitter(source_data.texts)
        split = [
            Document(page_content=clean_dividers(doc), metadata={"snippet": i})
            for i, doc in enumerate(split)
        ]
        response = split_list_of_docs(
            split, length_function, SETTINGS.default_llms.instruct_detailed.char_limit
        )
    else:
        print("Skip split")
        split = []
        response = []

    return {
        # "split_complete": True,
        # "semantic_contents": split,
        "contents": response,
        "instructions": (
            config["configurable"]["instructions"]
            if "instructions" in config["configurable"]
            else None
        ),
        "reformat_complete": True,
        # "reformat_contents": source_data.texts,
        "source_content_topics": source_data.source_contents.formatted_topics,
        "get_source_complete": True,
        "summary": source_data.source_contents.summary,
        "topics": source_data.source_contents.topics,
    }


class SummaryState(TypedDict):
    url: str
    file: str
    page: int
    content: Union[str, Document]
    instructions: str


def parse_content_dict(data):
    result = []
    for item in data["children"]:
        if item["tag"] == "output":
            topic = next(
                (
                    child["body"]
                    for child in item["children"]
                    if child["tag"] == "topic"
                ),
                None,
            )
            content = item["body"]
            summary = next(
                (
                    child["body"]
                    for child in item["children"]
                    if child["tag"] == "summary"
                ),
                None,
            )
            result.append({"topic": topic, "content": content, "summary": summary})
        else:
            result.extend(parse_content_dict(item))
    return result


# Here we generate a summary, given a document
async def reformat_content(state: SummaryState):
    response = {}
    content = (
        [state["content"]]
        if not isinstance(state["content"], list)
        else state["content"]
    )
    next_page = (
        [state["next_page_content"]]
        if not isinstance(state["next_page_content"], list)
        else state["next_page_content"]
    )
    prev_page = (
        [state["prev_page_content"]]
        if not isinstance(state["prev_page_content"], list)
        else state["prev_page_content"]
    )
    content = "\n".join(
        [
            ((item.page_content if isinstance(item, Document) else item))
            for item in content
        ]
    )
    next_page = "\n".join(
        [
            (item.page_content if isinstance(item, Document) else repr(item))
            for item in next_page
        ]
    )
    next_page = next_page[:1000] if len(next_page) > 1000 else next_page
    prev_page = "\n".join(
        [
            (item.page_content if isinstance(item, Document) else repr(item))
            for item in prev_page
        ]
    )
    prev_page = prev_page[-1000:] if len(prev_page) > 1000 else prev_page
    if state["instructions"] is not None:
        response = await get_chain("text_formatter_guided").ainvoke(
            {
                "context": content,
                "next_page": next_page,
                "prev_page": prev_page,
                "instructions": state["instructions"],
            }
        )
    else:
        response = await get_chain("text_formatter").ainvoke(
            {
                "context": content,
                "next_page": next_page,
                "prev_page": prev_page,
            }
        )
    metadata = {}
    if isinstance(state["content"], Document):
        metadata = state["content"].metadata.copy()

    if "file" in state and state["file"] is not None:
        metadata["filename"] = state["filename"]
    if "page" in state and state["page"] is not None:
        metadata["page"] = state["page"]
    if "url" in state and state["url"] is not None:
        metadata["url"] = state["url"]

    tags = None

    if "tags" in response:
        tags = response["tags"]
        if "thinking" in tags:
            metadata["thinking"] = tags["thinking"]

    if "parsed" in response:
        parsed_content = parse_content_dict(response["parsed"])
        items = []
        for i, topic in enumerate(parsed_content):
            items.append(
                {
                    "document": Document(
                        page_content=topic["content"],
                        metadata={**metadata, "topic": i + 1},
                    ),
                    "topic_index": i + 1,
                    "topic": topic["topic"],
                    "summary": topic["summary"],
                }
            )
        if len(items) > 1:
            average_lengths = [
                len(item["document"].page_content) for item in items[:-1]
            ]
            average_length = (
                sum(average_lengths) / len(average_lengths) if average_lengths else 0
            )
            last_item = items.pop()
            last_item["topic_index"] = 0
            last_item["summary"] = (
                last_item["summary"]
                if last_item["summary"] is not None
                else last_item["document"].page_content[:200] + "..."
            )
            last_item["topic"] = (
                f"Overview: " + last_item["topic"]
                if last_item["topic"] is not None
                else "Overview: "
            )

            items.insert(0, last_item)

    else:
        doc = Document(
            page_content=get_text_from_completion(response), metadata=metadata
        )
        topics = []
        summaries = []
        if tags is not None:
            if "topic" in tags:
                topics = str(tags["topic"]).split("\n\n")
            if "summary" in tags:
                summaries = str(tags["summary"]).split("\n\n")

        items = [
            {
                "document": doc,
                "topic": topics,
                "summary": summaries,
            }
        ]

    return {
        "reformatted_txt": [
            {"index": state["index"], "page": state["page"], "items": items}
        ]
    }


# Here we define the logic to map out over the documents
# We will use this an edge in the graph
def map_reformat(state: ProcessTextState):
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    return [
        Send(
            "reformat_content",
            {
                "content": content,
                "instructions": (
                    state["instructions"] if "instructions" in state else None
                ),
                "prev_page_content": (state["contents"][page - 1] if page > 0 else ""),
                "next_page_content": (
                    state["contents"][page + 1]
                    if page < len(state["contents"]) - 1
                    else ""
                ),
                "page": (
                    state["page"]
                    if "page" in state and state["page"] != -1
                    else (page + 1)
                ),
                "index": page,
                "filename": state["filename"] if "filename" in state else None,
                "url": state["url"] if "url" in state else None,
            },
        )
        for page, content in enumerate(state["contents"])
    ]


# async def combine_summaries(concept_summaries, concept_id, concepts: List[str]):
#     concept_summaries[concept_id] = concepts[0]
#     if len(concepts) > 1:
#         concept_summaries[concept_id] = get_text_from_completion(
#             await get_chain("combine_bullets").ainvoke(
#                 {"context": "\n- ".join(concepts)}
#             )
#         )


async def concat_reformat(state: ProcessTextState, config: RunnableConfig):
    sorted_reformat: List[Dict] = sorted(
        state["reformatted_txt"], key=lambda x: x["index"]
    )
    return {
        "reformat_complete": True,
        # "reformat_contents": [
        #     get_text_from_completion(result["document"]) for result in sorted_reformat
        # ],
        "source_content_topics": [
            SourceContentPage(
                page_content=get_text_from_completion(item["document"]),
                page_number=(
                    result["page"] if result["page"] != -1 else result["index"] + 1
                ),
                topic_index=item["topic_index"],
                metadata=item["document"].metadata,
                topic=item["topic"] or "",
            )
            for result in sorted_reformat
            for item in list(
                filter(
                    lambda x: not str(x.get("topic", "")).startswith("Overview:"),
                    result["items"],
                )
            )
        ],
        "summary": "\n".join(
            [
                "\n".join(
                    item["summary"] or ""
                    for result in sorted_reformat
                    for item in result["items"]
                )
            ]
        ),
        "topics": set(
            [
                item.get("topic", "")
                for result in sorted_reformat
                for item in list(
                    filter(
                        lambda x: not str(x.get("topic", "")).startswith("Overview:"),
                        result["items"],
                    )
                )
            ]
        ),
    }


async def finalize_content(state: ProcessTextState, config: RunnableConfig):
    instructions = state["instructions"] if "instructions" in state else None
    summary_collapsed = ""
    if config["configurable"]["summarize"]:
        if instructions is not None:
            response = await get_chain("summary_guided").ainvoke(
                {
                    "context": state["summary"],
                    "instructions": instructions,
                }
            )
        else:
            response = await get_chain("summary").ainvoke({"context": state["summary"]})
        summary_collapsed = get_text_from_completion(response)

    formatted_content = "\n".join(
        [page.page_content for page in state["source_content_topics"]]
    )

    flat_contents: List[Document] = []
    for item in state["contents"]:
        if isinstance(item, Document):
            flat_contents.append(item)
        elif isinstance(item, List):
            flat_contents.extend(item)
        else:
            # print(f"{state["contents"]}")
            raise ValueError(f"Unknown type {type(item)}: {item=}")

    return {
        "summarize_complete": True,
        "collapse_complete": True,
        "finalize_complete": True,
        "final_contents": flat_contents,
        "source_contents": SourceContents(
            formatted_topics=state["source_content_topics"],
            formatted_content=formatted_content,
            topics=state["topics"],
            summary=summary_collapsed,
        ),
    }


async def should_rag(state: ProcessTextState, config: RunnableConfig):
    if config["configurable"]["update_rag"]:
        return "rag_content"
    else:
        return END


async def rag_content(state: ProcessTextState, config: RunnableConfig):
    content = state["source_contents"]
    filetype = None
    if state["filename"] is not None:
        filetype = os.path.basename(state["filename"]).split(".")[-1]

    if config["configurable"]["overwrite_sources"] or state.get("split_complete", False):
        texts = [page.page_content for page in state["final_contents"]]
        update_db_source_rag(
            state["filename"] or state["url"],
            state["categories"],
            texts,
            content,
            filetype=filetype,
        )

    return {"rag_complete": True}


# Construct the graph
# Nodes:
process_text_graph = StateGraph(ProcessTextState, ProcessTextConfig)
process_text_graph.add_node("split_content", split_content)
process_text_graph.add_node("reformat_content", reformat_content)  # same as before
process_text_graph.add_node("concat_reformat", concat_reformat)
process_text_graph.add_node("get_source_content", get_source_content)
# process_text_graph.add_node("collapse_content", collapse_content)
process_text_graph.add_node("finalize_content", finalize_content)
process_text_graph.add_node("rag_content", rag_content)

# Edges:
process_text_graph.add_conditional_edges(START, should_rewrite_content)
# process_text_graph.add_edge(START, "split_content")
process_text_graph.add_conditional_edges(
    "split_content", map_reformat, ["reformat_content"]
)
process_text_graph.add_edge("reformat_content", "concat_reformat")
process_text_graph.add_edge("concat_reformat", "finalize_content")
# process_text_graph.add_conditional_edges("concat_reformat", should_collapse)
# process_text_graph.add_edge("collapse_content", "finalize_content")
process_text_graph.add_edge("get_source_content", "finalize_content")

process_text_graph.add_conditional_edges("finalize_content", should_rag)
process_text_graph.add_edge("rag_content", END)

process_text = process_text_graph.compile()
