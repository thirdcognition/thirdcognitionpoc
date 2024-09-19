import io
import re
import operator
from typing import Annotated, Dict, List, Set, TypedDict, Union
from langchain.chains.combine_documents.reduce import (
    split_list_of_docs,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from lib.chains.base_parser import get_text_from_completion
from lib.chains.init import get_chain

from lib.document_tools import (
    a_semantic_splitter,
)
from lib.helpers import parse_content_dict, prepare_contents
from lib.load_env import SETTINGS


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
class FindTopicsState(TypedDict):
    # Notice here we use the operator.add
    # This is because we want combine all the summaries we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    filename: str
    url: str
    source: str
    contents: List[Union[Document, str]]  # Annotated[list[Document], operator.add]
    instructions: str
    # generated
    content_topics: List[Dict]
    found_topics: Annotated[list, operator.add]
    summary: str
    topics: Set[str]


class FindTopicsConfig(TypedDict):
    instructions: str = None
    run_split: bool = False


# Here we generate a summary, given a document
async def setup_content(state: FindTopicsState, config: RunnableConfig):

    if config["configurable"].get("run_split", False):
        text = get_text_from_completion(state["contents"])

        split = await a_semantic_splitter(text)
        split = [
            Document(page_content=clean_dividers(doc), metadata={"snippet": i})
            for i, doc in enumerate(split)
        ]
        response = split_list_of_docs(
            split, length_function, SETTINGS.default_llms.instruct_detailed.char_limit
        )
    else:
        response = (
            [state["contents"]]
            if not isinstance(state["contents"], List)
            else state["contents"]
        )

    return {
        "split_complete": True,
        "contents": response,
        "instructions": (
            config["configurable"]["instructions"]
            if "instructions" in config["configurable"]
            else None
        ),
    }


class SummaryState(TypedDict):
    url: str
    filename: str
    page: int
    content: Union[str, Document]
    instructions: str


# Here we generate a summary, given a document
async def reformat_content(state: SummaryState):
    response = {}
    content, prev_page, next_page = prepare_contents(
        state["content"], state["prev_page_content"], state["next_page_content"]
    )
    if state["instructions"] is not None:
        response = await get_chain("topic_formatter_guided").ainvoke(
            {
                "context": content,
                "next_page": next_page,
                "prev_page": prev_page,
                "instructions": state["instructions"],
            }
        )
    else:
        response = await get_chain("topic_formatter").ainvoke(
            {
                "context": content,
                "next_page": next_page,
                "prev_page": prev_page,
            }
        )
    metadata = {}
    if isinstance(state["content"], Document):
        metadata = state["content"].metadata.copy()

    if "filename" in state and state["filename"] is not None:
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
        "found_topics": [
            {"index": state["index"], "page": state["page"], "items": items}
        ]
    }


# Here we define the logic to map out over the documents
# We will use this an edge in the graph
def map_reformat(state: FindTopicsState):
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


async def concat_reformat(state: FindTopicsState, config: RunnableConfig):
    sorted_reformat: List[Dict] = sorted(
        state["found_topics"], key=lambda x: x["index"]
    )
    return {
        "reformat_complete": True,
        "content_topics": [
            {
                "page_content": item["document"],
                "page_number": result["page"] if result["page"] != -1 else result["index"] + 1,
                "topic_index": item["topic_index"],
                "metadata": item["document"].metadata,
                "topic": item["topic"] or "",
            }
            for result in sorted_reformat
            for item in result["items"]
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
        "all_topics": set(
            [
                item.get("topic", "")
                for result in sorted_reformat
                for item in result["items"]
            ]
        ),
    }


# Construct the graph
# Nodes:
find_topics_graph = StateGraph(FindTopicsState, FindTopicsConfig)
find_topics_graph.add_node("setup_content", setup_content)
find_topics_graph.add_node("reformat_content", reformat_content)  # same as before
find_topics_graph.add_node("concat_reformat", concat_reformat)

# Edges:
find_topics_graph.add_edge(START, "setup_content")
find_topics_graph.add_conditional_edges(
    "setup_content", map_reformat, ["reformat_content"]
)
find_topics_graph.add_edge("reformat_content", "concat_reformat")
find_topics_graph.add_edge("concat_reformat", END)


find_topics = find_topics_graph.compile()
