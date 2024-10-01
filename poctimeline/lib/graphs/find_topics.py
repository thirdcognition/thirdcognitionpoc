from math import inf
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

from lib.chains.hierarchy_compiler import get_hierarchy
from lib.chains.init import get_chain

from lib.document_tools import (
    a_semantic_splitter,
)
from lib.helpers import (
    get_text_from_completion,
    get_unique_id,
)
from lib.load_env import SETTINGS
from lib.models.topics import (
    ParsedTopic,
    TopicDataTable,
    get_topic_doc_context,
    get_topic_str,
    parse_topic_items,
    prepare_topic_contents,
)
from lib.models.reference import Reference, ReferenceType

# from lib.prompts.topics import TOPIC_COMBINE_INSTRUCT_TAGS


def clean_dividers(text: str) -> str:
    text = re.sub(r"[pP]age [0-9]+", "", text)
    text = re.sub(r"[iI]mage [0-9]+", "", text)
    return text


# take text(s) and (join and) split them into max_text_len chunks using semantic splitter
# for each text chunk in parallel search_topics text chunk with llm
# join all search_topics text chunks into one text and run semantic split with max_text_len
# using ideas/concepts structure using previously found ideas/concepts iterate through all chunks to mine for ideas/concepts in a structured manner
# for each idea/concept generate combination document for all found content
# for each idea/concept generate summary
# return a structure with 1. search_topics content, 2. ideas/concepts, 3. general summary using ideas/concepts


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
    all_topics: Set[str]
    setup_content_complete: bool = False
    search_topics_content_complete: bool = False
    concat_search_topics_complete: bool = False


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
        "setup_content_complete": True,
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
async def search_topics_content(state: SummaryState):
    response = {}
    content, prev_page, next_page = prepare_topic_contents(
        state["content"],
        state["prev_page_content"],
        state["next_page_content"],
        format_callback=get_topic_doc_context,
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

    items = parse_topic_items(
        response,
        state,
        (
            state["content"].metadata.copy()
            if isinstance(state["content"], Document)
            else {}
        ),
    )

    return {
        "found_topics": [
            {"index": state["index"], "page": state["page"], "items": items}
        ]
    }


# Here we define the logic to map out over the documents
# We will use this an edge in the graph
def map_search_topics(state: FindTopicsState):
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    return [
        Send(
            "search_topics_content",
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


def get_result_dict(result, reserved_ids=None):
    if isinstance(result, Dict):
        return {
            "page_content": result["document"],
            "references": (
                Reference(
                    id=result["source"],
                    type=ReferenceType.source,
                    index=(result["page"] if result["page"] != -1 else result["index"] + 1),
                )
                if result["source"]
                else None
            ),
            "page": (result["page"] if result["page"] != -1 else result["index"] + 1),
            "topic_index": result["topic_index"],
            "metadata": result["document"].metadata,
            "topic": result["topic"] or "",
            "summary": result["summary"] or "",
            "instruct": result["instruct"] or "",
            "id": (
                result["id"]
                if "id" in result and result["id"]
                else get_unique_id(
                    f"{result['index']}-{result['topic_index']}-{result['topic']}",
                    reserved_ids,
                )
            ),
        }
    if isinstance(result, TopicDataTable):
        return result.__dict__


async def concat_search_topics(state: FindTopicsState, config: RunnableConfig):
    instructions = state["instructions"] if "instructions" in state else None
    sorted_search_topics: List[Dict] = sorted(
        state["found_topics"], key=lambda x: x["index"]
    )
    valid_ids = [
        item["id"]
        for result in sorted_search_topics
        for item in result["items"]
        if item["id"] is not None
    ]

    joined_topic_items: List[ParsedTopic]
    unwrapped_hierarchy, inverted_hierarchy, joined_topic_items, removed_ids = (
        await get_hierarchy(
            [item for result in sorted_search_topics for item in result["items"]],
            valid_ids=valid_ids,
            hierarchy_chain_name="topic_hierarchy",
            hierarchy_item_formatter=lambda x: get_topic_str(
                x,
                one_liner=True,
            ),
            join_chain_name="topic_combiner",
            join_item_formatter=lambda x: get_topic_str(
                x,
            ),
        )
    )

    # # store all topic ids to a variable
    filtered_topics_by_id = {
        item["id"]: item  # get_topic_document(item, result, instructions)
        for result in sorted_search_topics
        for item in result["items"]
        if item["id"] not in removed_ids
    }

    for new_topic in joined_topic_items:
        result = parse_topic_items(new_topic, state)
        filtered_topics_by_id[new_topic.id] = result[0]

    new_topics = sorted(
        filtered_topics_by_id.values(), key=lambda x: (x["page"], x["topic_index"])
    )
    reserved_ids = list(filtered_topics_by_id.keys())

    return {
        "search_topics_content_complete": True,
        "concat_search_topics_complete": True,
        "content_topics": [get_result_dict(result) for result in new_topics],
        "all_topics": set([item.get("topic", "") for item in new_topics]),
    }


# Construct the graph
# Nodes:
find_topics_graph = StateGraph(FindTopicsState, FindTopicsConfig)
find_topics_graph.add_node("setup_content", setup_content)
find_topics_graph.add_node(
    "search_topics_content", search_topics_content
)  # same as before
find_topics_graph.add_node("concat_search_topics", concat_search_topics)

# Edges:
find_topics_graph.add_edge(START, "setup_content")
find_topics_graph.add_conditional_edges(
    "setup_content", map_search_topics, ["search_topics_content"]
)
find_topics_graph.add_edge("search_topics_content", "concat_search_topics")
find_topics_graph.add_edge("concat_search_topics", END)


find_topics = find_topics_graph.compile()
