import asyncio
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
from lib.helpers import (
    convert_tags_to_dict,
    get_id_str,
    get_specific_tag,
    get_unique_id,
    parse_content_dict,
    prepare_contents,
    pretty_print,
)
from lib.load_env import SETTINGS
from lib.prompts.topics import TOPIC_COMBINE_INSTRUCT_TAGS


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
async def search_topics_content(state: SummaryState):
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
            pretty_print(topic, "topic", force=True)
            items.append(
                {
                    "document": (
                        Document(
                            page_content=topic["content"].strip(),
                            metadata={**metadata, "topic": i + 1},
                        )
                        if "content" in topic
                        else None
                    ),
                    "topic_index": i + 1,
                    "index": state["index"] if "index" in state else None,
                    "page": state["page"] if "page" in state else None,
                    "instruct": topic["instruct"] if "instruct" in state else None,
                    "topic": topic["topic"].strip() if "topic" in topic else None,
                    "summary": topic["summary"].strip() if "summary" in topic else None,
                    "id": f"{state['page' if 'page' in state else 'filename']}_{i+1}_{get_id_str(topic['id'])}",
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


async def concat_search_topics(state: FindTopicsState, config: RunnableConfig):
    instructions = state["instructions"] if "instructions" in state else None
    sorted_search_topics: List[Dict] = sorted(
        state["found_topics"], key=lambda x: x["index"]
    )

    combined_topics_response = await get_chain("topic_combiner").ainvoke(
        {
            "context": "\n".join(
                [
                    "<item>"
                    + f"<id>{item['id'].replace('\n', ' ').strip()}</id>"
                    + f"<topic>{item['topic'].replace('\n', ' ').strip()}</topic>"
                    + f"<summary>{item['summary'].replace('\n', ' ').strip()}</summary></item>"
                    for result in sorted_search_topics
                    for item in result["items"]
                ]
            )
        }
    )
    found_new_items = get_specific_tag(
        combined_topics_response["parsed"]["children"] or [], "item"
    )
    combined_topics = [
        convert_tags_to_dict(child, TOPIC_COMBINE_INSTRUCT_TAGS)["item"]
        for child in found_new_items
    ]

    # store all topic ids to a variable
    all_topics_by_id = {
        item["id"]: item for result in sorted_search_topics for item in result["items"]
    }
    all_topic_ids = [
        item["id"] for result in sorted_search_topics for item in result["items"]
    ]

    combined_ids = {}
    new_combined_topics = {}
    joined_ids = []
    for topic in combined_topics:
        id_list = []
        if topic["id"] in all_topic_ids:
            id_list.append(topic["id"])
        if isinstance(topic["child_topic"], list):
            for child_topic in topic["child_topic"]:
                if child_topic in all_topic_ids:
                    id_list.append(child_topic)
        elif len(topic["child_topic"]) > 0 and topic["child_topic"] in all_topic_ids:
            id_list.append(topic["child_topic"])
        id = topic["id"]
        joined_ids.extend(id_list)
        id_list = list(set(id_list))

        if len(id_list) > 1:
            if id in combined_ids:
                id = get_unique_id(id, list(combined_ids.keys()))
            combined_ids[id] = id_list
        elif len(id_list) == 1:
            if id in combined_ids:
                id = get_unique_id(id, list(new_combined_topics.keys()))
            new_combined_topics[id] = all_topics_by_id[id_list[0]]

    joined_ids = list(set(joined_ids))
    missing_ids = set(all_topic_ids) - set(joined_ids)
    for id in missing_ids:
        new_combined_topics[id] = all_topics_by_id[id]
    reserved_ids = list(new_combined_topics.keys())

    tasks = {}
    combined_metadata = {}

    pretty_print(combined_ids, "combined_ids", force=True)

    for combined_id, items in combined_ids.items():
        topics = sorted(
            [all_topics_by_id[item] for item in items],
            key=lambda x: (x["index"], x["topic_index"]),
        )
        content = "\n\n".join(
            [
                item["topic"].replace("\n", " ").strip()
                + f":\nSummary: {item['summary'].replace('\n', ' ').strip()}"
                + f"\n\nContent: {item['document'].page_content.strip()}"
                for item in topics
            ]
        )
        metadata = {
            "page": ", ".join([str(topic["page"]) for topic in topics]),
            "thinking": "\n".join(
                [
                    (
                        topic["document"].metadata["thinking"]
                        if "thinking" in topic["document"].metadata
                        else ""
                    )
                    for topic in topics
                ]
            ),
            "topic_index": min(topic["topic_index"] for topic in topics),
            "index": min(topic["index"] for topic in topics),
        }
        if "filename" in state and state["filename"] is not None:
            metadata["filename"] = state["filename"]
        if "url" in state and state["url"] is not None:
            metadata["url"] = state["url"]
        combined_metadata[combined_id] = metadata

        if instructions is not None:
            tasks[combined_id] = get_chain("page_formatter_guided").ainvoke(
                {
                    "context": content,
                    "next_page": "",
                    "prev_page": "",
                    "instructions": instructions,
                }
            )
        else:
            tasks[combined_id] = get_chain("page_formatter").ainvoke(
                {
                    "context": content,
                    "next_page": "",
                    "prev_page": "",
                }
            )

    for key in combined_ids.keys():
        result = await tasks[key]
        parsed_content = None
        response = ""
        if "parsed" in result:
            response = result["content"]
            parsed_content = parse_content_dict(result["parsed"])
        else:
            response = get_text_from_completion(result)

        response = Document(page_content=response, metadata=combined_metadata[key])
        if parsed_content is not None:
            pretty_print(parsed_content, "parsed_content", force=True)
            topic_index = combined_metadata[key]["topic_index"]
            index = combined_metadata[key]["index"]
            id = get_unique_id(key, reserved_ids)
            item = {
                "id": id,
                "topic": ", ".join(
                    [
                        (item["topic"] if "topic" in item else None)
                        for item in parsed_content
                    ]
                ),
                "summary": ", ".join(
                    [
                        (item["summary"] if "summary" in item else None)
                        for item in parsed_content
                    ]
                ),
                "document": response,
                "instruct": ", ".join(
                    [
                        (item["instruct"] if "instruct" in item else None)
                        for item in parsed_content
                    ]
                ),
                "topic_index": topic_index,
                "page": index + 1,
                "index": index,
            }
        else:
            item = {
                "id": id,
                "document": response,
                "topic_index": topic_index,
                "page": index + 1,
                "index": index,
            }
        new_combined_topics[id] = item

    new_topics = sorted(
        [topic for topic in new_combined_topics.values() if topic is not None],
        key=lambda x: (x["index"], x["topic_index"]),
    )

    pretty_print(sorted_search_topics, "sorted_search_topics", force=True)
    pretty_print(new_topics, "new_topics", force=True)

    return {
        "search_topics_complete": True,
        "content_topics": [
            {
                "page_content": result["document"],
                "page_number": (
                    result["page"] if result["page"] != -1 else result["index"] + 1
                ),
                "topic_index": result["topic_index"],
                "metadata": result["document"].metadata,
                "topic": result["topic"] or "",
                "summary": result["summary"] or "",
                "id": (
                    result["id"]
                    if "id" in result and result["id"]
                    else get_unique_id(
                        f"{result['index']}-{result['topic_index']}-{result['topic']}",
                        reserved_ids,
                    )
                ),
            }
            for result in new_topics
        ],
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
