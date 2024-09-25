import asyncio
import re
import operator
import textwrap
from typing import Annotated, Dict, List, Literal, TypedDict, Union
from langchain.chains.combine_documents.reduce import (
    collapse_docs,
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from lib.chains.init import get_chain
from lib.document_tools import a_semantic_splitter
from lib.helpers import (
    get_text_from_completion,
    pretty_print,
)
from lib.load_env import SETTINGS
from lib.models.topics import get_topic_doc_context, get_topic_document, parse_topic_items, prepare_topic_contents


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
# reformatted_contents, and a final summary.
class ProcessTextState(TypedDict):
    # Notice here we use the operator.add
    # This is because we want combine all the reformatted_contents we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    contents: List[Document]  # Annotated[list[Document], operator.add]
    instructions: str
    # generated
    reformatted_contents: Annotated[list, operator.add]
    content_pages: List[Document]
    summary: str
    collapsed_contents: List[Document]
    results: Dict
    setup_content_complete: bool = False
    reformat_content_complete: bool = False
    concat_content_complete: bool = False
    collapse_content_complete: bool = False
    finalize_content_complete: bool = False


class ProcessTextConfig(TypedDict):
    instructions: str = None
    collect_concepts: bool = False
    run_split: bool = True


# Here we generate a summary, given a document
async def setup_content(state: ProcessTextState, config: RunnableConfig):

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


# This will be the state of the node that we will "map" all
# documents to in order to generate reformatted_contents
class SummaryState(TypedDict):
    content: Union[str, Document]
    instructions: str


# Here we generate a summary, given a document
async def reformat_content(state: SummaryState):
    response = {}
    content, prev_page, next_page = prepare_topic_contents(
        state["content"], state["prev_page_content"], state["next_page_content"]
    )

    if state["instructions"] is not None:
        response = await get_chain("page_formatter_guided").ainvoke(
            {
                "context": content,
                "next_page": next_page,
                "prev_page": prev_page,
                "instructions": state["instructions"],
            }
        )
    else:
        response = await get_chain("page_formatter").ainvoke(
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
        "reformatted_contents": [
            {"index": state["index"], "page": state["page"], "items": items}
        ]
    }


# Here we define the logic to map out over the documents
# We will use this an edge in the graph
def map_reformatted_contents(state: ProcessTextState):
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    return [
        Send(
            "reformat_content",
            {
                "content": content,
                "prev_page_content": (state["contents"][page - 1] if page > 0 else ""),
                "next_page_content": (
                    state["contents"][page + 1]
                    if page < len(state["contents"]) - 1
                    else ""
                ),
                "index": page,
                "instructions": state["instructions"],
                "page": (
                    state["page"]
                    if "page" in state and state["page"] != -1
                    else (page + 1)
                ),
            },
        )
        for page, content in enumerate(state["contents"])
    ]


# async def combine_reformatted_contents(reformatted_contents, concept_id, concepts: List[str]):
#     reformatted_contents[concept_id] = concepts[0]
#     if len(concepts) > 1:
#         reformatted_contents[concept_id] = get_text_from_completion(
#             await get_chain("combine_bullets").ainvoke(
#                 {"context": "\n- ".join(concepts)}
#             )
#         )


async def concat_content(state: ProcessTextState, config: RunnableConfig):
    sorted_reformat: List[Dict] = sorted(
        state["reformatted_contents"], key=lambda x: x["index"]
    )
    content_pages = [
        get_topic_document(
            item, result, state["instructions"] if "instructions" in state else ""
        )
        for result in sorted_reformat
        for item in result["items"]
    ]

    return {
        "reformat_content_complete": True,
        "concat_content_complete": True,
        "content_pages": content_pages,
        "summary": "\n".join(
            [
                "\n".join(
                    (
                        f"{item['topic'].strip()}:\n"
                        if "topic" in item and item["topic"]
                        else ""
                    )
                    + (item["summary"] or "")
                    for result in sorted_reformat
                    for item in result["items"]
                )
            ]
        ),
        "collapsed_contents": content_pages,
    }


async def collapse(results: List, doc_list, callback):
    results.append(await acollapse_docs(doc_list, callback))


async def process_doc(doc: Document):
    if isinstance(doc, List):
        tasks = [process_doc(d) for d in doc]
        return await asyncio.gather(*tasks)

    metadata = doc.metadata
    instructions = metadata["instructions"] if "instructions" in metadata else None
    context = get_topic_doc_context(doc)

    if "instructions" in metadata and metadata["instructions"]:
        results = get_text_from_completion(
            await get_chain("text_formatter_compress_guided").ainvoke(
                {"context": context, "instructions": instructions}
            )
        )
    else:
        results = get_text_from_completion(
            await get_chain("text_formatter_compress").ainvoke({"context": context})
        )

    return Document(results, metadata=metadata)


# Add node to collapse reformatted_contents
async def collapse_content(state: ProcessTextState):
    doc_lists = split_list_of_docs(
        state["collapsed_contents"],
        length_function,
        SETTINGS.default_llms.instruct_detailed.context_size,
    )
    results = []

    tasks = [collapse(results, doc_list, process_doc) for doc_list in doc_lists]

    await asyncio.gather(*tasks)

    return {"collapse_content_complete": True, "collapsed_contents": results}


# This represents a conditional edge in the graph that determines
# if we should collapse the reformatted_contents or not
def should_collapse(
    state: ProcessTextState,
) -> Literal["collapse_content", "finalize_content"]:
    num_chars = length_function(state["collapsed_contents"])
    if num_chars > SETTINGS.default_llms.instruct_detailed.char_limit:
        return "collapse_content"
    else:
        return "finalize_content"


# Here we will generate the final summary
async def finalize_content(state: ProcessTextState, config: RunnableConfig):
    instructions = state["instructions"] if "instructions" in state else None
    if instructions is not None:
        response = await get_chain("summary_guided").ainvoke(
            {"context": state["summary"], "instructions": instructions}
        )
    else:
        response = await get_chain("summary").ainvoke({"context": state["summary"]})

    summary_collapsed = get_text_from_completion(response)

    flat_contents: List[Document] = []
    for item in state["contents"]:
        if isinstance(item, Document):
            flat_contents.append(item)
        elif isinstance(item, List):
            flat_contents.extend(item)
        else:
            pretty_print(state["contents"])
            raise ValueError(f"Unknown type {type(item)}: {item=}")

    return {
        "finalize_content_complete": True,
        "results": {
            "summary": summary_collapsed,
            "content": get_text_from_completion(state["collapsed_contents"]),
            "document": collapse_docs(state["collapsed_contents"], get_text_from_completion),
            "content_pages": state["content_pages"],
        },
    }


# Construct the graph
# Nodes:
process_text_graph = StateGraph(ProcessTextState, ProcessTextConfig)
process_text_graph.add_node("setup_content", setup_content)
process_text_graph.add_node("reformat_content", reformat_content)  # same as before
process_text_graph.add_node("concat_content", concat_content)
process_text_graph.add_node("collapse_content", collapse_content)
process_text_graph.add_node("finalize_content", finalize_content)

# Edges:
process_text_graph.add_edge(START, "setup_content")
process_text_graph.add_conditional_edges(
    "setup_content", map_reformatted_contents, ["reformat_content"]
)
process_text_graph.add_edge("reformat_content", "concat_content")
process_text_graph.add_conditional_edges("concat_content", should_collapse)
process_text_graph.add_conditional_edges("collapse_content", should_collapse)
process_text_graph.add_edge("finalize_content", END)

process_text = process_text_graph.compile()
