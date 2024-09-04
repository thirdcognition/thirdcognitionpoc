import asyncio
import io
import os
import re
import operator
from collections import defaultdict
from typing import Annotated, Dict, List, Literal, TypedDict, Union
from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langchain_community.document_loaders.url_playwright import PlaywrightURLLoader

from chains.base_parser import get_text_from_completion
from chains.init import get_chain
from lib.db_tools import update_db_file_and_rag
from lib.document_parse import process_file_contents
from lib.document_tools import a_semantic_splitter, markdown_to_text, semantic_splitter
from lib.load_env import SETTINGS
from models.sqlite_tables import (
    SourceConcept,
    SourceConceptList,
    SourceContents,
    SourceData,
    SourceDataTable,
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
class OverallState(TypedDict):
    # Notice here we use the operator.add
    # This is because we want combine all the summaries we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    category: List[str]
    file: io.BytesIO
    filename: str
    url: str
    contents: List[Document] #Annotated[list[Document], operator.add]
    final_contents: List[Document]
    reformat_contents: List[Document] #Annotated[list[Document], operator.add]
    # concepts: List[SourceConcept] #Annotated[list, operator.add]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    collected_concepts: List[SourceConcept]
    source_contents: SourceContents
    content_summaries: Dict
    instructions: str
    split_complete: bool = False
    reformat_complete: bool = False
    concept_complete: bool = False
    summary_complete: bool = False
    collapse_complete: bool = False
    final_contents_complete: bool = False
    rag_complete: bool = False


class ProcessTextConfig(TypedDict):
    instructions: str = None
    collect_concepts: bool = False
    overwrite_sources: bool = False


# Here we generate a summary, given a document
async def split_content(state: OverallState, config: RunnableConfig):
    if state["category"] is None or state["category"] == [] or state["category"] == "":
        raise ValueError("Category must be provided")
    if state["filename"] is not None:
        if state["file"] is None or len(state["file"].getvalue()) == 0:
            raise ValueError("File must be provided")
        filename = state["filename"]
        filetype = os.path.basename(filename).split(".")[-1]
        texts = await process_file_contents(
            state["file"],
            filename,
            state["category"],
            overwrite=config["configurable"]["overwrite_sources"],
        )
        text = "\n\n".join(texts)
        if filetype == "md":
            text = markdown_to_text(text)

        split = await a_semantic_splitter(text)
        split = [Document(page_content=clean_dividers(doc)) for doc in split]
        response = split_list_of_docs(
            split, length_function, SETTINGS.default_llms.instruct_detailed.char_limit
        )
    elif state["url"] is not None:
        # TODO: Figure out image parsing.
        loader = PlaywrightURLLoader(
            urls=[state["url"]],
            remove_selectors=["header", "footer"],
        )
        text = await loader.aload()
        split = await a_semantic_splitter(text)
        split = [Document(page_content=clean_dividers(doc)) for doc in split]
        response = split_list_of_docs(
            split, length_function, SETTINGS.default_llms.instruct_detailed.char_limit
        )
    else:
        response = state["contents"]
    return {
        "split_complete": True,
        "contents": response,
        "instructions": (
            config["configurable"]["instructions"]
            if "instructions" in config["configurable"]
            else None
        ),
    }


# This will be the state of the node that we will "map" all
# documents to in order to generate summaries
class SummaryState(TypedDict):
    url: str
    file: str
    page: int
    content: Union[str, Document]
    instructions: str


# Here we generate a summary, given a document
async def reformat_content(state: SummaryState):
    response = ""
    if state["instructions"] is not None:
        response = await get_chain("text_formatter_guided").ainvoke(
            {"context": state["content"].page_content if isinstance(state["content"], Document) else state["content"], "instructions": state["instructions"]}
        )
    else:
        response = await get_chain("text_formatter").ainvoke(
            {"context": state["content"].page_content if isinstance(state["content"], Document) else state["content"]}
        )
    metadata = {}
    if isinstance("content", Document):
        metadata = state["content"].metadata.copy()

    if "file" in state:
        metadata["filename"] = state["filename"]
    if "page" in state:
        metadata["page"] = state["page"]
    if "url" in state:
        metadata["url"] = state["url"]
    if isinstance(response, tuple) and len(response) == 2:
        metadata["thougths"] = response[1]

    doc = Document(page_content=get_text_from_completion(response), metadata=metadata)

    return {"summaries": [doc]}


# Here we define the logic to map out over the documents
# We will use this an edge in the graph
def map_summaries(state: OverallState):
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    return [
        Send(
            "reformat_content",
            {
                "content": content,
                "instructions": state["instructions"],
                "page": page,
                "filename": state["filename"],
            },
        )
        for page, content in enumerate(state["contents"])
    ]


async def combine_summaries(summaries, concept_id, concepts: List[str]):
    summaries[concept_id] = concepts[0]
    if len(concepts) > 1:
        summaries[concept_id] = get_text_from_completion(
            await get_chain("combine_bullets").ainvoke(
                {"context": "\n- ".join(concepts)}
            )
        )


async def summarize_content(state: OverallState, config: RunnableConfig):
    concepts: List[SourceConcept] = []
    if (
        config is not None
        and "configurable" in config
        and config["configurable"]["collect_concepts"] is True
    ):
        for i, summary in enumerate(state["summaries"]):
            existing_categories = {}
            if len(concepts) > 0:
                for concept in concepts:
                    for category in concept.category:
                        if category.tag not in existing_categories.keys():
                            existing_categories[category.tag] = (
                                f"{category.tag}: {category.description}"
                            )

            params = {
                "context": (f"File: {state["filename"]}\nPage: {i}\n" if state["filename"] else f"URL: {state["url"]}\n") + f"Content: {get_text_from_completion(summary)}"
            }
            if len(existing_categories.keys()):
                params["chat_history"] = [
                    AIMessage(
                        "Existing categories (tag: description):\n"
                        + "\n".join(existing_categories.values()),
                    )
                ]

            newConcepts: SourceConceptList = await get_chain(
                "concept_structured"
            ).ainvoke(params)
            newConcepts.concepts = [
                concept
                for concept in newConcepts.concepts
                if len(concept.content.strip()) > 10
            ]
            concepts += newConcepts.concepts

    return {
        "reformat_complete": True,
        "summary_complete": True,
        "concept_complete": True,
        "reformat_contents": [
            Document(
                get_text_from_completion(summary),
                metadata=summary.metadata if isinstance(summary, Document) else {
                    "thoughts": summary[1] if isinstance(summary, tuple) else None
                },
            )
            for summary in state["summaries"]
        ],
        "collapsed_summaries": [
            Document(
                get_text_from_completion(summary),
                metadata=summary.metadata if isinstance(summary, Document) else {
                    "thoughts": summary[1] if isinstance(summary, tuple) else None
                },
            )
            for summary in state["summaries"]
        ],
        "collected_concepts": concepts,
    }


async def collapse(results: List, doc_list, callback):
    results.append(await acollapse_docs(doc_list, callback))


# Add node to collapse summaries
async def collapse_content(state: OverallState):
    doc_lists = split_list_of_docs(
        state["collapsed_summaries"],
        length_function,
        SETTINGS.default_llms.instruct_detailed.context_size,
    )
    results = []
    instructions = state["instructions"] if "instructions" in state else None

    async def process_doc_with_instructions(x):
        nonlocal instructions
        return get_text_from_completion(
            await get_chain("text_formatter_compress_guided").ainvoke(
                {"context": x, "instructions": instructions}
            )
        )

    async def process_doc(x):
        return get_text_from_completion(
            await get_chain("text_formatter_compress").ainvoke({"context": x})
        )

    tasks = None
    if instructions is not None:
        tasks = [
            collapse(results, doc_list, process_doc_with_instructions)
            for doc_list in doc_lists
        ]
    else:
        tasks = [collapse(results, doc_list, process_doc) for doc_list in doc_lists]

    await asyncio.gather(*tasks)

    return {
        "collapse_complete": True,
        "collapsed_summaries": results
    }


# This represents a conditional edge in the graph that determines
# if we should collapse the summaries or not
def should_collapse(
    state: OverallState,
) -> Literal["collapse_content", "finalize_content"]:
    num_chars = length_function(state["collapsed_summaries"])
    if num_chars > SETTINGS.default_llms.instruct_detailed.char_limit:
        return "collapse_content"
    else:
        return "finalize_content"


async def collapse_concept_summaries(category, concept_summaries, contents):
    joined = "\n".join(contents)

    if len(joined) < 200:
        concept_summaries[category] = joined
    else:
        concept_summaries[category] = get_text_from_completion(
            await get_chain("summary").ainvoke({"context": joined})
            if len(contents) == 1
            else await get_chain("summary_documents").ainvoke({"context": contents})
        )


# Here we will generate the final summary
async def finalize_content(state: OverallState, config: RunnableConfig):
    instructions = state["instructions"] if "instructions" in state else None
    if instructions is not None:
        response = await get_chain("summary_guided").ainvoke(
            {"context": state["collapsed_summaries"], "instructions": instructions}
        )
    else:
        response = await get_chain("summary").ainvoke(
            {"context": state["collapsed_summaries"]}
        )

    summary_thoughts = ""
    summary_collapsed = get_text_from_completion(response)
    if isinstance(response, tuple) and len(response) == 2:
        summary_thoughts = response[1]

    concept_summaries = defaultdict(list)
    if (
        config is not None
        and "configurable" in config
        and config["configurable"]["collect_concepts"] is True
    ):
        concepts = defaultdict(list[SourceConcept])
        for concept in state["collected_concepts"]:
            for category in concept.category:
                concepts[category.tag].append(concept)

        tasks = []
        for concept_category in concepts.keys():
            contents = "\n".join(concept.content for concept in concepts[concept_category])
            num_chars = length_function(contents)
            if num_chars > SETTINGS.default_llms.instruct_detailed.char_limit:
                contents = semantic_splitter(contents)
            else:
                contents = [contents]

            tasks.append(
                collapse_concept_summaries(concept_category, concept_summaries, contents)
            )

        await asyncio.gather(*tasks)

    formatted_content = ""
    formatted_content_toughts = ""
    for page in state["reformat_contents"]:
        formatted_content += page.page_content + "\n"
        if page.metadata is not None and "thoughts" in page.metadata.keys() and page.metadata["thoughts"] is not None:
            formatted_content_toughts += page.metadata["thoughts"] + "\n"

    flat_contents:List[Document] = []
    for item in state["contents"]:
        if isinstance(item, Document):
            flat_contents.append(item)
        elif isinstance(item, List):
            flat_contents.extend(item)
        else:
            print(f"{state["contents"]}")
            raise ValueError(f"Unknown type {type(item)}: {item=}")

    if (
        config is not None
        and "configurable" in config
        and config["configurable"]["collect_concepts"] is True
    ):
        return {
            "final_contents_complete": True,
            "final_contents": flat_contents,
            "source_contents": SourceContents(
                formatted_content=formatted_content,
                formatted_content_thoughts=formatted_content_toughts,
                summary=summary_collapsed,
                summary_thoughts=summary_thoughts,
                concepts=state["collected_concepts"],
                summaries=concept_summaries,
            )
        }
    else:
        return {
            "final_contents_complete": True,
            "final_contents": flat_contents,
            "content_summaries": {
                "formatted_content": formatted_content,
                "formatted_content_toughts": formatted_content_toughts,
                "summary": summary_collapsed,
                "summary_thoughts": summary_thoughts,
            }
        }


async def rag_content(state: OverallState, config: RunnableConfig):
    if config["configurable"]["collect_concepts"] is True:
        content = state["source_contents"]
        filetype = None
        if state["filename"] is not None:
            filetype = os.path.basename(state["filename"]).split(".")[-1]

        texts = [page.page_content for page in state["final_contents"]]
        update_db_file_and_rag(
            state["filename"] or state["url"],
            state["category"],
            texts,
            content,
            filetype,
        )

    return {
        "rag_complete": True
    }


# Construct the graph
# Nodes:
process_text_graph = StateGraph(OverallState, ProcessTextConfig)
process_text_graph.add_node("split_content", split_content)
process_text_graph.add_node("reformat_content", reformat_content)  # same as before
process_text_graph.add_node("summarize_content", summarize_content)
process_text_graph.add_node("collapse_content", collapse_content)
process_text_graph.add_node("finalize_content", finalize_content)
process_text_graph.add_node("rag_content", rag_content)

# Edges:
process_text_graph.add_edge(START, "split_content")
process_text_graph.add_conditional_edges(
    "split_content", map_summaries, ["reformat_content"]
)
process_text_graph.add_edge("reformat_content", "summarize_content")
process_text_graph.add_conditional_edges("summarize_content", should_collapse)
process_text_graph.add_conditional_edges("collapse_content", should_collapse)
process_text_graph.add_edge("finalize_content", "rag_content")
process_text_graph.add_edge("rag_content", END)

process_text = process_text_graph.compile()


# def llm_process(chain_id, texts, guidance=None, force=False) -> tuple[str, str]:
#     text = ""
#     thoughts = ""

#     i = 0
#     total = len(texts)

#     if total > 1 and chain_id == "summary":
#         total += 1

#     if not force and (
#         texts == None or texts[0] == None or total == 1 and len(texts[0]) < 1000
#     ):
#         return None, None

#     guided_llm = ""
#     if guidance is not None and guidance != "":
#         guided_llm = "_guided"

#     chain = get_chain(chain_id + guided_llm)

#     if total > 1:
#         inputs = []
#         for sub_text in texts:
#             i += 1
#             input = {"context": clean_dividers(sub_text)}
#             if guidance is not None and guidance != "":
#                 input["question"] = guidance

#             inputs.append(input)

#             result = chain.invoke(input)

#             if isinstance(result, tuple) and len(result) == 1:
#                 result = result[0]

#             mid_thoughts = ""
#             if isinstance(result, tuple) and len(result) == 2:
#                 mid_results, mid_thoughts = result
#             else:
#                 mid_results = result

#             mid_results = (
#                 mid_results.content
#                 if isinstance(mid_results, BaseMessage)
#                 else mid_results
#             )

#             text += mid_results + "\n\n"
#             thoughts += mid_thoughts + "\n\n"

#     else:
#         text = clean_dividers(texts[0])

#     if chain_id == "summary":
#         text = markdown_to_text(text)

#         input = {"context": text}

#         if guidance is not None and guidance != "":
#             input["question"] = guidance

#         result = chain.invoke(input)
#         thoughts = ""

#         if isinstance(result, tuple) and len(result) == 1:
#             result = result[0]

#         if isinstance(result, tuple) and len(result) == 2:
#             text, thoughts = result
#         else:
#             text = result

#         text = text.content if isinstance(text, BaseMessage) else text

#     return text.strip(), thoughts.strip()
