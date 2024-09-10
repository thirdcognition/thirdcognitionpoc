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

from lib.chains.base_parser import get_text_from_completion
from lib.chains.init import get_chain
from lib.db_tools import update_db_file_rag_concepts
from lib.document_parse import process_file_contents
from lib.document_tools import (
    a_semantic_splitter,
    markdown_to_text,
    semantic_splitter,
    split_text,
)
from lib.load_env import SETTINGS
from lib.models.prompts import TitledSummary
from lib.models.sqlite_tables import (
    ParsedConceptList,
    SourceConcept,
    SourceContents,
    SourceReference,
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
    contents: List[Union[Document, str]]  # Annotated[list[Document], operator.add]
    final_contents: List[Document]
    reformat_contents: List[Document]  # Annotated[list[Document], operator.add]
    concepts: List[SourceConcept]  # Annotated[list, operator.add]
    reformatted_txt: Annotated[list, operator.add]
    collapsed_reformatted_txt: List[Document]
    collected_concepts: List[SourceConcept]
    source_contents: SourceContents
    content_result: Dict
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
    update_rag: bool = False
    summarize: bool = True


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
        texts = await process_file_contents(
            state["file"],
            filename,
            state["categories"],
            overwrite=config["configurable"]["overwrite_sources"],
        )
        text = "\n\n".join(texts)
        if filetype == "md":
            text = markdown_to_text(text)
    elif "url" in state.keys() and state["url"] is not None:
        # TODO: Figure out image parsing.
        loader = PlaywrightURLLoader(
            urls=[state["url"]],
            remove_selectors=["header", "footer"],
        )
        text = await loader.aload()
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
    split = [Document(page_content=clean_dividers(doc)) for doc in split]
    response = split_list_of_docs(
        split, length_function, SETTINGS.default_llms.instruct_detailed.char_limit
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
            {
                "context": (
                    state["content"].page_content
                    if isinstance(state["content"], Document)
                    else state["content"]
                ),
                "instructions": state["instructions"],
            }
        )
    else:
        response = await get_chain("text_formatter").ainvoke(
            {
                "context": (
                    state["content"].page_content
                    if isinstance(state["content"], Document)
                    else state["content"]
                )
            }
        )
    metadata = {}
    if isinstance("content", Document):
        metadata = state["content"].metadata.copy()

    if "file" in state and state["file"] is not None:
        metadata["filename"] = state["filename"]
    if "page" in state and state["page"] is not None:
        metadata["page"] = state["page"]
    if "url" in state and state["url"] is not None:
        metadata["url"] = state["url"]
    if isinstance(response, tuple) and len(response) == 2:
        metadata["thougths"] = response[1]

    doc = Document(page_content=get_text_from_completion(response), metadata=metadata)

    return {"reformatted_txt": [doc]}


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
                "page": page if "page" in state and page != -1 else None,
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
    return {
        "reformat_complete": True,
        "reformat_contents": [
            get_text_from_completion(txt) for txt in state["reformatted_txt"]
        ],
        "collapsed_reformatted_txt": [
            Document(
                get_text_from_completion(txt),
            )
            for txt in state["reformatted_txt"]
        ],
    }


def should_conceptualize(
    state: ProcessTextState,
    config: RunnableConfig,
) -> Literal["concept_content", "collapse_content", "finalize_content"]:
    if config["configurable"]["collect_concepts"]:
        return "concept_content"
    else:
        return should_collapse(state=state)


async def concept_content(state: ProcessTextState, config: RunnableConfig):
    concepts: List[SourceConcept] = state["concepts"] if "concepts" in state else []
    if config["configurable"]["collect_concepts"] is True:
        for i, txt in enumerate(state["reformatted_txt"]):
            existing_categories = {}
            if len(concepts) > 0:
                for concept in concepts:
                    for categories in concept.tags:
                        if categories.tag not in existing_categories.keys():
                            existing_categories[categories.tag] = (
                                f"{categories.tag}: {categories.description}"
                            )

            params = {
                "context": (
                    f"File: {state['filename']}\nPage: {i}\n"
                    if state["filename"]
                    else f"URL: {state['url']}\n"
                )
                + f"Content: {get_text_from_completion(txt)}"
            }
            if len(existing_categories.keys()):
                params["chat_history"] = [
                    AIMessage(
                        "Existing categories (tag: description):\n"
                        + "\n".join(existing_categories.values()),
                    )
                ]

            newConcepts: ParsedConceptList = await get_chain(
                "concept_structured"
            ).ainvoke(params)

            newConcepts.concepts = [
                concept
                for concept in newConcepts.concepts
                if len("\n".join(concept.content).strip()) > 10
            ]
            if len(newConcepts.concepts) > 0:
                concepts.extend(
                    [
                        SourceConcept(
                            id=parsed_concept.id,
                            title=parsed_concept.title,
                            contents=[parsed_concept.content],
                            tags=parsed_concept.tags,
                            references=[
                                SourceReference(
                                    source=(
                                        state["filename"]
                                        if "filename" in state
                                        else state["url"]
                                    ),
                                    page_number=i,
                                )
                            ],
                        )
                        for parsed_concept in newConcepts.concepts
                    ]
                )

            filtered_concepts: Dict[str, SourceConcept] = {}
            for concept in concepts:
                if concept.id not in filtered_concepts.keys():
                    filtered_concepts[concept.id] = concept
                else:
                    filtered_concepts[concept.id].contents = list(
                        set(filtered_concepts[concept.id].contents + concept.contents)
                    )
                    for new_tag in concept.tags:
                        if new_tag.tag not in [
                            tag.tag for tag in filtered_concepts[concept.id].tags
                        ]:
                            filtered_concepts[concept.id].tags.append(new_tag)
                    for ref in concept.references:
                        if f"{ref.source}_{(ref.page_number or 0)}" not in [
                            f"{ref.source}_{(ref.page_number or 0)}"
                            for ref in filtered_concepts[concept.id].references
                        ]:
                            filtered_concepts[concept.id].references.append(ref)

            concepts = list(filtered_concepts.values())

    return {
        "concept_complete": True,
        "collected_concepts": concepts,
    }


async def collapse(results: List, doc_list, callback):
    results.append(await acollapse_docs(doc_list, callback))


# Add node to collapse summaries
async def collapse_content(state: ProcessTextState):
    doc_lists = split_list_of_docs(
        state["collapsed_reformatted_txt"],
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

    return {"collapse_complete": True, "collapsed_reformatted_txt": results}


# This represents a conditional edge in the graph that determines
# if we should collapse the summaries or not
def should_collapse(
    state: ProcessTextState,
) -> Literal["collapse_content", "finalize_content"]:
    num_chars = length_function(state["collapsed_reformatted_txt"])
    if num_chars > SETTINGS.default_llms.instruct_detailed.char_limit:
        return "collapse_content"
    else:
        return "finalize_content"


async def collapse_concept(
    i: int, concepts: List[SourceConcept], concept: SourceConcept
):
    joined = "\n".join(concept.contents)

    if len(concept.contents) > 1:
        reformat = get_text_from_completion(
            await get_chain("text_formatter").ainvoke({"context": joined})
        )
        concept.contents = [reformat]
        joined = reformat

    if len(joined) < 200:
        concept.summary = concept.summary or joined
    else:
        contents = split_text(joined)
        if len(concept.contents) > 1 or concept.summary is None:
            summary: TitledSummary = get_text_from_completion(
                await get_chain("summary_with_title").ainvoke({"context": joined})
                if len(contents) == 1
                else await get_chain("summary_documents_with_title").ainvoke(
                    {"context": contents}
                )
            )
            concept.title = summary.title
            concept.summary = summary.summary

    concepts[i] = concept


# Here we will generate the final summary
async def finalize_content(state: ProcessTextState, config: RunnableConfig):
    instructions = state["instructions"] if "instructions" in state else None
    summary_collapsed = ""
    if config["configurable"]["summarize"]:
        if instructions is not None:
            response = await get_chain("summary_guided").ainvoke(
                {
                    "context": state["collapsed_reformatted_txt"],
                    "instructions": instructions,
                }
            )
        else:
            response = await get_chain("summary").ainvoke(
                {"context": state["collapsed_reformatted_txt"]}
            )
        summary_collapsed = get_text_from_completion(response)

    # summaries = defaultdict(list)
    if config["configurable"]["collect_concepts"] is True:
        collected_concepts = state["collected_concepts"]
        # concepts = defaultdict(list[SourceConcept])
        tasks = []
        for i, concept in enumerate[collected_concepts]:
            tasks.append(collapse_concept(i, collected_concepts, concept))
            # for concept_tag in concept.tags:
            #     concepts[concept_tag.tag].append(concept)

        # for tag in concepts.keys():
        #     contents = "\n".join(
        #         ["\n".join(concept.contents) for concept in concepts[tag]]
        #     )
        #     num_chars = length_function(contents)
        #     if num_chars > SETTINGS.default_llms.instruct_detailed.char_limit:
        #         contents = semantic_splitter(contents)
        #     else:
        #         contents = [contents]

        #     tasks.append(
        #         collapse_concept_summaries(
        #             tag, summaries, contents
        #         )
        #     )
        # for concept in collected_concepts.concepts:

        await asyncio.gather(*tasks)

        # collected_concepts.summaries = summaries

    formatted_content = "\n".join(state["reformat_contents"])

    flat_contents: List[Document] = []
    for item in state["contents"]:
        if isinstance(item, Document):
            flat_contents.append(item)
        elif isinstance(item, List):
            flat_contents.extend(item)
        else:
            # print(f"{state["contents"]}")
            raise ValueError(f"Unknown type {type(item)}: {item=}")

    if config["configurable"]["collect_concepts"] is True:
        return {
            "final_contents_complete": True,
            "final_contents": flat_contents,
            "source_contents": SourceContents(
                formatted_content=formatted_content,
                summary=summary_collapsed,
                # concepts=state["collected_concepts"],
                # concept_summaries=summaries,
            ),
            "collected_concepts": collected_concepts,
        }
    else:
        return {
            "final_contents_complete": True,
            "final_contents": flat_contents,
            "content_result": {
                "formatted_content": formatted_content,
                "summary": summary_collapsed,
            },
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

    texts = [page.page_content for page in state["final_contents"]]
    update_db_file_rag_concepts(
        state["filename"] or state["url"],
        state["categories"],
        texts,
        content,
        concepts=state["collected_concepts"] if "collected_concepts" in state else None,
        filetype=filetype,
    )

    return {"rag_complete": True}


# Construct the graph
# Nodes:
process_text_graph = StateGraph(ProcessTextState, ProcessTextConfig)
process_text_graph.add_node("split_content", split_content)
process_text_graph.add_node("reformat_content", reformat_content)  # same as before
process_text_graph.add_node("concat_reformat", concat_reformat)
process_text_graph.add_node("concept_content", concept_content)
process_text_graph.add_node("collapse_content", collapse_content)
process_text_graph.add_node("finalize_content", finalize_content)
process_text_graph.add_node("rag_content", rag_content)

# Edges:
process_text_graph.add_edge(START, "split_content")
process_text_graph.add_conditional_edges(
    "split_content", map_reformat, ["reformat_content"]
)
process_text_graph.add_edge("reformat_content", "concat_reformat")
process_text_graph.add_conditional_edges("concat_reformat", should_conceptualize)
process_text_graph.add_conditional_edges("concept_content", should_collapse)
process_text_graph.add_conditional_edges("collapse_content", should_collapse)
process_text_graph.add_conditional_edges("finalize_content", should_rag)
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
