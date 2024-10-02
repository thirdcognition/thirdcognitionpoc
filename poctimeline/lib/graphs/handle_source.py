import io
import os
import re
from typing import Dict, List, TypedDict, Union
from langchain.chains.combine_documents.reduce import (
    split_list_of_docs,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langchain_community.document_loaders.url_playwright import PlaywrightURLLoader

from lib.db.topics import get_topic_by_id, update_db_topic_rag
from lib.db.concept import get_concept_by_id, update_db_concept_rag
from lib.db.source import (
    db_source_exists,
    get_db_sources,
    update_db_source_rag,
)
from lib.document_parse import SourceType, process_source_contents
from lib.document_tools import (
    a_semantic_splitter,
    markdown_to_text,
)
from lib.helpers.shared import get_number, get_text_from_completion, pretty_print
from lib.load_env import SETTINGS
from lib.models.concepts import ConceptDataTable
from lib.models.taxonomy import Taxonomy, TaxonomyDataTable
from lib.models.source import (
    SourcePage,
    SourceContents,
)
from lib.graphs.process_text import process_text
from lib.graphs.find_topics import find_topics, get_result_dict
from lib.graphs.find_taxonomy import find_taxonomy
from lib.graphs.find_concepts import find_concepts
from lib.models.topics import TopicDataTable, get_topic_item, topic_to_dict


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
class ProcessSourceState(TypedDict):
    # Notice here we use the operator.add
    # This is because we want combine all the summaries we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    categories: List[str]
    file: io.BytesIO
    filename: str
    url: str
    source: str
    instructions: str
    # generated
    contents: List[Union[Document, str]]  # Annotated[list[Document], operator.add]
    split_content_complete: bool
    get_source_content_complete: bool
    process_text_complete: bool
    process_text_result: Dict
    find_topics_complete: bool
    find_topics_result: Dict
    find_taxonomy_complete: bool
    find_taxonomy_result: List[Taxonomy]
    find_concepts_complete: bool
    find_concepts_result: List[ConceptDataTable]
    rag_update_complete: bool


class ProcessSourceConfig(TypedDict):
    instructions: str = None
    overwrite_sources: bool = False
    update_rag: bool = False
    rewrite_text: bool = True
    run_split_for_source: bool = True
    update_concepts: bool = True


async def should_rewrite_content(state: ProcessSourceState, config: RunnableConfig):
    source = state["filename"] if "filename" in state else state["url"]
    if config["configurable"]["rewrite_text"] or not db_source_exists(source):
        return "split_content"
    else:
        return "get_source_content"


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
        response = source_data.texts

    topics: List[TopicDataTable] = []
    for topic_id in source_data.source_topics:
        db_topic = get_topic_by_id(topic_id)
        topics.append(db_topic)

    concepts: List[ConceptDataTable] = []
    for concept_id in source_data.source_concepts:
        db_concept = get_concept_by_id(concept_id)
        concepts.append(db_concept)

    contents: SourceContents = source_data.source_contents
    return {
        "contents": response,
        "instructions": (
            config["configurable"]["instructions"]
            if "instructions" in config["configurable"]
            else None
        ),
        "process_text_complete": True,
        "process_text_result": {
            "summary": contents.summary,
            "content": contents.formatted_content,
            "document": Document(page_content=contents.formatted_content, metadata={"source": source, "topic": contents.topic}),
            "content_pages": [Document(page_content=page.content, metadata=page.metadata) for page in contents.pages],
        },
        "find_topics_complete": True,
        "find_topics_result": {
            "content_topics": [
                get_result_dict(get_topic_item(result))
                for result in topics
            ],
            "all_topics": set([item.get("topic", "") for item in topics]),
        },
        "find_taxonomy_complete": True,
        "find_taxonomy_result": [],
        "find_concepts_complete": True,
        "find_concepts_result": concepts
    }


# Here we generate a summary, given a document
async def split_content(state: ProcessSourceState, config: RunnableConfig):
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
        text = get_text_from_completion(state["contents"])

    split = await a_semantic_splitter(text)
    split = [
        Document(page_content=clean_dividers(doc), metadata={"snippet": i})
        for i, doc in enumerate(split)
    ]
    response = split_list_of_docs(
        split, length_function, SETTINGS.default_llms.instruct_detailed.char_limit
    )

    return {
        "split_content_complete": True,
        # "semantic_contents": split,
        "contents": response,
        "instructions": (
            config["configurable"]["instructions"]
            if "instructions" in config["configurable"]
            else None
        ),
    }


async def should_update_concepts(state: ProcessSourceState, config: RunnableConfig):
    if config["configurable"].get("update_concepts", False):
        return "find_topics"
    elif config["configurable"].get("update_rag", False):
        return "rag_update"
    else:
        return END


async def process_text_step(state: ProcessSourceState, config: RunnableConfig):
    if "contents" not in state.keys() or state["contents"] is None:
        raise ValueError("Contents must be provided")

    results = await process_text.ainvoke(
        {
            "contents": state["contents"],
            "filename": state.get("filename", None),
            "url": state.get("url", None),
            "source": state.get("source", None),
            "instructions": state.get("instructions", None),
        }
    )

    return {
        "process_text_complete": True,
        "process_text_result": results["results"],
    }


async def find_topics_step(state: ProcessSourceState, config: RunnableConfig):
    if "contents" not in state.keys() or state["contents"] is None:
        raise ValueError("Contents must be provided")

    topics_results = await find_topics.ainvoke(
        {
            "contents": (
                state["process_text_result"]["content_pages"]
                if "process_text_result" in state
                else state["contents"]
            ),
            "filename": state.get("filename", None),
            "url": state.get("url", None),
            "source": state.get("source", None),
            "instructions": state.get("instructions", None),
        }
    )

    return {
        "find_topics_complete": True,
        "find_topics_result": topics_results,
    }


async def find_taxonomy_step(state: ProcessSourceState, config: RunnableConfig):
    if "contents" not in state.keys() or state["contents"] is None:
        raise ValueError("Contents must be provided")

    taxonomy_results = await find_taxonomy.ainvoke(
        {
            "categories": state["categories"],
            "filename": state.get("filename", None),
            "url": state.get("url", None),
            "source": state.get("source", None),
            "content_topics": state["find_topics_result"]["content_topics"],
            "instructions": state.get("instructions", None),
        }
    )

    return {
        "find_taxonomy_complete": True,
        "find_taxonomy_result": taxonomy_results["results"],
    }


async def find_concepts_step(state: ProcessSourceState, config: RunnableConfig):
    if "contents" not in state.keys() or state["contents"] is None:
        raise ValueError("Contents must be provided")

    concepts_results = await find_concepts.ainvoke(
        {
            "categories": state["categories"],
            "filename": state.get("filename", None),
            "url": state.get("url", None),
            "source": state.get("source", None),
            "content_topics": state["find_topics_result"]["content_topics"],
            "instructions": state.get("instructions", None),
        }
    )

    return {
        "find_concepts_complete": True,
        "find_concepts_result": concepts_results["collected_concepts"],
    }


async def should_rag(state: ProcessSourceState, config: RunnableConfig):
    if config["configurable"]["update_rag"] and state["contents"] is not None:
        return "rag_update"
    else:
        return END


async def rag_update(state: ProcessSourceState, config: RunnableConfig):
    # texts = state["contents"]
    final_topic = (
        state["process_text_result"]["document"].metadata["topic"]
        if isinstance(
            state["process_text_result"]["document"], Document
        )
        else ", ".join(state["find_topics_result"]["all_topics"])
    )
    final_contents = state["process_text_result"]["content"]
    content_pages: List[Document] = state["process_text_result"]["content_pages"]
    # all_topics = state["find_topics_result"]["all_topics"]
    summary = state["process_text_result"]["summary"]
    content_topics = state["find_topics_result"]["content_topics"]

    # source_topics = [
    #     TopicModel(
    #         page_content=get_text_from_completion(item["page_content"]),
    #         page_number=get_number(item["page"]) if "page" in item else 0,
    #         topic_index=get_number(item["topic_index"]) if "topic_index" in item else 0,
    #         metadata=item["page_content"].metadata,
    #         topic=item["topic"] or "",
    #         instruct=(
    #             item["page_content"].metadata["instruct"]
    #             if "instruct" in item["page_content"].metadata
    #             else ""
    #         ),
    #         summary=item["summary"] or "",
    #         id=item["id"] or "",
    #         chroma_ids=[],
    #         chroma_collections=[],
    #     )
    #     for item in content_topics
    # ]

    source_pages = [
        SourcePage(
            content=get_text_from_completion(item),
            page_metadata=item.metadata,
        )
        for item in content_pages
    ]

    contents = SourceContents(
        formatted_content=final_contents,
        topic=final_topic,
        summary=summary,
        pages=source_pages,
    )

    filetype = None
    if state["filename"] is not None:
        filetype = os.path.basename(state["filename"]).split(".")[-1]

    if config["configurable"]["overwrite_sources"] or state.get(
        "split_complete", False
    ):
        texts = [get_text_from_completion(page) for page in state["contents"]]
        update_db_source_rag(
            state["filename"] or state["url"],
            state["categories"],
            texts,
            contents,
            filetype=filetype,
        )
        update_db_topic_rag(
            state["filename"] or state["url"],
            state["categories"],
            content_topics,
        )

        if config["configurable"]["update_concepts"]:
            update_db_concept_rag(
                state["filename"] or state["url"],
                state["categories"],
                concepts=(
                    state["find_concepts_result"]
                    if "find_concepts_result" in state
                    else None
                ),
            )

    return {"rag_update_complete": True}


handle_source_graph = StateGraph(ProcessSourceState, ProcessSourceConfig)
handle_source_graph.add_node("split_content", split_content)
handle_source_graph.add_node("get_source_content", get_source_content)
handle_source_graph.add_node("process_text", process_text_step)
handle_source_graph.add_node("find_topics", find_topics_step)
handle_source_graph.add_node("find_taxonomy", find_taxonomy_step)
handle_source_graph.add_node("find_concepts", find_concepts_step)
handle_source_graph.add_node("rag_update", rag_update)

# Edges:
handle_source_graph.add_conditional_edges(START, should_rewrite_content)
handle_source_graph.add_edge("split_content", "process_text")
handle_source_graph.add_conditional_edges("process_text", should_update_concepts)
handle_source_graph.add_conditional_edges("get_source_content", should_update_concepts)
handle_source_graph.add_edge("find_topics", "find_taxonomy")
handle_source_graph.add_edge("find_taxonomy", "find_concepts")
handle_source_graph.add_conditional_edges("find_concepts", should_rag)
handle_source_graph.add_edge("rag_update", END)

handle_source = handle_source_graph.compile()
