import asyncio
import io
import json
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
from lib.db_tools import (
    db_commit,
    db_source_exists,
    delete_db_concept,
    get_db_concepts,
    get_db_concept_taxonomy,
    get_db_sources,
    get_existing_concept_ids,
    update_concept_category,
    update_db_source_rag_concepts,
)
from lib.document_parse import SourceType, process_source_contents
from lib.document_tools import (
    a_semantic_splitter,
    markdown_to_text,
    split_text,
)
from lib.helpers import pretty_print
from lib.load_env import SETTINGS
from lib.models.prompts import TitledSummary
from lib.models.sqlite_tables import (
    ConceptTaxonomy,
    ConceptDataTable,
    ParsedConcept,
    ParsedConceptIds,
    ParsedConceptTaxonomyList,
    ParsedConceptList,
    ConceptData,
    ParsedConceptStructure,
    ParsedConceptStructureList,
    ParsedUniqueConceptList,
    SourceContents,
    SourceReference,
    convert_concept_category_tag_to_dict,
    convert_taxonomy_dict_to_tag_structure_string,
    convert_taxonomy_tags_to_dict,
)
from lib.prompts.formatters import TAXONOMY_ALL_TAGS


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
    semantic_contents: List[Document]
    contents: List[Union[Document, str]]  # Annotated[list[Document], operator.add]
    final_contents: List[Document]
    reformat_contents: List[Document]  # Annotated[list[Document], operator.add]
    concepts: List[ConceptData]  # Annotated[list, operator.add]
    reformatted_txt: Annotated[list, operator.add]
    snippets: List[Document]
    taxonomy: List[Dict]
    found_concepts: Annotated[list, operator.add]
    new_concepts: List[ParsedConcept]
    collapsed_reformatted_txt: List[Document]
    collected_concepts: List[ConceptData]
    source_contents: SourceContents
    content_result: Dict
    instructions: str
    split_complete: bool = False
    reformat_complete: bool = False
    split_reformatted_complete: bool = False
    find_concepts_complete: bool = False
    collapse_concepts_complete: bool = False
    concept_complete: bool = False
    summary_complete: bool = False
    collapse_complete: bool = False
    finalize_complete: bool = False
    rag_complete: bool = False


class ProcessTextConfig(TypedDict):
    instructions: str = None
    collect_concepts: bool = False
    overwrite_sources: bool = False
    update_rag: bool = False
    summarize: bool = True
    rewrite_text: bool = True


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
        "semantic_contents": split,
        "contents": response,
        "instructions": (
            config["configurable"]["instructions"]
            if "instructions" in config["configurable"]
            else None
        ),
    }


async def get_source_content(state: ProcessTextState, config: RunnableConfig):
    source = state["filename"] if state["filename"] is not None else state["url"]
    sources = get_db_sources(source)
    source_data = sources[source]
    if source_data is None:
        raise ValueError(f"Source {source} not found")

    split = await a_semantic_splitter(source_data.texts)
    split = [
        Document(page_content=clean_dividers(doc), metadata={"snippet": i})
        for i, doc in enumerate(split)
    ]
    response = split_list_of_docs(
        split, length_function, SETTINGS.default_llms.instruct_detailed.char_limit
    )

    return {
        "split_complete": True,
        "semantic_contents": split,
        "contents": response,
        "instructions": (
            config["configurable"]["instructions"]
            if "instructions" in config["configurable"]
            else None
        ),
        "reformat_complete": True,
        "reformat_contents": source_data.texts,
        "collapsed_reformatted_txt": [
            Document(
                txt,
            )
            for txt in source_data.texts
        ],
    }


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
) -> Literal["split_reformatted_content", "collapse_content", "finalize_content"]:
    if config["configurable"]["collect_concepts"]:
        return "split_reformatted_content"
    else:
        return should_collapse(state=state)


class ProcessConceptsState(TypedDict):
    url: str
    file: str
    content: Document
    taxonomy: Dict[str, ConceptTaxonomy]


_new_ids = {}


def get_taxonomy_item_list(categories: List[str], reset=False) -> List[ConceptTaxonomy]:
    concepts = get_db_concept_taxonomy(categories=categories, reset=reset)
    all_concepts:List[ConceptTaxonomy] = []

    for category, concept_list in concepts.items():
        for concept in concept_list:
            all_concepts.append(concept)
    return all_concepts


def get_specific_tag(items, tag="category_tag") -> List[dict]:
    found_items = []
    for item in items:
        if item["tag"] == tag:
            found_items.append(item)
        if 0 < len(item["children"]):
            found_items.extend(get_specific_tag(item["children"], tag))
    return found_items


async def split_reformatted_content(state: ProcessTextState):
    snippets: List[Document] = []
    flat_contents: List[Document] = []
    for item in state["contents"]:
        if isinstance(item, Document):
            flat_contents.append(item)
        elif isinstance(item, List):
            flat_contents.extend(item)
        else:
            # print(f"{state["contents"]}")
            raise ValueError(f"Unknown type {type(item)}: {item=}")
    for page, content in enumerate(flat_contents):
        if isinstance(content, Document):
            content = content.page_content

        # print(f"Snippet {page}:\n{content[:100]}...")

        split = await a_semantic_splitter(
            content, threshold_type="gradient", breakpoint_threshold=90
        )
        snippets += [
            Document(
                page_content=clean_dividers(doc), metadata={"snippet": i, "page": page}
            )
            for i, doc in enumerate(split)
        ]

    existing_concept_taxonomy: List[ConceptTaxonomy] = get_taxonomy_item_list(
        reset=True, categories=state["categories"]
    )

    pretty_print(existing_concept_taxonomy, "Existing taxonomy")

    existing_taxonomy = ""
    existing_taxonomy += "\n\n".join(
        [
            convert_taxonomy_dict_to_tag_structure_string(
                convert_concept_category_tag_to_dict(v)
            )
            for v in existing_concept_taxonomy
        ]
        if 0 < len(existing_concept_taxonomy)
        else []
    )

    nex_taxonomy = ""
    new_taxonomy_items = []
    for snippet in snippets:
        snippet_taxonomy = await get_chain("concept_taxonomy").ainvoke(
            {
                "existing_taxonomy": existing_taxonomy + nex_taxonomy,
                "context": snippet.page_content,
            }
        )
        category_tags = get_specific_tag(snippet_taxonomy["parsed"]["children"] or [])
        if len(category_tags) == 0:
            print(f"No category tag found for snippet: {snippet.page_content[:100]}...")
            continue
        if len(category_tags) > 0:
            for item in category_tags:
                new_taxonomy_items.append(
                    convert_taxonomy_tags_to_dict(item, TAXONOMY_ALL_TAGS)
                )

    global _new_ids
    _new_ids[state["filename"] if "filename" in state else state["url"]] = []
    return {
        "split_reformatted_complete": True,
        "taxonomy": new_taxonomy_items,
        "snippets": snippets,
    }


async def map_find_concepts(state: ProcessTextState):
    taxonomy: List[ConceptTaxonomy] = get_taxonomy_item_list(
        categories=state["categories"]
    )

    new_taxonomy: Dict[str, Dict] = state["taxonomy"]
    state_taxonomy = [
        convert_taxonomy_dict_to_tag_structure_string(
            convert_concept_category_tag_to_dict(item)
        )
        for item in taxonomy
    ] + [convert_taxonomy_dict_to_tag_structure_string(item) for item in new_taxonomy]

    pretty_print(new_taxonomy, "Taxonomy items")

    return [
        Send(
            "find_concepts",
            {
                "url": state["url"] if "url" in state else None,
                "filename": state["filename"] if "filename" in state else None,
                "content": snippet,
                "taxonomy": state_taxonomy,
            },
        )
        for snippet in state["snippets"]
    ]


def get_unique_id(id_str: str, existing_ids: List[str]):
    id = f"{id_str.lower().strip().replace(' ', '_')}"
    if id not in existing_ids:
        return id
    id_index = 0
    while True:
        id_index += 1
        new_id = f"{id}_{id_index}"
        if new_id not in existing_ids:
            return new_id


_divider = "±!add!±"


async def find_concepts(state: ProcessConceptsState, config: RunnableConfig):
    ident = (
        f"File: {state['filename']} --\tPage: {state['content'].metadata['page']} --\tSnippet: {state['content'].metadata['snippet']}"
        if state["filename"]
        else f"URL: {state['url']} --\tPage: {state['content'].metadata['page']} --\tSnippet: {state['content'].metadata['snippet']}"
    )
    taxonomy = "\n".join(state["taxonomy"] or "")

    params = {
        "context": ident + f"Content: {get_text_from_completion(state['content'])}",
        "taxonomy": taxonomy,
    }

    new_concepts: List[ParsedConcept] = []

    # print(f"{ident}: Finding concepts in")
    more_concepts: ParsedConceptList = await get_chain("concept_structured").ainvoke(
        params
    )

    max_repeat = 1
    repeat = 0
    try:
        while len(more_concepts.concepts) > 0:
            repeat += 1

            new_concepts += more_concepts.concepts
            if repeat > max_repeat:
                break
            # print(f"{ident}: Finding more concepts after found: {len(more_concepts.concepts)} {repeat=}")
            more_concepts: ParsedConceptList = await get_chain("concept_more").ainvoke(
                {
                    "context": params["context"],
                    "taxonomy": taxonomy,
                    "existing_concepts": "\n".join(
                        f"{concept.title.replace('\n', ' ').strip()}:\n"
                        f"Summary: {concept.summary.replace('\n', ' ').strip()}\n"
                        f"Taxonomy IDs: {', '.join(concept.taxonomy).replace('\n', ' ').strip()}\n"
                        f"Content:\n{concept.content.strip()}\n"
                        for concept in new_concepts
                    ),
                }
            )
    except Exception as e:
        print(e)

    global _new_ids
    global_new_ids = _new_ids[
        state["filename"] if "filename" in state else state["url"]
    ]
    new_ids = []
    for concept in new_concepts:
        concept.id = get_unique_id(
            concept.title, get_existing_concept_ids() + new_ids + global_new_ids
        )
        new_ids.append(concept.id)
        concept.page_number = state["content"].metadata["page"]

    global_new_ids += new_ids
    # print(f"{ident}: Finding unique concepts in")
    unique_concept_ids: ParsedUniqueConceptList = await get_chain(
        "concept_unique"
    ).ainvoke(
        {
            "existing_concepts": "\n".join(
                f"Id({concept.id}): {concept.title.replace('\n', ' ').strip()}\n"
                f"Summary: {concept.summary.replace('\n', ' ').strip()}\n"
                f"Taxonomy IDs: {', '.join(concept.taxonomy).replace('\n', ' ').strip()}\n"
                f"Content:\n{concept.content.strip()}\n"
                for concept in new_concepts
            )
        }
    )
    global _divider

    concepts: Dict[str, ParsedConcept] = {}
    for combined_concept in unique_concept_ids.concepts:
        if combined_concept.id in concepts.keys():
            continue
        for concept in new_concepts:
            if concept.id == combined_concept.id:
                concept.summary = combined_concept.summary.replace("\n", " ").strip()
                concept.title = combined_concept.title.replace("\n", " ").strip()
                concepts[combined_concept.id] = concept

    for combined_concept in unique_concept_ids.concepts:
        for concept in new_concepts:
            if (
                combined_concept.id in concepts.keys()
                and combined_concept.combined_ids is not None
                and concept.id in combined_concept.combined_ids
            ):
                concepts[combined_concept.id].content += (
                    "\n" + _divider + "\n" + concept.content.strip()
                )
                concepts[combined_concept.id].taxonomy = list(
                    set(concepts[combined_concept.id].taxonomy + concept.taxonomy)
                )

    return {"found_concepts": list(concepts.values())}


async def combine_concepts(state: ProcessTextState, config: RunnableConfig):
    found_concepts = state["found_concepts"]
    new_concepts = []
    for item in found_concepts:
        if isinstance(item, ParsedConcept):
            new_concepts.append(item)
        if isinstance(item, List):
            new_concepts.extend(item)

    return {"find_concepts_complete": True, "new_concepts": new_concepts}


# def unwrap_concept_hierarchy(
#     structure: Union[
#         ParsedConceptStructure, List[ParsedConceptStructure], ParsedConceptStructureList
#     ]
# ) -> Dict[str, List]:
#     concept_children: Dict[str, List[str]] = {}
#     if isinstance(structure, ParsedConceptStructureList):
#         structure = structure.structure

#     if isinstance(structure, ParsedConceptStructure) and structure.children is not None and len(structure.children) > 0:
#         if structure.id not in concept_children.keys():
#             concept_children[structure.id] = []
#         for child in structure.children:
#             concept_children[structure.id].append(child.id)
#             concept_children.update(unwrap_concept_hierarchy(child))

#     return concept_children


def unwrap_concept_hierarchy(
    structure: ParsedConceptStructureList,
) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}

    def traverse(node: ParsedConceptStructure, parent_id=None):
        if parent_id:
            if parent_id not in result:
                result[parent_id] = []
            result[parent_id].append(node.id)

        for child in node.children:
            traverse(child, node.id)

    for node in structure.structure:
        traverse(node)

    return result


async def collapse_concepts(state: ProcessTextState, config: RunnableConfig):
    concepts: List[ConceptData] = []  # state["concepts"] if "concepts" in state else []
    all_concept_ids = get_existing_concept_ids(refresh=True)
    found_concepts: List[ParsedConcept] = state["new_concepts"]
    source = state["filename"] if "filename" in state else state["url"]

    if found_concepts is not None and len(found_concepts) > 0:
        found_concepts_by_id: Dict[str, ParsedConcept] = {
            concept.id: concept for concept in found_concepts
        }

        existing_taxonomy_items: List[ConceptTaxonomy] = get_taxonomy_item_list(
            reset=True, categories=state["categories"]
        )
        existing_taxonomy_by_id: Dict[str, ConceptTaxonomy] = {
            concept.id: concept for concept in existing_taxonomy_items
        }

        existing_taxonomy = [convert_concept_category_tag_to_dict(v) for v in existing_taxonomy_items] if 0 < len(existing_taxonomy_items) else []

        new_concept_taxonomy: List[Dict] = state["taxonomy"]
        previous_concept_data: List[ConceptDataTable] = get_db_concepts(source=source)
        previous_concepts: Dict[str, ConceptData] = {}
        if len(previous_concept_data) > 0:
            for concept in previous_concept_data:
                previous_concepts[concept.id] = ConceptData(**concept.concept_contents)

        get_unique_concepts = lambda: get_chain("concept_unique").ainvoke(
            {
                "existing_concepts": "Previously defined concepts:\n"
                + "\n".join(
                    f"Id({concept.id}): {concept.title}\nSummary: {concept.summary}\nTaxonomy IDs: {', '.join(concept.taxonomy)}"
                    for concept in previous_concepts.values()
                )
                + "\n\nNewly defined concepts:\n"
                + "\n".join(
                    f"Id({concept.id}): {concept.title}\nSummary: {concept.summary}\nTaxonomy IDs: {', '.join(concept.taxonomy)}"
                    for concept in found_concepts
                )
            }
        )

        # print(f"Found {len(found_concepts)} concepts, optimizing for unique concepts.")
        unique_concepts: ParsedUniqueConceptList = await get_unique_concepts()
        if isinstance(unique_concepts, AIMessage):
            print("\n\nRetrying get unique concepts...")
            unique_concepts = await get_unique_concepts()
        # await get_chain(
        #     "concept_unique"
        # ).ainvoke(
        #     {
        #         "existing_concepts": "Previously defined concepts:\n"
        #         + "\n".join(
        #             f"Id({concept.id}): {concept.title}\nSummary: {concept.summary}\nTaxonomy IDs: {', '.join(concept.taxonomy)}"
        #             for concept in previous_concepts.values()
        #         )
        #         + "\n\nNewly defined concepts:\n"
        #         + "\n".join(
        #             f"Id({concept.id}): {concept.title}\nSummary: {concept.summary}\nTaxonomy IDs: {', '.join(concept.taxonomy)}"
        #             for concept in found_concepts
        #         )
        #     }
        # )

        pretty_print(unique_concepts, "New unique concepts")

        concept_hierarchy_task = lambda: get_chain("concept_hierarchy").ainvoke(
            {
                "existing_concepts": "\n".join(
                    f"Id({concept.id}): {concept.title}\n{concept.summary}"
                    for concept in unique_concepts.concepts
                )
            }
        )

        concept_taxonomy_task = lambda: get_chain(
            "concept_taxonomy_structured"
        ).ainvoke(
            {
                "existing_taxonomy": "\n".join(
                    convert_taxonomy_dict_to_tag_structure_string(item)
                    for item in existing_taxonomy
                ),
                "new_taxonomy": "\n".join(
                    convert_taxonomy_dict_to_tag_structure_string(item)
                    for item in new_concept_taxonomy
                ),
                "concepts": "\n".join(
                    f"Id({concept.id}): {concept.title}\nSummary: {concept.summary}\nTaxonomy IDs: {', '.join(concept.taxonomy)}"
                    for concept in unique_concepts.concepts
                ),
            }
        )

        # : ParsedConceptStructure
        # concept_hierarchy_task = get_chain("concept_hierarchy").ainvoke(
        #     {
        #         "existing_concepts": "\n".join(
        #             f"Id({concept.id}): {concept.title}\n{concept.summary}"
        #             for concept in unique_concepts.concepts
        #         )
        #     }
        # )
        # # : ParsedConceptTaxonomyList
        # concept_taxonomy_task = get_chain("concept_taxonomy_structured").ainvoke(
        #     {
        #         "existing_taxonomy": "\n".join(
        #             convert_taxonomy_dict_to_tag_structure_string(item)
        #             for item in existing_taxonomy
        #         ),
        #         "new_taxonomy": "\n".join(
        #             convert_taxonomy_dict_to_tag_structure_string(item)
        #             for item in new_concept_taxonomy
        #         ),
        #         "concepts": "\n".join(
        #             f"Id({concept.id}): {concept.title}\nSummary: {concept.summary}\nTaxonomy IDs: {', '.join(concept.taxonomy)}"
        #             for concept in unique_concepts.concepts
        #         ),
        #     }
        # )

        concept_taxonomy: ParsedConceptTaxonomyList
        concept_hierarchy: ParsedConceptStructureList
        concept_taxonomy, concept_hierarchy = await asyncio.gather(
            concept_taxonomy_task(), concept_hierarchy_task()
        )
        global _divider

        # Retry once if fail.
        if isinstance(concept_taxonomy, AIMessage):
            print("\n\nRetrying concept taxonomy...")
            concept_taxonomy = await concept_taxonomy_task()

        if isinstance(concept_hierarchy, AIMessage):
            print("\n\nRetrying concept hierarchy...")
            concept_hierarchy = await concept_hierarchy_task()

        pretty_print(concept_taxonomy, "Concept taxonomy", force=True)
        pretty_print(concept_hierarchy, "Concept hierarchy", force=True)

        unwrapped_hierarchy = unwrap_concept_hierarchy(concept_hierarchy)
        inverted_hierarchy: Dict[str, str] = {}
        for key, value in unwrapped_hierarchy.items():
            for item in value:
                inverted_hierarchy[item] = key

        pretty_print(unwrapped_hierarchy, "Unwrapped hierarchy")
        pretty_print(inverted_hierarchy, "Inverted hierarchy")

        existing_concept_category_ids = existing_taxonomy_by_id.keys()
        new_category_ids = []
        for category in concept_taxonomy.taxonomy:
            category.id = (
                category.id
                or get_unique_id(
                    category.tag, list(existing_concept_category_ids) + new_category_ids
                )
            ).strip()
            new_id = category.id
            changes = False

            if new_id in existing_concept_category_ids:
                changes = (
                    changes
                    or existing_taxonomy_by_id[new_id].tag != category.tag
                    or existing_taxonomy_by_id[new_id].title != category.title
                    or existing_taxonomy_by_id[new_id].description
                    != category.description
                    or existing_taxonomy_by_id[new_id].taxonomy != category.taxonomy
                    or existing_taxonomy_by_id[new_id].parent_taxonomy
                    != category.parent_taxonomy
                )
                existing_taxonomy_by_id[new_id].tag = category.tag.strip()
                existing_taxonomy_by_id[new_id].title = category.title.replace(
                    "\n", " "
                ).strip()
                existing_taxonomy_by_id[new_id].description = (
                    category.description.replace("\n", " ").strip()
                )
                existing_taxonomy_by_id[new_id].taxonomy = category.taxonomy.strip()
                existing_taxonomy_by_id[new_id].parent_taxonomy = (
                    category.parent_taxonomy.strip()
                ) if category.parent_taxonomy else None
            else:
                new_category_ids.append(new_id)
                changes = True
                existing_taxonomy_by_id[new_id] = ConceptTaxonomy(
                    id=new_id,
                    parent_id=category.parent_id,
                    title=category.title.replace("\n", " ").strip(),
                    tag=category.tag.strip(),
                    description=category.description.replace("\n", " ").strip(),
                    taxonomy=category.taxonomy.strip(),
                    parent_taxonomy=category.parent_taxonomy.strip() if category.parent_taxonomy else None,
                    type=category.type.strip(),
                )

            if category.parent_id:
                changes = (
                    changes
                    or existing_taxonomy_by_id[new_id].parent_id != category.parent_id
                )
                existing_taxonomy_by_id[new_id].parent_id = category.parent_id

            if changes:
                pretty_print(
                    existing_taxonomy_by_id[new_id], "Updated category"
                )
                # print(f"\n\n\nNew/Updated category tag:\n\n{existing_taxonomy_by_id[new_id].model_dump_json(indent=4)}")
                update_concept_category(
                    existing_taxonomy_by_id[new_id], categories=state["categories"]
                )

        removed_ids = []

        all_ids = list(
            set(
                [concept.id for concept in unique_concepts.concepts]
                + list(previous_concepts.keys())
            )
        )

        all_taxonomy_ids = list(existing_taxonomy_by_id.keys())

        # Create new concepts
        for sum_concept in unique_concepts.concepts:
            id = sum_concept.id
            old_concept_item = False
            new_concept: ConceptData
            if id is None or id not in all_concept_ids:
                concept = (
                    found_concepts_by_id[id]
                    if id is not None and id in found_concepts_by_id.keys()
                    else sum_concept
                )
                new_id = concept.id or get_unique_id(concept.title, all_ids)
                new_concept = ConceptData(
                    id=new_id,
                    parent_id=(
                        inverted_hierarchy[new_id]
                        if new_id in inverted_hierarchy.keys()
                        else None
                    ),
                    title=sum_concept.title,
                    summary=sum_concept.summary,
                    contents=(
                        [concept.summary]
                        if isinstance(concept, ParsedConceptIds)
                        else concept.content.split(_divider)
                    ),
                    references=[
                        SourceReference(
                            source=source,
                            page_number=(
                                1
                                if isinstance(concept, ParsedConceptIds)
                                else concept.page_number
                            ),
                        )
                    ],
                    taxonomy=[],
                    sources=[source],
                    children=[
                        child_id
                        for child_id in unwrapped_hierarchy.get(id, [])
                        if child_id in all_ids
                    ],
                )
            elif id in found_concepts_by_id.keys():
                concept = found_concepts_by_id[id]
                new_concept = previous_concepts[id]
                new_concept.contents.extend(concept.content.split(_divider))
                new_concept.children = list(
                    set(
                        new_concept.children
                        + [
                            child_id
                            for child_id in unwrapped_hierarchy.get(id, [])
                            if child_id in all_ids
                        ]
                    )
                )
                new_concept.taxonomy = [
                    taxonomy_id
                    for taxonomy_id in list(
                        set(new_concept.taxonomy + sum_concept.taxonomy)
                    )
                    if taxonomy_id in all_taxonomy_ids
                ]

                references = [
                    ref for ref in new_concept.references if ref.source == source
                ]
                if concept.page_number not in [ref.page_number for ref in references]:
                    new_concept.references.append(
                        SourceReference(
                            source=source,
                            page_number=concept.page_number,
                        )
                    )
            elif id in previous_concepts.keys():
                old_concept_item = True
                new_concept = previous_concepts[id]

            if sum_concept.combined_ids is not None:
                for id in sum_concept.combined_ids:
                    if old_concept_item:
                        removed_ids.append(new_concept.id)
                    if id in previous_concepts.keys():
                        removed_ids.append(id)
                        new_concept.contents.extend(previous_concepts[id].contents)
                        for ref in previous_concepts[id].references:
                            if f"{ref.source}_{(ref.page_number or 0)}" not in [
                                f"{ref.source}_{(ref.page_number or 0)}"
                                for ref in new_concept.references
                            ]:
                                new_concept.references.append(ref)
                        new_concept.sources = list(
                            set(new_concept.sources + previous_concepts[id].sources)
                        )
                        new_concept.children.append(previous_concepts[id].id)
                        new_concept.parent_id = (
                            new_concept.parent_id or previous_concepts[id].parent_id
                        )
                    if id in found_concepts_by_id.keys():
                        concept = found_concepts_by_id[id]
                        new_concept.contents.extend(concept.content.split(_divider))
                        references = [
                            ref
                            for ref in new_concept.references
                            if ref.source == source
                        ]
                        if concept.page_number not in [
                            ref.page_number for ref in references
                        ]:
                            new_concept.references.append(
                                SourceReference(
                                    source=source,
                                    page_number=concept.page_number,
                                )
                            )

            for taxonomy_item in concept_taxonomy.taxonomy:
                if new_concept.id in taxonomy_item.connected_concepts:
                    if taxonomy_item.id not in new_concept.taxonomy:
                        new_concept.taxonomy.append(taxonomy_item.id)

            concepts.append(new_concept)
        for removed_id in removed_ids:
            if removed_id in previous_concepts.keys():
                del previous_concepts[removed_id]
                delete_db_concept(removed_id, commit=False)

        db_commit()

    return {
        "concept_complete": True,
        "collapse_concepts_complete": True,
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


async def collapse_concept(i: int, concepts: List[ConceptData], concept: ConceptData):
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
            summary: TitledSummary = (
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
        # concepts = defaultdict(list[ConceptData])
        tasks = []
        for i, concept in enumerate(collected_concepts):
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
            "summarize_complete": True,
            "collapse_complete": True,
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
            "summarize_complete": True,
            "collapse_complete": True,
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
    update_db_source_rag_concepts(
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
process_text_graph.add_node("get_source_content", get_source_content)
process_text_graph.add_node("split_reformatted_content", split_reformatted_content)
process_text_graph.add_node("find_concepts", find_concepts)
process_text_graph.add_node("combine_concepts", combine_concepts)
process_text_graph.add_node("collapse_concepts", collapse_concepts)
process_text_graph.add_node("collapse_content", collapse_content)
process_text_graph.add_node("finalize_content", finalize_content)
process_text_graph.add_node("rag_content", rag_content)

# Edges:
process_text_graph.add_conditional_edges(START, should_rewrite_content)
# process_text_graph.add_edge(START, "split_content")
process_text_graph.add_conditional_edges(
    "split_content", map_reformat, ["reformat_content"]
)
process_text_graph.add_edge("reformat_content", "concat_reformat")
process_text_graph.add_conditional_edges("concat_reformat", should_conceptualize)
process_text_graph.add_conditional_edges("get_source_content", should_conceptualize)
process_text_graph.add_conditional_edges(
    "split_reformatted_content", map_find_concepts, ["find_concepts"]
)
process_text_graph.add_edge("find_concepts", "combine_concepts")
process_text_graph.add_edge("combine_concepts", "collapse_concepts")
process_text_graph.add_conditional_edges("collapse_concepts", should_collapse)
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
