import asyncio
import itertools
import operator
from typing import Annotated, Dict, List, Set, TypedDict, Union

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.constants import Send

from lib.chains.hierarchy_compiler import get_hierarchy
from lib.chains.init import get_chain
from lib.db.concept import delete_db_concept, get_db_concepts, get_existing_concept_ids
from lib.db.sqlite import db_commit
from lib.db.taxonomy import get_taxonomy_item_list
from lib.helpers import (
    get_text_from_completion,
    get_unique_id,
    pretty_print,
    unwrap_hierarchy,
)
from lib.load_env import SETTINGS
from lib.models.prompts import TitledSummary

from lib.document_tools import (
    split_text,
)
from lib.models.taxonomy import (
    Taxonomy,
    convert_taxonomy_to_dict,
    convert_taxonomy_dict_to_tag_simple_structure_string,
)
from lib.models.concepts import (
    ParsedConcept,
    ConceptDataTable,
    ParsedConceptIds,
    ParsedConceptList,
    ConceptData,
    ParsedConceptStructureList,
    ParsedUniqueConceptList,
    SourceReference,
    get_concept_str,
)
from lib.models.source import (
    split_topics,
)


class FindConceptsState(TypedDict):
    categories: List[str]
    filename: str
    url: str
    source: str
    content_topics: List[Dict]
    instructions: str
    # Generated
    concepts: List[ConceptData]
    found_concepts: Annotated[list, operator.add]
    new_concepts: List[ParsedConcept]
    collected_concepts: List[ConceptData]
    # semantic_contents: List[Document]
    search_concepts_complete: bool = False
    collapse_concepts_complete: bool = False
    concept_complete: bool = False
    taxonomy: Dict


class FindConceptsConfig(TypedDict):
    instructions: str = None


class ProcessConceptsState(TypedDict):
    categories: List[str]
    filename: str
    url: str
    topics: List[Dict]
    taxonomy: Dict
    taxonomy_str: str


_new_ids = {}
_divider = "±!s!±"


async def map_search_concepts(state: FindConceptsState):
    taxonomy: List[Taxonomy] = get_taxonomy_item_list(categories=state["categories"])

    state_taxonomy = [
        convert_taxonomy_dict_to_tag_simple_structure_string(
            convert_taxonomy_to_dict(item)
        )
        for item in taxonomy
    ]

    return [
        Send(
            "search_concepts",
            {
                "url": state["url"] if "url" in state else None,
                "filename": state["filename"] if "filename" in state else None,
                "categories": state["categories"],
                "topics": topics,
                "taxonomy": taxonomy,
                "taxonomy_str": state_taxonomy,
            },
        )
        for topics in split_topics(state["content_topics"])
    ]


async def search_concepts(state: ProcessConceptsState, config: RunnableConfig):
    topics: List[Dict] = state["topics"]
    content = ""
    for topic in topics:
        ident = (
            f"File: {state['filename']} --\tPage: {topic['page_number']} --\tTopics: {topic['topic_index']}"
            if state["filename"]
            else f"URL: {state['url']} --\tPage: {topic['page_number']} --\tTopic: {topic['topic_index']}"
        )
        content += (
            ident
            + f"{topic['topic']}: {get_text_from_completion(topic['page_content'])}\n\n"
        )

    taxonomy = "\n".join(state["taxonomy_str"]) or ""
    params = {
        "context": content,
        "taxonomy": taxonomy,
    }

    new_concepts: List[ParsedConcept] = []

    # print(f"{ident}: Finding concepts in")
    more_concepts: ParsedConceptList = await get_chain("concept_structured").ainvoke(
        params
    )
    if isinstance(more_concepts, AIMessage):
        print("Retry concept_structured once")
        await asyncio.sleep(30)
        more_concepts: ParsedConceptList = await get_chain(
            "concept_structured"
        ).ainvoke(params)

    max_repeat = 1
    repeat = 0
    try:
        while len(more_concepts.concepts) > 0:
            repeat += 1

            new_concepts += more_concepts.concepts
            if repeat > max_repeat:
                break
            params = {
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
            # print(f"{ident}: Finding more concepts after found: {len(more_concepts.concepts)} {repeat=}")
            more_concepts: ParsedConceptList = await get_chain("concept_more").ainvoke(
                params
            )
            if isinstance(more_concepts, AIMessage):
                print("Retry concept_more once")
                await asyncio.sleep(30)
                more_concepts: ParsedConceptList = await get_chain(
                    "concept_more"
                ).ainvoke(params)
    except Exception as e:
        print(e)

    global _new_ids
    global_new_ids = _new_ids.get(
        state["filename"] if "filename" in state else state["url"], []
    )
    _new_ids[state["filename"] if "filename" in state else state["url"]] = (
        global_new_ids
    )
    new_ids = []
    cat_for_id = "-".join(state["categories"]).replace(" ", "-").lower()
    for concept in new_concepts:
        concept.id = get_unique_id(
            cat_for_id + "-" + concept.title,
            get_existing_concept_ids() + new_ids + global_new_ids,
        )
        new_ids.append(concept.id)
        concept.page_number = concept.page_number or -1

    global_new_ids += new_ids
    params = {
        "existing_concepts": "",
        "new_concepts": "\n".join(
            f"Id({concept.id}): {concept.title.replace('\n', ' ').strip()}\n"
            f"Summary: {concept.summary.replace('\n', ' ').strip()}\n"
            f"Taxonomy IDs: {', '.join(concept.taxonomy).replace('\n', ' ').strip()}\n"
            f"Content:\n{concept.content.strip()}\n"
            for concept in new_concepts
        ),
    }
    # print(f"{ident}: Finding unique concepts in")
    unique_concept_ids: ParsedUniqueConceptList = await get_chain(
        "concept_unique"
    ).ainvoke(params)
    if isinstance(unique_concept_ids, AIMessage):
        print("Retry concept_unique once")
        await asyncio.sleep(30)
        unique_concept_ids: ParsedUniqueConceptList = await get_chain(
            "concept_unique"
        ).ainvoke(params)
    global _divider

    concepts: Dict[str, ParsedConcept] = {}
    try:
        for combined_concept in unique_concept_ids.concepts:
            if combined_concept.id in concepts.keys():
                continue
            for concept in new_concepts:
                if concept.id == combined_concept.id:
                    concept.summary = combined_concept.summary.replace(
                        "\n", " "
                    ).strip()
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
    except Exception as e:
        print(e)

    return {"found_concepts": list(concepts.values())}


async def combine_concepts(state: FindConceptsState, config: RunnableConfig):
    found_concepts = state["found_concepts"]
    new_concepts = []
    for item in found_concepts:
        if isinstance(item, ParsedConcept):
            new_concepts.append(item)
        if isinstance(item, List):
            new_concepts.extend(item)

    return {"search_concepts_complete": True, "new_concepts": new_concepts}


concept_hierarchy_task = lambda concepts: get_chain("concept_hierarchy").ainvoke(
    {
        "hierarchy_items": get_concept_str(concepts, summary=True, taxonomy=True),
    }
)


async def collapse_concepts(state: FindConceptsState, config: RunnableConfig):
    concepts: List[ConceptData] = []  # state["concepts"] if "concepts" in state else []
    all_concept_ids = get_existing_concept_ids(refresh=True)
    found_concepts: List[ParsedConcept] = state["new_concepts"]
    source = state["filename"] if "filename" in state else state["url"]

    if found_concepts is not None and len(found_concepts) > 0:
        found_concepts_by_id: Dict[str, ParsedConcept] = {
            concept.id: concept for concept in found_concepts
        }

        previous_concept_data: List[ConceptDataTable] = get_db_concepts(source=source)
        previous_concepts: Dict[str, ConceptData] = {}
        sorted_ids = []
        if len(previous_concept_data) > 0:
            for concept in previous_concept_data:
                previous_concepts[concept.id] = ConceptData(
                    **concept.concept_contents.__dict__
                )
                previous_concepts[concept.id].taxonomy = sorted(
                    previous_concepts[concept.id].taxonomy
                )
            sorted_ids = sorted(
                previous_concepts.keys(),
                key=lambda x: (
                    "-".join(previous_concepts[x].taxonomy),
                    previous_concepts[x].id,
                ),
            )

        global _divider

        existing_concepts_content = get_concept_str(
            [previous_concepts[id] for id in sorted_ids],
            summary=True,
            taxonomy=True,
            as_array=True,
        )

        found_concepts = sorted(
            found_concepts, key=lambda x: ("-".join(x.taxonomy), x.id)
        )
        new_concepts_content = get_concept_str(
            found_concepts, summary=True, taxonomy=True, as_array=True
        )

        unique_contents: List[tuple] = []
        if (
            len("\n".join(existing_concepts_content) + "\n".join(new_concepts_content))
            > SETTINGS.default_llms.instruct.char_limit
        ):
            print("Chunk new concepts...")
            new_content = ""
            for item in new_concepts_content:
                new_content += item + "\n\n"
                if len(new_content) > SETTINGS.default_llms.instruct.char_limit / 2:
                    unique_contents.append((existing_concepts_content, new_content))
                    new_content = ""
            if len(new_content) > 0:
                unique_contents.append((existing_concepts_content, new_content))
        else:
            unique_contents = [
                ("\n".join(existing_concepts_content), "\n".join(new_concepts_content))
            ]

        get_unique_concepts = lambda existing_concepts, new_concepts: get_chain(
            "concept_unique"
        ).ainvoke(
            {"existing_concepts": existing_concepts, "new_concepts": new_concepts}
        )

        # print(f"Found {len(found_concepts)} concepts, optimizing for unique concepts.")
        uniq_concepts: List[ParsedConceptIds] = []
        combined_ids: Dict[str, List[str]] = {}
        for uniq_item in unique_contents:
            item_unique_concepts: ParsedUniqueConceptList = await get_unique_concepts(
                uniq_item[0], uniq_item[1]
            )
            if isinstance(item_unique_concepts, AIMessage):
                print("\n\nRetry concept_unique once...")
                await asyncio.sleep(30)
                item_unique_concepts = await get_unique_concepts(
                    uniq_item[0], uniq_item[1]
                )
            for item in item_unique_concepts.concepts:
                if (
                    item.id is not None
                    and item.combined_ids is not None
                    and len(item.combined_ids) > 0
                ):
                    if item.id in combined_ids:
                        combined_ids[item.id] = list(
                            set(item.combined_ids + combined_ids[item.id])
                        )
                    else:
                        combined_ids[item.id] = item.combined_ids
            uniq_concepts.extend(item_unique_concepts.concepts)

        # if len(unique_contents) > 1:
        #     new_concepts_content = get_concept_str(
        #         uniq_concepts, summary=True, taxonomy=True, as_array=True
        #     )

        #     item_unique_concepts: ParsedUniqueConceptList = await get_unique_concepts(
        #         existing_concepts_content, new_concepts_content
        #     )
        #     if isinstance(item_unique_concepts, AIMessage):
        #         print("\n\nRetry concept_unique once...")
        #         await asyncio.sleep(30)
        #         item_unique_concepts = await get_unique_concepts(
        #             existing_concepts_content, new_concepts_content
        #         )
        #     uniq_concepts = item_unique_concepts.concepts

        for item in uniq_concepts:
            if (
                item.id is not None
                and item.combined_ids is not None
                and len(item.combined_ids) > 0
            ):
                if item.id in combined_ids:
                    combined_ids[item.id] = list(
                        set(item.combined_ids + combined_ids[item.id])
                    )
                else:
                    combined_ids[item.id] = item.combined_ids

        reverse_combined_ids: Dict[str, str] = {}
        all_combined_ids = list(set(itertools.chain(*combined_ids.values())))
        for parent_id, value in combined_ids.items():
            found = True
            max_repeat = len(all_combined_ids) * len(combined_ids)
            irepeat = 0
            did_break = False
            if parent_id is not None:
                while parent_id in all_combined_ids and found:
                    irepeat += 1
                    if irepeat > max_repeat:
                        did_break = True
                        break
                    found = True
                    for new_parent_id, par_values in combined_ids.items():
                        if parent_id in par_values:
                            parent_id = new_parent_id
                            found = True
                            break
                    found = False

            if did_break:
                print("Failed to find parent_id for ", parent_id)

            for child_id in value:
                reverse_combined_ids[child_id] = parent_id

        for concept in uniq_concepts:
            if (
                concept.parent_id is not None
                and concept.parent_id in reverse_combined_ids
            ):
                concept.parent_id = reverse_combined_ids[concept.parent_id]

        pretty_print(uniq_concepts, "New unique concepts")

        unwrapped_hierarchy, inverted_hierarchy, flat_hierarchy = await get_hierarchy(
            uniq_concepts,
            lambda x: get_concept_str(x, summary=True, taxonomy=True, as_array=True),
            "concept_hierarchy",
        )

        # concept_hierarchy: ParsedConceptStructureList = await concept_hierarchy_task(
        #     uniq_concepts
        # )
        # if isinstance(concept_hierarchy, AIMessage):
        #     print("\n\nRetrying concept hierarchy...")
        #     concept_hierarchy = await concept_hierarchy_task(uniq_concepts)

        # unwrapped_hierarchy = unwrap_hierarchy(concept_hierarchy)
        # inverted_hierarchy: Dict[str, str] = {}
        # for key, value in unwrapped_hierarchy.items():
        #     for item in value:
        #         inverted_hierarchy[item] = key

        for child_id, parent_id in inverted_hierarchy.items():
            for concept in uniq_concepts:
                if concept.id == child_id and len(concept.taxonomy) == 0:
                    parent_concept = next(
                        (
                            concept
                            for concept in uniq_concepts
                            if concept.id == parent_id
                        ),
                        None,
                    )
                    concept.taxonomy.extend(parent_concept.taxonomy)

        # pretty_print(concept_hierarchy, "Concept hierarchy")
        pretty_print(uniq_concepts, "Uniq concepts", force=True)

        # pretty_print(unwrapped_hierarchy, "Unwrapped hierarchy")
        # pretty_print(inverted_hierarchy, "Inverted hierarchy")

        cat_for_id = "-".join(state["categories"]).replace(" ", "-").lower()

        removed_ids = []

        all_ids = list(
            set(
                [concept.id for concept in uniq_concepts]
                + list(previous_concepts.keys())
            )
        )
        existing_taxonomy: List[Taxonomy] = get_taxonomy_item_list(
            categories=state["categories"], reset=True
        )
        existing_taxonomy_by_id = {item.id: item for item in existing_taxonomy}
        all_taxonomy_ids = list(existing_taxonomy_by_id.keys())

        # Create new concepts
        for sum_concept in uniq_concepts:
            id = sum_concept.id
            old_concept_item = False
            new_concept: ConceptData
            if (
                id is None
                or id not in all_concept_ids
                or id not in previous_concepts.keys()
            ):
                concept = (
                    found_concepts_by_id[id]
                    if id is not None and id in found_concepts_by_id.keys()
                    else sum_concept
                )
                new_id = concept.id or cat_for_id + "-" + get_unique_id(
                    concept.title, all_ids
                )
                parent_id = inverted_hierarchy.get(new_id, None)

                if parent_id is not None and parent_id in reverse_combined_ids:
                    parent_id = reverse_combined_ids[parent_id]

                new_concept = ConceptData(
                    id=new_id,
                    parent_id=parent_id,
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
            elif id in found_concepts_by_id.keys() and id in previous_concepts.keys():
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
                delete_db_concept(id, commit=False)

            elif id in previous_concepts.keys():
                old_concept_item = True
                new_concept = previous_concepts[id]

            if sum_concept.combined_ids is not None:
                if old_concept_item:
                    removed_ids.append(new_concept.id)
                for id in sum_concept.combined_ids:
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
                            set(
                                new_concept.sources
                                + previous_concepts[id].sources
                                + [source]
                            )
                        )
                        new_concept.children.append(previous_concepts[id].id)
                        parent_id = (
                            new_concept.parent_id or previous_concepts[id].parent_id
                        )
                        if parent_id in reverse_combined_ids.keys():
                            parent_id = reverse_combined_ids[parent_id]
                        new_concept.parent_id = parent_id
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


async def collapse_concept(i: int, concepts: List[ConceptData], concept: ConceptData):
    joined = "\n".join(concept.contents)

    if len(concept.contents) > 1:
        reformat = get_text_from_completion(
            await get_chain("text_formatter_simple").ainvoke({"context": joined})
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


async def finalize_concepts(state: FindConceptsState, config: RunnableConfig):
    collected_concepts = state["collected_concepts"]
    tasks = []
    for i, concept in enumerate(collected_concepts):
        tasks.append(collapse_concept(i, collected_concepts, concept))

    await asyncio.gather(*tasks)

    return {
        "collected_concepts": collected_concepts,
        "finalize_concepts_complete": True,
    }


find_concepts_graph = StateGraph(FindConceptsState, FindConceptsConfig)
find_concepts_graph.add_node("search_concepts", search_concepts)
find_concepts_graph.add_node("combine_concepts", combine_concepts)
find_concepts_graph.add_node("collapse_concepts", collapse_concepts)
find_concepts_graph.add_node("finalize_concepts", finalize_concepts)

find_concepts_graph.add_conditional_edges(
    START, map_search_concepts, ["search_concepts"]
)

find_concepts_graph.add_edge("search_concepts", "combine_concepts")
find_concepts_graph.add_edge("combine_concepts", "collapse_concepts")
find_concepts_graph.add_edge("collapse_concepts", "finalize_concepts")
find_concepts_graph.add_edge("finalize_concepts", END)

find_concepts = find_concepts_graph.compile()
