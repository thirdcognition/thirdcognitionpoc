import asyncio
import operator
import textwrap
from typing import Annotated, List, Union
from pydantic import BaseModel
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig, RunnableSequence
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from lib.chains.rag_chain import get_rag_chain
from lib.helpers.shared import get_text_from_completion
from lib.models.journey import (
    ModuleStructure,
    JourneyDataTable,
    SubsubjectModel,
    SubjectModel,
    SubsubjectStructure,
)
from lib.models.concepts import ConceptData
from lib.models.source import SourceData, SourceDataTable
from lib.prompts.journey import JourneyPrompts, Subsubject
from lib.chains.init import get_base_chain, get_chain
from lib.streamlit_tools import llm_edit
from lib.models.reference import Reference


class SubsubjectTemplate(BaseModel):
    id: str
    instructions: str
    module_amount: int


class SubjectTemplate(BaseModel):
    id: str
    subject_instruction: str
    subject_prompt_instructions: str
    subject_prompts: JourneyPrompts
    subsubject_templates: List[SubsubjectTemplate]
    subsubject_amount: 3
    default_module_amount: int = 5


class JourneyTemplate(BaseModel):
    id: str
    journey_instruction: str
    subject_templates: List[SubjectTemplate]
    subject_amount: int = 2


class JourneyCreationConfig(TypedDict):
    journey_template: JourneyTemplate
    amount_of_documents: int = 5


class JourneyCreationState(TypedDict):
    journey_name: str
    chroma_collections: List[str]
    categories: List[str]
    journey: JourneyDataTable
    references: List[Reference]
    subjects: Annotated[list[SubjectModel], operator.add]
    subjects_sorted: List[SubjectModel]
    subjects_done: bool = False
    journey_done: bool = False


class SujectCreationConfig(TypedDict):
    journey_template: JourneyTemplate
    subject_index: int
    amount_of_documents: int = 5


class SubjectCreationState(TypedDict):
    journey: JourneyDataTable
    subject: SubjectModel
    references: List[Reference]
    initial_plan: List[Subsubject]
    initial_plan_done: bool = False
    plan: Annotated[list[SubsubjectModel], operator.add]
    plan_sorted: List[SubsubjectModel]
    plan_done: bool = False
    subject_done: bool = False


class SubsubjectCreationConfig(TypedDict):
    journey_template: JourneyTemplate
    subsubject_index: int
    subject_index: int
    amount_of_documents: int = 5


class SubsubjectCreationState(TypedDict):
    journey: JourneyDataTable
    # concepts: List[ConceptData]
    subsubject: SubsubjectModel
    concepts_content: str
    subject_title: str
    modules: str
    modules_done: bool = False
    subsubject_structured: SubsubjectStructure
    subsubject_structured_done: bool = False
    content_prepare: str
    content_prepare_done: bool = False
    content: str
    content_done: bool = False
    intro: str
    intro_done: bool = False
    subsubject_done: bool = False


def get_journey_items(
    state: Union[JourneyCreationState, SubjectCreationState, SubsubjectCreationState], config
) -> tuple[
    JourneyDataTable,
    SubjectModel,
    SubsubjectModel,
    JourneyTemplate,
    SubjectTemplate,
    SubsubjectTemplate,
]:
    journey: JourneyDataTable = state["journey"] if "journey" in state else None
    subject: SubjectModel = (
        journey.subjects[config["configurable"]["subject_index"]]
        if journey is not None
        else None
    )
    subsubject: SubsubjectModel = (
        state["subsubject"]
        if "subsubject" in state
        else (
            subject.plan[config["configurable"]["subsubject_index"]]
            if subject is not None
            else None
        )
    )
    journey_template: JourneyTemplate = config["configurable"]["journey_template"] if "configurable" in config else None
    if "subject_index" in config["configurable"]:
        subject_template: SubjectTemplate = journey_template.subject_templates[
            config["configurable"]["subject_index"]
        ]
    else:
        subject_template: SubjectTemplate = None
    if "subsubject_index" in config["configurable"] and subject_template is not None:
        subsubject_template: SubsubjectTemplate = subject_template.subsubject_templates[
            config["configurable"]["subsubject_index"]
        ]
    else:
        subsubject_template: SubsubjectTemplate = None

    return journey, subject, subsubject, journey_template, subject_template, subsubject_template


async def modules_build(
    state: SubsubjectCreationState, config: RunnableConfig
) -> SubsubjectCreationState:
    journey, subject, subsubject, journey_template, subject_template, subsubject_template = (
        get_journey_items(state, config)
    )
    prompt = (
        subject.prompts.subsubject_modules
        if subject is not None
        else subject_template.subject_prompts.subsubject_modules
    )

    module_amount = (
        subsubject_template.module_amount
        if subsubject_template.module_amount is not None
        else subject_template.default_module_amount
    )

    concepts = state["concepts"] if "concepts" in state else subsubject.concepts
    class_content = "\n".join(
        [
            f"{concept.title}:\n{concept.content}\nReferences: {", ".join([f"{item.source} ({item.page_number})" for item in concept.references])}"
            for concept in concepts
        ]
    )
    subject_title = f"Title: {subsubject.title}\nSubject: {subsubject.subject}"
    class_modules = await get_base_chain("subsubject_modules")(
        (prompt.system, prompt.user)
    ).ainvoke(
        {
            "context": class_content,
            "journey_instructions": journey_template.journey_instruction,
            "subject_instructions": subject_template.subject_instruction,
            "subject": subject_title,
            "amount": module_amount,
            # "chat_history": previous_class_subjects + previous_class_modules
        }
    )

    return {
        "subject_title": subject_title,
        "concepts_content": class_content,
        "modules": get_text_from_completion(class_modules),
        "modules_done": True,
    }


async def subsubject_structured_build(
    state: SubsubjectCreationState, config: RunnableConfig
) -> SubsubjectCreationState:
    journey, subject, subsubject, journey_template, subject_template, subsubject_template = (
        get_journey_items(state, config)
    )

    instructions = (
        (
            journey.instructions
            if journey is not None
            else (
                journey_template.journey_instruction + "\n" + subject.instructions
                if subject is not None
                else subject_template.subject_instruction
            )
        ).strip()
        + "\n"
        + (
            subsubject.instructions
            if subsubject.instructions is not None
            else subsubject_template.instructions
        ).strip()
    )

    structured = await get_chain("subsubject_structured").ainvoke(
        {
            "context": f"""
            Title:
            {subsubject.title}
            Summary:
            {subsubject.summary}
            Content:
            {state["concepts_content"]}
            Modules:
            {state["modules"]}
        """
            + (
                """
            Instructions:
            {instructions}
        """
                if instructions
                else ""
            ),
        }
    )

    return {
        "subsubject_structured": structured,
        "subsubject_structured_done": True,
    }


async def process_module_to_content(
    module: ModuleStructure, doc_chain: RunnableSequence, context: str
):
    content = (
        "\n\n"
        + f"Subject: {module.title.strip()}\n\nSubject description: {module.description.strip()}"
    )
    content += (
        "\n\n"
        + "\n\nSubject content:"
        + await doc_chain.ainvoke(
            {
                "question": f"Subject: {module.title.strip()}\n\nSubject description: {module.description.strip()}",
                "context": context,
            }
        )["answer"]
    )
    return content


async def content_prepare_build(
    state: SubsubjectCreationState, config: RunnableConfig
) -> SubsubjectCreationState:
    journey = state["journey"] if "journey" in state else None
    structured = state["subsubject_structured"]

    doc_chain = get_rag_chain(
        journey.chroma_collections,
        "hyde_document",
        amount_of_documents=config["configurable"]["amount_of_documents"],
    )
    content = state["subject_title"] + "\n\n" + state["concepts_content"].strip()

    if structured is not None and isinstance(structured, SubsubjectStructure):
        tasks = [
            process_module_to_content(module, doc_chain, state)
            for module in structured.modules
        ]
        contents = await asyncio.gather(*tasks)
        return "\n\n".join(contents)
    else:
        content += "\n\n" + state["modules"]

    return {
        "content_prepare": get_text_from_completion(content),
        "content_prepare_done": True,
    }


async def content_build(
    state: SubsubjectCreationState, config: RunnableConfig
) -> SubsubjectCreationState:
    journey, subject, subsubject, journey_template, subject_template, subsubject_template = (
        get_journey_items(state, config)
    )
    prompt = (
        subject.prompts.subsubject_content
        if subject is not None
        else subject_template.subject_prompts.subsubject_content
    )

    prep_content = state["content_prepare"]

    content = await get_base_chain("subsubject_content")(
        (prompt.system, prompt.user)
    ).ainvoke(
        {
            "context": prep_content,
            "journey_instructions": journey_template.journey_instruction,
            "subject_instructions": subject_template.subject_instruction,
            "subsubject_instructions": (
                subsubject.instructions
                if subject is not None
                else subsubject_template.instructions
            ),
            "subject": state["subject_title"],
        }
    )

    return {
        "content": get_text_from_completion(content),
        "content_done": True,
    }


async def intro_build(
    state: SubsubjectCreationState, config: RunnableConfig
) -> SubsubjectCreationState:
    journey, subject, subsubject, journey_template, subject_template, subsubject_template = (
        get_journey_items(state, config)
    )
    prompt = (
        subject.prompts.subsubject_intro
        if subject is not None
        else subject_template.subject_prompts.subsubject_intro
    )

    class_intro = await get_base_chain("subsubject_intro")(
        (prompt.system, prompt.user)
    ).ainvoke(
        {
            "context": state["content"],
            "journey_instructions": (
                journey.instructions
                if journey is not None
                else journey_template.journey_instruction
            ),
            "subject_instructions": (
                subject.instructions
                if subject is not None
                else subject_template.subject_instruction
            ),
            "subsubject_instructions": (
                subsubject.instructions
                if subject is not None
                else subsubject_template.instructions
            ),
            "subject": state["subject_title"],
            # "chat_history": previous_class_intros
        }
    )

    return {
        "intro": get_text_from_completion(class_intro),
        "intro_done": True,
    }


async def subsubject_build(
    state: SubsubjectCreationState, config: RunnableConfig
) -> SubsubjectCreationState:
    journey, subject, subsubject, journey_template, subject_template = get_journey_items(
        state, config
    )

    new_subsubject = SubsubjectModel(
        title=subsubject.title,
        subject=subsubject.subject,
        summary=subsubject.summary,
        concepts=subsubject.concepts,
        content=state["content"].strip(),
        intro=state["intro"].strip(),  # class_intro.strip(),
        modules=state["modules"].strip(),
        structured=state["subsubject_structured"],
    )

    return {
        "subsubject": new_subsubject,
        "subsubject_done": True,
    }


subsubject_creation_graph = StateGraph(SubsubjectCreationState, JourneyCreationConfig)
subsubject_creation_graph.add_node("modules_build", modules_build)
subsubject_creation_graph.add_node("subsubject_structured_build", subsubject_structured_build)
# TODO after subsubject_structured build ask user for files/links to use for references
subsubject_creation_graph.add_node("content_prepare_build", content_prepare_build)
subsubject_creation_graph.add_node("content_build", content_build)
subsubject_creation_graph.add_node("intro_build", intro_build)
subsubject_creation_graph.add_node("subsubject_build", subsubject_build)

subsubject_creation_graph.add_edge(START, "modules_build")
subsubject_creation_graph.add_edge("modules_build", "subsubject_structured_build")
subsubject_creation_graph.add_edge("subsubject_structured_build", "content_prepare_build")
subsubject_creation_graph.add_edge("content_prepare_build", "content_build")
subsubject_creation_graph.add_edge("content_build", "intro_build")
subsubject_creation_graph.add_edge("intro_build", "subsubject_build")
subsubject_creation_graph.add_edge("subsubject_build", END)

build_subsubject = subsubject_creation_graph.compile()


async def new_plan_build(
    state: SubjectCreationState, config: RunnableConfig
) -> SubjectCreationState:
    journey, subject, subsubject, journey_template, subject_template = get_journey_items(
        state, config
    )
    prompt = (
        subject.prompts.plan
        if subject is not None
        else subject_template.subject_prompts.plan
    )

    concepts = (
        state["concepts"]
        if "concepts" in state
        else subject.concepts if subject is not None else None
    )
    if concepts is None:
        raise Exception("Concepts required for building plan plan")
    content = "\n".join(
        [
            f"{concept.title}:\n{concept.content}\nReferences: {", ".join([f"{item.source} ({item.page_number})" for item in concept.references])}"
            for concept in concepts
        ]
    )

    plan: List[Subsubject] = await get_base_chain("plan")(
        (prompt.system, prompt.user)
    ).ainvoke(
        {
            "context": content,
            "amount": subject_template.subsubject_amount,
            "journey_instructions": (
                journey.instructions
                if journey is not None
                else journey_template.journey_instruction
            ),
            "subject_instructions": (
                subject.instructions
                if subject is not None
                else subject_template.subject_instruction
            ),
        }
    )

    return {
        "initial_plan": plan,
        "initial_plan_done": True,
    }


class SubsubjectBuildState(TypedDict):
    journey: JourneyDataTable
    concepts: List[ConceptData]
    subsubject: Subsubject
    subject_index: int
    subsubject_index: int


async def subsubject_build(state: SubsubjectBuildState) -> SubjectCreationState:
    subsubject: Subsubject = state["subsubject"]
    concept_ids = subsubject.concept_ids
    concepts = []
    for concecpt_id in concept_ids:
        if "concept_id" in state["concepts"]:
            concepts.append(state["concepts"][concecpt_id])

    subsubject = SubsubjectModel(
        title=subsubject.title, subject=subsubject.subject, summary=subsubject.summary, concepts=concepts
    )

    response = await build_subsubject.ainvoke(
        {
            "journey": state["journey"],
            "subsubject": state["subsubject"],
            "concepts": concepts,
        },
        {
            "configurable": {
                "journey_template": state["journey_template"],
                "subject_index": state["subject_index"],
                "subsubject_index": state["subsubject_index"],
                "amount_of_documents": state["amount_of_documents"],
            }
        },
    )

    return {"plan": [response["subsubject"]]}


async def map_plan(
    state: SubjectCreationState, config: RunnableConfig
) -> List[RunnableConfig]:
    return [
        Send(
            "subsubject_build",
            {
                "subsubject": state["initial_plan"][i],
                "journey": state["journey"],
                "concepts": state["concepts"],
                "journey_template": config["configurable"]["journey_template"],
                "subject_index": config["configurable"]["subject_index"],
                "subsubject_index": i,
                "amount_of_documents": config["configurable"]["amount_of_documents"],
            },
        )
        for i in range(len(state["initial_plan"]))
    ]


async def combine_mapped_plan(
    state: SubjectCreationState, config: RunnableConfig
) -> SubjectCreationState:
    plan = sorted(
        state["plan"],
        key=lambda x: next(
            (subsubject.title for subsubject in state["initial_plan"] if subsubject.title == x.title),
            None,
        ),
    )
    return {
        "plan_sorted": plan,
        "plan_done": True,
    }


async def summary_build(
    state: SubjectCreationState, config: RunnableConfig
) -> SubjectCreationState:
    journey, subject, subsubject, journey_template, subject_template = get_journey_items(
        state, config
    )
    plan = state["plan_sorted"]

    summary = await llm_edit(
        [
            textwrap.dedent(
                f"""
                Title: {subsubject.title}
                Subject:
                {subsubject.subject.replace("\n", " ")}
                Summary:
                {subsubject.summary.replace("\n", " ")}
                Content:
                {(subsubject.content or "").replace("\n", " ")}
                """
            )
            for subsubject in state["plan"]
        ],
        "Summarize the following content into a description.",
        summarize=True,
    )

    title = await get_chain("module").ainvoke(
        {
            "context": summary,
            "module": "Summarize context with 10 words or less to a title",
        }
    )

    new_subject = SubjectModel(
        title=title,
        summary=summary,
        plan=plan,
        prompts=subject.prompts if subject else subject_template.subject_prompts,
        instructions=(
            subject.instructions if subject else subject_template.subject_instruction
        ),
        subsubject_amount=(
            subject.subsubjects_amount
            if subject
            else (
                len(subject_template.subsubject_templates)
                if subject_template.subsubject_templates
                else subject_template.subsubject_amount
            )
        ),
        module_amount=(
            subject.module_amount if subject else subject_template.default_module_amount
        ),
    )

    return {"subject": new_subject, "subject_done": True}


subject_creation_graph = StateGraph(SubjectCreationState, SujectCreationConfig)
subject_creation_graph.add_node("new_plan_build", new_plan_build)
subject_creation_graph.add_node("subsubject_build", subsubject_build)
subject_creation_graph.add_node("combine_mapped_plan", combine_mapped_plan)
subject_creation_graph.add_node("summary_build", summary_build)

subject_creation_graph.add_edge(START, "new_plan_build")
subject_creation_graph.add_conditional_edges(
    "new_plan_build", map_plan, ["subsubject_build"]
)
subject_creation_graph.add_edge("subsubject_build", "combine_mapped_plan")
subject_creation_graph.add_edge("combine_mapped_plan", "summary_build")
subject_creation_graph.add_edge("summary_build", END)

async def journey_build(state: JourneyCreationState, config: RunnableConfig) -> JourneyCreationState:
    journey, subject, subsubject, journey_template, subject_template = get_journey_items(state, config)

    if journey is None:
        journey = JourneyDataTable(
            journey_name=state["journey_name"],
            chroma_collections=["rag_" + categories for categories in state["categories"]] if state["chroma_collections" is None] else state["chroma_collections"],
            journey_template_id=journey_template.id if journey_template else None,
        )

    sources = state["sources"] if "sources" in state else None
    concepts = state["concepts"] if "concepts" in state else None

    if concepts is None and sources is not None:
        concepts = []
        for source in sources:
            concepts.extend(source.source_concepts)
    else:
        raise ValueError("Concepts must be provided if sources are not provided.")

