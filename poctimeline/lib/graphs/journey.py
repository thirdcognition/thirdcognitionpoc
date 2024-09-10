import asyncio
import operator
import textwrap
from typing import Annotated, List, Union
from pydantic import BaseModel
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig, RunnableSequence
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from lib.chains.base_parser import get_text_from_completion
from lib.chains.rag_chain import get_rag_chain
from lib.models.journey import (
    TaskStructure,
    JourneyModel,
    StepModel,
    SubjectModel,
    StepStructure,
)
from lib.models.sqlite_tables import SourceConcept, SourceData
from lib.prompts.journey import JourneyPrompts
from lib.prompts.journey_structured import Step
from lib.chains.init import get_base_chain, get_chain
from lib.streamlit_tools import llm_edit


class StepTemplate(BaseModel):
    id: str
    instructions: str
    task_amount: int


class SubjectTemplate(BaseModel):
    id: str
    subject_instruction: str
    subject_prompt_instructions: str
    subject_prompts: JourneyPrompts
    step_templates: List[StepTemplate]
    step_amount: 3
    default_task_amount: int = 5


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
    journey: JourneyModel
    concepts: List[SourceConcept]
    sources: List[SourceData]
    subjects: Annotated[list[SubjectModel], operator.add]
    subjects_sorted: List[SubjectModel]
    subjects_done: bool = False
    journey_done: bool = False


class SujectCreationConfig(TypedDict):
    journey_template: JourneyTemplate
    subject_index: int
    amount_of_documents: int = 5


class SubjectCreationState(TypedDict):
    journey: JourneyModel
    subject: SubjectModel
    concepts: List[SourceConcept]
    initial_plan: List[Step]
    initial_plan_done: bool = False
    plan: Annotated[list[StepModel], operator.add]
    plan_sorted: List[StepModel]
    plan_done: bool = False
    subject_done: bool = False


class StepCreationConfig(TypedDict):
    journey_template: JourneyTemplate
    step_index: int
    subject_index: int
    amount_of_documents: int = 5


class StepCreationState(TypedDict):
    journey: JourneyModel
    # concepts: List[SourceConcept]
    step: StepModel
    concepts_content: str
    subject_title: str
    tasks: str
    tasks_done: bool = False
    step_structured: StepStructure
    step_structured_done: bool = False
    content_prepare: str
    content_prepare_done: bool = False
    content: str
    content_done: bool = False
    intro: str
    intro_done: bool = False
    step_done: bool = False


def get_journey_items(
    state: Union[JourneyCreationState, SubjectCreationState, StepCreationState], config
) -> tuple[
    JourneyModel,
    SubjectModel,
    StepModel,
    JourneyTemplate,
    SubjectTemplate,
    StepTemplate,
]:
    journey: JourneyModel = state["journey"] if "journey" in state else None
    subject: SubjectModel = (
        journey.subjects[config["configurable"]["subject_index"]]
        if journey is not None
        else None
    )
    step: StepModel = (
        state["step"]
        if "step" in state
        else (
            subject.plan[config["configurable"]["step_index"]]
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
    if "step_index" in config["configurable"] and subject_template is not None:
        step_template: StepTemplate = subject_template.step_templates[
            config["configurable"]["step_index"]
        ]
    else:
        step_template: StepTemplate = None

    return journey, subject, step, journey_template, subject_template, step_template


async def tasks_build(
    state: StepCreationState, config: RunnableConfig
) -> StepCreationState:
    journey, subject, step, journey_template, subject_template, step_template = (
        get_journey_items(state, config)
    )
    prompt = (
        subject.prompts.step_tasks
        if subject is not None
        else subject_template.subject_prompts.step_tasks
    )

    task_amount = (
        step_template.task_amount
        if step_template.task_amount is not None
        else subject_template.default_task_amount
    )

    concepts = state["concepts"] if "concepts" in state else step.concepts
    class_content = "\n".join(
        [
            f"{concept.title}:\n{concept.content}\nReference: {concept.reference}"
            for concept in concepts
        ]
    )
    subject_title = f"Title: {step.title}\nSubject: {step.subject}"
    class_tasks = await get_base_chain("step_tasks")(
        (prompt.system, prompt.user)
    ).ainvoke(
        {
            "context": class_content,
            "journey_instructions": journey_template.journey_instruction,
            "subject_instructions": subject_template.subject_instruction,
            "subject": subject_title,
            "amount": task_amount,
            # "chat_history": previous_class_subjects + previous_class_tasks
        }
    )

    return {
        "subject_title": subject_title,
        "concepts_content": class_content,
        "tasks": get_text_from_completion(class_tasks),
        "tasks_done": True,
    }


async def step_structured_build(
    state: StepCreationState, config: RunnableConfig
) -> StepCreationState:
    journey, subject, step, journey_template, subject_template, step_template = (
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
            step.instructions
            if step.instructions is not None
            else step_template.instructions
        ).strip()
    )

    structured = await get_chain("step_structured").ainvoke(
        {
            "context": f"""
            Title:
            {step.title}
            Summary:
            {step.summary}
            Content:
            {state["concepts_content"]}
            Tasks:
            {state["tasks"]}
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
        "step_structured": structured,
        "step_structured_done": True,
    }


async def process_task_to_content(
    task: TaskStructure, doc_chain: RunnableSequence, context: str
):
    content = (
        "\n\n"
        + f"Subject: {task.title.strip()}\n\nSubject description: {task.description.strip()}"
    )
    content += (
        "\n\n"
        + "\n\nSubject content:"
        + await doc_chain.ainvoke(
            {
                "question": f"Subject: {task.title.strip()}\n\nSubject description: {task.description.strip()}",
                "context": context,
            }
        )["answer"]
    )
    return content


async def content_prepare_build(
    state: StepCreationState, config: RunnableConfig
) -> StepCreationState:
    journey = state["journey"] if "journey" in state else None
    structured = state["step_structured"]

    doc_chain = get_rag_chain(
        journey.chroma_collection,
        "hyde_document",
        amount_of_documents=config["configurable"]["amount_of_documents"],
    )
    content = state["subject_title"] + "\n\n" + state["concepts_content"].strip()

    if structured is not None and isinstance(structured, StepStructure):
        tasks = [
            process_task_to_content(task, doc_chain, state)
            for task in structured.tasks
        ]
        contents = await asyncio.gather(*tasks)
        return "\n\n".join(contents)
    else:
        content += "\n\n" + state["tasks"]

    return {
        "content_prepare": get_text_from_completion(content),
        "content_prepare_done": True,
    }


async def content_build(
    state: StepCreationState, config: RunnableConfig
) -> StepCreationState:
    journey, subject, step, journey_template, subject_template, step_template = (
        get_journey_items(state, config)
    )
    prompt = (
        subject.prompts.step_content
        if subject is not None
        else subject_template.subject_prompts.step_content
    )

    prep_content = state["content_prepare"]

    content = await get_base_chain("step_content")(
        (prompt.system, prompt.user)
    ).ainvoke(
        {
            "context": prep_content,
            "journey_instructions": journey_template.journey_instruction,
            "subject_instructions": subject_template.subject_instruction,
            "step_instructions": (
                step.instructions
                if subject is not None
                else step_template.instructions
            ),
            "subject": state["subject_title"],
        }
    )

    return {
        "content": get_text_from_completion(content),
        "content_done": True,
    }


async def intro_build(
    state: StepCreationState, config: RunnableConfig
) -> StepCreationState:
    journey, subject, step, journey_template, subject_template, step_template = (
        get_journey_items(state, config)
    )
    prompt = (
        subject.prompts.step_intro
        if subject is not None
        else subject_template.subject_prompts.step_intro
    )

    class_intro = await get_base_chain("step_intro")(
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
            "step_instructions": (
                step.instructions
                if subject is not None
                else step_template.instructions
            ),
            "subject": state["subject_title"],
            # "chat_history": previous_class_intros
        }
    )

    return {
        "intro": get_text_from_completion(class_intro),
        "intro_done": True,
    }


async def step_build(
    state: StepCreationState, config: RunnableConfig
) -> StepCreationState:
    journey, subject, step, journey_template, subject_template = get_journey_items(
        state, config
    )

    new_step = StepModel(
        title=step.title,
        subject=step.subject,
        summary=step.summary,
        concepts=step.concepts,
        content=state["content"].strip(),
        intro=state["intro"].strip(),  # class_intro.strip(),
        tasks=state["tasks"].strip(),
        structured=state["step_structured"],
    )

    return {
        "step": new_step,
        "step_done": True,
    }


step_creation_graph = StateGraph(StepCreationState, JourneyCreationConfig)
step_creation_graph.add_node("tasks_build", tasks_build)
step_creation_graph.add_node("step_structured_build", step_structured_build)
# TODO after step_structured build ask user for files/links to use for references
step_creation_graph.add_node("content_prepare_build", content_prepare_build)
step_creation_graph.add_node("content_build", content_build)
step_creation_graph.add_node("intro_build", intro_build)
step_creation_graph.add_node("step_build", step_build)

step_creation_graph.add_edge(START, "tasks_build")
step_creation_graph.add_edge("tasks_build", "step_structured_build")
step_creation_graph.add_edge("step_structured_build", "content_prepare_build")
step_creation_graph.add_edge("content_prepare_build", "content_build")
step_creation_graph.add_edge("content_build", "intro_build")
step_creation_graph.add_edge("intro_build", "step_build")
step_creation_graph.add_edge("step_build", END)

build_step = step_creation_graph.compile()


async def new_plan_build(
    state: SubjectCreationState, config: RunnableConfig
) -> SubjectCreationState:
    journey, subject, step, journey_template, subject_template = get_journey_items(
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
            f"{concept.title}:\n{concept.content}\nReference: {concept.reference}"
            for concept in concepts
        ]
    )

    plan: List[Step] = await get_base_chain("plan")(
        (prompt.system, prompt.user)
    ).ainvoke(
        {
            "context": content,
            "amount": subject_template.step_amount,
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


class StepBuildState(TypedDict):
    journey: JourneyModel
    concepts: List[SourceConcept]
    step: Step
    subject_index: int
    step_index: int


async def step_build(state: StepBuildState) -> SubjectCreationState:
    step: Step = state["step"]
    concept_ids = step.concept_ids
    concepts = []
    for concecpt_id in concept_ids:
        if "concept_id" in state["concepts"]:
            concepts.append(state["concepts"][concecpt_id])

    step = StepModel(
        title=step.title, subject=step.subject, summary=step.summary, concepts=concepts
    )

    response = await build_step.ainvoke(
        {
            "journey": state["journey"],
            "step": state["step"],
            "concepts": concepts,
        },
        {
            "configurable": {
                "journey_template": state["journey_template"],
                "subject_index": state["subject_index"],
                "step_index": state["step_index"],
                "amount_of_documents": state["amount_of_documents"],
            }
        },
    )

    return {"plan": [response["step"]]}


async def map_plan(
    state: SubjectCreationState, config: RunnableConfig
) -> List[RunnableConfig]:
    return [
        Send(
            "step_build",
            {
                "step": state["initial_plan"][i],
                "journey": state["journey"],
                "concepts": state["concepts"],
                "journey_template": config["configurable"]["journey_template"],
                "subject_index": config["configurable"]["subject_index"],
                "step_index": i,
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
            (step.title for step in state["initial_plan"] if step.title == x.title),
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
    journey, subject, step, journey_template, subject_template = get_journey_items(
        state, config
    )
    plan = state["plan_sorted"]

    summary = await llm_edit(
        [
            textwrap.dedent(
                f"""
                Title: {step.title}
                Subject:
                {step.subject.replace("\n", " ")}
                Summary:
                {step.summary.replace("\n", " ")}
                Content:
                {(step.content or "").replace("\n", " ")}
                """
            )
            for step in state["plan"]
        ],
        "Summarize the following content into a description.",
        summarize=True,
    )

    title = await get_chain("task").ainvoke(
        {
            "context": summary,
            "task": "Summarize context with 10 words or less to a title",
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
        step_amount=(
            subject.step_amount
            if subject
            else (
                len(subject_template.step_templates)
                if subject_template.step_templates
                else subject_template.step_amount
            )
        ),
        task_amount=(
            subject.task_amount if subject else subject_template.default_task_amount
        ),
    )

    return {"subject": new_subject, "subject_done": True}


subject_creation_graph = StateGraph(SubjectCreationState, SujectCreationConfig)
subject_creation_graph.add_node("new_plan_build", new_plan_build)
subject_creation_graph.add_node("step_build", step_build)
subject_creation_graph.add_node("combine_mapped_plan", combine_mapped_plan)
subject_creation_graph.add_node("summary_build", summary_build)

subject_creation_graph.add_edge(START, "new_plan_build")
subject_creation_graph.add_conditional_edges(
    "new_plan_build", map_plan, ["step_build"]
)
subject_creation_graph.add_edge("step_build", "combine_mapped_plan")
subject_creation_graph.add_edge("combine_mapped_plan", "summary_build")
subject_creation_graph.add_edge("summary_build", END)

async def journey_build(state: JourneyCreationState, config: RunnableConfig) -> JourneyCreationState:
    journey, subject, step, journey_template, subject_template = get_journey_items(state, config)

    if journey is None:
        journey = JourneyModel(
            journeyname=state["journey_name"],
            chroma_collection=["rag_" + category for category in state["categories"]] if state["chroma_collections" is None] else state["chroma_collections"],
            journey_template_id=journey_template.id if journey_template else None,
        )

    sources = state["sources"] if "sources" in state else None
    concepts = state["concepts"] if "concepts" in state else None

    if concepts is None and sources is not None:
        concepts = []
        for source in sources:
            concepts.extend(source.source_contents.concepts)
    else:
        raise ValueError("Concepts must be provided if sources are not provided.")

