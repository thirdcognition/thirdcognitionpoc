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
    JourneyItem,
    JourneyDataTable,
    ModuleStructure,
)
from lib.models.concepts import ConceptData
from lib.models.source import SourceDataTable
from lib.prompts.journey import JourneyPrompts, Module
from lib.chains.init import get_base_chain, get_chain
from lib.streamlit_tools import llm_edit
from lib.models.reference import Reference


class ModuleTemplate(BaseModel):
    id: str
    instructions: str
    action_amount: int


class SectionTemplate(BaseModel):
    id: str
    section_instruction: str
    section_prompt_content_instructions: str
    section_prompts: JourneyPrompts
    module_templates: List[ModuleTemplate]
    module_amount: 3
    default_action_amount: int = 5


class JourneyTemplate(BaseModel):
    id: str
    journey_instruction: str
    section_templates: List[SectionTemplate]
    section_amount: int = 2


class JourneyCreationConfig(TypedDict):
    journey_template: JourneyTemplate
    amount_of_documents: int = 5


class JourneyCreationState(TypedDict):
    journey_name: str
    chroma_collections: List[str]
    categories: List[str]
    journey: JourneyDataTable
    references: List[Reference]
    sections: Annotated[list[JourneyItem], operator.add]
    sections_sorted: List[JourneyItem]
    sections_done: bool = False
    journey_done: bool = False


class SujectCreationConfig(TypedDict):
    journey_template: JourneyTemplate
    section_index: int
    amount_of_documents: int = 5


class SectionCreationState(TypedDict):
    journey: JourneyDataTable
    section: JourneyItem
    references: List[Reference]
    initial_plan: List[Module]
    initial_plan_done: bool = False
    plan: Annotated[list[JourneyItem], operator.add]
    plan_sorted: List[JourneyItem]
    plan_done: bool = False
    section_done: bool = False


class ModuleCreationConfig(TypedDict):
    journey_template: JourneyTemplate
    module_index: int
    section_index: int
    amount_of_documents: int = 5


class ModuleCreationState(TypedDict):
    journey: JourneyDataTable
    # concepts: List[ConceptData]
    module: JourneyItem
    concepts_content: str
    section_title: str
    actions: str
    actions_done: bool = False
    module_structured: ModuleStructure
    module_structured_done: bool = False
    content_prepare: str
    content_prepare_done: bool = False
    content: str
    content_done: bool = False
    intro: str
    intro_done: bool = False
    module_done: bool = False


def get_journey_items(
    state: Union[JourneyCreationState, SectionCreationState, ModuleCreationState], config
) -> tuple[
    JourneyDataTable,
    JourneyItem,
    JourneyItem,
    JourneyTemplate,
    SectionTemplate,
    ModuleTemplate,
]:
    journey: JourneyDataTable = state["journey"] if "journey" in state else None
    section: JourneyItem = (
        journey.children[config["configurable"]["subject_index"]]
        if journey is not None
        else None
    )
    module: JourneyItem = (
        state["module"]
        if "module" in state
        else (
            section.children[config["configurable"]["module_index"]]
            if section is not None
            else None
        )
    )
    journey_template: JourneyTemplate = config["configurable"]["journey_template"] if "configurable" in config else None
    if "section_index" in config["configurable"]:
        section_template: SectionTemplate = journey_template.section_templates[
            config["configurable"]["section_index"]
        ]
    else:
        section_template: SectionTemplate = None
    if "module_index" in config["configurable"] and section_template is not None:
        module_template: ModuleTemplate = section_template.module_templates[
            config["configurable"]["module_index"]
        ]
    else:
        module_template: ModuleTemplate = None

    return journey, section, module, journey_template, section_template, module_template


async def actions_build(
    state: ModuleCreationState, config: RunnableConfig
) -> ModuleCreationState:
    journey, section, module, journey_template, section_template, module_template = (
        get_journey_items(state, config)
    )
    prompt = (
        section.prompts.module_actions
        if section is not None
        else section_template.section_prompts.module_actions
    )

    action_amount = (
        module_template.action_amount
        if module_template.action_amount is not None
        else section_template.default_action_amount
    )

    concepts = state["concepts"] if "concepts" in state else module.references
    class_content = "\n".join(
        [
            f"{concept.title}:\n{concept.content}\nReferences: {", ".join([f"{item.source} ({item.page_number})" for item in concept.references])}"
            for concept in concepts
        ]
    )
    section_title = f"Title: {module.title}\nSection: {module.section}"
    class_actions = await get_base_chain("module_actions")(
        (prompt.system, prompt.user)
    ).ainvoke(
        {
            "context": class_content,
            "journey_content_instructions": journey_template.journey_instruction,
            "section_content_instructions": section_template.section_instruction,
            "subject": section_title,
            "amount": action_amount,
            # "chat_history": previous_class_sections + previous_class_actions
        }
    )

    return {
        "section_title": section_title,
        "concepts_content": class_content,
        "actions": get_text_from_completion(class_actions),
        "actions_done": True,
    }


async def module_structured_build(
    state: ModuleCreationState, config: RunnableConfig
) -> ModuleCreationState:
    journey, section, module, journey_template, section_template, module_template = (
        get_journey_items(state, config)
    )

    instructions = (
        (
            journey.instructions
            if journey is not None
            else (
                journey_template.journey_instruction + "\n" + section.instructions
                if section is not None
                else section_template.section_instruction
            )
        ).strip()
        + "\n"
        + (
            module.instructions
            if module.instructions is not None
            else module_template.instructions
        ).strip()
    )

    structured = await get_chain("module_structured").ainvoke(
        {
            "context": f"""
            Title:
            {module.title}
            Summary:
            {module.summary}
            Content:
            {state["concepts_content"]}
            Actions:
            {state["actions"]}
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
        "module_structured": structured,
        "module_structured_done": True,
    }


async def process_action_to_content(
    action: JourneyItem, doc_chain: RunnableSequence, context: str
):
    content = (
        "\n\n"
        + f"Section: {action.title.strip()}\n\nSection description: {action.description.strip()}"
    )
    content += (
        "\n\n"
        + "\n\nSection content:"
        + await doc_chain.ainvoke(
            {
                "question": f"Section: {action.title.strip()}\n\nSection description: {action.description.strip()}",
                "context": context,
            }
        )["answer"]
    )
    return content


async def content_prepare_build(
    state: ModuleCreationState, config: RunnableConfig
) -> ModuleCreationState:
    journey = state["journey"] if "journey" in state else None
    structured = state["module_structured"]

    doc_chain = get_rag_chain(
        journey.chroma_collections,
        "hyde_document",
        amount_of_documents=config["configurable"]["amount_of_documents"],
    )
    content = state["subject_title"] + "\n\n" + state["concepts_content"].strip()

    if structured is not None and isinstance(structured, ModuleStructure):
        tasks = [
            process_action_to_content(action, doc_chain, state)
            for action in structured.children
        ]
        contents = await asyncio.gather(*tasks)
        return "\n\n".join(contents)
    else:
        content += "\n\n" + state["actions"]

    return {
        "content_prepare": get_text_from_completion(content),
        "content_prepare_done": True,
    }


async def content_build(
    state: ModuleCreationState, config: RunnableConfig
) -> ModuleCreationState:
    journey, section, module, journey_template, section_template, module_template = (
        get_journey_items(state, config)
    )
    prompt = (
        section.prompts.module_content
        if section is not None
        else section_template.section_prompts.module_content
    )

    prep_content = state["content_prepare"]

    content = await get_base_chain("module_content")(
        (prompt.system, prompt.user)
    ).ainvoke(
        {
            "context": prep_content,
            "journey_content_instructions": journey_template.journey_instruction,
            "section_content_instructions": section_template.section_instruction,
            "module_content_instructions": (
                module.instructions
                if section is not None
                else module_template.instructions
            ),
            "subject": state["subject_title"],
        }
    )

    return {
        "content": get_text_from_completion(content),
        "content_done": True,
    }


async def intro_build(
    state: ModuleCreationState, config: RunnableConfig
) -> ModuleCreationState:
    journey, section, module, journey_template, section_template, module_template = (
        get_journey_items(state, config)
    )
    prompt = (
        section.prompts.module_intro
        if section is not None
        else section_template.section_prompts.module_intro
    )

    class_intro = await get_base_chain("module_intro")(
        (prompt.system, prompt.user)
    ).ainvoke(
        {
            "context": state["content"],
            "journey_content_instructions": (
                journey.instructions
                if journey is not None
                else journey_template.journey_instruction
            ),
            "section_content_instructions": (
                section.instructions
                if section is not None
                else section_template.section_instruction
            ),
            "module_content_instructions": (
                module.instructions
                if section is not None
                else module_template.instructions
            ),
            "subject": state["subject_title"],
            # "chat_history": previous_class_intros
        }
    )

    return {
        "intro": get_text_from_completion(class_intro),
        "intro_done": True,
    }


async def module_build(
    state: ModuleCreationState, config: RunnableConfig
) -> ModuleCreationState:
    journey, section, module, journey_template, section_template = get_journey_items(
        state, config
    )

    new_module = JourneyItem(
        title=module.title,
        section=module.section,
        summary=module.summary,
        references=module.references,
        content=state["content"].strip(),
        intro=state["intro"].strip(),  # class_intro.strip(),
        actions=state["actions"].strip(),
        structured=state["module_structured"],
    )

    return {
        "module": new_module,
        "module_done": True,
    }


module_creation_graph = StateGraph(ModuleCreationState, JourneyCreationConfig)
module_creation_graph.add_node("actions_build", actions_build)
module_creation_graph.add_node("module_structured_build", module_structured_build)
# TODO after module_structured build ask user for files/links to use for references
module_creation_graph.add_node("content_prepare_build", content_prepare_build)
module_creation_graph.add_node("content_build", content_build)
module_creation_graph.add_node("intro_build", intro_build)
module_creation_graph.add_node("module_build", module_build)

module_creation_graph.add_edge(START, "actions_build")
module_creation_graph.add_edge("actions_build", "module_structured_build")
module_creation_graph.add_edge("module_structured_build", "content_prepare_build")
module_creation_graph.add_edge("content_prepare_build", "content_build")
module_creation_graph.add_edge("content_build", "intro_build")
module_creation_graph.add_edge("intro_build", "module_build")
module_creation_graph.add_edge("module_build", END)

build_module = module_creation_graph.compile()


async def new_plan_build(
    state: SectionCreationState, config: RunnableConfig
) -> SectionCreationState:
    journey, section, module, journey_template, section_template = get_journey_items(
        state, config
    )
    prompt = (
        section.prompts.plan
        if section is not None
        else section_template.section_prompts.plan
    )

    concepts = (
        state["concepts"]
        if "concepts" in state
        else section.concepts if section is not None else None
    )
    if concepts is None:
        raise Exception("Concepts required for building plan plan")
    content = "\n".join(
        [
            f"{concept.title}:\n{concept.content}\nReferences: {", ".join([f"{item.source} ({item.page_number})" for item in concept.references])}"
            for concept in concepts
        ]
    )

    plan: List[Module] = await get_base_chain("plan")(
        (prompt.system, prompt.user)
    ).ainvoke(
        {
            "context": content,
            "amount": section_template.module_amount,
            "journey_content_instructions": (
                journey.instructions
                if journey is not None
                else journey_template.journey_instruction
            ),
            "section_content_instructions": (
                section.instructions
                if section is not None
                else section_template.section_instruction
            ),
        }
    )

    return {
        "initial_plan": plan,
        "initial_plan_done": True,
    }


class ModuleBuildState(TypedDict):
    journey: JourneyDataTable
    concepts: List[ConceptData]
    module: Module
    subject_index: int
    module_index: int


async def module_build(state: ModuleBuildState) -> SectionCreationState:
    module: Module = state["module"]
    concept_ids = module.concept_ids
    concepts = []
    for concecpt_id in concept_ids:
        if "concept_id" in state["concepts"]:
            concepts.append(state["concepts"][concecpt_id])

    module = JourneyItem(
        title=module.title, subject=module.subject, summary=module.summary, concepts=concepts
    )

    response = await build_module.ainvoke(
        {
            "journey": state["journey"],
            "module": state["module"],
            "concepts": concepts,
        },
        {
            "configurable": {
                "journey_template": state["journey_template"],
                "subject_index": state["subject_index"],
                "module_index": state["module_index"],
                "amount_of_documents": state["amount_of_documents"],
            }
        },
    )

    return {"plan": [response["module"]]}


async def map_plan(
    state: SectionCreationState, config: RunnableConfig
) -> List[RunnableConfig]:
    return [
        Send(
            "module_build",
            {
                "module": state["initial_plan"][i],
                "journey": state["journey"],
                "concepts": state["concepts"],
                "journey_template": config["configurable"]["journey_template"],
                "subject_index": config["configurable"]["subject_index"],
                "module_index": i,
                "amount_of_documents": config["configurable"]["amount_of_documents"],
            },
        )
        for i in range(len(state["initial_plan"]))
    ]


async def combine_mapped_plan(
    state: SectionCreationState, config: RunnableConfig
) -> SectionCreationState:
    plan = sorted(
        state["plan"],
        key=lambda x: next(
            (module.title for module in state["initial_plan"] if module.title == x.title),
            None,
        ),
    )
    return {
        "plan_sorted": plan,
        "plan_done": True,
    }


async def summary_build(
    state: SectionCreationState, config: RunnableConfig
) -> SectionCreationState:
    journey, section, module, journey_template, section_template = get_journey_items(
        state, config
    )
    plan = state["plan_sorted"]

    summary = await llm_edit(
        [
            textwrap.dedent(
                f"""
                Title: {module.title}
                Section:
                {module.subject.replace("\n", " ")}
                Summary:
                {module.summary.replace("\n", " ")}
                Content:
                {(module.content or "").replace("\n", " ")}
                """
            )
            for module in state["plan"]
        ],
        "Summarize the following content into a description.",
        summarize=True,
    )

    title = await get_chain("action").ainvoke(
        {
            "context": summary,
            "action": "Summarize context with 10 words or less to a title",
        }
    )

    new_section = JourneyItem(
        title=title,
        summary=summary,
        children=plan,
        prompts=section.prompts if section else section_template.section_prompts,
        instructions=(
            section.instructions if section else section_template.section_instruction
        ),
        module_amount=(
            section.modules_amount
            if section
            else (
                len(section_template.module_templates)
                if section_template.module_templates
                else section_template.module_amount
            )
        ),
        action_amount=(
            section.action_amount if section else section_template.default_action_amount
        ),
    )

    return {"section": new_section, "section_done": True}


section_creation_graph = StateGraph(SectionCreationState, SujectCreationConfig)
section_creation_graph.add_node("new_plan_build", new_plan_build)
section_creation_graph.add_node("module_build", module_build)
section_creation_graph.add_node("combine_mapped_plan", combine_mapped_plan)
section_creation_graph.add_node("summary_build", summary_build)

section_creation_graph.add_edge(START, "new_plan_build")
section_creation_graph.add_conditional_edges(
    "new_plan_build", map_plan, ["module_build"]
)
section_creation_graph.add_edge("module_build", "combine_mapped_plan")
section_creation_graph.add_edge("combine_mapped_plan", "summary_build")
section_creation_graph.add_edge("summary_build", END)

async def journey_build(state: JourneyCreationState, config: RunnableConfig) -> JourneyCreationState:
    journey, section, module, journey_template, section_template = get_journey_items(state, config)

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

