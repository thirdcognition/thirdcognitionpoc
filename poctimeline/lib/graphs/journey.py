import asyncio
from typing import List
from pydantic import BaseModel
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig, RunnableSequence
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from lib.chains.rag_chain import get_rag_chain
from lib.models.journey import ActionStructure, JourneyModel, StepModel, SubjectModel, SubjectStructure
from lib.models.sqlite_tables import SourceConcept
from lib.prompts.journey import JourneyPrompts
from lib.chains.init import get_base_chain, get_chain

class SubjectTemplate(BaseModel):
    subject_instruction: str
    subject_prompt_instructions: str
    subject_prompts: JourneyPrompts
    step_amount: int = 5
    action_amounts: List[int]
    default_action_amount: int = 5

class JourneyTemplate(BaseModel):
    journey_instruction: str
    subject_templates: List[SubjectTemplate]
    subject_amount: int = 2

class JourneyCreationConfig(TypedDict):
    journey_template: JourneyTemplate

class JourneyCreationState(TypedDict):
    journey: JourneyModel

class SujectCreationConfig(TypedDict):
    journey_template: JourneyTemplate
    subject_index: int

class SubjectCreationState(TypedDict):
    journey: JourneyModel
    subject: SubjectModel

class StepCreationConfig(TypedDict):
    journey_template: JourneyTemplate
    step_index: int
    subject_index: int
    amount_of_documents: int = 5

class StepCreationState(TypedDict):
    journey: JourneyModel
    concepts: List[SourceConcept]
    concepts_content: str
    step: StepModel
    actions: str
    actions_done: bool = False
    actions_structured: SubjectStructure
    actions_structured_done: bool = False
    content_prepare:str
    content_prepare_done: bool = False
    content: str
    content_done: bool = False
    intro_content: str
    intro_done: bool = False
    step_done: bool = False


async def actions_build(state: StepCreationState, config: RunnableConfig) -> StepCreationState:
    journey = state["journey"]
    subject:SubjectModel = journey.subjects[config["configurable"]["subject_index"]]
    step:StepModel = subject.steps[config["configurable"]["step_index"]]
    journey_template:JourneyTemplate = config["configurable"]["journey_template"]
    subject_template:SubjectTemplate = journey_template.subject_templates[config["configurable"]["subject_index"]]
    prompt = subject_template.subject_prompts.step_actions

    action_amount = subject_template.action_amounts[config["step_index"]] if config["step_index"] < len(subject_template.action_amounts) else subject_template.default_action_amount

    concepts = state["concepts"]
    class_content = "\n".join([f"{concept.title}:\n{concept.content}\nReference: {concept.reference}" for concept in concepts])
    subject_string = f"Title: {step.title}\nSubject: {step.subject}"
    class_actions = await get_base_chain("journey_step_actions")((prompt.system, prompt.user)).ainvoke(
        {
            "context": class_content,
            "journey_instructions": journey_template.journey_instruction,
            "instructions": subject_template.subject_instruction,
            "subject": subject_string,
            "amount": action_amount,
            # "chat_history": previous_class_subjects + previous_class_actions
        }
    )

    return {
        "concepts_content": class_content,
        "actions" : class_actions,
        "actions_done": True,
    }

async def actions_structured_build(state: StepCreationState, config: RunnableConfig) -> StepCreationState:
    journey = state["journey"]
    subject:SubjectModel = journey.subjects[config["configurable"]["subject_index"]]
    step:StepModel = subject.steps[config["configurable"]["step_index"]]
    journey_template:JourneyTemplate = config["configurable"]["journey_template"]
    subject_template:SubjectTemplate = journey_template.subject_templates[config["configurable"]["subject_index"]]

    instructions = (journey_template.journey_instruction + "\n" + subject_template.subject_instruction).strip()

    structured = await get_chain("journey_structured").ainvoke({
        "context": f"""
            Title:
            {step.title}
            Content:
            {state["concepts_content"]}
            Actions:
            {state["actions"]}
        """ + ("""
            Instructions:
            {instructions}
        """ if instructions else ""),
    })

    return {
        "actions_structured": structured,
        "actions_structured_done": True,
    }

async def process_action_to_content(action:ActionStructure, doc_chain:RunnableSequence, context:str):
    content = "\n\n" + f"Section: {action.title.strip()}\n\nSection description: {action.description.strip()}"
    content += "\n\n" + "\n\nSection content:" + await doc_chain.ainvoke({"question": f"Section: {action.title.strip()}\n\nSection description: {action.description.strip()}", "context": context})["answer"]
    return content

async def content_prepare_build(state: StepCreationState, config: RunnableConfig) -> StepCreationState:
    journey = state["journey"]
    subject:SubjectModel = journey.subjects[config["configurable"]["subject_index"]]
    step:StepModel = subject.steps[config["configurable"]["step_index"]]
    journey_template:JourneyTemplate = config["configurable"]["journey_template"]
    subject_template:SubjectTemplate = journey_template.subject_templates[config["configurable"]["subject_index"]]
    prompt = subject_template.subject_prompts.step_content_redo
    structured = state["actions_structured"]

    doc_chain = get_rag_chain(journey.chroma_collection, "hyde_document", amount_of_documents=config["configurable"]["amount_of_documents"])
    subject_string = f"Title: {step.title}\nSubject: {step.subject}"
    content = f"{subject_string.strip()}\n\n{state["concepts_content"].strip()}"

    if structured is not None and isinstance(structured, SubjectStructure):
        tasks = [process_action_to_content(action, doc_chain, state) for action in structured.actions]
        contents = await asyncio.gather(*tasks)
        return "\n\n".join(contents)
    else:
        content += "\n\n" + step.actions.strip()

    return {
        "content_prepare": content,
        "content_prepare_done": True,
    }


async def content_build(state: StepCreationState, config: RunnableConfig) -> StepCreationState:
    journey = state["journey"]
    subject:SubjectModel = journey.subjects[config["configurable"]["subject_index"]]
    step:StepModel = subject.steps[config["configurable"]["step_index"]]
    journey_template:JourneyTemplate = config["configurable"]["journey_template"]
    subject_template:SubjectTemplate = journey_template.subject_templates[config["configurable"]["subject_index"]]
    prompt = subject_template.subject_prompts.step_content_redo

    subject_string = f"Title: {step.title}\nSubject: {step.subject}"
    prep_content = state["content_prepare"]

    content = await get_base_chain("journey_step_content_redo")((prompt.system, prompt.user)).ainvoke(
        {
                "context": prep_content,
                "journey_instructions": journey_template.journey_instruction,
                "instructions": subject_template.subject_instruction,
                "subject": subject_string
        }
    )

    return {
        "content": content,
        "content_done": True,
    }

step_creation_graph = StateGraph(StepCreationState, JourneyCreationConfig)
step_creation_graph.add_node("actions_build", actions_build)
step_creation_graph.add_node("actions_structured_build", actions_structured_build)
# TODO after actions_structured build ask user for files/links to use for references
step_creation_graph.add_node("content_prepare_build", content_prepare_build)
step_creation_graph.add_node("content_build", content_build)
step_creation_graph.add_node("intro_build", intro_build)
step_creation_graph.add_node("step_build", step_build)