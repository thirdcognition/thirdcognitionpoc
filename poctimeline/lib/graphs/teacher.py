import operator
import textwrap
from typing import Annotated, Dict

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from lib.chains.chain import Chain
from lib.chat import DELIMITER, init_journey_chat
from lib.db_tools import (
    JourneyModel,
    get_db_journey,
    get_vectorstore,
)
from lib.chains.init import (
    get_chain,
    get_llm,
)

from typing import Annotated, List, TypedDict

from lib.models.journey import TaskStructure, SubjectModel, SubjectStructure
from lib.models.teaching import TeachingTask, TeachingItemPlan, UserData
from lib.prompts.journey import JourneyPrompts, plan

journey_teaching_plan_parser = PydanticOutputParser(pydantic_object=TeachingItemPlan)

_planner_chain: dict[str, Chain] = {}
def get_planner_chain(subject: SubjectModel):
    id = f"{subject.title}:{subject.summary}"
    if id in _planner_chain:
        return _planner_chain[id]

    prompts:JourneyPrompts = subject.prompts
    prompt = None
    if prompts and prompts.plan:
        prompt = prompts.plan
    else:
        prompt = plan

    _planner_chain[id] = Chain(
        llm=get_llm("json"),
        retry_llm=get_llm("json"),
        validation_llm=get_llm("instruct_detailed"),
        prompt=prompt,
        output_parser=journey_teaching_plan_parser,
    )
    return _planner_chain[id]

def prepare_planner_input(subject_data: SubjectStructure, task_data: TaskStructure, previous_tasks: List[TeachingTask], amount=5):

    subject_description = textwrap.dedent(f"""
        Main subject: {subject_data.title}
        Detailed subject: {task_data.title}
        Detailed objective: {task_data.description}
        """)

    subject_content = textwrap.dedent(f"""
        Main content:
        {subject_data.content.replace('\n', '').strip()}

        Detailed content:{ (f"\n        - {resource.title.strip()}: {resource.summary.replace('\n', '').strip()}" for resource in task_data.resources) }
        """)

    chat_history:List[BaseMessage] = []
    for teaching_task in previous_tasks:
        chat_history += teaching_task.messages
        # chat_history += (f"\n{message.type}: {message.content.replace('\n', '').strip()}" for message in teaching_task.messages)

    return {
        "context": subject_description + "\n" + subject_content,
        "amount": amount,
        "chat_history": chat_history,
    }

class TeachingConfig(TypedDict):
    journey_name: str
    subject_name: str
    sub_subject_index: int
    step_index: int
    subject_data: SubjectStructure
    user_data: UserData

class TeachingState(TypedDict):
    id: str
    current_task_index: int

    current_task: TeachingTask
    past_tasks: Annotated[List[TeachingTask], operator.add]
    pre_class_messages: Annotated[List[BaseMessage], add_messages]
    post_class_messages: Annotated[List[BaseMessage], add_messages]

def init_state(state:TeachingState, config: RunnableConfig):
    journey_name=config["journey_name"]
    subject_name=config["subject_name"]
    sub_subject_index:int= config["sub_subject_index"]
    step_index:int=config["step_index"]
    init_journey_chat(config["journey_name"])
    journey:Dict[str, JourneyModel] = get_db_journey(journey_name)
    step = journey[subject_name].subjects[sub_subject_index].plan[step_index].structured

    id = f"{journey_name}{DELIMITER}{subject_name}{DELIMITER}{sub_subject_index}{DELIMITER}{step_index}"

    return {
        "id": id,
        "subject_data": step,
        "current_task_index": 0,
        "current_task": None,
        "past_tasks": [],
        "pre_class_messages": [],
        "post_class_messages": [],
    }

def introduce_class(state: TeachingState):
    subject_data:SubjectStructure = state["subject_data"]
    messages = state["pre_class_messages"]
    messages = messages + [AIMessage(content=subject_data.intro)]
    return {"pre_class_messages": messages}

def chat_in_class(state: TeachingState, message: str):
    subject_data:SubjectStructure = state["subject_data"]
    current_task_index:int = state["current_task_index"] or 0
    past_tasks:List[TeachingTask] = state["past_tasks"]
    current_task:TeachingTask = state["current_task"]

def continue_class(state: TeachingState):
    subject_data:SubjectStructure = state["subject_data"]
    current_task_index:int = state["current_task_index"] or 0
    past_tasks:List[TeachingTask] = state["past_tasks"]
    current_task:TeachingTask = state["current_task"]

def pre_plan_task(state: TeachingState):
    subject_data:SubjectStructure = state["subject_data"]
    current_task_index:int = state["current_task_index"]
    past_tasks:List[TeachingTask] = state["past_tasks"]

# State should contain:
# Messages/History
# Current task index
# User data
# Subject data

# graph
# 0. Take intro message from journey data and push it to messages
# 1. Wait for user input/Continue
# 1.1 while user input execute chat chain with state and history data
# 2. While tasks left
# 2.1 Take task
# 2.2 Take title, description, and resources from task
# 2.3 Plan teaching tasks to execute description with the help of resources and history
# 2.4 Take teaching task and execute it
# 2.5. Wait for user input/Continue
# 2.5.1 while user input execute chat chain with state and history data
# 2.6 Take test task from 2.1 and plan test and verification with the history and task resources
# 2.6.0 While user input does not pass test verification
# 2.6.1 Wait for user input
# 2.6.1 Test user input against test verification. If answer assumed verify that it is correct, otherwise use chat chain
# 2.6.2 If test fails converse with user to help them understand about their mistakes, not giving the right answer but encouraging user to think
# 2.7 Summarize discussion and relevant details from the task.
# 2.8. Wait for user input/Continue
# 2.8.1 while user input execute chat chain with state and history data
# 3. Summarize all the content from the journey and the user's learning
# 3.1. Wait for user input/Continue
# 3.1.1 while user input execute chat chain with state and history data


# main loop
# 1. Send the introduction message which is optimized to user specific details
# 2. (wait)
# 3. Iterate through tasks with teaching loop

# Wait loop
# 1. Wait for continue or user input
# 2. If user input go to user input loop
# 3. If continue, resume

# Teaching loop
# 1. Describe the next subject from the description to the user using learning material and history as context
# 2. (wait)
# 3. Iteratate through tasks using the task loop
# 4. Summarize all the content from the task loop
# 5. (wait)
# 6. Using the task test verify that the user has learned the content from the task

# Task loop
# 3. Take the task description and the learning material and determine approximately 3 distinct items to teach
# 4. Iterate through these plan in following fashion:
#   4.1. Teach about the item in up to 5 sentences
#   4.2. (wait)
#   4.3. Give an example of the item with up to 5 sentences
#   4.4. (wait)
#   4.5. Ask a question about the item
#   4.6. (wait)

# user input loop
# 1. Is the user input:
#   a. a statement?
#   b. a question?
#   c. a response to a question?
# 1.1. If statement add message to history and return to main loop
# 1.2. If question, go to question loop
# 1.3. If response, go to verify loop

# question loop
# 1. Send the question to the llm to determine the next step
# 2. If the llm determines that the question is about history respond directly from the history
# 3. If the llm determines that the question is about subject respond from learning material
# 4. If the llm determines that the question is about journey search the journey database and respond from documents
# 5. If the llm determines that the question is not relevant respond with a message that the matter doesn't fit the subject

# verify loop
# 1. take the message history and check if the last user input is a valid response to the question (history[-2]) using the expected response details
# 2. If the response is valid, add the response to the history send a verification message and return to main loop
# 3. If the response is not valid, analyse the response and determine the next step
# 3.1. If the analysis reveals a mistake use the learning material to explain the mistake to the user and go to user input loop
# 3.2 If the response is irrelevant restart response loop with a request for the answer to the question

class QueryState(TypedDict):
    messages: Annotated[list, add_messages]
    db_results: list

def get_chat_graph(vectorstore_id: str = None):
    graph_builder = StateGraph(QueryState)
    llm_json = get_llm("json")
    grader_chain = get_chain("grader")()
    retriever = get_vectorstore(vectorstore_id, "hyde").as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 40, "score_threshold": 0.3},
    )


    return graph_builder.build()