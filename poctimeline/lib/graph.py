from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from lib.db_tools import (
    Base,
    FileDataTable,
    JourneyDataTable,
    get_db_files,
    get_db_journey,
    get_vectorstore,
    init_db,
)
from lib.chain import (
    EMBEDDING_CHAR_LIMIT,
    EMBEDDING_OVERLAP,
    INSTRUCT_CHAR_LIMIT,
    client_host,
    SQLITE_DB,
    get_chain,
    get_llm,
    # get_vectorstore,
)
from prompts import JourneyStructure

# main loop
# 1. Send the introduction message which is optimized to user specific details
# 2. (wait)
# 3. Iterate through actions with teaching loop

# Wait loop
# 1. Wait for continue or user input
# 2. If user input go to user input loop
# 3. If continue, resume

# Teaching loop
# 1. Describe the next subject from the description to the user using learning material and history as context
# 2. (wait)
# 3. Iteratate through actions using the action loop
# 4. Summarize all the content from the action loop
# 5. (wait)
# 6. Using the action test verify that the user has learned the content from the action

# Action loop
# 3. Take the action description and the learning material and determine approximately 3 distinct items to teach
# 4. Iterate through these steps in following fashion:
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
    grader_chain = get_chain("grader")
    retriever = get_vectorstore(vectorstore_id, "hyde").as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 40, "score_threshold": 0.3},
    )


    return graph_builder.build()