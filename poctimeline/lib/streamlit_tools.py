from datetime import datetime, timedelta
from io import BytesIO
import re
import time
from typing import Dict, List
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from langchain_core.messages import BaseMessage
from langgraph.graph.state import CompiledStateGraph

from lib.chains.init import get_chain
from lib.models.user import user_db_get_session
from lib.document_tools import markdown_to_text

from lib.graphs.handle_source import handle_source
from lib.graphs.process_text import process_text
from lib.graphs.find_concepts import find_concepts
from lib.graphs.find_topics import find_topics
from lib.graphs.find_taxonomy import find_taxonomy
from lib.models.source import SourceDataTable

def get_all_categories():
    database_session = user_db_get_session()

    if database_session is None:
        return None

    if "file_categories" not in st.session_state:
        uniq_categories = []
        categories = database_session.query(SourceDataTable.category_tags).distinct()
        for items in categories:
            for categories_list in items:
                if len(categories_list) > 0:
                    for categories in categories_list:
                        if categories not in uniq_categories:
                            uniq_categories.append(categories)

        st.session_state.file_categories = uniq_categories
    else:
        uniq_categories = st.session_state.file_categories

    return uniq_categories


async def graph_call(
    show_progress=True,
    texts: List[str] = None,
    source: str = None,
    filename: str = None,
    file: BytesIO = None,
    url: str = None,
    categories: List[str] = None,
    overwrite: bool = False,
    guidance: str = None,
    summarize: bool = True,
    graph: CompiledStateGraph = process_text,
    title = ""
) -> Dict:
    config = {
        "configurable": {
            "overwrite_sources": overwrite,
            "rewrite_text": overwrite,
            "update_rag": (source != None or file != None or url != None)
            and categories != None,
            "update_concepts": (source != None or file != None or url != None)
            and categories != None,
            "guidance": guidance,
            "summarize": summarize,
        }
    }
    states = {}
    if graph == handle_source:
        states.update({"split_content": "Split document into pages", "get_source_content": "Get source content", "process_text": "Process document contents"})
    if graph == process_text or graph == handle_source:
        states.update(
            {
                "setup_content": "Initial setup for content",
                "reformat_content": "Reformat content pages",
                "collapse_content": "Compress content to one page",
                "finalize_content": "Finalize and write summary",
            }
        )
    if graph == handle_source:
        states.update({"find_topics": "Look for document topics"})
    if graph == find_topics or graph == handle_source:
        states.update(
            {
                "setup_content": "Initial setup for content",
                "search_topics_content": "Search for topics in content",
                "concat_search_topics": "Organize found topics",
            }
        )
    if graph == handle_source:
        states.update({"find_taxonomy": "Look for document taxonomy"})
    if graph == find_taxonomy or graph == handle_source:
        states.update(
            {
                "search_taxonomy": "Search for taxonomy items within topics",
                "combine_taxonomy_items": "Organize found taxonomy"
            }
        )
    if graph == handle_source:
        states.update({"find_concepts": "Look for document concepts"})
    if graph == find_concepts or graph == handle_source:
        states.update(
            {
                "search_concepts": "Search for concepts within topics",
                "collapse_concepts": "Organize found concepts"
            }
        )

    if len(states.keys()) == 0:
        states.update(
            {
                "get_source_content": "Get source content",
                "split_content": "Split document into pages",
                "process_text": "Reformat document",
                "find_topics": "Identify document topics",
                "find_taxonomy": "Build document taxonomy",
                "find_concepts": "Identify document concepts based on taxonomy",
                "combine_concepts": "Combine found concepts",
                "collapse_concepts": "Format concepts and taxonomy"
            }
        )

    if config["configurable"]["update_rag"]:
        states.update({"rag_update": "Save changes to RAG"})

    state_keys = list(states.keys())
    total = len(state_keys)
    cur_step = 0

    progress = None
    if show_progress:
        progress = st.progress(0, text=f"{title} - Initializing...")


    show_progress_items = False

    state_status = {}
    if show_progress_items:
        for state in state_keys:
            state_status[state] = st.empty()
            # state_status[state].warning(f"{states[state]}: Not started")
    events = []

    params = {}
    if source is not None:
        params["source"] = source
    if filename is not None:
        params["filename"] = filename
    if file is not None:
        params["file"] = file
    if url is not None:
        params["url"] = url
    if categories is not None:
        params["categories"] = categories
    if texts is not None:
        params["contents"] = texts

    white_space = ""

    prev_time = datetime.now()
    prev_times = [prev_time]

    async for event in graph.astream_events(
        params,
        config=config,
        version="v2",
    ):
        if event["event"] != "on_chat_model_stream" and event["name"] in state_keys:
            now = datetime.now()
            if "_end" in  event["event"]:
                white_space = white_space[:-2]
                prev_time = prev_times.pop()
            # if now - prev_time > timedelta(seconds=1):
            print(f"{title} - +{round((now-prev_time).total_seconds(), 2)}s:" + white_space + f"{event['name']} - {event['event']=}")
            if "_start" in event["event"]:
                white_space += "  "
                prev_times.append(now)
        if (show_progress_items or show_progress) and (
            event["event"] == "on_chain_start" or event["event"] == "on_chain_end"
        ):
            input = (
                event["data"]["input"]
                if "input" in event["data"] and isinstance(event["data"]["input"], dict)
                else None
            )
            output = (
                event["data"]["output"]
                if "output" in event["data"]
                and isinstance(event["data"]["output"], dict)
                else None
            )

            for state in state_keys:
                update_state_status = (
                    input[f"{state}_complete"]
                    if input is not None and f"{state}_complete" in input
                    else False
                ) or (
                    output[f"{state}_complete"]
                    if output is not None and f"{state}_complete" in output
                    else False
                )
                if show_progress:
                    # print("Show progress", state, states[state])
                    if update_state_status:
                        cur_step = max(state_keys.index(state) + 1, cur_step)
                        progress.progress(
                            min(cur_step, total) / total,
                            text=f"{title} - {states[state]} complete",
                        )
                    elif (
                        state == event["name"]
                        and event["event"] == "on_chain_start"
                    ):
                        cur_step = max(state_keys.index(state), cur_step)
                        progress.progress(
                            min(cur_step, total) / total,
                            text=f"{title} - {states[state]} in progress",
                        )

                if show_progress_items:
                    if update_state_status:
                        # if isinstance(state_status[state], StatusContainer):
                        #     state_status[state].update(label=f"{states[state]} complete", state="complete")
                        # else:
                        state_status[state].empty()
                        state_status[state].success(f"{title} - {states[state]} complete")
                    elif (
                        state == event["name"]
                        and event["event"] == "on_chain_start"
                    ):
                        # if not isinstance(state_status[state], StatusContainer):
                        state_status[state].empty()
                        state_status[state].info(
                            f"{title} - {states[state]} in progress"
                        )  # , state="running")


            # print(f"\n\n\n{event["name"]=}: {event["event"]}\n\n\n")
        events.append(event)
    # print(f"\n\n{result=}\n\n")
    result = events[-1]["data"]["output"]

    if show_progress:
        progress.progress(1.0, text="Done")
        time.sleep(1)
        progress.empty()
    if show_progress_items:
        for state in states.keys():
            state_status[state].empty()

    return result


async def llm_edit(
    texts: List[str],
    guidance=None,
    force: bool = False,
    show_process=True,
    summarize: bool = False,
) -> str:
    if texts == None or texts[0] == None:
        raise ValueError("No text provided")

    if not force and (len(texts) == 1 and len(texts[0]) < 1000):
        return texts[0] or ""

    result = await graph_call(
        show_progress=show_process,
        texts=texts,
        guidance=guidance,
        collect_concepts=False,
        summarize=summarize,
    )

    contents = result["content_result"]

    if summarize:
        return contents["summary"].strip()
    else:
        return contents["formatted_content"].strip()

def nav_to(url):
    nav_script = """
        <meta http-equiv="refresh" content="0; url='%s'">
    """ % (
        url
    )
    st.write(nav_script, unsafe_allow_html=True)
