from io import BytesIO
import re
from typing import List
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from lib.chains.init import get_chain
from lib.db_tools import SourceDataTable, init_db
from lib.document_tools import markdown_to_text
from langchain_core.messages import BaseMessage

from lib.graphs.process_text import ProcessTextState, process_text

with open("admin_auth.yaml") as file:
    auth_config = yaml.load(file, Loader=SafeLoader)


def check_auth():
    # global authenticator
    # if authenticator is None:
    authenticator = stauth.Authenticate(
        auth_config["credentials"],
        auth_config["cookie"]["name"],
        auth_config["cookie"]["key"],
        auth_config["cookie"]["expiry_days"],
        auth_config["pre-authorized"],
    )

    authenticator.login()
    if st.session_state["authentication_status"]:
        col1, col2 = st.columns([7, 1])
        col1.write(f'Welcome *{st.session_state["name"]}*')
        with col2:
            authenticator.logout()
        return True
    else:
        if st.session_state["authentication_status"] is False:
            st.error("Username/password is incorrect")
        if st.session_state["authentication_status"] is None:
            st.warning("Please enter your username and password")

        return False


def get_all_categories():
    database_session = init_db()

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


async def process_text_call(
    show_progress=True,
    texts: List[str] = None,
    source: str = None,
    filename: str = None,
    file: BytesIO = None,
    url: str = None,
    categories: List[str] = None,
    overwrite: bool = False,
    guidance: str = None,
    collect_concepts=True,
    summarize: bool = True,
) -> ProcessTextState:
    config = {
        "configurable": {
            "collect_concepts": collect_concepts,
            "overwrite_sources": overwrite,
            "update_rag": (source != None or file != None or url != None)
            and categories != None,
            "guidance": guidance,
            "summarize": summarize,
        }
    }
    states = {
        "split": "Read and analysis of document",
        "reformat": "Document re-formatting",
    }
    if config["configurable"]["collect_concepts"]:
        states.update(
            {
                "split_reformatted": "Prepare content for concept search",
                "find_concepts": "Searching for concepts",
                "collapse_concepts": "Formatting concepts"
            }
        )

    states.update(
        {
            "summary": "Summary writing",
            "collapse": "Collapsing contents",
            "finalize": "Combining contents",
        }
    )


    if config["configurable"]["update_rag"]:
        states.update({"rag": "RAG database update"})

    state_status = {}
    if show_progress:
        for state in states.keys():
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

    async for event in process_text.astream_events(
        params,
        config=config,
        version="v2",
    ):
        # if event["event"] != "on_chat_model_stream":
        #     print(f"\n\n\n{event['name']=} - {event['event']=}\n\n")
        if show_progress and (
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

            for state in states.keys():
                update_state_status = (
                    input[f"{state}_complete"]
                    if input is not None and f"{state}_complete" in input
                    else False
                ) or (
                    output[f"{state}_complete"]
                    if output is not None and f"{state}_complete" in output
                    else False
                )
                if update_state_status:
                    # if isinstance(state_status[state], StatusContainer):
                    #     state_status[state].update(label=f"{states[state]} complete", state="complete")
                    # else:
                    state_status[state].empty()
                    state_status[state].success(f"{states[state]} complete")
                elif (
                    state == event["name"] or f"{state}_content" == event["name"]
                    and event["event"] == "on_chain_start"
                ):
                    # if not isinstance(state_status[state], StatusContainer):
                    state_status[state].empty()
                    state_status[state].info(
                        f"{states[state]} in progress"
                    )  # , state="running")


            # print(f"\n\n\n{event["name"]=}: {event["event"]}\n\n\n")
        events.append(event)
    # print(f"\n\n{result=}\n\n")
    result = events[-1]["data"]["output"]

    if show_progress:
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

    result = await process_text_call(
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


# text = ""

# i = 0
# total = len(texts)

# if total > 1 and chain == "summary":
#     total += 1

# if not force and (
#     texts == None or texts[0] == None or total == 1 and len(texts[0]) < 1000
# ):
#     return None, None

# bar = st.progress(0, text="Processing...")

# text =

# if total > 1:
#     inputs = []
#     for sub_text in texts:
#         bar.progress(i / total, f"Processing {i+1} / {total}...")
#         i += 1
#         _text = re.sub(r"[pP]age [0-9]+:", "", sub_text)
#         _text = re.sub(r"[iI]mage [0-9]+:", "", _text)
#         input = {"context": _text}

#         guided_llm = ""
#         if guidance is not None and guidance != "":
#             input["question"] = guidance
#             guided_llm = "_guided"

#         inputs.append(input)

#         result = get_chain(chain + guided_llm).invoke(input)

#         if isinstance(result, tuple) and len(result) == 2:
#             result = result[1]

#         mid_results = mid_results.content if isinstance(mid_results, BaseMessage) else mid_results

#         text += mid_results + "\n\n"

# else:
#     _text = re.sub(r"[pP]age [0-9]+", "", texts[0])
#     _text = re.sub(r"[iI]mage [0-9]+", "", _text)
#     text = _text

# bar.progress((total - 1) / total, text="Summarizing...")

# if chain == "summary":
#     text = markdown_to_text(text)

#     input = {"context": text}

#     guided_llm = ""

#     if guidance is not None and guidance != "":
#         guided_llm = "_guided"
#         input["question"] = guidance

#     text = get_chain(chain + guided_llm).invoke(input)

#     if isinstance(text, tuple) and len(text) == 2:
#         _,  text = text

#     text = text.content if isinstance(text, BaseMessage) else text

# bar.empty()

# return text.strip()


def nav_to(url):
    nav_script = """
        <meta http-equiv="refresh" content="0; url='%s'">
    """ % (
        url
    )
    st.write(nav_script, unsafe_allow_html=True)
