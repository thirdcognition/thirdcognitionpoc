import pprint as pp
import streamlit as st
from lib.load_env import DEBUGMODE
from langchain_core.runnables import (
    RunnableSequence,
    RunnableWithMessageHistory,
)
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

def print_params(msg = "", params = ""):
    if DEBUGMODE:
        if msg: print(f"\n\n\n{msg}")
        if params: print(f"'\n\n{pp.pformat(params).replace("\\n", "\n")}\n\n")

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = {}

    if session_id not in st.session_state["chat_history"]:
        st.session_state["chat_history"][session_id] = ChatMessageHistory()
    return st.session_state["chat_history"][session_id]

def get_chain_with_history(chain_id: str, chain:RunnableSequence):
    if "history_chains" not in st.session_state:
        st.session_state["history_chains"] = {}
    if chain_id in st.session_state["history_chains"]:
        return st.session_state["history_chains"][chain_id]

    history_chain = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        output_messages_key="answer",
        history_messages_key="chat_history"
    )

    st.session_state["history_chains"][chain_id] = history_chain
    return history_chain