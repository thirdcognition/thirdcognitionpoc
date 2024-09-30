import streamlit as st
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))

from lib.models.user import AuthStatus, UserLevel
from lib.streamlit.user import check_auth
from lib.chat import chat_elements, init_journey_chat
from lib.streamlit_tools import get_all_categories

st.set_page_config(
    page_title="TC POC: Admin",
    page_icon="static/icon.png",
    layout="centered",
    menu_items={
        "About": """# ThirdCognition PoC
[ThirdCognition](https://thirdcognition.com)
This is an *extremely* cool admin tool!
        """
    },
)


def main():

    auth_valid = check_auth()

    st.write("# Welcome! 👋")

    st.markdown(
        """
Select 👈 a the subject from sidebar to edit the content!
    """
    )

    if auth_valid == AuthStatus.NO_LOGIN:
        st.write("Please log in for more functionality.")
        return

    with st.container():
        st.write("### Engage with the RAG:")
        categories = get_all_categories()
        selected_categories = st.selectbox(
            "Select a categories", categories, key="categories", index=0
        )

        if selected_categories:
            journey_found = init_journey_chat(
                rag_collection=f"rag_{selected_categories}"
            )
            # if "chat_state" not in st.session_state:
            #     st.session_state.chat_state = "default"

            # if journey_found:
            chat_elements("default")
        else:
            st.write("Please select a categories to start chatting.")


if __name__ == "__main__":
    main()
