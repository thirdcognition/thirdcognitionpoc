# import required dependencies
import os
import sys
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))

from lib.chat import DELIMITER, chat_elements, init_journey_chat
from lib.models.journey import JourneyDataTable


# chat_history = { "default": [] } #[[] for _ in range(11)]
# query_history = { "default": [] }
# chat_state = "default"


def page_not_found():
    st.title("Journey not found")
    st.write("The Journey you are looking for does not exist.")
    st.write("Please check the URL and try again.")


def main():
    st.set_page_config(
        page_title="TC POC",
        page_icon="static/icon.png",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            # 'Get Help': 'https://www.extremelycoolapp.com/help',
            # 'Report a bug': "https://www.extremelycoolapp.com/bug",
            "About": """# ThirdCognition PoC
[ThirdCognition](https://thirdcognition.com)
This is an *extremely* cool app!
            """
        },
    )

    journey_name = (
        st.query_params.get("journey", None)
        or ("active_journey" in st.session_state and st.session_state.active_journey)
        or None
    )
    chat_module = (
        st.query_params.get("state", None)
        or ("active_module" in st.session_state and st.session_state.active_module)
        or ""
    )
    journey_found = init_journey_chat(journey_name)

    if not journey_found and journey_name is not None:
        page_not_found()
        return

    if "active_journey" not in st.session_state:
        st.session_state.active_journey = journey_name

    if (
        "active_module" not in st.session_state
        or chat_module != st.session_state.active_module
    ):
        st.session_state.active_module = chat_module

    # st.header("ThirdCognition Proof of concept demostration")
    if "chat_state" not in st.session_state:
        st.session_state.chat_state = "default"
    chat_state = st.session_state.chat_state

    if chat_state != "default":
        if st.button(
            ":house: Return home", use_container_width=True, disabled=(0 == chat_state)
        ):
            chat_state = "default"
            st.session_state.chat_state = chat_state
            st.session_state.chat_journey = None
            st.rerun()

    if journey_name != None:
        journey_list = st.session_state.journey_list
        journey: JourneyDataTable = st.session_state.journey_list[journey_name]
        if chat_state == "default":
            st.subheader("ThirdCognition Virtual Buddy", divider=True)
        else:
            section_index = int(chat_state.split(DELIMITER)[1])
            st.subheader(journey.children[section_index].title, divider=True)
        # st.subheader(journey["title"], divider=True)
        # st.write(journey["summary"])

        with st.sidebar:
            st.markdown(
                """<style>
            button[data-testid=baseButton-secondary] {
                display: block;
                text-align: left;
                color: inherit;
                text-decoration: none;
                background-color: unset;
                border: none;
                padding-top: 0;
                padding-bottom: 0;
            }
            button[data-testid=baseButton-secondary]:active {
                text-decoration: underline;
                background-color: unset;
                color: inherit;
            }
            button[data-testid=baseButton-secondary]:disabled {
                border: none;
                cursor: auto !important;
            }
            </style>""",
                unsafe_allow_html=True,
            )

            print(f"{chat_state=}")

            for i, section in enumerate(journey.children):
                with st.expander(
                    f"{section.title}", expanded=(f"{journey_name}{DELIMITER}{i}" in chat_state or (0 == i and chat_state == "default"))
                ):
                    for j, module in enumerate(section.children):
                        module_id = f"{journey_name}{DELIMITER}{i}{DELIMITER}{j}"
                        if st.button(
                            module.title,
                            use_container_width=True,
                            disabled=(module_id == chat_state),
                            key=f"module_{module_id}",
                        ):  # , on_click=set_chat_state, args=(i, task)
                            chat_state = module_id
                            st.session_state.chat_state = chat_state
                            st.session_state.chat_journey = journey_name
                            st.rerun()

        for journey_name in journey_list:
            if (
                "chat_journey" in st.session_state
                and st.session_state.chat_journey == journey_name
                and "chat_state" in st.session_state
            ):
                chat_elements(st.session_state.chat_state, journey_name)

    if chat_state == "default" and st.session_state.chat_state == chat_state:
        chat_elements("default", journey_name)


if __name__ == "__main__":
    main()
