import os
import random
import sys
import time
import streamlit as st
from streamlit_theme import st_theme

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))

# Hide image fullscreen, control page wirth, adjust typeface


def get_theme(reset=False):
    theme = st.session_state.get("theme")
    retry = st.session_state.get("theme_retry", 0)
    retry += 1
    st.session_state["theme_retry"] = retry

    load_theme = False

    if theme is None or reset:
        try:
            theme = st_theme(key="theme_request")
        except Exception as e:
            print(e)

    if theme is None:
        theme = {
            "primaryColor": "#ff4b4b",
            "backgroundColor": "#ffffff",
            "secondaryBackgroundColor": "#f0f2f6",
            "textColor": "#31333F",
            "base": "light",
            "font": '"Source Sans Pro", sans-serif',
            "linkText": "#0068c9",
            "fadedText05": "rgba(49, 51, 63, 0.1)",
            "fadedText10": "rgba(49, 51, 63, 0.2)",
            "fadedText20": "rgba(49, 51, 63, 0.3)",
            "fadedText40": "rgba(49, 51, 63, 0.4)",
            "fadedText60": "rgba(49, 51, 63, 0.6)",
            "bgMix": "rgba(248, 249, 251, 1)",
            "darkenedBgMix100": "hsla(220, 27%, 68%, 1)",
            "darkenedBgMix25": "rgba(151, 166, 195, 0.25)",
            "darkenedBgMix15": "rgba(151, 166, 195, 0.15)",
            "lightenedBg05": "hsla(0, 0%, 100%, 1)",
            "borderColor": "rgba(49, 51, 63, 0.2)",
            "borderColorLight": "rgba(49, 51, 63, 0.1)",
        }
        st.session_state['theme_load_failed'] = True
    else:
        st.session_state['theme_load_failed'] = False

    st.session_state["theme"] = theme

    return theme


def init_css():
    with open(current_dir + "/static/global_styles.css") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

    # st.markdown("""
    #         <style>
    #             button[title="View fullscreen"] {
    #                 visibility: hidden;
    #             }

    #             .block-container {
    #                 max-width: 1000px;  #control page width
    #                 padding-top: 0rem;
    #                 padding-bottom: 2rem;
    #                 padding-left: 1rem;
    #                 padding-right: 1rem;
    #             }
    #         </style>
    #         """, unsafe_allow_html=True)
