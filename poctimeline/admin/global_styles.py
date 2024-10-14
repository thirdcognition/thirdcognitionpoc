import os
import sys
import streamlit as st
from streamlit_theme import st_theme
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))

#Hide image fullscreen, control page wirth, adjust typeface

theme = None

def get_theme():
    global theme
    if theme is None:
        theme = st_theme()
    return theme

def init_css():
    with open(current_dir+"/static/global_styles.css") as css:
        st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

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