import streamlit as st
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))

from lib.models.user import AuthStatus, UserLevel, check_auth_level
from lib.streamlit.user import check_auth


def init_sidebar(req_user_level: UserLevel = UserLevel.anonymous) -> AuthStatus:
    with st.sidebar:
        menu_items = st.empty()

    auth_valid = check_auth(req_user_level)
    user_level: UserLevel = UserLevel.anonymous
    if auth_valid != AuthStatus.NO_LOGIN:
        user_level = check_auth_level()

    # st.markdown(
    #     """
    #     <style>
    #         section[data-testid="stSidebar"] {
    #             width: 300px !important; # Set the width to your desired value
    #         }
    #     </style>
    #     """,
    #     unsafe_allow_html=True, )

    # st.logo("logo.png")
    menu_container = menu_items.container()
    with menu_container:
        st.page_link("Admin_Home.py", label="Home")
        if user_level >= UserLevel.org_admin:
            st.page_link("pages/6_Manage_Organization.py", label="Manage Organization")
            st.page_link("pages/1_Upload.py", label="Upload files")
            st.page_link("pages/2_Manage_Uploads.py", label="Manage files")
            st.page_link("pages/3_Browse_Concepts.py", label="Browse topics")

    return auth_valid
