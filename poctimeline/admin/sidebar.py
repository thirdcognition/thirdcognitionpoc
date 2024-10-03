import os
import sys
import streamlit as st
from streamlit_theme import st_theme

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))

from lib.models.user import AuthStatus, UserLevel, check_auth_level
from lib.streamlit.user import check_auth


def init_sidebar(req_user_level: UserLevel = UserLevel.anonymous) -> AuthStatus:
    # from streamlit_javascript import st_javascript
    # st_theme = st_javascript("""window.getComputedStyle(window.parent.document.getElementsByClassName("stApp")[0]).getPropertyValue("color-scheme")""")
    # if st_theme == "dark":
    #     ...
    # else:
    #     ...
    theme = st_theme()
    # print(theme)

    if theme is not None and theme["base"] == "dark":
        st.logo(f"{current_dir}/static/logo.png")
    else:
        st.logo(f"{current_dir}/static/logo-white.png")

    with st.sidebar:
        st.markdown(
            """<style>
        div[data-testid=stVerticalBlock]:has(div[data-testid=stPageLink]) {
            display: block;
            text-align: left;
            color: inherit;
            text-decoration: none;
            background-color: unset;
            border: none;
            padding-top: 0;
            padding-bottom: 0;
            margin: .5rem 0 .5rem;
        }
        div[data-testid=stPageLink]:active {
            text-decoration: underline;
            background-color: unset;
            color: inherit;
        }
        div[data-testid=stPageLink]:disabled {
            border: none;
            cursor: auto !important;
        }
        </style>""",
            unsafe_allow_html=True,
        )
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
            st.page_link("pages/journey_simple_template.py", label="Create Journey")
            st.page_link("pages/organization_manage.py", label="Manage Organization")
            st.page_link("pages/source_upload.py", label="Upload files")
            st.page_link("pages/source_manage.py", label="Manage files")
            st.page_link("pages/concepts_view.py", label="Browse topics")
        st.divider()

    return auth_valid
