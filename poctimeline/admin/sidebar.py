import os
import sys
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))

from admin.global_styles import get_theme, init_css
from lib.models.user import AuthStatus, UserLevel, check_auth_level
from lib.streamlit.user import check_auth

def get_image(id, path="") -> str:
    theme = get_theme()
    # theme = st_theme()
    # print(theme)

    if theme is not None and theme["base"] == "dark":
        id = id + "_dark"

    path = (
        os.path.join(current_dir, "static", path)
        if path != ""
        else os.path.join(current_dir, "static")
    )
    # print(os.path.join(path, f"{id}.png"))
    return os.path.join(path, f"{id}.png")


def init_sidebar(req_user_level: UserLevel = UserLevel.anonymous) -> AuthStatus:
    # from streamlit_javascript import st_javascript
    # st_theme = st_javascript("""window.getComputedStyle(window.parent.document.getElementsByClassName("stApp")[0]).getPropertyValue("color-scheme")""")
    # if st_theme == "dark":
    #     ...
    # else:
    #     ...
    theme = get_theme()
    init_css()
    # print(theme)

    st.logo(get_image("logo"))
    # if theme is not None and theme["base"] == "dark":
    # else:
    #     st.logo(get_image("logo-white", ""))

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

    auth_valid = check_auth(
        req_user_level
    )  # st.session_state.get("auth_level", check_auth(req_user_level))
    print(f"auth_valid: {auth_valid}")
    # st.session_state["auth_level"] = auth_valid
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
            st.page_link("pages/journey_simple_create.py", label="Create Journey")
            st.page_link("pages/journey_simple_manage.py", label="Manage Journeys")
            st.page_link("pages/organization_manage.py", label="Manage Organization")
        if user_level >= UserLevel.super_admin:
            st.page_link("pages/source_upload.py", label="Upload files")
            st.page_link("pages/source_manage.py", label="Manage files")
            st.page_link("pages/concepts_view.py", label="Browse topics")
        st.divider()

    return auth_valid
