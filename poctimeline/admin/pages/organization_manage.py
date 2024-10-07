import os
import sys

import pandas as pd
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../../lib"))

from admin.sidebar import init_sidebar
from lib.helpers.shared import is_valid_email
from lib.streamlit.user import check_auth
from lib.models.user import (
    AuthStatus,
    UserLevel,
    add_org,
    add_user,
    get_all_orgs,
    get_all_users,
    get_org_by_id,
    get_user_org,
    get_user_org_id,
    is_org_admin,
    is_super_admin,
    set_disable_org,
    set_disable_user,
    set_org_name,
    set_user_journeys,
    set_user_level,
    set_user_org,
    set_user_name,
)
from lib.models.user import AuthStatus, UserLevel
from lib.streamlit.user import check_auth


st.set_page_config(
    page_title="TC POC: Manage organization",
    page_icon="static/icon.png",
    layout="centered",
    menu_items={
        "About": """# ThirdCognition PoC
[ThirdCognition](https://thirdcognition.com)
This is an *extremely* cool admin tool!
        """
    },
)


def manage_organizations():
    if is_super_admin():
        st.subheader("Manage Organizations")

        orgs = get_all_orgs()

        org_data = []
        for org in orgs:
            org_data.append(
                {
                    "Disabled": org.disabled,
                    "Organization ID": org.organization_id,
                    "Organization Name": org.organization_name,
                }
            )

        df = pd.DataFrame(org_data)
        df['Organization ID'] = df['Organization ID'].astype(str)
        column_config = {
            "Organization ID": st.column_config.TextColumn(
                "Organization ID",
                disabled=True,
            ),
            "Organization Name": st.column_config.TextColumn(
                "Organization Name",
                disabled=False,
            ),
            "Disabled": st.column_config.CheckboxColumn(
                "Disabled",
                disabled=False,
            ),
        }
        edited_df = st.data_editor(df, column_config=column_config, use_container_width=True, hide_index=True)

        with st.container(border=True):
            st.write("#### Add New Organization")
            new_org_id = st.text_input("New Organization ID", key="new_org_id")
            new_org_name = st.text_input("New Organization Name", key="new_org_name")
            new_admin_email = st.text_input("New Admin Email", key="new_admin_email")
            if new_admin_email and not is_valid_email(new_admin_email):
                st.warning("Please enter a valid email address.")
            if st.button("Add New Organization"):
                if (
                    new_org_id
                    and new_org_name
                    and new_admin_email
                    and is_valid_email(new_admin_email)
                ):
                    add_org(new_org_id, new_org_name)
                    add_user(
                        new_admin_email, level=UserLevel.org_admin, org_id=new_org_id
                    )
                else:
                    st.warning(
                        "Please fill all the fields and enter a valid email address."
                    )
                st.success("Organization created successfully!")
                get_all_orgs(reset=True)
                get_all_users(reset=True)
                st.rerun()

        for index, row in edited_df.iterrows():
            if row["Disabled"] != df.at[index, "Disabled"]:
                set_disable_org(row["Organization ID"], row["Disabled"])
            if row["Organization Name"] != df.at[index, "Organization Name"]:
                set_org_name(row["Organization ID"], row["Organization Name"])
    else:
        user_org = get_user_org(st.session_state["username"])
        with st.container(border=True):
            st.subheader("Modify Organization")
            # st.write(f"Organization ID: {user_org.organization_id}")
            new_org_name = st.text_input(
                "New Organization Name",
                key="new_org_name",
                value=user_org.organization_name,
            )
            if st.button("Save") and new_org_name != user_org.organization_name:
                set_org_name(user_org.organization_id, new_org_name)
                st.success("Organization name updated successfully!")
                get_user_org(reset=True)
                st.rerun()


def manage_users():
    st.subheader("Manage Users")
    users = get_all_users()
    user_data = []
    for user in users:
        user_data.append(
            {
                "Disabled": user.disabled,
                "Name": user.name,
                "Email": user.email,
                "Username": user.username,
                "Level": UserLevel(user.level).name,
                "Organization ID": user.organization_id,
            }
        )

    df = pd.DataFrame(user_data)
    df['Email'] = df['Email'].astype(str)
    df['Username'] = df['Username'].astype(str)
    df['Organization ID'] = df['Organization ID'].astype(str)
    column_config = {
        "Disabled": st.column_config.CheckboxColumn(
            "Disabled",
            disabled=False,
        ),
        "Name": st.column_config.TextColumn(
            "Name",
            disabled=False,
        ),
        "Email": st.column_config.TextColumn(
            "Email",
            disabled=True,
        ),
        "Username": st.column_config.TextColumn(
            "Username",
            disabled=True,
        ),
        "Level": st.column_config.SelectboxColumn(
            "Level",
            default=UserLevel.user.name,
            options=[UserLevel.user.name, UserLevel.org_admin.name, UserLevel.super_admin.name] if is_super_admin() else [UserLevel.user.name, UserLevel.org_admin.name],
            required=True,
            disabled=False,
        ),
        "Organization ID": st.column_config.TextColumn(
            "Organization ID",
            disabled=True,
        ),
    }
    edited_df = st.data_editor(df, column_config=column_config, use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.write("#### Add New User")
        new_email = st.text_input("New User Email", key="new_user_email")
        # new_username = st.text_input("New Username", key="new_username")
        if is_super_admin():
            new_org_id = st.selectbox(
                "New Organization ID",
                options=[org.organization_id for org in get_all_orgs()],
                key="new_user_org_id",
            )
        else:
            new_org_id = get_user_org_id()
        new_level = st.selectbox(
            "New User Level",
            [UserLevel.user, UserLevel.org_admin],
            key="new_user_level",
        )
        new_journeys = st.text_input(
            "New Journeys (comma separated)", key="new_journeys"
        )
        if new_email and not is_valid_email(new_email):
            st.warning("Please enter a valid email address.")
        if st.button("Add new user"):
            if new_email and new_org_id and new_level and is_valid_email(new_email):
                add_user(
                    new_email,
                    org_id=new_org_id,
                    level=new_level,
                    journeys=new_journeys.split(",") if new_journeys else None,
                )
            else:
                st.warning(
                    "Please fill all the fields and enter a valid email address."
                )
            st.success("User created successfully!")
            st.rerun()

    for index, row in edited_df.iterrows():
        if row["Disabled"] != df.at[index, "Disabled"]:
            set_disable_user(row["Email"], row["Disabled"])
        if row["Username"] != df.at[index, "Username"]:
            if row["Username"] != df.at[index, "Username"]:
                st.warning(f"Username for {row['Email']} cannot be changed.")
        if row["Name"] != df.at[index, "Name"]:
            set_user_name(row["Email"], row["Name"])
        if row["Level"] != df.at[index, "Level"]:
            set_user_level(row["Email"], UserLevel[row["Level"]])
        if row["Organization ID"] != df.at[index, "Organization ID"]:
            set_user_org(row["Email"], row["Organization ID"])
        # You'll need to implement a function to handle changing journeys


def main():
    st.title("Organization and User Management")

    if init_sidebar(UserLevel.org_admin) != AuthStatus.LOGGED_IN:
        return

    if is_super_admin():
        tab1, tab2 = st.tabs(["Organizations", "Users"])

        with tab1:
            manage_organizations()

        with tab2:
            manage_users()
    else:
        manage_users()
        manage_organizations()


if __name__ == "__main__":
    main()
