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
    delete_user,
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
    set_user_level,
    set_user_org,
    set_user_name,
)
from lib.models.user import AuthStatus, UserLevel
from lib.streamlit.user import check_auth


st.set_page_config(
    page_title="TC POC: Manage organization",
    page_icon="static/icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
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

        orgs = get_all_orgs(reset=True)

        org_data = []
        for org in orgs:
            org_data.append(
                {
                    "Disabled": org.disabled,
                    "Name": org.organization_name,
                    "ID": org.id,
                }
            )

        df = pd.DataFrame(org_data)
        df['ID'] = df['ID'].astype(str)
        column_config = {
            "ID": st.column_config.TextColumn(
                "ID",
                disabled=True,
            ),
            "Name": st.column_config.TextColumn(
                "Name",
                width="large",
                disabled=False,
            ),
            "Disabled": st.column_config.CheckboxColumn(
                "Disabled",
                width="small",
                disabled=False,
            ),
        }
        edited_df = st.data_editor(df, column_config=column_config, use_container_width=True, hide_index=True)

        with st.container(border=True):
            st.write("#### Add New Organization")
            # new_org_id = st.text_input("New Organization ID", key="new_org_id")
            new_org_name = st.text_input("New Organization Name", key="new_org_name")
            new_admin_email = st.text_input("New Admin Email", key="new_admin_email")
            if new_admin_email and not is_valid_email(new_admin_email):
                st.warning("Please enter a valid email address.")
            if st.button("Add New Organization"):
                if (
                    new_org_name
                    and new_admin_email
                    and is_valid_email(new_admin_email)
                ):
                    org = add_org(new_org_name)
                    add_user(
                        new_admin_email, level=UserLevel.org_admin, org_id=org.id
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
                set_disable_org(row["ID"], row["Disabled"])
            if row["Name"] != df.at[index, "Name"]:
                set_org_name(row["ID"], row["Name"])
    else:
        user_org = get_user_org(st.session_state["username"])
        with st.container(border=False):
            st.divider()
            st.subheader("Modify Organization")
            # st.write(f"Organization ID: {user_org.organization_id}")
            new_org_name = st.text_input(
                "New Organization Name",
                key="new_org_name",
                value=user_org.organization_name,
            )
            if st.button("Save") and new_org_name != user_org.organization_name:
                set_org_name(user_org.id, new_org_name)
                st.success("Organization name updated successfully!")
                # get_user_org(st.session_state["username"],  reset=True)
                st.rerun()


def manage_users():
    st.subheader("Manage Users")
    users = get_all_users(reset=True)
    orgs = get_all_orgs(reset=True)
    cur_user_org = get_user_org(email = st.session_state["username"])

    user_data = []
    for user in users:
        user_info = {
                "Disabled": user.disabled,
                "Name": user.name,
                "Email": user.email,
                "ID": user.id,
                "Level": UserLevel(user.level).name,
            }

        if is_super_admin():
            user_info["Organization"] = next(
                (org.organization_name for org in orgs if org.id == user.organization_id),
                "Unknown"
        )

        user_data.append(user_info)

    df = pd.DataFrame(user_data)
    df['Email'] = df['Email'].astype(str)
    if "Organization" in df:
        df['Organization'] = df['Organization'].astype(str)

    org_options = {org.organization_name: org.id for org in orgs}

    column_config = {
        "Disabled": st.column_config.CheckboxColumn(
            "Disabled",
            disabled=False,
            width="small",
        ),
        "Name": st.column_config.TextColumn(
            "Name",
            disabled=False,
        ),
        "Email": st.column_config.TextColumn(
            "Email",
            disabled=True,
        ),
        "ID": st.column_config.TextColumn(
            "ID",
            disabled=True,
        ),
        "Level": st.column_config.SelectboxColumn(
            "Level",
            default=UserLevel.user.name,
            options=[UserLevel.user.name, UserLevel.org_admin.name, UserLevel.super_admin.name] if is_super_admin() else [UserLevel.user.name, UserLevel.org_admin.name],
            required=True,
            disabled=False,
        ),
    }

    if is_super_admin():
        column_config["Organization"] = st.column_config.SelectboxColumn(
            "Organization",
            options=list(org_options.keys()),
            required=True,  # This ensures the organization cannot be None
            disabled=not is_super_admin(),
        )

    edited_df = st.data_editor(df, column_config=column_config, use_container_width=True, hide_index=True)

    filtered_df = edited_df[~(edited_df["Email"] == st.session_state["username"]) & (edited_df["Level"] != UserLevel.super_admin.name)]



    user_id_title_map = {
        record['ID']: f"{record['Organization'] + ': ' if 'Organization' in record else ''}{record['Email']}" + (f" - {record['Name']}" if record['Name'] else "")
        for record in filtered_df.to_dict('records')
    }

    if len(user_id_title_map.keys()) > 0:
        st.divider()
        st.write("#### Remove user")
        col1, col2 = st.columns([10, 1], vertical_alignment="bottom")
        delete_user_id = col1.selectbox(
            "Select a user to remove",
            options=list(user_id_title_map.keys()),
            format_func=lambda x: user_id_title_map[x],
            key="delete_user_id",
        )

        selected_user_record = filtered_df.set_index('ID').loc[delete_user_id].to_dict()
        organization_id = org_options[selected_user_record['Organization']] if "Organization" in selected_user_record else cur_user_org.id

        with col2.popover(":x:"):
            st.write("Warning: This is permanent and cannot be undone.")
            if st.button(f"Are you sure you want to remove {user_id_title_map[delete_user_id]}?", key="delete_user_" + delete_user_id):
                delete_user(delete_user_id, organization_id)
                st.success(f"User {user_id_title_map[delete_user_id]} deleted successfully!")
                st.rerun()

    st.divider()
    with st.container():
        st.write("#### Add New User")
        new_email = st.text_input("New User Email", key="new_user_email")

        if is_super_admin():
            new_org_name = st.selectbox(
                "Organization",
                options=list(org_options.keys()),
                key="new_user_org_id"
            )
            new_org_id = org_options[new_org_name]
        else:
            new_org_id = get_user_org_id()

        new_level = st.selectbox(
            "New User Level",
            [UserLevel.user, UserLevel.org_admin],
            key="new_user_level",
        )

        if new_email and not is_valid_email(new_email):
            st.warning("Please enter a valid email address.")
        if st.button("Add new user"):
            if new_email and new_org_id and new_level and is_valid_email(new_email):
                add_user(
                    new_email,
                    org_id=new_org_id,
                    level=new_level,
                )
                st.success("User created successfully!")
                st.rerun()
            else:
                st.warning(
                    "Please fill all the fields and enter a valid email address."
                )

    for index, row in edited_df.iterrows():
        if row["Disabled"] != df.at[index, "Disabled"]:
            set_disable_user(row["ID"], row["Disabled"])
        if row["Name"] != df.at[index, "Name"]:
            set_user_name(row["ID"], row["Name"])
        if row["Level"] != df.at[index, "Level"]:
            set_user_level(row["ID"], UserLevel[row["Level"]])
        if "Organization" in row and row["Organization"] != df.at[index, "Organization"]:
            set_user_org(row["ID"], org_options[row["Organization"]])

# def manage_users():
#     st.subheader("Manage Users")
#     users = get_all_users(reset=True)
#     orgs = get_all_orgs(reset=True)

#     user_data = []
#     for user in users:
#         user_data.append(
#             {
#                 "Disabled": user.disabled,
#                 "Name": user.name,
#                 "Email": user.id,
#                 "ID": user.email,
#                 # "Username": user.username,
#                 "Level": UserLevel(user.level).name,
#                 "Organization": user.organization_id,
#             }
#         )

#     df = pd.DataFrame(user_data)
#     df['Email'] = df['Email'].astype(str)
#     # df['Username'] = df['Username'].astype(str)
#     df['Organization'] = df['Organization'].astype(str)
#     column_config = {
#         "Disabled": st.column_config.CheckboxColumn(
#             "Disabled",
#             disabled=False,
#         ),
#         "Name": st.column_config.TextColumn(
#             "Name",
#             disabled=False,
#         ),
#         "Email": st.column_config.TextColumn(
#             "Email",
#             disabled=True,
#         ),
#         "ID": st.column_config.TextColumn(
#             "ID",
#             disabled=True,
#         ),
#         # "Username": st.column_config.TextColumn(
#         #     "Username",
#         #     disabled=True,
#         # ),
#         "Level": st.column_config.SelectboxColumn(
#             "Level",
#             default=UserLevel.user.name,
#             options=[UserLevel.user.name, UserLevel.org_admin.name, UserLevel.super_admin.name] if is_super_admin() else [UserLevel.user.name, UserLevel.org_admin.name],
#             required=True,
#             disabled=False,
#         ),
#         "Organization": st.column_config.SelectboxColumn(
#             "Organization",
#             options=[org.id for org in orgs],
#             format_func=lambda x: next(org for org in orgs if org.id == x).organization_name,
#             disabled=False,
#         ),
#     }
#     edited_df = st.data_editor(df, column_config=column_config, use_container_width=True, hide_index=True)


#     # Filter out users own account and org_admin accounts
#     filtered_df = edited_df[~(edited_df["Email"] == st.session_state["username"]) & (edited_df["Level"] != UserLevel.super_admin.name)]

#     col1, col2 = st.columns([10, 1], vertical_alignment="bottom")
#     # Create a list of user titles for display
#     # Extract emails from the filtered dataframe
#     # Create a list of user titles incorporating emails
#     # Create a mapping of user IDs to user titles for display
#     user_id_title_map = {
#         record['ID']: f"{next(org.organization_name for org in orgs if org.id == record['Organization'])}: {record['Email']} - {record['Name']}"
#         for record in filtered_df.to_dict('records')
#     }
#     # Create a selectbox to choose a user to delete based on displayed title using format_func
#     delete_user_id = col1.selectbox(
#         "Select a user to delete",
#         options=list(user_id_title_map.keys()),
#         format_func=lambda x: user_id_title_map[x],
#         key="delete_user_id",
#     )

#     # Directly get the user's record from filtered_df using the selected user ID
#     selected_user_record = filtered_df.set_index('ID').loc[delete_user_id].to_dict()
#     organization_id = selected_user_record['Organization']
#     with col2.popover(":x:"):
#         if st.button(f"Are you sure you want to remove {user_id_title_map[delete_user_id]}?", key="delete_user_" + delete_user_id):
#             # Find the organization ID using the organization name
#             delete_user(delete_user_id, organization_id)
#             st.success(f"User {user_id_title_map[delete_user_id]} deleted successfully!")
#             st.rerun()

#     with st.container(border=True):
#         st.write("#### Add New User")
#         new_email = st.text_input("New User Email", key="new_user_email")
#         # new_username = st.text_input("New Username", key="new_username")
#         if is_super_admin():
#             new_org_id = st.selectbox(
#                 "Organization",
#                 options={org.id: org.organization_name for org in orgs},
#                 format_func=lambda x: orgs[x].organization_name,
#                 key="new_user_org_id"
#             )
#         else:
#             new_org_id = get_user_org_id()
#         new_level = st.selectbox(
#             "New User Level",
#             [UserLevel.user, UserLevel.org_admin],
#             key="new_user_level",
#         )
#         # new_journeys = st.text_input(
#         #     "New Journeys (comma separated)", key="new_journeys"
#         # )
#         if new_email and not is_valid_email(new_email):
#             st.warning("Please enter a valid email address.")
#         if st.button("Add new user"):
#             if new_email and new_org_id and new_level and is_valid_email(new_email):
#                 add_user(
#                     new_email,
#                     org_id=new_org_id,
#                     level=new_level,
#                     # journeys=new_journeys.split(",") if new_journeys else None,
#                 )
#             else:
#                 st.warning(
#                     "Please fill all the fields and enter a valid email address."
#                 )
#             st.success("User created successfully!")
#             st.rerun()

#     for index, row in edited_df.iterrows():
#         if row["Disabled"] != df.at[index, "Disabled"]:
#             set_disable_user(row["ID"], row["Disabled"])
#         # if row["Username"] != df.at[index, "Username"]:
#         #     if row["Username"] != df.at[index, "Username"]:
#         #         st.warning(f"Username for {row['ID']} cannot be changed.")
#         if row["Name"] != df.at[index, "Name"]:
#             set_user_name(row["ID"], row["Name"])
#         if row["Level"] != df.at[index, "Level"]:
#             set_user_level(row["ID"], UserLevel[row["Level"]])
#         if row["Organization"] != df.at[index, "Organization"]:
#             set_user_org(row["ID"], row["Organization"])
#         # You'll need to implement a function to handle changing journeys


def main():
    st.title("Organization and User Management")

    if init_sidebar(UserLevel.org_admin) != AuthStatus.LOGGED_IN:
        st.switch_page("login.py")
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
