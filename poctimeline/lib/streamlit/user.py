import datetime
from enum import Enum
import hashlib
import jwt
import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities import Validator
from streamlit_extras.stylable_container import stylable_container
import yaml

from admin.global_styles import get_theme
from lib.helpers.shared import pretty_print
from lib.load_env import SETTINGS
from lib.models.user import (
    AuthStatus,
    UserLevel,
    add_user,
    check_auth_level,
    get_all_orgs,
    get_all_users,
    get_db_user,
    get_user_org,
    init_org_db,
    is_org_admin,
    is_super_admin,
    load_auth_config,
    remove_preauth_email,
    set_disable_user,
    set_user_org,
    set_user_name,
    write_auth_config,
)


class MyValidator(Validator):
    def validate_name(self, name: str = None) -> bool:
        return True

    def validate_password(self, password: str = None) -> bool:
        return True


def add_preregistered_email_ui():
    # Check if the user is a super_admin or an org_admin
    if is_super_admin() or is_org_admin():
        # If the user is a super_admin, show a select box for organizations
        if is_super_admin():
            orgs = get_all_orgs()
            org_ids = [org.organization_id for org in orgs]
            selected_org_id = st.selectbox("Select an organization", org_ids)
        # If the user is an org_admin, use their organization
        else:
            selected_org_id = get_user_org().organization_id

        # Show a text input for the email and username
        email = st.text_input("Email")
        username = st.text_input("Username")

        # Show a select box for the user level
        user_levels = [UserLevel.user, UserLevel.org_admin]
        selected_user_level = st.selectbox("User Level", user_levels)

        # Show a button to add the user
        if st.button("Add User"):
            # Add the user to the UserDataTable
            add_user(
                email,
                username,
                selected_org_id,
                level=selected_user_level,
                disabled=False,
            )
            st.success("User added successfully!")
    else:
        st.error("Not enough access: Only super_admin and org_admin can add users.")


def list_all_preregistered_emails(org_id=None):
    # Check if the user is a super_admin or an org_admin
    if is_super_admin() or is_org_admin():
        # If the user is a super_admin and no org_id is provided, show a select box for organizations
        if is_super_admin() and org_id is None:
            orgs = get_all_orgs()
            org_ids = [org.organization_id for org in orgs]
            selected_org_id = st.selectbox("Select an organization", org_ids)
        # If the user is an org_admin or an org_id is provided, use that organization
        else:
            selected_org_id = org_id if org_id else get_user_org().organization_id

        # Load the auth configuration
        auth_config = load_auth_config()

        # Get the pre-authorized emails for the selected organization
        preauthorized_emails = auth_config["pre-authorized"]["emails"]

        # Get all users for the selected organization
        users = get_all_users(selected_org_id)

        # Filter the users to only include those with pre-authorized emails
        filtered_emails = [
            user.email for user in users if user.email in preauthorized_emails
        ]
        # Show the filtered emails in a table
        st.write("Pre-authorized emails:")
        for email in filtered_emails:
            st.write(email)

            # Show a button to remove the email
            if st.button(f"Remove {email}"):
                # Remove the email from the pre-authorized list
                remove_preauth_email(email)

                st.success("Email removed successfully!")
    else:
        st.error(
            "Not enough access: Only super_admin and org_admin can view pre-registered emails."
        )


def list_all_users(org_id=None):
    # Check if the user is a super_admin or an org_admin
    if is_super_admin() or is_org_admin():
        # If the user is a super_admin and no org_id is provided, show a select box for organizations
        if is_super_admin() and org_id is None:
            orgs = get_all_orgs()
            org_ids = [org.organization_id for org in orgs]
            selected_org_id = st.selectbox("Select an organization", org_ids)
        # If the user is an org_admin or an org_id is provided, use that organization
        else:
            selected_org_id = org_id if org_id else get_user_org().organization_id

        # Get all users for the selected organization
        users = get_all_users(selected_org_id)

        # Show the users in a table
        st.write("Users:")
        for user in users:
            st.write(
                f"{user.email} - {user.username} - {user.level} - {user.organization_id} - {user.disabled}"
            )

            # Show a button to edit the user
            if st.button(f"Edit {user.email}"):
                update_user_details(user.email)

            # Show a button to disable/enable the user
            if user.disabled:
                if st.button(f"Enable {user.email}"):
                    set_disable_user(user.email, False)
            else:
                if st.button(f"Disable {user.email}"):
                    set_disable_user(user.email, True)
    else:
        st.error("Not enough access: Only super_admin and org_admin can view users.")


@st.fragment
def get_login_state():
    theme = get_theme()
    with stylable_container(
        key="login_selector",
        css_styles=[
            """
        button {
            border: None;
        }
        """,
            """
        button:not(:disabled):hover {
            border: None;
            background-color: red;
            color: #fff;
        }
        """,
            f"""
        button:disabled {{
            background-color: {"#2e2e2e" if theme["base"] == "dark" else "#eee"};
            color: {"#fff" if theme["base"] == "dark" else "#000"};
        }}
        """,
        ],
    ):
        tab1, tab2 = st.columns([1, 1])
        login_state = st.session_state.get("auth_state", "login")
        if st.session_state.get("username") is None:
            if tab1.button(
                "I have an account",
                key="auth_state_login",
                use_container_width=True,
                disabled=login_state == "login",
            ):
                login_state = "login"
            if tab2.button(
                "I need to register",
                key="auth_state_register",
                use_container_width=True,
                disabled=login_state == "register",
            ):
                login_state = "register"
        else:
            login_state = "login"
        if login_state != st.session_state.get("auth_state"):
            st.session_state["auth_state"] = login_state
            st.rerun()
        return login_state


def get_authenticator(reset=False, reload=False) -> stauth.Authenticate:
    auth_config = load_auth_config(reset=reset, reload=reload)
    authenticator: stauth.Authenticate = st.session_state.get(
        "st_auth"
    )  # get_authenticator()
    # auth_config_str = str(auth_config)
    # auth_config_hashed = (
    #     st.session_state.get("auth_config_hash")
    #     or hashlib.md5(auth_config_str.encode()).hexdigest()
    # )

    if reset or reload or authenticator is None:
        new_authenticator: stauth.Authenticate = (
            authenticator
            if isinstance(authenticator, stauth.Authenticate)
            else stauth.Authenticate(
                auth_config["credentials"],
                auth_config["cookie"]["name"],
                auth_config["cookie"]["key"],
                auth_config["cookie"]["expiry_days"],
                validator=MyValidator(),
            )
        )
        if authenticator != new_authenticator and new_authenticator is not None:
            authenticator = new_authenticator
            st.session_state["st_auth"] = authenticator
        return authenticator
    else:
        return authenticator


def check_auth(
    user_level: UserLevel = UserLevel.user, reset: bool = False, container=None
) -> AuthStatus:
    cur_user_level = st.session_state.get("user_level")
    authenticator = get_authenticator("authentication_status" not in st.session_state)
    if authenticator:
        get_manual_login(authenticator)
    # print(f"Authenticator: {authenticator}")
    if (
        user_level is not None
        and st.session_state.get("username") is not None
        and not reset
        and authenticator
    ):
        # print(f"check_auth: {st.session_state.get('username', 'username none')} - {cur_user_level} - {user_level}")
        user = get_db_user(st.session_state["username"])
        org = get_user_org(st.session_state["username"])
        if user is not None and org is not None:
            with st.sidebar:
                message = st.columns([2, 1], vertical_alignment="bottom")
                message[0].write(
                    f'{org.organization_name if org is not None else ""}\n\nWelcome *{user.name}*'
                )

                with message[1]:
                    authenticator.logout("Logout")
                auth_status = (
                    AuthStatus.NO_ACCESS
                    if check_auth_level() < user_level
                    else AuthStatus.LOGGED_IN
                )
            return auth_status

    auth_config = load_auth_config()
    init_org_db()
    # with st.sidebar:

    container = st.sidebar if container is None else container.container()

    with container:
        message = st.empty()
        login_container = st.empty()

        if not st.session_state.get("authentication_status"):
            if authenticator:
                get_manual_login(authenticator)

            # tab1, tab2 = login_container.tabs(["Login", "Register"])
            login_state = get_login_state()
            container = st.container()

            if login_state == "login":
                # with tab1:
                try:
                    authenticator.login()
                    if st.session_state["authentication_status"]:
                        manual_login(
                            authenticator,
                            st.session_state["username"],
                            st.session_state["name"],
                        )
                except Exception as e:
                    if "There are multiple" not in str(e):
                        manual_logout(authenticator)
                        authenticator.login()
            # with tab2:
            else:
                if st.session_state.get("username") is None:
                    register_user()
                else:
                    st.write("Already logged in")

            write_auth_config(auth_config)
            # st.rerun()

            if st.session_state.get("authentication_status"):
                login_container.empty()
                user = get_db_user(st.session_state["username"], reset=True)
                org = get_user_org(st.session_state["username"], reset=True)
                if user.disabled or org.disabled:
                    print("Logout")
                    manual_logout(authenticator)
                    st.rerun()

                message = st.columns([2, 1], vertical_alignment="bottom")
                message[0].write(
                    f'{org.organization_name if org is not None else ""}\n\nWelcome *{st.session_state["name"]}*'
                )

                with message[1]:
                    authenticator.logout("Logout")
                cur_user_level = check_auth_level()
                auth_status = (
                    AuthStatus.NO_ACCESS
                    if cur_user_level < user_level
                    else AuthStatus.LOGGED_IN
                )
            else:
                if st.session_state.get("authentication_status") is False:
                    message.error("Username/password is incorrect")
                    auth_status = AuthStatus.NO_LOGIN
                if st.session_state.get("authentication_status") is None:
                    message.info(
                        "Please login if you have an account or register if you have a registered email address."
                    )
                    auth_status = AuthStatus.NO_LOGIN
                else:
                    cur_user_level = check_auth_level()
                    auth_status = (
                        AuthStatus.NO_ACCESS
                        if cur_user_level < user_level
                        else AuthStatus.LOGGED_IN
                    )

        st.session_state["user_level"] = cur_user_level
        return auth_status


def user_details():
    # print(f"{('authentication_status' not in st.session_state)=}")
    authenticator = get_authenticator("authentication_status" not in st.session_state)
    auth_config = load_auth_config()
    logged_in = False

    message = st.empty()
    if st.session_state.get("prev_login") is None:
        tab1 = st.empty()
        tab2 = st.empty()
        tab3 = st.empty()
    if st.session_state.get("authentication_status") is not True:
        tab1, tab2 = st.tabs(["Login", "Register"])
        tab3 = st.empty()
    else:
        tab1, tab2, tab3 = st.tabs(["Logout", "My details", "Reset Password"])

    with tab1:
        authenticator.login()

    if st.session_state.get("authentication_status"):
        tab1.write(f'Welcome *{st.session_state["name"]}*')
        with tab2:
            update_user_details(st.session_state["username"])
        with tab3:
            reset_password()
        with tab1:
            authenticator.logout("Logout")
        logged_in = True
    else:
        if st.session_state.get("authentication_status") is False:
            message.error("Username/password is incorrect")
        # if st.session_state.get("authentication_status") is None:
        #     message.warning("Please enter your username and password")

        with tab2:
            register_user()

        logged_in = False

    write_auth_config(auth_config)
    return logged_in


def get_manual_login(authenticator: stauth.Authenticate):
    auth_config = load_auth_config()
    # try:
    token: str = authenticator.cookie_controller.cookie_model.cookie_manager.get(
        auth_config["cookie"]["name"]
    )
    if token is None:
        # print("token is none")
        return False
    # print(f"{token=}")
    cookie = jwt.decode(
        token.encode(),
        auth_config["cookie"]["key"],
        algorithms=["HS256"],
    )
    # print(f"{cookie=}")
    if cookie["exp_date"] < datetime.datetime.utcnow().timestamp():
        # print("Cookie expired")
        authenticator.cookie_controller.cookie_model.cookie_manager.delete(
            auth_config["cookie"]["name"]
        )
        return False
    else:
        # print("Cookie valid")
        st.session_state["username"] = cookie["username"]
        st.session_state["name"] = get_db_user(cookie["username"]).name
        st.session_state["authentication_status"] = True
        return True
    # except Exception as e:
    #     print(e)


def manual_login(
    authenticator: stauth.Authenticate,
    username_of_registered_user: str,
    name_of_registered_user: str,
):
    # print("Manual login")
    auth_config = load_auth_config()
    # token = authenticator.cookie_controller.cookie_model.cookie_manager.get(auth_config["cookie"]["name"])
    token = authenticator.cookie_controller.get_cookie()

    if token is not None:
        # print("Cookie already set")
        authenticator.cookie_controller.delete_cookie()
        # authenticator.cookie_controller.cookie_model.cookie_manager.delete(auth_config["cookie"]["name"])

    st.session_state["username"] = username_of_registered_user
    st.session_state["name"] = name_of_registered_user

    token = jwt.encode(
        {
            "username": username_of_registered_user,
            "exp_date": (
                datetime.datetime.utcnow() + datetime.timedelta(days=30)
            ).timestamp(),
        },
        auth_config["cookie"]["key"],
        algorithm="HS256",
    )
    try:
        authenticator.cookie_controller.cookie_model.cookie_manager.set(
            auth_config["cookie"]["name"],
            token,
            expires_at=datetime.datetime.now() + datetime.timedelta(days=30),
            key="manual_login_" + username_of_registered_user,
        )
        st.session_state["authentication_status"] = True
        st.session_state["auth_state"] = 'login'
    except Exception as e:
        print("cookie set error", e)


def manual_logout(authenticator: stauth.Authenticate):
    print("Manual logout")
    auth_config = load_auth_config()
    try:
        authenticator.cookie_controller.cookie_model.cookie_manager.delete(
            auth_config["cookie"]["name"]
        )
    except Exception as e:
        print(e)

    authenticator.authentication_controller.authentication_model.credentials["usernames"][st.session_state["username"]]["logged_in"] = False
    st.session_state["logout"] = True
    st.session_state["name"] = None
    st.session_state["username"] = None
    st.session_state["authentication_status"] = None


def complete_registration(data):
    # print("Complete registration", data)
    # callback({'widget': 'Register user', 'new_name': new_first_name,
    #           'new_last_name': new_last_name, 'new_email': new_email,
    #           'new_username': new_username})
    auth_config = load_auth_config(reload=True)
    authenticator = get_authenticator()
    name_of_registered_user = data["new_name"] + " " + data["new_last_name"]
    username_of_registered_user = data["new_username"]
    email_of_registered_user = data["new_email"]
    manual_login(
        authenticator, username_of_registered_user, name_of_registered_user
    )
    password = authenticator.authentication_controller.authentication_model.credentials["usernames"][username_of_registered_user]["password"]
    # hashed_password = stauth.Hasher([password]).generate()
    add_user(
        email_of_registered_user,
        username_of_registered_user,
        name_of_registered_user
    )
    write_auth_config(auth_config)


def register_user():
    # Using st_auth register new user and reflect changes to database
    authenticator: stauth.Authenticate = get_authenticator(
        reload=st.session_state.get("auth_dirty", False)
    )
    auth_config = load_auth_config(reload=st.session_state.get("auth_dirty", False))

    st.session_state["auth_dirty"] = False
    no_error = True
    error_container = st.empty()
    try:
        (
            email_of_registered_user,
            username_of_registered_user,
            name_of_registered_user,
        ) = authenticator.register_user(
            pre_authorized=auth_config["pre-authorized"]["emails"],
            captcha=False,
            key="user_registration",
            fields={
                "Form name": "Register",
            },
            callback=complete_registration
        )
    except stauth.RegisterError as e:
        print(e)
        st.session_state["auth_dirty"] = True
        no_error = False
        error_container.error(e)
    # password = authenticator.credentials["usernames"][username_of_registered_user]["password"]

    if no_error and email_of_registered_user:
        write_auth_config(auth_config)
        complete_registration({
            "new_name": name_of_registered_user.split(" ")[0],
            "new_last_name": " ".join(name_of_registered_user.split(" ")[1:]),
            "new_username": username_of_registered_user,
            "new_email": email_of_registered_user
        })

        # auth_config["credentials"]["usernames"][username_of_registered_user] = {
        #     "email": email_of_registered_user,
        #     "name": username_of_registered_user,
        #     "password": hashed_password[0],
        # }
        # with open(SETTINGS.auth_filename, "w", encoding="utf-8") as file:
        #     yaml.dump(auth_config, file, default_flow_style=False)
        # st.success("User registered successfully!")


def update_user_details(user_id):
    # Using st_auth update user details and reflect changes to database
    user = get_db_user(user_id)
    if user is None:
        error_message = "User not found."
        st.error(error_message)
        return

    container = st.container()

    with container:
        new_username = st.text_input("Username", value=user.username)
        new_name = st.text_input("Real Name", value=user.name)
        if st.button("Update"):
            # set_user_org(user.email, new_org_id)
            auth_config = load_auth_config()
            auth_config["credentials"]["usernames"][new_username] = auth_config[
                "credentials"
            ]["usernames"].pop(user.username)
            set_user_name(user.email, new_name)
            if write_auth_config(auth_config):
                st.success("User details updated successfully!")


def reset_password():
    container = st.container()

    with container:
        # Using st_auth reset user password and reflect changes to database
        email = st.text_input("Email")
        new_password = st.text_input("New Password", type="password")

        if st.button("Reset Password"):
            user = get_db_user(email)
            if user is None:
                st.error("User not found.")
                return

            hashed_password = stauth.Hasher([new_password]).generate()
            auth_config = load_auth_config()
            auth_config["credentials"]["usernames"][user.username]["password"] = (
                hashed_password[0]
            )
            if write_auth_config(auth_config):
                st.success("Password reset successfully!")
