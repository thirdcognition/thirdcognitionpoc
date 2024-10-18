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
from lib.models.user import (
    AuthStatus,
    UserLevel,
    add_user,
    check_auth_level,
    get_db_user,
    get_user_org,
    init_org_db,
    load_auth_config,
    set_user_name,
    write_auth_config,
)


class MyValidator(Validator):
    def validate_name(self, name: str = None) -> bool:
        return True

    def validate_password(self, password: str = None) -> bool:
        return True


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
        user = get_db_user(email=st.session_state["username"])
        org = get_user_org(user=user)
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
                org = get_user_org(user=user, reset=True)
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
                    message.error("Email/password is incorrect")
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


# def user_details():
#     # print(f"{('authentication_status' not in st.session_state)=}")
#     authenticator = get_authenticator("authentication_status" not in st.session_state)
#     auth_config = load_auth_config()
#     logged_in = False

#     message = st.empty()
#     if st.session_state.get("prev_login") is None:
#         tab1 = st.empty()
#         tab2 = st.empty()
#         tab3 = st.empty()
#     if st.session_state.get("authentication_status") is not True:
#         tab1, tab2 = st.tabs(["Login", "Register"])
#         tab3 = st.empty()
#     else:
#         tab1, tab2, tab3 = st.tabs(["Logout", "My details", "Reset Password"])

#     with tab1:
#         authenticator.login()

#     if st.session_state.get("authentication_status"):
#         tab1.write(f'Welcome *{st.session_state["name"]}*')
#         with tab2:
#             update_user_details(st.session_state["username"])
#         with tab3:
#             reset_password()
#         with tab1:
#             authenticator.logout("Logout")
#         logged_in = True
#     else:
#         if st.session_state.get("authentication_status") is False:
#             message.error("Username/password is incorrect")
#         # if st.session_state.get("authentication_status") is None:
#         #     message.warning("Please enter your username and password")

#         with tab2:
#             register_user()

#         logged_in = False

#     write_auth_config(auth_config)
#     return logged_in


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
    if cookie["exp_date"] < datetime.datetime.now(datetime.timezone.utc).timestamp():
        # print("Cookie expired")
        authenticator.cookie_controller.cookie_model.cookie_manager.delete(
            auth_config["cookie"]["name"]
        )
        return False
    else:
        db_user = get_db_user(cookie["username"])
        # print("Cookie valid")
        st.session_state["username"] = cookie["username"]
        st.session_state["name"] = db_user.name if db_user else ""
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
        try:
            authenticator.cookie_controller.delete_cookie()
        except Exception as e:
            print(e)
        # authenticator.cookie_controller.cookie_model.cookie_manager.delete(auth_config["cookie"]["name"])

    st.session_state["username"] = username_of_registered_user
    st.session_state["name"] = name_of_registered_user

    token = jwt.encode(
        {
            "username": username_of_registered_user,
            "exp_date": (
                datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
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
        st.session_state["auth_state"] = "login"
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

    try:
        authenticator.authentication_controller.authentication_model.credentials[
            "usernames"
        ][st.session_state["username"]]["logged_in"] = False
    except:
        print("Failed to reset logged_in")
    st.session_state["logout"] = True
    st.session_state["name"] = None
    st.session_state["username"] = None
    st.session_state["authentication_status"] = None


def complete_registration(data):
    auth_config = load_auth_config(reload=True)
    authenticator = get_authenticator()
    name_of_registered_user = data["new_name"] + " " + data["new_last_name"]
    username_of_registered_user = data["new_username"]
    email_of_registered_user = data["new_email"]
    manual_login(authenticator, username_of_registered_user, name_of_registered_user)
    password = authenticator.authentication_controller.authentication_model.credentials[
        "usernames"
    ][username_of_registered_user]["password"]
    # hashed_password = stauth.Hasher([password]).generate()
    add_user(
        email=email_of_registered_user, name=name_of_registered_user, password=password, register=True
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
            merge_username_email=True,
            callback=complete_registration,
        )
    except stauth.RegisterError as e:
        print(e)
        st.session_state["auth_dirty"] = True
        no_error = False
        error_container.error(e)
    # password = authenticator.credentials["usernames"][username_of_registered_user]["password"]

    if no_error and email_of_registered_user:
        write_auth_config(auth_config)
        complete_registration(
            {
                "new_name": name_of_registered_user.split(" ")[0],
                "new_last_name": " ".join(name_of_registered_user.split(" ")[1:]),
                "new_username": username_of_registered_user,
                "new_email": email_of_registered_user,
            }
        )

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
        # new_username = st.text_input("Username", value=user.email)
        new_name = st.text_input("Real Name", value=user.name)
        if st.button("Update"):
            # set_user_org(user.email, new_org_id)
            auth_config = load_auth_config()
            auth_config["credentials"]["usernames"][user.email] = auth_config[
                "credentials"
            ]["usernames"].pop(user.email)
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
            auth_config["credentials"]["usernames"][user.email]["password"] = (
                hashed_password[0]
            )
            if write_auth_config(auth_config):
                st.success("Password reset successfully!")
