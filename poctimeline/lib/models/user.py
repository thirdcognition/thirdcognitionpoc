import datetime
from enum import Enum
from functools import cache
import hashlib
import os
from typing import List
import uuid
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import sqlalchemy as sqla
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import Session

from lib.db.sqlite import Base, db_commit, db_session, init_system_db
from lib.helpers.shared import get_id_str, pretty_print
from lib.load_env import SETTINGS


class AuthStatus(Enum):
    LOGGED_IN = 1
    NO_LOGIN = 2
    NO_ACCESS = 3


class UserLevel(Enum):
    super_admin = 1000
    org_admin = 500
    manager = 200
    user = 100
    anonymous = 50

    def __lt__(self, other):
        if isinstance(other, UserLevel):
            return self.value < other.value
        elif isinstance(other, int):
            return self.value < other
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, UserLevel):
            return self.value > other.value
        elif isinstance(other, int):
            return self.value > other
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, UserLevel):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        elif isinstance(other, UserDataTable):
            return self.value == other.level
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, UserLevel):
            return self.value >= other.value
        elif isinstance(other, int):
            return self.value >= other
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, UserLevel):
            return self.value <= other.value
        elif isinstance(other, int):
            return self.value <= other
        return NotImplemented


class UserDataTable(Base):
    __tablename__ = SETTINGS.users_tablename

    id = sqla.Column(sqla.String, primary_key=True)
    email = sqla.Column(sqla.String, unique=True)
    username = sqla.Column(sqla.String, nullable=True)  # same as email
    name = sqla.Column(sqla.String, nullable=True)
    password = sqla.Column(sqla.String, nullable=True)
    level = sqla.Column(sqla.Integer, default=100)
    organization_id = sqla.Column(sqla.String)
    disabled = sqla.Column(sqla.Boolean, default=False)
    failed_login_attempts = sqla.Column(sqla.Integer, default=0)
    logged_in = sqla.Column(sqla.Boolean, default=False)
    registered_at = sqla.Column(sqla.DateTime, default=None)


class OrganizationDataTable(Base):
    __tablename__ = SETTINGS.organizations_tablename

    # id = Column(Integer, primary_key=True)
    id = sqla.Column(sqla.String, primary_key=True)
    organization_name = sqla.Column(sqla.String)
    created_at = sqla.Column(sqla.DateTime, default=None)
    db_name = sqla.Column(sqla.String)
    disabled = sqla.Column(sqla.Boolean, default=False)


def init_org_db():
    session = init_system_db()
    # Check if the user is already in the UserDataTable
    users = session.query(UserDataTable).all()
    user_count = len(users)

    # print("User len", user_count)
    # pretty_print([user.__dict__ for user in users], force=True)

    if user_count == 0:
        # If the user does not exist, add them as an admin
        users = []
        for user in SETTINGS.super_admin:
            print(f"Adding super admin: {user=}")
            new_user = UserDataTable(
                id=str(uuid.uuid4()),
                email=user[0],
                username=user[0],
                name=None,  # Assuming name is provided in the SETTINGS.super_admin list
                password=None,
                level=UserLevel.super_admin.value,
                organization_id=SETTINGS.default_organization[0],
                disabled=False,
            )
            # create_preauth_email(user[0])
            session.add(new_user)
        session.commit()

    # Check if the organization exists in the OrganizationDataTable
    org_exists = (
        session.query(OrganizationDataTable)
        .filter_by(id=SETTINGS.default_organization[0])
        .first()
    )

    if not org_exists:
        # If the organization does not exist, add it to the OrganizationDataTable
        third_cognition_org = {
            "id": SETTINGS.default_organization[0],
            "organization_name": SETTINGS.default_organization[1],
            "db_name": SETTINGS.default_organization[
                2
            ],  # Replace with the actual database name
            "disabled": False,
        }
        new_org = OrganizationDataTable(**third_cognition_org)
        session.add(new_org)
        session.commit()


def user_db_get_session() -> Session:
    init_system_db()
    if st.session_state.get("username"):
        organization = get_user_org(email=st.session_state.get("username"))
        return db_session(organization.db_name) if organization else None


def get_user_chroma_path():
    init_system_db()
    if st.session_state.get("username"):
        organization = get_user_org(email=st.session_state.get("username"))
        return os.path.join(
            SETTINGS.db_path, organization.db_name, SETTINGS.chroma_path
        )


def user_db_commit():
    init_system_db()
    if st.session_state.get("username"):
        organization = get_user_org(email=st.session_state.get("username"))
        return db_commit(organization.db_name)


def write_auth_config(auth_config: dict) -> bool:
    session = init_system_db()

    # Get current user data from database
    current_users = session.query(UserDataTable).all()

    # Create a dictionary of current user data for easy lookup
    current_user_data = {user.email: user for user in current_users}

    # Initialize a flag to track if there have been any changes
    changes_made = False

    # Iterate over the user data in auth_config
    for username, user_data in auth_config["credentials"]["usernames"].items():
        # If the user is not in the database, add them
        email = user_data["email"] or username
        if email not in current_user_data:
            new_user = UserDataTable(
                email=email,
                username=email,
                name=user_data.get("name"),
                password=user_data.get("password"),
                failed_login_attempts=user_data.get("failed_login_attempts"),
                logged_in=user_data.get("logged_in"),
            )
            session.add(new_user)
            changes_made = True
        # If the user is in the database, check if their data has changed
        else:
            current_user = current_user_data[email]
            if (
                current_user.name != user_data.get("name", current_user.name)
                or current_user.password
                != user_data.get("password", current_user.password)
                or current_user.failed_login_attempts
                != user_data.get(
                    "failed_login_attempts", current_user.failed_login_attempts
                )
                or current_user.logged_in
                != user_data.get("logged_in", current_user.logged_in)
            ):
                # If the user data has changed, update the database
                current_user.name = user_data.get("name", current_user.name)
                current_user.password = user_data.get("password", current_user.password)
                current_user.failed_login_attempts = user_data.get(
                    "failed_login_attempts", current_user.failed_login_attempts
                )
                current_user.logged_in = user_data.get(
                    "logged_in", current_user.logged_in
                )
                changes_made = True

    # Iterate over the pre-authorized emails in auth_config
    for email in auth_config["pre-authorized"]["emails"]:
        # If the email is not in the database, add it as an empty user
        if email not in current_user_data:
            new_user = UserDataTable(email=email)
            session.add(new_user)
            changes_made = True

    # If there have been any changes, commit them to the database
    if changes_made:
        session.commit()

    # Update the session state with the new auth_config
    st.session_state["auth_config"] = auth_config

    return changes_made


def load_auth_config(reset=False, reload=False):
    if st.session_state.get("auth_config") is not None and not reset and not reload:
        return st.session_state["auth_config"]
    if reset and st.session_state.get("auth_config") is not None:
        del st.session_state["auth_config"]
        if st.session_state.get("st_auth"):
            del st.session_state["st_auth"]

    auth_config: dict = st.session_state.get("auth_config") if reload else None

    session = init_system_db()
    users = session.query(UserDataTable).all()
    new_auth_config = {
        "credentials": {"usernames": {}},
        "cookie": {
            "expiry_days": SETTINGS.auth_cookie_expiry,
            "key": SETTINGS.auth_cookie_secret,  # Must be string
            "name": SETTINGS.auth_cookie_key,
        },
        "pre-authorized": {"emails": []},
    }
    if auth_config is not None:
        auth_config.update(new_auth_config)
    else:
        auth_config = new_auth_config

    for user in users:
        if user.registered_at is not None:
            auth_config["credentials"]["usernames"][user.email] = {
                "email": user.email,
                "failed_login_attempts": user.failed_login_attempts,
                "logged_in": user.logged_in,
                "name": user.name,
                "password": user.password,  # Password should be hashed
            }
        elif user.email:
            auth_config["pre-authorized"]["emails"].append(user.email)

    write_auth_config(auth_config)

    # pretty_print([user.__dict__ for user in users], force=True)
    # pretty_print(auth_config, force=True)

    # if len(users) == 0:
    #     for user in SETTINGS.super_admin:
    #         create_preauth_email(user[0])

    return auth_config


# authenticator: stauth.Authenticate = None


# def get_authenticator():
#     authenticator = st.session_state.get("st_auth")
#     return authenticator


# def set_authenticator(auth: stauth.Authenticate):
#     st.session_state["st_auth"] = auth


# super_admin can access all, org_admin can access all in their org
def get_all_users(org_id: str = None, reset=False) -> List[UserDataTable]:
    if reset or f"db_users_{org_id}" not in st.session_state:
        st.session_state[f"db_users_{org_id}"] = None

    if st.session_state[f"db_users_{org_id}"] is not None:
        return st.session_state[f"db_users_{org_id}"]

    session = init_system_db()
    if is_super_admin() and org_id is None:
        # If the user is a super_admin, return all users
        users = session.query(UserDataTable).all()
    elif is_org_admin() or (is_super_admin() and org_id is not None):
        org_id = org_id or get_user_org_id()
        if org_id is not None:
            # If the user is an org_admin, return all users in their org
            users = session.query(UserDataTable).filter_by(organization_id=org_id).all()
    else:
        # If the user is not a super_admin or an org_admin, return an empty list
        users = []

    st.session_state[f"db_users_{org_id}"] = users

    return users


def get_all_orgs(reset=False) -> List[OrganizationDataTable]:
    if reset or "db_orgs" not in st.session_state:
        st.session_state["db_orgs"] = None

    if st.session_state["db_orgs"] is not None:
        return st.session_state["db_orgs"]

    session = init_system_db()
    if is_super_admin():
        # If the user is a super_admin, return all organizations
        orgs = session.query(OrganizationDataTable).all()
    else:
        # If the user is not a super_admin, return a list with their organization
        orgs = [get_user_org(email=st.session_state["username"])]

    st.session_state["db_orgs"] = orgs

    return orgs


# @cache
def get_org_by_id(org_id: str, reset=False) -> OrganizationDataTable:
    # if reset:
    #     get_org_by_id.cache_clear()

    session = init_system_db()
    # Query OrganizationDataTable for the organization
    org_data = session.query(OrganizationDataTable).filter_by(id=org_id).first()

    return org_data


def check_auth_level(email: str = None) -> UserLevel:
    if (email or st.session_state.get("username")) is None:
        return UserLevel.anonymous

    # Query UserDataTable for the user
    user_data = get_db_user(email or st.session_state["username"])
    # Check if the user exists in the UserDataTable
    if user_data:
        # If the user exists, return their level
        return UserLevel(user_data.level)
    else:
        # If the user does not exist, return anonymous level
        return UserLevel.anonymous


def is_super_admin() -> bool:
    return check_auth_level() == UserLevel.super_admin


def is_org_admin() -> bool:
    return check_auth_level() == UserLevel.org_admin


def is_user() -> bool:
    return check_auth_level() == UserLevel.user


# @cache
def get_db_user(
    email: str = None, id: str = None, reset: bool = False
) -> UserDataTable:
    # if reset:
    #     get_db_user.cache_clear()

    if email is None and id is None:
        raise ValueError("Either email or id must be provided")

    session = init_system_db()
    user_data = None
    # Query UserDataTable for the user
    if id is not None:
        user_data = session.query(UserDataTable).filter_by(id=id).first()

    if email is not None and user_data is None:
        user_data = session.query(UserDataTable).filter_by(email=email).first()

    return user_data


# @cache
def get_user_org(
    email: str = None, id: str = None, user: UserDataTable = None, reset: bool = False
) -> OrganizationDataTable:
    # if reset:
    #     get_user_org.cache_clear()

    if user is not None:
        id = user.id

    if email is None and id is None:
        raise ValueError("Either email or id must be provided")

    session = init_system_db()
    # Query UserDataTable for the user
    # Query UserDataTable and OrganizationDataTable in one query
    join = session.query(OrganizationDataTable).join(
        UserDataTable,
        UserDataTable.organization_id == OrganizationDataTable.id,
    )
    org_data = None
    if id is not None:
        org_data = join.filter(UserDataTable.id == id).first()
    elif email is not None:
        org_data = join.filter(UserDataTable.email == email).first()

    return org_data


def get_user_org_id(email: str = None, id: str = None) -> str:
    if email is None:
        email = st.session_state.get("username")
    if email is None and id is None:
        return None
    org = get_user_org(email=email, id=id)
    return org.id if org is not None else None


def has_access(
    email=None, user_id=None, org_id=None, user_level: UserLevel = UserLevel.user
):
    cur_user_level = check_auth_level(st.session_state.get("username"))
    if user_level > cur_user_level:
        raise Exception("User level too low")

    cur_user = get_db_user(email=st.session_state.get("username", email), id=user_id)
    cur_org = get_user_org(id=cur_user.id) if cur_user is not None else None

    if is_super_admin():
        pass
    elif is_org_admin() and not (org_id is None or cur_org.id == org_id):
        print(repr(cur_org.id), repr(org_id))
        raise Exception("Not enough access: Org admin can only modify their org")
    elif is_user() and not (email is None or cur_user.email == email):
        raise Exception("Not enough access: User can only modify their own account")

        # raise Exception("Not enough access: Anonymous user can't modify users")


def add_user(
    email: str,
    name: str = None,
    org_id: str = None,
    level: UserLevel = None,
    disabled: bool = False,
    password: str = False,
    register: bool = False,
) -> UserDataTable:
    session = init_system_db()
    # Check if the user is already in the UserDataTable

    cur_user = get_db_user(email=st.session_state.get("username"))
    cur_org = get_user_org(id=cur_user.id) if cur_user is not None else None
    org_id = org_id or (cur_org.id if cur_org else None)

    user = get_db_user(email=email)

    if org_id is None and (cur_user is not None and email != cur_user.email):
        has_access(user_level=UserLevel.super_admin)
    if cur_user is not None:
        has_access(email, org_id=org_id)
    # org admin can only modify users in their org

    if user:
        # If the user exists, check for changes and update their details
        updated = False
        if email is not None and user.username != email:
            user.username = email
            updated = True
        if level is not None and user.level != level.value:
            if not isinstance(level, UserLevel):
                level = UserLevel(level)
            user.level = level.value if level else None
            updated = True
        if org_id is not None and user.organization_id != org_id:
            user.organization_id = org_id
            updated = True
        if name is not None and user.name != name:
            user.name = name
            updated = True
        if disabled is not None and user.disabled != disabled:
            user.disabled = disabled
            updated = True
        if password is not None and user.password != password:
            user.password = password
            updated = True
        if register and user.registered_at is None:
            user.registered_at = datetime.datetime.now(datetime.timezone.utc)
            updated = True
        if updated:
            session.commit()
    else:
        has_access(org_id=org_id, user_level=UserLevel.org_admin)
        if level is not None and not isinstance(level, UserLevel):
            level = UserLevel(level)
        # If the user does not exist, add them to the UserDataTable
        user = UserDataTable(
            id=str(uuid.uuid4()),
            email=email,
            username=email,
            name=name,
            level=(level or UserLevel.user).value,
            organization_id=org_id,
            disabled=disabled,
            registered_at=datetime.datetime.now(datetime.timezone.utc) if register else None
        )
        session.add(user)
        session.commit()
        # create_preauth_email(email)
    get_all_users(reset=True)

    return user
    # get_db_user.cache_clear()


def delete_user(id: str, org_id: str):
    session = init_system_db()

    # Query UserDataTable for the user
    user = get_db_user(id=id)
    email = user.email
    has_access(org_id=org_id, user_level=UserLevel.org_admin)

    if user:
        # If the user exists, delete them from the UserDataTable
        session.delete(user)
        session.commit()

        # Remove the user's preauthorized email
        remove_preauth_email(email)

        # Remove the user's username from the auth_config
        auth_config = load_auth_config()
        if email in auth_config["credentials"]["usernames"]:
            del auth_config["credentials"]["usernames"][email]
            write_auth_config(auth_config)

        get_all_users(reset=True)
    else:
        print(f"User with id {id} does not exist.")


def set_user_org(id: str, org_id: str):
    session = init_system_db()

    has_access(id, org_id)

    # Query UserDataTable for the user
    user = get_db_user(id=id)

    if user:
        # If the user exists, update their organization_id
        # Check if the organization exists in the OrganizationDataTable
        org = session.query(OrganizationDataTable).filter_by(id=org_id).first()
        if org:
            user.organization_id = org_id
            session.commit()
            get_all_users(reset=True)
            # get_db_user.cache_clear()
        else:
            print(f"Organization with ID {org_id} does not exist.")
    else:
        print(f"User with id {id} does not exist.")


def set_user_name(id: str, name: str):
    session = init_system_db()

    has_access(user_id=id)

    # Query UserDataTable for the user
    user = get_db_user(id=id)

    if user:
        # If the user exists, update their name
        user.name = name
        session.commit()
    else:
        print(f"User with id {id} does not exist.")


def set_user_level(id: str, level: UserLevel):
    session = init_system_db()

    # Query UserDataTable for the user
    user = get_db_user(id=id)
    has_access(id, user.organization_id, UserLevel.org_admin)

    if user:
        # Check if the level is valid
        if level in UserLevel:
            # If the user exists and the level is valid, update their level
            user.level = level.value
            session.commit()
            get_all_users(reset=True)
            # get_db_user.cache_clear()
        else:
            print(f"Invalid level: {level}")
    else:
        print(f"User with id {id} does not exist.")


def set_disable_user(id: str, state: bool):
    session = init_system_db()

    # If user level is less than org_admin, raise exception
    has_access(user_level=UserLevel.org_admin)

    # Query UserDataTable for the user
    user = get_db_user(id=id)

    if user:
        # If the user exists, update their disabled state
        user.disabled = state
        session.commit()
        get_all_users(reset=True)
        # get_db_user.cache_clear()
    else:
        print(f"User with id {id} does not exist.")


def add_org(
    name: str, db_name: str = None, disabled: bool = False
) -> OrganizationDataTable:
    session = init_system_db()

    org_id = str(uuid.uuid4())
    # Take the name, make it lowercase, replace special characters with '_', and reduce multiple '_' to one
    db_name = db_name or get_id_str(name)

    has_access(user_level=UserLevel.super_admin)

    # Check if the organization is already in the OrganizationDataTable
    org: OrganizationDataTable = (
        session.query(OrganizationDataTable).filter_by(id=org_id).first()
    )

    if org:
        # If the organization exists, update its details
        org.organization_name = name
        org.db_name = db_name
        org.disabled = disabled
        session.commit()
    else:
        # If the organization does not exist, add it to the OrganizationDataTable
        org = OrganizationDataTable(
            id=org_id,
            organization_name=name,
            db_name=db_name,
            disabled=disabled,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )
        session.add(org)

        session.commit()
    get_all_orgs(reset=True)
    # get_org_by_id.cache_clear()
    return org


def set_org_name(org_id: str, name: str):
    session = init_system_db()
    # Query OrganizationDataTable for the organization
    org = session.query(OrganizationDataTable).filter_by(id=org_id).first()
    has_access(org_id=org.id, user_level=UserLevel.org_admin)

    if org:
        # If the organization exists, update its name
        org.organization_name = name
        session.commit()
        get_all_orgs(reset=True)
        # get_org_by_id.cache_clear()
    else:
        print(f"Organization with ID {org_id} does not exist.")


def set_disable_org(org_id: str, state: bool):
    session = init_system_db()
    has_access(user_level=UserLevel.super_admin)
    # Query OrganizationDataTable for the organization
    org = session.query(OrganizationDataTable).filter_by(id=org_id).first()

    if org:
        # If the organization exists, update its disabled state
        org.disabled = state
        session.commit()
        get_all_orgs(reset=True)
        # get_org_by_id.cache_clear()
    else:
        print(f"Organization with ID {org_id} does not exist.")


def create_preauth_email(preauthorized_email):
    # Check if the file exists
    auth_config = load_auth_config(reset=True)
    # Add a preauthorized email for admin access
    # Check if the email is already in the preauthorized list

    if preauthorized_email not in auth_config["pre-authorized"]["emails"]:
        # If not, add it to the preauthorized list
        auth_config["pre-authorized"]["emails"].append(preauthorized_email)

    # Save the updated configuration back to the YAML file
    write_auth_config(auth_config)


def remove_preauth_email(preauthorized_email):
    # Load the auth configuration
    auth_config = load_auth_config(reset=True)

    # Check if the email is in the preauthorized list
    if preauthorized_email in auth_config["pre-authorized"]["emails"]:
        # If it is, remove it from the preauthorized list
        auth_config["pre-authorized"]["emails"].remove(preauthorized_email)

    # Check if the email is in the UserDataTable
    session = init_system_db()
    user = session.query(UserDataTable).filter_by(email=preauthorized_email).first()
    if user:
        # Make sure user has super_admin or org_admin rights for the user org
        has_access(user.email, user.organization_id, UserLevel.org_admin)

        # If the user exists, remove them from the UserDataTable
        session.delete(user)
        session.commit()

        # Save the updated configuration back to the YAML file
        write_auth_config(auth_config)

        return True
