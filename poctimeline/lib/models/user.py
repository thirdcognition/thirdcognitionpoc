from enum import Enum
from functools import cache
import hashlib
import os
from typing import List
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import sqlalchemy as sqla
from sqlalchemy.ext.mutable import MutableList

from lib.db.sqlite import Base, db_commit, db_session, init_system_db
from lib.helpers.shared import pretty_print
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

    # id = Column(Integer, primary_key=True)
    email = sqla.Column(sqla.String, primary_key=True)
    username = sqla.Column(sqla.String, nullable=True)
    realname = sqla.Column(sqla.String, nullable=True)
    level = sqla.Column(sqla.Integer, default=100)
    organization_id = sqla.Column(sqla.String)
    journey_ids = sqla.Column(MutableList.as_mutable(sqla.PickleType), nullable=True)
    disabled = sqla.Column(sqla.Boolean, default=False)


class OrganizationDataTable(Base):
    __tablename__ = SETTINGS.organizations_tablename

    # id = Column(Integer, primary_key=True)
    organization_id = sqla.Column(sqla.String, primary_key=True)
    organization_name = sqla.Column(sqla.String)
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
            new_user = UserDataTable(
                email=user[0],
                username=user[1],
                realname=None,  # Assuming realname is provided in the SETTINGS.super_admin list
                level=UserLevel.super_admin.value,
                organization_id=SETTINGS.default_organization[0],
                journey_ids=[],
                disabled=False,
            )
            create_preauth_email(user[0])
            session.add(new_user)
        session.commit()

    # Check if the organization exists in the OrganizationDataTable
    org_exists = (
        session.query(OrganizationDataTable)
        .filter_by(organization_id=SETTINGS.default_organization[0])
        .first()
    )

    if not org_exists:
        # If the organization does not exist, add it to the OrganizationDataTable
        third_cognition_org = {
            "organization_id": SETTINGS.default_organization[0],
            "organization_name": SETTINGS.default_organization[1],
            "db_name": SETTINGS.default_organization[
                2
            ],  # Replace with the actual database name
            "disabled": False,
        }
        new_org = OrganizationDataTable(**third_cognition_org)
        session.add(new_org)
        session.commit()


def user_db_get_session():
    init_system_db()
    organization = get_user_org(st.session_state.get("username"))
    return db_session(organization.db_name)

def get_user_chroma_path():
    init_system_db()
    organization = get_user_org(st.session_state.get("username"))
    return os.path.join(SETTINGS.db_path, organization.db_name, SETTINGS.chroma_path)

def user_db_commit():
    init_system_db()
    organization = get_user_org(st.session_state.get("username"))
    return db_commit(organization.db_name)


def write_auth_config(auth_config: dict) -> bool:
    st.session_state["auth_config"] = auth_config
    auth_config_str = str(auth_config)
    auth_config_hash = hashlib.md5(auth_config_str.encode()).hexdigest()
    if (
        "auth_config_hash" not in st.session_state
        or st.session_state["auth_config_hash"] != auth_config_hash
    ):
        st.session_state["auth_config_hash"] = auth_config_hash
        with open(SETTINGS.auth_filename, "w", encoding="utf-8") as file:
            yaml.dump(auth_config, file, default_flow_style=False)
        return True

    return False


def load_auth_config():
    if st.session_state.get("auth_config") is not None:
        return st.session_state["auth_config"]

    auth_config: dict = None

    if os.path.isfile(SETTINGS.auth_filename):
        with open(SETTINGS.auth_filename) as file:
            auth_config = yaml.load(file, Loader=SafeLoader)

    try:
        users = len(auth_config["credentials"]["usernames"].keys()) if auth_config is not None else 0
    except:
        users = 0

    if users == 0:
        # If the file does not exist, initialize the dict using the example
        auth_config = {
            "credentials": {
                "usernames": {
                    # "username": {
                    #     "email": "markus@thirdcognition.com",
                    #     "failed_login_attempts": 0,
                    #     "logged_in": False,
                    #     "name": "Markus Haverinen",
                    #     "password": ""  # Will be hashed automatically
                    # }
                }
            },
            "cookie": {
                "expiry_days": SETTINGS.auth_cookie_expiry,
                "key": SETTINGS.auth_cookie_secret,  # Must be string
                "name": SETTINGS.auth_cookie_key,
            },
            "pre-authorized": {"emails": []},
        }
        # Save the initialized configuration to the YAML file

    write_auth_config(auth_config)

    if users == 0:
        for user in SETTINGS.super_admin:
            create_preauth_email(user[0])

    return auth_config


authenticator: stauth.Authenticate = None


def get_authenticator():
    global authenticator
    return authenticator


def set_authenticator(auth: stauth.Authenticate):
    global authenticator
    authenticator = auth


# super_admin can access all, org_admin can access all in their org
def get_all_users(org_id: str = None, reset=False) -> List[UserDataTable]:
    if reset or "db_users" not in st.session_state:
        st.session_state["db_users"] = None

    if st.session_state["db_users"] is not None:
        return st.session_state["db_users"]

    session = init_system_db()
    if is_super_admin():
        # If the user is a super_admin, return all users
        users = session.query(UserDataTable).all()
    elif is_org_admin():
        org_id = get_user_org_id()
        if org_id is not None:
        # If the user is an org_admin, return all users in their org
            users = session.query(UserDataTable).filter_by(organization_id=org_id).all()
    else:
        # If the user is not a super_admin or an org_admin, return an empty list
        users = []

    st.session_state["db_users"] = users

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
        orgs = [get_user_org(st.session_state["username"])]

    st.session_state["db_orgs"] = orgs

    return orgs


@cache
def get_org_by_id(org_id: str, reset=False) -> OrganizationDataTable:
    if reset:
        get_org_by_id.cache_clear()

    session = init_system_db()
    # Query OrganizationDataTable for the organization
    org_data = (
        session.query(OrganizationDataTable).filter_by(organization_id=org_id).first()
    )

    return org_data


def check_auth_level(db_user: str = None) -> UserLevel:
    if (db_user or st.session_state.get("username")) is None:
        return UserLevel.anonymous

    # Query UserDataTable for the user
    user_data = get_db_user(db_user or st.session_state["username"])
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


@cache
def get_db_user(db_user: str = None, email: str = None) -> UserDataTable:
    if db_user is None and email is None:
        raise ValueError("Either db_user or email must be provided")

    session = init_system_db()
    user_data = None
    # Query UserDataTable for the user
    if db_user is not None:
        user_data = session.query(UserDataTable).filter_by(username=db_user).first()
    elif email is not None:
        user_data = session.query(UserDataTable).filter_by(email=email).first()

    return user_data


@cache
def get_user_org(db_user: str = None, email: str = None) -> OrganizationDataTable:
    if db_user is None and email is None:
        raise ValueError("Either db_user or email must be provided")

    session = init_system_db()
    # Query UserDataTable for the user
    # Query UserDataTable and OrganizationDataTable in one query
    join = session.query(OrganizationDataTable).join(
        UserDataTable,
        UserDataTable.organization_id == OrganizationDataTable.organization_id,
    )
    org_data = None
    if db_user is not None:
        org_data = join.filter(UserDataTable.username == db_user).first()
    elif email is not None:
        org_data = join.filter(UserDataTable.email == email).first()
    return org_data


def get_user_org_id(db_user: str = None) -> str:
    if db_user is None:
        db_user = st.session_state.get("username")
    if db_user is None:
        return None
    org = get_user_org(db_user)
    return org.organization_id if org is not None else None


def has_access(email=None, org_id=None, user_level: UserLevel = UserLevel.user):
    cur_user_level = check_auth_level(st.session_state.get("username"))
    if user_level > cur_user_level:
        raise Exception("User level too low")

    cur_user = get_db_user(email=email, db_user=st.session_state.get("username"))
    cur_org = get_user_org(email=cur_user.email) if cur_user is not None else None

    if is_super_admin():
        pass
    elif is_org_admin() and not (org_id is None or cur_org.organization_id == org_id):
        print(repr(cur_org.organization_id), repr(org_id))
        raise Exception("Not enough access: Org admin can only modify their org")
    elif is_user() and not (email is None or cur_user.email == email):
        raise Exception("Not enough access: User can only modify their own account")

        # raise Exception("Not enough access: Anonymous user can't modify users")


def add_user(
    email: str,
    username: str = None,
    realname: str = None,
    org_id: str = None,
    journeys: list = None,
    level: UserLevel = None,
    disabled: bool = False,
):
    session = init_system_db()
    # Check if the user is already in the UserDataTable

    cur_user = get_db_user(db_user=st.session_state.get("username"))
    cur_org = get_user_org(email=cur_user.email) if cur_user is not None else None
    org_id = org_id or (cur_org.organization_id if cur_org else None)

    user = get_db_user(email=email)

    if org_id is None and (cur_user is not None and email != cur_user.email):
        has_access(user_level=UserLevel.super_admin)
    if cur_user is not None:
        has_access(email, org_id)
    # org admin can only modify users in their org

    if user:
        # If the user exists, check for changes and update their details
        updated = False
        if username is not None and user.username != username:
            user.username = username
            updated = True
        if level is not None and user.level != level.value:
            if not isinstance(level, UserLevel):
                level = UserLevel(level)
            user.level = level.value if level else None
            updated = True
        if org_id is not None and user.organization_id != org_id:
            user.organization_id = org_id
            updated = True
        if realname is not None and user.realname != realname:
            user.realname = realname
            updated = True
        if journeys is not None and user.journey_ids != journeys:
            user.journey_ids = journeys
            updated = True
        if disabled is not None and user.disabled != disabled:
            user.disabled = disabled
            updated = True
        if updated:
            session.commit()
    else:
        has_access(user_level=UserLevel.org_admin)
        if level is not None and not isinstance(level, UserLevel):
            level = UserLevel(level)
        # If the user does not exist, add them to the UserDataTable
        new_user = UserDataTable(
            email=email,
            username=username,
            realname=realname,
            level=(level or UserLevel.user).value,
            organization_id=org_id,
            journey_ids=journeys,
            disabled=disabled,
        )
        session.add(new_user)
        session.commit()
        create_preauth_email(email)
    get_all_users(reset=True)
    get_db_user.cache_clear()


def set_user_org(email: str, org_id: str):
    session = init_system_db()

    has_access(email, org_id)

    # Query UserDataTable for the user
    user = get_db_user(email=email)

    if user:
        # If the user exists, update their organization_id
        # Check if the organization exists in the OrganizationDataTable
        org = (
            session.query(OrganizationDataTable)
            .filter_by(organization_id=org_id)
            .first()
        )
        if org:
            user.organization_id = org_id
            session.commit()
            get_all_users(reset=True)
            get_db_user.cache_clear()
        else:
            print(f"Organization with ID {org_id} does not exist.")
    else:
        print(f"User with email {email} does not exist.")


def set_user_realname(email: str, realname: str):
    session = init_system_db()

    has_access(email)

    # Query UserDataTable for the user
    user = get_db_user(email=email)

    if user:
        # If the user exists, update their realname
        user.realname = realname
        session.commit()
    else:
        print(f"User with email {email} does not exist.")


def set_user_level(email: str, level: UserLevel):
    session = init_system_db()

    # Query UserDataTable for the user
    user = get_db_user(email=email)
    has_access(email, user.organization_id, UserLevel.org_admin)

    if user:
        # Check if the level is valid
        if level in UserLevel:
            # If the user exists and the level is valid, update their level
            user.level = level.value
            session.commit()
            get_all_users(reset=True)
            get_db_user.cache_clear()
        else:
            print(f"Invalid level: {level}")
    else:
        print(f"User with email {email} does not exist.")


def set_user_journeys(email: str, journeys: list):
    session = init_system_db()
    # Query UserDataTable for the user
    user = get_db_user(email=email)
    has_access(email, user.organization_id, UserLevel.org_admin)

    if user:
        # If the user exists, update their journey_ids
        user.journey_ids = journeys
        session.commit()
        get_all_users(reset=True)
        get_db_user.cache_clear()
    else:
        print(f"User with email {email} does not exist.")


def set_disable_user(email: str, state: bool):
    session = init_system_db()

    # If user level is less than org_admin, raise exception
    has_access(user_level=UserLevel.org_admin)

    # Query UserDataTable for the user
    user = get_db_user(email=email)

    if user:
        # If the user exists, update their disabled state
        user.disabled = state
        session.commit()
        get_all_users(reset=True)
        get_db_user.cache_clear()
    else:
        print(f"User with email {email} does not exist.")


def add_org(org_id: str, name: str, db_name: str = None, disabled: bool = False):
    session = init_system_db()

    has_access(user_level=UserLevel.super_admin)

    # Check if the organization is already in the OrganizationDataTable
    org = session.query(OrganizationDataTable).filter_by(organization_id=org_id).first()

    if org:
        # If the organization exists, update its details
        org.organization_name = name
        org.db_name = db_name or f"{org_id}_db"
        org.disabled = disabled
        session.commit()
    else:
        # If the organization does not exist, add it to the OrganizationDataTable
        new_org = OrganizationDataTable(
            organization_id=org_id,
            organization_name=name,
            db_name=db_name or f"{org_id}_db",
            disabled=disabled,
        )
        session.add(new_org)

        session.commit()
    get_all_orgs(reset=True)
    get_org_by_id.cache_clear()


def set_org_name(org_id: str, name: str):
    session = init_system_db()
    # Query OrganizationDataTable for the organization
    org = session.query(OrganizationDataTable).filter_by(organization_id=org_id).first()
    has_access(org_id=org.organization_id, user_level=UserLevel.org_admin)

    if org:
        # If the organization exists, update its name
        org.organization_name = name
        session.commit()
        get_all_orgs(reset=True)
        get_org_by_id.cache_clear()
    else:
        print(f"Organization with ID {org_id} does not exist.")


def set_disable_org(org_id: str, state: bool):
    session = init_system_db()
    has_access(user_level=UserLevel.super_admin)
    # Query OrganizationDataTable for the organization
    org = session.query(OrganizationDataTable).filter_by(organization_id=org_id).first()

    if org:
        # If the organization exists, update its disabled state
        org.disabled = state
        session.commit()
        get_all_orgs(reset=True)
        get_org_by_id.cache_clear()
    else:
        print(f"Organization with ID {org_id} does not exist.")


def create_preauth_email(preauthorized_email):
    # Check if the file exists
    auth_config = load_auth_config()
    # Add a preauthorized email for admin access
    # Check if the email is already in the preauthorized list

    if preauthorized_email not in auth_config["pre-authorized"]["emails"]:
        # If not, add it to the preauthorized list
        auth_config["pre-authorized"]["emails"].append(preauthorized_email)

    # Save the updated configuration back to the YAML file
    write_auth_config(auth_config)


def remove_preauth_email(preauthorized_email):
    # Load the auth configuration
    auth_config = load_auth_config()

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
