import time
from typing import List

import streamlit as st

import os
import sys

from admin.sidebar import init_sidebar

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir + "/../lib"))


from lib.models.journey_progress import JourneyItemProgress, JourneyItemProgressState, JourneyProgressDataTable
from lib.models.user import AuthStatus, UserLevel, add_user, get_all_users, get_db_user, get_user_org
from lib.models.journey import (
    JourneyItem,
    get_all_journeys_from_db,
)


st.set_page_config(
    page_title="TC POC: Admin",
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

@st.fragment
def assign_journey(journey_id: str):
    journey = JourneyItem.get(journey_id=journey_id)
    org_id = get_user_org(st.session_state["username"]).id
    users = get_all_users(org_id, reset=True)
    all_users = [user.email for user in users]
    available_users = [user.email for user in users]
    assigned_users = []

    # Display current assignments and user progress

    # Load all progress items for the current journey
    progress_items = JourneyProgressDataTable.load_all_from_db(
        journey_item_id=journey_id
    )

    if len(progress_items) > 0:
        st.subheader("Connected Users")
        # Display current assignments and user progress
        count = 0
        for progress_item in progress_items:
            # pretty_print(progress_item.__dict__, force=True)
            # try:
            # Get the user for the current progress item
            user = get_db_user(id=progress_item.user_id)
            del available_users[available_users.index(user.email)]
            assigned_users.append(user.email)

            container = st.container(border=True)
            col1, col2, col3 = container.columns(
                [0.3, 0.6, 0.15], vertical_alignment="center"
            )

            if user:
                col1.markdown(f"**{user.name +"**\n**" if user.name else ""}{user.email}**")
                count += 1

                journey_progress = JourneyItemProgress.from_db(progress_item)
                if JourneyItemProgressState.NOT_STARTED > journey_progress.get_state():
                    next_modules_progress = journey_progress.get_next(reset=True)
                    # pretty_print(next_modules_progress, force=True)
                    all_children = journey.all_children_by_id()
                    next_module = all_children[next_modules_progress[0].journey_item_id]
                    # journey_item = JourneyItem.get(journey_id = journey_progress.item_id)
                    # st.write(f"Journey: {journey_item.title}")
                    col2.progress(
                        journey_progress.get_progress(), f"Next module: {next_module}"
                    )
                else:
                    col2.write("Not yet started")

                with col3.popover("Manage"):
                    if st.button(
                        f"Remove",
                        key=f"remove_{journey_progress.id}",
                        use_container_width=True,
                    ):
                        journey_progress.removed = True
                        journey_progress.save_to_db()
                        st.rerun()

        if count == 0:
            st.write("No assigned users have yet signed up and started the journey")

        st.divider()

    st.subheader("Assign to Users")
    assign_users = st.multiselect(
        "Select users to assign journeys",
        [user for user in users if user.email in available_users],
        format_func=lambda u: f"{u.name} ({u.email})" if u.name else u.email,
    )

    if st.button(
        "Assign to User(s)", key="assing_journey_users_" + journey_id, disabled=(not assign_users)
    ):
        # journey_id = st.session_state.get("journey_edit_id")
        for user in assign_users:
            user_id = user.id
            journey_item = JourneyItem.load_from_db(journey_id)
            JourneyProgressDataTable.from_journey_item(journey_item, user_id=user_id)

        journey_item.reset_cache()
        st.success("Journeys assigned successfully.")
        st.rerun(scope="fragment")

    st.divider()

    st.subheader("Assign to New User")
    new_user_email = st.text_input(
        "New User Email", placeholder="e.g., newuser@example.com"
    )

    if new_user_email and new_user_email not in all_users and st.button(
        "Assign to New User", key="assing_journey_new_user_" + journey_id
    ):
        # journey_id = st.session_state.get("journey_edit_id")
        new_user = add_user(email=new_user_email, org_id=org_id)
        journey_item = JourneyItem.load_from_db(journey_id)
        JourneyProgressDataTable.from_journey_item(journey_item, user_id=new_user.id)
        journey_item.reset_cache()

        st.success("New user created and journey assigned.")
        st.rerun(scope="fragment")
    if new_user_email and new_user_email in all_users:
        st.write("User already has an account")

async def main():
    st.title("Modify Journey")

    if init_sidebar(UserLevel.org_admin) != AuthStatus.LOGGED_IN:
        st.switch_page("login.py")
        return

    st.markdown(" ")
    db_journey_items = get_all_journeys_from_db()

    journeys = [
        JourneyItem.get(journey_item=db_journey) for db_journey in db_journey_items
    ]

    journey: JourneyItem

    if len(journeys) > 1:
        journey_titles = [journey.title for journey in journeys]
        journey_title = st.selectbox("Choose journey", options=journey_titles)
        journey = journeys[journey_titles.index(journey_title)]
    else:
        journey = journeys[0]

    if journey is not None:
        assign_journey(journey.id)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
