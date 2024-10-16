from datetime import datetime, timedelta
from enum import Enum
from functools import cache
from typing import Any, Dict, List, Optional, Union
import uuid
from pydantic import BaseModel, Field
import sqlalchemy as sqla
from sqlalchemy.orm import Session
import streamlit as st
from lib.db.sqlite import Base
from lib.helpers.shared import pretty_print
from lib.load_env import SETTINGS
from lib.models.journey import JourneyItem, JourneyItemType
from lib.models.user import user_db_get_session


class JourneyProgressDataTable(Base):
    __tablename__ = SETTINGS.journey_progress_tablename

    id = sqla.Column(sqla.String, primary_key=True)
    parent_id = sqla.Column(sqla.String)
    journey_id = sqla.Column(sqla.String)
    item_id = sqla.Column(sqla.String)
    item_type = sqla.Column(sqla.String)
    user_id = sqla.Column(sqla.String)
    disabled = sqla.Column(sqla.Boolean)
    removed = sqla.Column(sqla.Boolean)
    assigned_at = sqla.Column(sqla.DateTime)
    started_at = sqla.Column(sqla.DateTime, default=None)
    completed_at = sqla.Column(sqla.DateTime, default=None)
    due_at = sqla.Column(sqla.DateTime, default=None)
    end_of_day = sqla.Column(sqla.Integer)
    length_in_days = sqla.Column(sqla.Integer)
    test_results = sqla.Column(sqla.PickleType, default=None)
    chat_id = sqla.Column(sqla.String, default=None)

    @classmethod
    def load_from_db(
        cls,
        id: str = None,
        item_id: str = None,
        journey_id: str = None,
        user_id: str = None,
        session=None,
    ) -> "JourneyProgressDataTable":
        if session is None:
            session = user_db_get_session()
        query = session.query(JourneyProgressDataTable)

        if id is not None:
            query = query.filter(JourneyProgressDataTable.id == id)
        if item_id is not None:
            query = query.filter(JourneyProgressDataTable.item_id == item_id)
        if journey_id is not None:
            query = query.filter(JourneyProgressDataTable.journey_id == journey_id)
        if user_id is not None:
            query = query.filter(JourneyProgressDataTable.user_id == user_id)

        db_journey_item_progress = query.first()
        if db_journey_item_progress is None:
            raise ValueError(
                f"Journey Progress Item with provided parameters not found in the database."
            )

        return db_journey_item_progress

    @classmethod
    def load_all_children(
        cls, parent_id: str, user_id: str = None, session=None
    ) -> dict[str, list["JourneyProgressDataTable"]]:
        if session is None:
            session = user_db_get_session()

        # Query for direct children
        query = session.query(JourneyProgressDataTable).filter(
            JourneyProgressDataTable.parent_id == parent_id
        )

        # If user_id is provided, filter the results
        if user_id is not None:
            query = query.filter(JourneyProgressDataTable.user_id == user_id)

        direct_children = query.all()

        # Initialize the result dictionary with direct children
        result = {parent_id: direct_children}

        # Recursively load children's children
        for child in direct_children:
            result.update(cls.load_all_children(child.item_id, user_id, session))

        return result

    @classmethod
    def load_all_from_db(
        cls,
        user_id: str = None,
        item_id: str = None,
        journey_id: str = None,
        parent_id: str = None,
        started: Union[datetime, bool, None] = None,
        completed: Union[datetime, bool, None] = None,
        item_type: JourneyItemType = None,
        session=None,
    ) -> list["JourneyProgressDataTable"]:
        if session is None:
            session = user_db_get_session()

        query = session.query(JourneyProgressDataTable)
        if user_id is not None:
            query = query.filter(JourneyProgressDataTable.user_id == user_id)

        if item_id is not None:
            query = query.filter(JourneyProgressDataTable.item_id == item_id)

        if journey_id is not None:
            query = query.filter(JourneyProgressDataTable.journey_id == journey_id)

        if parent_id is not None:
            query = query.filter(JourneyProgressDataTable.parent_id == parent_id)

        if item_type is not None:
            query = query.filter(JourneyProgressDataTable.item_type == item_type.value)

        if started is not None:
            if isinstance(started, bool):
                if started:
                    query = query.filter(
                        JourneyProgressDataTable.started_at.isnot(None)
                    )
                else:
                    query = query.filter(JourneyProgressDataTable.started_at.is_(None))
            elif isinstance(started, datetime):
                query = query.filter(JourneyProgressDataTable.started_at >= started)

        if completed is not None:
            if isinstance(completed, bool):
                if completed:
                    query = query.filter(
                        JourneyProgressDataTable.completed_at.isnot(None)
                    )
                else:
                    query = query.filter(
                        JourneyProgressDataTable.completed_at.is_(None)
                    )
            elif isinstance(completed, datetime):
                query = query.filter(JourneyProgressDataTable.completed_at <= completed)

        db_journey_item_progresss = query.all()

        return db_journey_item_progresss

    @classmethod
    def from_journey_item(
        cls,
        journey_item: "JourneyItem",
        user_id: str,
        session: Session = None,
        commit=True,
    ) -> "JourneyProgressDataTable":
        if session is None:
            session = user_db_get_session()
        journey = journey_item.get_journey()
        all_children: dict[str, JourneyItem] = journey.all_children_by_id()

        if journey is not None and journey != journey_item:
            journey_id = journey.id
            parent_id = journey_item.parent_id
        else:
            journey_id = None
            parent_id = None

        # Create a new JourneyProgressDataTable item from the JourneyItem
        db_journey_item_progress: JourneyProgressDataTable = None
        try:
            db_journey_item_progress = cls.load_from_db(
                item_id=journey_item.id,
                journey_id=journey_id,
                user_id=user_id,
                session=session,
            )
        except Exception as e:
            print(e)

        changes = False

        parent_id = journey_item.parent_id

        # pretty_print(journey_item, force=True)
        # pretty_print([child.to_json() for child in all_children.values()], force=True)

        if parent_id is not None and (
            db_journey_item_progress is None
            or db_journey_item_progress.end_of_day != journey_item.end_of_day
        ):
            parent: JourneyItem = (
                all_children[parent_id] if parent_id != journey.id else journey
            )
            grand_parent_id = parent.parent_id
            grand_parent: JourneyItem = (
                all_children[grand_parent_id]
                if grand_parent_id and grand_parent_id != journey.id
                else (journey if grand_parent_id == journey.id else None)
            )
            if grand_parent is not None and len(grand_parent.children) > 1:
                parent_index = grand_parent.children.index(parent)
            else:
                parent_index = 0

            current_index = next(
                (
                    index
                    for index, child in enumerate(parent.children)
                    if child.id == journey_item.id
                ),
                None,
            )

            previous_ancestor_eod = 0
            if current_index == 0 and parent_index > 0 and grand_parent is not None:
                previous_ancestor: JourneyItem = grand_parent.children[parent_index - 1]
                previous_ancestor_eod = (
                    previous_ancestor.end_of_day
                    if previous_ancestor.end_of_day is not None
                    else 0
                )

            previous_sibling_eod = 0
            if current_index > 0:
                previous_sibling: JourneyItem = parent.children[current_index - 1]

                previous_sibling_eod = (
                    previous_sibling.end_of_day
                    if previous_sibling.end_of_day is not None
                    else 0
                )
            length_in_days = (
                journey_item.end_of_day - previous_ancestor_eod
                if current_index == 0
                else journey_item.end_of_day - previous_sibling_eod
            )
        else:
            length_in_days = journey_item.end_of_day

        if db_journey_item_progress is None:
            db_journey_item_progress = JourneyProgressDataTable(
                id=str(uuid.uuid4()),
                journey_id=journey_id if journey_id != journey_item.id else None,
                item_id=journey_item.id,
                item_type=journey_item.item_type.value,
                user_id=user_id,
                assigned_at=datetime.now(),
                end_of_day=journey_item.end_of_day,
                length_in_days=length_in_days,
                parent_id=parent_id,
            )
            session.add(db_journey_item_progress)
            changes = True
        else:
            if db_journey_item_progress.journey_id != (
                journey_id if journey_id != journey_item.id else None
            ):
                db_journey_item_progress.journey_id = (
                    journey_id if journey_id != journey_item.id else None
                )
                changes = True
            if db_journey_item_progress.item_type != journey_item.item_type.value:
                db_journey_item_progress.item_type = journey_item.item_type.value
                changes = True
            if db_journey_item_progress.user_id != user_id:
                db_journey_item_progress.user_id = user_id
                changes = True

            if db_journey_item_progress.length_in_days != length_in_days:
                db_journey_item_progress.length_in_days = length_in_days
                changes = True

            if db_journey_item_progress.end_of_day != journey_item.end_of_day:
                db_journey_item_progress.end_of_day = journey_item.end_of_day
                changes = True

            if db_journey_item_progress.parent_id != parent_id:
                db_journey_item_progress.parent_id = parent_id
                changes = True

        if commit and changes:
            session.commit()

        return db_journey_item_progress

    @classmethod
    def from_journey_item_progress(
        cls,
        item: "JourneyItemProgress",
        session: Session = None,
        commit=True,
        include_children=True,
    ) -> tuple[bool, "JourneyProgressDataTable"]:
        if session is None:
            session = user_db_get_session()

        db_journey_item_progress = cls.load_from_db(
            id=item.id,
            item_id=item.item_id,
            journey_id=item.journey_id,
            user_id=item.user_id,
            session=session,
        )

        changes = False
        if db_journey_item_progress is None:
            db_journey_item_progress = cls(
                id=item.id,
                journey_id=item.journey_id,
                item_id=item.item_id,
                item_type=item.item_type.value,
                user_id=item.user_id,
                assigned_at=item.assigned_at,
                started_at=item.started_at,
                completed_at=item.completed_at,
                due_at=item.due_at,
                length_in_days=item.length_in_days,
                test_results=item.test_results,
                chat_id=item.chat_id,
                disabled=item.disabled,
                removed=item.removed,
            )
            session.add(db_journey_item_progress)
            changes = True
        else:
            if db_journey_item_progress.journey_id != item.journey_id:
                db_journey_item_progress.journey_id = item.journey_id
                changes = True
            if db_journey_item_progress.item_type != item.item_type.value:
                db_journey_item_progress.item_type = item.item_type.value
                changes = True
            if db_journey_item_progress.user_id != item.user_id:
                db_journey_item_progress.user_id = item.user_id
                changes = True
            if db_journey_item_progress.assigned_at != item.assigned_at:
                db_journey_item_progress.assigned_at = item.assigned_at
                changes = True
            if db_journey_item_progress.started_at != item.started_at:
                db_journey_item_progress.started_at = item.started_at
                changes = True
            if db_journey_item_progress.completed_at != item.completed_at:
                db_journey_item_progress.completed_at = item.completed_at
                changes = True
            if db_journey_item_progress.due_at != item.due_at:
                db_journey_item_progress.due_at = item.due_at
                changes = True
            if db_journey_item_progress.length_in_days != item.length_in_days:
                db_journey_item_progress.length_in_days = item.length_in_days
                changes = True
            if db_journey_item_progress.test_results != item.test_results:
                db_journey_item_progress.test_results = item.test_results
                changes = True
            if db_journey_item_progress.chat_id != item.chat_id:
                db_journey_item_progress.chat_id = item.chat_id
                changes = True
            if db_journey_item_progress.disabled != item.disabled:
                db_journey_item_progress.disabled = item.disabled
                changes = True
            if db_journey_item_progress.removed != item.removed:
                db_journey_item_progress.removed = item.removed
                changes = True

        if include_children:
            for child in item.children:
                child_changes, _ = cls.from_journey_item_progress(
                    child, session=session, commit=False, include_children=True
                )
                changes = changes or child_changes

        if changes and commit:
            session.commit()

        return changes, db_journey_item_progress

    def to_journey_item_progress(self) -> "JourneyItemProgress":
        return JourneyItemProgress.from_db(self)


# Has statistics based on assigned_at, length of completion (via started_at to completed_at and expected length in days)


class JourneyItemProgressState(Enum):
    COMPLETED = "completed"
    STARTED = "started"
    NOT_STARTED = "not_started"
    OVERDUE = "overdue"

    def __eq__(self, other):
        if isinstance(other, Enum):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other

    def __lt__(self, other):
        if isinstance(other, str):
            other = JourneyItemProgressState(other)
        if isinstance(other, Enum):
            other = JourneyItemProgressState(other.value)

        if isinstance(other, JourneyItemProgressState):
            return list(JourneyItemProgressState).index(self) < list(
                JourneyItemProgressState
            ).index(other)
        return NotImplemented

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        if isinstance(other, (str, Enum)):
            other = JourneyItemProgressState(other)

        if isinstance(other, JourneyItemProgressState):
            return list(JourneyItemProgressState).index(self) > list(
                JourneyItemProgressState
            ).index(other)
        return NotImplemented

    def __ge__(self, other):
        return self > other or self == other


class JourneyItemProgress(BaseModel):
    id: str = Field(
        default=None,
        title="Unique Identifier",
        description="Unique identifier for the journey item progress.",
    )
    journey_id: Optional[str] = Field(
        default=None,
        title="Journey ID",
        description="Identifier of the journey this progress is associated with. Can be None.",
    )
    parent_id: Optional[str] = Field(
        default=None,
        title="Parent ID",
        description="Identifier of the parent item this progress is associated with. Can be None.",
    )
    item_id: str = Field(
        default=None,
        title="Item ID",
        description="Identifier of the journey item this progress is associated with.",
    )
    item_type: JourneyItemType = Field(
        default=JourneyItemType.JOURNEY,
        title="Item Type",
        description="Type of the journey item. Can be 'journey', 'section', 'module' or 'action'.",
    )
    user_id: str = Field(
        default=None,
        title="User ID",
        description="User ID of the user this progress is associated with.",
    )
    assigned_at: datetime = Field(
        default=None,
        title="Assigned At",
        description="Date and time when the journey item was assigned to the user.",
    )
    started_at: Optional[datetime] = Field(
        default=None,
        title="Started At",
        description="Date and time when the user started working on the journey item.",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        title="Completed At",
        description="Date and time when the user completed the journey item.",
    )
    end_of_day: int = Field(
        default=None,
        title="End of Day",
        description="Day by which progress should be completed",
    )
    due_at: Optional[datetime] = Field(
        default=None,
        title="Due At",
        description="Date and time when the journey item is due to be completed.",
    )
    length_in_days: int = Field(
        default=None,
        title="Length in Days",
        description="Number of days it takes to complete the journey item.",
    )
    test_results: Optional[Dict[str, Any]] = Field(
        default=None,
        title="Test Results",
        description="Results of the tests associated with the journey item.",
    )
    chat_id: Optional[str] = Field(
        default=None,
        title="Chat ID",
        description="Identifier of the chat associated with the journey item.",
    )
    children: Optional[List["JourneyItemProgress"]] = Field(
        default_factory=list,
        title="Children",
        description="List of child journey item progresses for this item.",
    )
    disabled: bool = Field(
        default=False,
        title="Disabled",
        description="Indicates whether the journey item progress is disabled.",
    )
    removed: bool = Field(
        default=False,
        title="Removed",
        description="Indicates whether the journey item progress has been removed.",
    )

    cache: Dict[str, Any] = Field(
        default_factory=dict,
        title="Cache",
        description="Dictionary to store cached data related to the journey item. This field is ignored during serialization.",
        exclude=True,
    )

    def get_state(self) -> JourneyItemProgressState:
        if self.completed_at is not None:
            return JourneyItemProgressState.COMPLETED
        elif self.started_at is not None:
            if self.due_at is not None and datetime.now() > self.due_at:
                return JourneyItemProgressState.OVERDUE
            else:
                return JourneyItemProgressState.STARTED
        else:
            return JourneyItemProgressState.NOT_STARTED

    def start(self):
        self.started_at = datetime.now()
        self.due_at = JourneyItemProgress.calculate_due_date(self)

        self.save_to_db()

    def complete(self):
        self.completed_at = datetime.now()
        self.save_to_db()

    @classmethod
    def calculate_due_date(cls, journey_item_progress: "JourneyItemProgress"):
        start_date = (
            journey_item_progress.started_at
            if journey_item_progress.started_at
            else journey_item_progress.assigned_at
        )
        due_date = start_date
        days_to_add = journey_item_progress.length_in_days

        while days_to_add > 0:
            due_date += timedelta(days=1)
            if (
                due_date.weekday() < 5
            ):  # Monday is 0 and Sunday is 6. If it's less than 5, it's a weekday.
                days_to_add -= 1

        return due_date

    @classmethod
    def from_db(
        cls, item: JourneyProgressDataTable, reset=False
    ) -> "JourneyItemProgress":

        # Use Streamlit's session_state for caching
        cache_key = f"journey_item_progress_{item.id}"
        if cache_key in st.session_state and not reset:
            return st.session_state[cache_key]

        # Fetch corresponding journey item and its children
        journey_item = JourneyItem.get(journey_id=item.item_id)
        session = user_db_get_session()
        children = [
            cls.from_db(
                JourneyProgressDataTable.from_journey_item(
                    child, item.user_id, session=session
                )
            )
            for child in journey_item.children
        ]

        # Create JourneyItemProgress instance
        journey_item_progress = JourneyItemProgress(
            id=item.id,
            journey_id=item.journey_id,
            item_id=item.item_id,
            parent_id=item.parent_id,
            item_type=JourneyItemType(item.item_type),
            user_id=item.user_id,
            assigned_at=item.assigned_at,
            started_at=item.started_at,
            completed_at=item.completed_at,
            end_of_day=item.end_of_day or 0,
            due_at=item.due_at,
            length_in_days=item.length_in_days,
            test_results=item.test_results,
            chat_id=item.chat_id,
            children=children,
            disabled=item.disabled or False,
            removed=item.removed or False,
        )

        # Store in session_state
        st.session_state[cache_key] = journey_item_progress

        return journey_item_progress

    def save_to_db(self) -> tuple[bool, "JourneyProgressDataTable"]:
        session = user_db_get_session()
        return JourneyProgressDataTable.from_journey_item_progress(
            self, session=session
        )

    def get_progress(self) -> float:
        if not self.children:
            if self.get_state() == JourneyItemProgressState.COMPLETED:
                return 1.0
            else:
                return 0.0
        else:
            total_progress = 0.0
            total_length = 0.0
            for child in self.children:
                child_progress = child.get_progress()
                total_progress += child_progress * child.length_in_days
                total_length += child.length_in_days
            return total_progress / total_length if total_length > 0 else 0.0

    def flatten(
        self,
        type_filter: Optional[JourneyItemType] = None,
        state_filter: Optional[JourneyItemProgressState] = None,
        reset=False,
    ) -> List[str]:
        cache_key = "flattened" + ("_" + type_filter.value if type_filter else "")
        if self.cache.get(cache_key) and not reset:
            return self.cache.get(cache_key)
        items: list[str] = []
        if (not type_filter or type_filter == self.item_type) and (
            not state_filter or state_filter <= self.get_state()
        ):
            items.append(self.id)
        if self.children:
            for child in self.children:
                items.extend(child.flatten(type_filter, state_filter, reset))

        # if type_filter:
        #     items = [item for item in items if item.item_type == type_filter]

        self.cache[cache_key] = items
        return items

    def all_children_by_id(self, reset=False) -> dict[str, "JourneyItemProgress"]:
        cache_key = "all_children"
        if self.cache.get(cache_key) and not reset:
            return self.cache.get(cache_key)

        all_children_by_id = {}  # [child for child in self.children]
        if self.children and len(self.children) > 0:
            for child in self.children:
                all_children_by_id[child.id] = child
                all_children_by_id.update(child.all_children_by_id(reset=reset))

        self.cache[cache_key] = all_children_by_id
        return all_children_by_id

    def get_last_completed(self, reset=False) -> Optional["JourneyItemProgress"]:
        all_children = self.all_children_by_id(reset)

        last_completed = None
        latest_completion_time = None

        for child in all_children.values():
            if child.completed_at:
                if (
                    not latest_completion_time
                    or child.completed_at > latest_completion_time
                ):
                    last_completed = child
                    latest_completion_time = child.completed_at

        return last_completed

    def get_next(self, amount=3, stick_with_one_parent=True, reset=False):
        all_children = self.all_children_by_id()
        if not self.cache.get("incomplete_modules") or reset:
            print("foof")
            all_incomplete_modules = sorted(
                self.flatten(
                    type_filter=JourneyItemType.MODULE,
                    state_filter=JourneyItemProgressState.NOT_STARTED,
                    reset=reset,
                ),
                key=lambda module_id: all_children[module_id].end_of_day,
            )
            incomplete_modules: List[JourneyItemProgress] = []
            for module_id in all_incomplete_modules:
                incomplete_modules.append(all_children[module_id])

            self.cache["incomplete_modules"] = incomplete_modules
        else:
            incomplete_modules = self.cache["incomplete_modules"]

        parent_id = None
        if stick_with_one_parent:
            parent_id = incomplete_modules[0].parent_id

            incomplete_modules = [
                module
                for module in incomplete_modules
                if (parent_id == None or module.parent_id == parent_id)
            ]

        return incomplete_modules[:amount]

    @classmethod
    def get_overall_progress(
        cls,
        user_id: str,
        journey_id: str = None,
        item_id: str = None,
        session=None,
        reset=False,
    ) -> Dict[str, Union[float, int]]:
        cache_key = f"overall_progress_{user_id}_{journey_id}_{item_id}"

        if cache_key not in st.session_state or reset:
            if session is None:
                session = user_db_get_session()

            db_journey_item_progresss = JourneyProgressDataTable.load_all_from_db(
                user_id=user_id,
                journey_id=journey_id,
                item_id=item_id,
                session=session,
            )

            journey_item_progresses = [
                JourneyItemProgress.from_db(item) for item in db_journey_item_progresss
            ]

            total_progress = 0.0
            state_counts = {state.value: 0 for state in JourneyItemProgressState}
            for item_progress in journey_item_progresses:
                total_progress += item_progress.get_progress()
                state_counts[item_progress.get_state().value] += 1

            overall_progress = (
                total_progress / len(journey_item_progresses)
                if journey_item_progresses
                else 0.0
            )

            st.session_state[cache_key] = {
                "overall_progress": overall_progress,
                "state_counts": state_counts,
            }

        return st.session_state[cache_key]
