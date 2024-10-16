from datetime import datetime, timedelta
from enum import Enum
from functools import cache
from typing import Any, Dict, List, Optional, Union
import uuid
from pydantic import BaseModel, Field
import sqlalchemy as sqla
from sqlalchemy.orm import Session
from lib.db.sqlite import Base
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
    username = sqla.Column(sqla.String)
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
        username: str = None,
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
        if username is not None:
            query = query.filter(JourneyProgressDataTable.username == username)

        db_journey_item_progress = query.first()
        if db_journey_item_progress is None:
            raise ValueError(
                f"Journey Progress Item with provided parameters not found in the database."
            )

        return db_journey_item_progress

    @classmethod
    def load_all_children(cls, parent_id: str, username: str = None, session=None) -> dict[str, list["JourneyProgressDataTable"]]:
        if session is None:
            session = user_db_get_session()

        # Query for direct children
        query = session.query(JourneyProgressDataTable).filter(JourneyProgressDataTable.parent_id == parent_id)

        # If username is provided, filter the results
        if username is not None:
            query = query.filter(JourneyProgressDataTable.username == username)

        direct_children = query.all()

        # Initialize the result dictionary with direct children
        result = {parent_id: direct_children}

        # Recursively load children's children
        for child in direct_children:
            result.update(cls.load_all_children(child.item_id, username, session))

        return result

    @classmethod
    def load_all_items(
        cls,
        username: str = None,
        journey_id: str = None,
        parent_id: str = None,
        started: Union[datetime, bool, None] = None,
        completed: Union[datetime, bool, None] = None,
        session=None,
    ) -> list["JourneyProgressDataTable"]:
        if session is None:
            session = user_db_get_session()

        query = session.query(JourneyProgressDataTable)
        if username is not None:
            query = query.filter(JourneyProgressDataTable.username == username)

        if journey_id is not None:
            query = query.filter(JourneyProgressDataTable.journey_id == journey_id)

        if parent_id is not None:
            query = query.filter(JourneyProgressDataTable.parent_id == parent_id)

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
        username: str,
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
        db_journey_item_progress: JourneyProgressDataTable = cls.load_from_db(
            item_id=journey_item.id,
            journey_id=journey_id,
            username=username,
            session=session,
        )

        changes = False

        parent_id = journey_item.parent_id
        if parent_id is not None and (
            db_journey_item_progress is None
            or db_journey_item_progress.end_of_day != journey_item.end_of_day
        ):
            parent: JourneyItem = all_children[parent_id]
            grand_parent_id = parent.parent_id
            grand_parent: JourneyItem = (
                all_children[grand_parent_id] if grand_parent_id is not None else None
            )
            if grand_parent is not None and len(grand_parent.children) > 1:
                parent_index = grand_parent.children.index(parent)
            else:
                parent_index = 0

            current_index = parent.children.index(journey_item)

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
                username=username,
                assigned_at=datetime.now(),
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
            if db_journey_item_progress.username != username:
                db_journey_item_progress.username = username
                changes = True

            if db_journey_item_progress.length_in_days != length_in_days:
                db_journey_item_progress.length_in_days = length_in_days
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
            username=item.username,
            session=session,
        )

        changes = False
        if db_journey_item_progress is None:
            db_journey_item_progress = cls(
                id=item.id,
                journey_id=item.journey_id,
                item_id=item.item_id,
                item_type=item.item_type.value,
                username=item.username,
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
            if db_journey_item_progress.username != item.username:
                db_journey_item_progress.username = item.username
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
    NOT_STARTED = "not_started"
    STARTED = "started"
    OVERDUE = "overdue"
    COMPLETED = "completed"


class JourneyItemProgress(BaseModel):
    id: str = Field(
        default=None,
        title="Unique Identifier",
        description="Unique identifier for the journey item progress.",
    )
    journey_id: str = Field(
        default=None,
        title="Journey ID",
        description="Identifier of the journey this progress is associated with.",
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
    username: str = Field(
        default=None,
        title="Username",
        description="Username of the user this progress is associated with.",
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
        cls, item: JourneyProgressDataTable
    ) -> "JourneyItemProgress":
        # get corresponding journey item and use its children to define JourneyItemProgress children
        journey_item = JourneyItem.get(item.item_id)
        session = user_db_get_session()
        children = [
            cls.from_db(
                JourneyProgressDataTable.from_journey_item(
                    child, item.username, session=session
                )
            )
            for child in journey_item.children
        ]

        return JourneyItemProgress(
            id=item.id,
            journey_id=item.journey_id,
            item_id=item.item_id,
            item_type=item.item_type,
            username=item.username,
            assigned_at=item.assigned_at,
            started_at=item.started_at,
            completed_at=item.completed_at,
            due_at=item.due_at,
            length_in_days=item.length_in_days,
            test_results=item.test_results,
            chat_id=item.chat_id,
            children=children,
            disabled=item.disabled,
            removed=item.removed
        )

    def save_to_db(self) -> tuple[bool, "JourneyProgressDataTable"]:
        session = user_db_get_session()
        return JourneyProgressDataTable.from_journey_item_progress(self, session=session)

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

    @classmethod
    def get_overall_progress(
        cls,
        username: str,
        journey_id: str = None,
        item_id: str = None,
        session=None,
    ) -> Dict[str, Union[float, int]]:
        if session is None:
            session = user_db_get_session()

        db_journey_item_progresss = JourneyProgressDataTable.load_all_items(
            username=username,
            journey_id=journey_id,
            item_id=item_id,
            session=session,
        )

        journey_item_progresses = [
            JourneyItemProgress.from_db(item)
            for item in db_journey_item_progresss
        ]

        total_progress = 0.0
        state_counts = {state.value: 0 for state in JourneyItemProgressState}
        for item_progress in journey_item_progresses:
            total_progress += item_progress.get_progress()
            state_counts[item_progress.get_state().value] += 1

        overall_progress = total_progress / len(journey_item_progresses) if journey_item_progresses else 0.0

        return {
            "overall_progress": overall_progress,
            "state_counts": state_counts,
        }
