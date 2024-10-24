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
from lib.models.journey import JourneyItem, JourneyItemType, get_journey_cache
from lib.models.user import get_db_user, user_db_get_session


class JourneyProgressDataTable(Base):
    __tablename__ = SETTINGS.journey_progress_tablename

    id = sqla.Column(sqla.String, primary_key=True)
    parent_id = sqla.Column(sqla.String)
    journey_item_parent_id = sqla.Column(sqla.String)
    journey_id = sqla.Column(sqla.String)
    journey_item_id = sqla.Column(sqla.String)
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
    extras = sqla.Column(sqla.PickleType, default=None)
    chat_id = sqla.Column(sqla.String, default=None)

    def __eq__(self, other):
        if (
            isinstance(other, JourneyProgressDataTable)
            or type(other).__name__ == JourneyProgressDataTable.__name__
        ):
            return self.id == other.id
        return False

    @classmethod
    def load_from_db(
        cls,
        id: str = None,
        journey_item_id: str = None,
        journey_id: str = None,
        user_id: str = None,
        session=None,
    ) -> "JourneyProgressDataTable":
        if session is None:
            session = user_db_get_session()
        query = session.query(JourneyProgressDataTable)

        if id is not None:
            query = query.filter(JourneyProgressDataTable.id == id)
        if journey_item_id is not None:
            query = query.filter(
                JourneyProgressDataTable.journey_item_id == journey_item_id
            )
        if journey_id is not None:
            query = query.filter(JourneyProgressDataTable.journey_id == journey_id)
        if user_id is not None:
            query = query.filter(JourneyProgressDataTable.user_id == user_id)

        db_journey_item_progress = query.first()
        # if db_journey_item_progress is None:
        #     raise ValueError(
        #         f"Journey Progress Item with provided parameters not found in the database."
        #     )

        return db_journey_item_progress

    @classmethod
    def load_all_children(
        cls,
        journey_item_id: str = None,
        user_id: str = None,
        id: str = None,
        session=None,
        load_grand_children=False,
    ) -> dict[str, list["JourneyProgressDataTable"]]:
        if session is None:
            session = user_db_get_session()

        query = None

        if id is not None:
            query = session.query(JourneyProgressDataTable).filter(
                JourneyProgressDataTable.parent_id == id
            )
            # db_item = session.query(JourneyProgressDataTable).filter(
            #     JourneyProgressDataTable.id == id
            # ).first()
            # parent_id = db_item.journey_item_id
            # user_id = db_item.user_id
        elif journey_item_id is None:
            # Query for direct children
            query = session.query(JourneyProgressDataTable).filter(
                JourneyProgressDataTable.journey_item_parent_id == journey_item_id
            )

        # If user_id is provided, filter the results
        if user_id is not None:
            query = query.filter(JourneyProgressDataTable.user_id == user_id)

        direct_children = query.all()

        # Initialize the result dictionary with direct children
        if id:
            result = {id: direct_children}
            # Recursively load children's children
            if load_grand_children:
                for child in direct_children:
                    result.update(
                        cls.load_all_children(
                            user_id=user_id, id=child.id, session=session
                        )
                    )
        else:
            result = {journey_item_id: direct_children}

            # Recursively load children's children
            if load_grand_children:
                for child in direct_children:
                    result.update(
                        cls.load_all_children(
                            user_id=user_id,
                            journey_item_id=child.journey_item_id,
                            session=session,
                        )
                    )

        return result

    @classmethod
    def load_all_from_db(
        cls,
        user_id: str = None,
        journey_item_id: str = None,
        journey_id: str = None,
        journey_item_parent_id: str = None,
        started: Union[datetime, bool, None] = None,
        completed: Union[datetime, bool, None] = None,
        item_type: JourneyItemType = None,
        removed=False,
        disabled=False,
        session=None,
    ) -> list["JourneyProgressDataTable"]:
        if session is None:
            session = user_db_get_session()

        query = session.query(JourneyProgressDataTable)
        if user_id is not None:
            query = query.filter(JourneyProgressDataTable.user_id == user_id)

        if journey_item_id is not None:
            query = query.filter(
                JourneyProgressDataTable.journey_item_id == journey_item_id
            )

        if journey_id is not None:
            query = query.filter(JourneyProgressDataTable.journey_id == journey_id)

        if journey_item_parent_id is not None:
            query = query.filter(
                JourneyProgressDataTable.journey_item_parent_id
                == journey_item_parent_id
            )

        if item_type is not None:
            query = query.filter(JourneyProgressDataTable.item_type == item_type.value)

        query = query.filter(
            sqla.or_(
                JourneyProgressDataTable.removed == False,
                JourneyProgressDataTable.removed == None,
            )
            if removed == False
            else (JourneyProgressDataTable.removed == removed)
        )
        query = query.filter(
            sqla.or_(
                JourneyProgressDataTable.disabled == False,
                JourneyProgressDataTable.disabled == None,
            )
            if disabled == False
            else (JourneyProgressDataTable.disabled == disabled)
        )

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
        parent_id: str = None,
        session: Session = None,
        commit=True,
        reset=False,
    ) -> "JourneyProgressDataTable":
        if user_id is None or journey_item is None:
            raise ValueError(
                f"Both Journey Item and User_id must be provided when creating JourneyItemProgress"
            )

        journey = journey_item.get_root()

        cache_key = (
            "create_from_journey_item_"
            + journey.id
            + "_"
            + journey_item.id
            + "_"
            + user_id
        )

        if not reset and cache_key in get_journey_cache():
            return get_journey_cache()[cache_key]

        if session is None:
            session = user_db_get_session()
        all_children: dict[str, JourneyItem] = journey.all_children_by_id()

        if journey is not None and journey != journey_item:
            journey_id = journey.id
            journey_item_parent_id = journey_item.parent_id
        else:
            journey_id = None
            journey_item_parent_id = None

        # Create a new JourneyProgressDataTable item from the JourneyItem
        db_journey_item_progress: JourneyProgressDataTable = None
        try:
            db_journey_item_progress = cls.load_from_db(
                journey_item_id=journey_item.id,
                journey_id=journey_id,
                user_id=user_id,
                session=session,
            )
        except Exception as e:
            print(e)

        changes = False

        journey_item_parent_id = journey_item.parent_id

        # pretty_print(journey_item, force=True)
        # pretty_print([child.to_json() for child in all_children.values()], force=True)

        if journey_item_parent_id is not None and (
            db_journey_item_progress is None
            or db_journey_item_progress.end_of_day != journey_item.end_of_day
        ):
            parent: JourneyItem = (
                all_children[journey_item_parent_id]
                if journey_item_parent_id != journey.id
                else journey
            )
            journey_item_grand_parent_id = parent.parent_id
            grand_parent: JourneyItem = (
                all_children[journey_item_grand_parent_id]
                if journey_item_grand_parent_id
                and journey_item_grand_parent_id != journey.id
                else (journey if journey_item_grand_parent_id == journey.id else None)
            )
            if grand_parent is not None and len(grand_parent.children) > 1:
                parent_index = next(
                    index
                    for index, child in enumerate(grand_parent.children)
                    if child.id == parent.id
                )
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
                parent_id=parent_id,
                journey_id=journey_id if journey_id != journey_item.id else None,
                journey_item_id=journey_item.id,
                item_type=journey_item.item_type.value,
                user_id=user_id,
                assigned_at=datetime.now(),
                end_of_day=journey_item.end_of_day,
                length_in_days=length_in_days,
                journey_item_parent_id=journey_item_parent_id,
            )
            session.add(db_journey_item_progress)

            for child in journey_item.children:
                cls.from_journey_item(
                    child,
                    user_id,
                    db_journey_item_progress.id,
                    session=session,
                    reset=True,
                    commit=False,
                )

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

            if (
                db_journey_item_progress.journey_item_parent_id
                != journey_item_parent_id
            ):
                db_journey_item_progress.journey_item_parent_id = journey_item_parent_id
                changes = True

        if commit and changes:
            session.commit()

        get_journey_cache()[cache_key] = db_journey_item_progress

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
            # journey_item_id=item.journey_item_id,
            # journey_id=item.journey_id,
            # user_id=item.user_id,
            session=session,
        )

        changes = False
        if db_journey_item_progress is None:
            db_journey_item_progress = cls(
                id=item.id,
                parent_id=item.parent_id,
                journey_id=item.journey_id,
                journey_item_id=item.journey_item_id,
                journey_item_parent_id=item.journey_item_parent_id,
                item_type=item.item_type.value,
                user_id=item.user_id,
                assigned_at=item.assigned_at,
                started_at=item.started_at,
                completed_at=item.completed_at,
                due_at=item.due_at,
                length_in_days=item.length_in_days,
                test_results=item.test_results,
                extras=item.extras,
                chat_id=item.chat_id,
                disabled=item.disabled,
                removed=item.removed,
            )
            session.add(db_journey_item_progress)
            changes = True
        else:
            if db_journey_item_progress.journey_id != item.journey_id:
                db_journey_item_progress.journey_id = item.journey_id
                #print(f"Update {item.id=} {item.journey_id=}")
                changes = True
            if db_journey_item_progress.journey_item_id != item.journey_item_id:
                db_journey_item_progress.journey_item_id = item.journey_item_id
                #print(f"Update {item.id=} {item.journey_item_id=}")
                changes = True
            if (
                db_journey_item_progress.journey_item_parent_id
                != item.journey_item_parent_id
            ):
                db_journey_item_progress.journey_item_parent_id = (
                    item.journey_item_parent_id
                )
                #print(f"Update {item.id=} {item.journey_item_parent_id=}")
                changes = True
            if db_journey_item_progress.item_type != item.item_type.value:
                db_journey_item_progress.item_type = item.item_type.value
                #print(f"Update {item.id=} {item.item_type.value=}")
                changes = True
            if db_journey_item_progress.user_id != item.user_id:
                db_journey_item_progress.user_id = item.user_id
                #print(f"Update {item.id=} {item.user_id=}")
                changes = True
            if db_journey_item_progress.assigned_at != item.assigned_at:
                db_journey_item_progress.assigned_at = item.assigned_at
                #print(f"Update {item.id=} {item.assigned_at=}")
                changes = True
            if db_journey_item_progress.started_at != item.started_at:
                db_journey_item_progress.started_at = item.started_at
                #print(f"Update {item.id=} {item.started_at=}")
                changes = True
            if db_journey_item_progress.completed_at != item.completed_at:
                db_journey_item_progress.completed_at = item.completed_at
                #print(f"Update {item.id=} {item.completed_at=}")
                changes = True
            if db_journey_item_progress.due_at != item.due_at:
                db_journey_item_progress.due_at = item.due_at
                #print(f"Update {item.id=} {item.due_at=}")
                changes = True
            if db_journey_item_progress.length_in_days != item.length_in_days:
                db_journey_item_progress.length_in_days = item.length_in_days
                #print(f"Update {item.id=} {item.length_in_days=}")
                changes = True
            if db_journey_item_progress.test_results != item.test_results:
                db_journey_item_progress.test_results = item.test_results
                #print(f"Update {item.id=} {item.test_results=}")
                changes = True
            if db_journey_item_progress.extras != item.extras:
                db_journey_item_progress.extras = item.extras
                #print(f"Update {item.id=} {item.extras=}")
                changes = True
            if db_journey_item_progress.chat_id != item.chat_id:
                db_journey_item_progress.chat_id = item.chat_id
                #print(f"Update {item.id=} {item.chat_id=}")
                changes = True
            if db_journey_item_progress.disabled != item.disabled:
                db_journey_item_progress.disabled = item.disabled
                #print(f"Update {item.id=} {item.disabled=}")
                changes = True
            if db_journey_item_progress.removed != item.removed:
                db_journey_item_progress.removed = item.removed
                #print(f"Update {item.id=} {item.removed=}")
                changes = True

        if include_children and item.children:
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
    parent_id: Optional[str] = Field(
        default=None,
        title="Parent ID",
        description="Identifier of the parent item this progress is associated with. Can be None.",
    )
    journey_id: Optional[str] = Field(
        default=None,
        title="Journey ID",
        description="Identifier of the journey this progress is associated with. Can be None.",
    )
    journey_item_parent_id: Optional[str] = Field(
        default=None,
        title="Journey item parent ID",
        description="Identifier of the parent item this progress is associated with. Can be None.",
    )
    journey_item_id: str = Field(
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
    extras: Optional[Any] = Field(
        default=None,
        title="Extras",
        description="Extra dict to store random bits",
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

    def __eq__(self, other):
        if (
            isinstance(other, JourneyItemProgress)
            or type(other).__name__ == JourneyItemProgress.__name__
        ):
            return self.id == other.id
        return False

    def reset_cache(self):
        self.cache = {}
        if self.children and len(self.children) > 0:
            for child in self.children:
                child.reset_cache()
        get_journey_cache().clear()

    def get_state(self) -> JourneyItemProgressState:
        # print(f"{self.get_ident()} {self.completed_at=} {self.started_at=}")
        if self.completed_at is not None:
            return JourneyItemProgressState.COMPLETED
        elif self.started_at is not None:
            # if self.due_at is not None and datetime.now() > self.due_at:
            #     return JourneyItemProgressState.OVERDUE
            # else:
            return JourneyItemProgressState.STARTED
        else:
            return JourneyItemProgressState.NOT_STARTED

    def get_root(self, reset=False) -> Optional["JourneyItemProgress"]:
        if self.parent_id is None:
            return self

        if "root" in self.cache and not reset:
            return self.cache["root"]

        parent = self.get(progress_id=self.parent_id)
        while parent.parent_id is not None:
            # print(f"{parent.item_type=} {parent.id=}")
            parent = self.get(progress_id=parent.parent_id)
        # print(f"{parent.item_type=} {parent.id=}")
        self.cache["root"] = parent
        return parent

    def start(self, session=None, commit=True, solo=False):
        session = session or user_db_get_session()
        self.started_at = datetime.now()
        self.due_at = JourneyItemProgress.calculate_due_date(self)
        root = None

        if not solo:
            root = self.get_root()
            if self.id != root.id:
                ancestry = self.get_ancestry(root)
                all_children = root.all_children_by_id()
                for ancestor in ancestry:
                    if (
                        ancestor != root.id
                        and all_children[ancestor].started_at is None
                    ):
                        all_children[ancestor].start(
                            session=session, commit=False, solo=True
                        )
                    if ancestor == root.id:
                        root.start(session=session, commit=False, solo=True)

            for child in self.children:
                if child.started_at is None:
                    child.start(session=session, commit=False, solo=True)

        self.save_to_db(session=session, commit=commit, include_children=False)

        if not solo:
            root.reset_cache()
            get_journey_cache().clear()

    def complete(self, feedback: str = None, session=None, commit=True, solo=False):
        session = session or user_db_get_session()
        self.completed_at = datetime.now()
        if feedback != None:
            self.extras = self.extras or {}
            self.extras["completion_feedback"] = feedback
        if not solo and self.children:
            for child in self.children:
                if child.completed_at is None:
                    child.complete(session=session, commit=False)
        root = self.get_root()
        if self.id != root.id:
            ancestry = self.get_ancestry(root)
            all_children = root.all_children_by_id()
            for ancestor in ancestry:
                if ancestor != root.id and all_children[ancestor].completed_at is None:
                    ancestor = all_children[ancestor]
                    all_children_completed = all(
                        child.get_state() == JourneyItemProgressState.COMPLETED
                        for child in ancestor.children
                    )
                    if len(ancestor.children) == 1 or all_children_completed:
                        ancestor.complete(session=session, commit=False, solo=True)
                if ancestor == root.id:
                    all_children_completed = all(
                        child.get_state() == JourneyItemProgressState.COMPLETED
                        for child in root.children
                    )
                    if all_children_completed:
                        root.complete(session=session, commit=False, solo=True)
        self.save_to_db(session=session, commit=commit)
        root = self.get_root()
        root.reset_cache()
        get_journey_cache().clear()

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
    def reinit(cls, item: "JourneyItemProgress", key=None) -> "JourneyItemProgress":
        if isinstance(item, cls):
            return item

        children = (
            [cls.reinit(child) for child in item.children] if item.children else None
        )

        new_item = cls(
            id=item.id,
            parent_id=item.parent_id,
            journey_id=item.journey_id,
            journey_item_id=item.journey_item_id,
            journey_item_parent_id=item.journey_item_parent_id,
            item_type=JourneyItemType(item.item_type),
            user_id=item.user_id,
            assigned_at=item.assigned_at,
            started_at=item.started_at,
            completed_at=item.completed_at,
            end_of_day=item.end_of_day or 0,
            due_at=item.due_at,
            length_in_days=item.length_in_days,
            test_results=item.test_results,
            extras=item.extras,
            chat_id=item.chat_id,
            children=children,
            disabled=item.disabled or False,
            removed=item.removed or False,
        )

        if key:
            get_journey_cache()[key] = new_item

        return new_item

    @classmethod
    def from_db(
        cls, item: JourneyProgressDataTable = None, id=None, session=None, reset=False
    ) -> "JourneyItemProgress":

        if item is None and id is None:
            raise ValueError("Either item or id must be defined")

        if id is None and item is not None:
            id = item.id

        # Use Streamlit's session_state for caching
        cache_key = f"journey_item_progress_{id}"
        if cache_key in get_journey_cache() and not reset:
            return cls.reinit(get_journey_cache()[cache_key], cache_key)

        session = user_db_get_session()
        if item is None and id is not None:
            item = JourneyProgressDataTable.load_from_db(id=id, session=session)

        if item is None:
            raise ValueError("Item with " + id + " could not be found")

        # Fetch corresponding journey item and its children
        # journey_item = JourneyItem.get(journey_id=item.journey_item_id)

        db_children = JourneyProgressDataTable.load_all_children(
            id=item.id, session=session
        )

        # pretty_print(db_children, f"Children {id}", force=True)

        children = [
            cls.from_db(id=child.id, session=session) for child in db_children[item.id]
        ]

        # children = [
        #     cls.from_db(
        #         JourneyProgressDataTable.from_journey_item(
        #             child, item.user_id, session=session
        #         )
        #     )
        #     for child in journey_item.children
        # ]

        # pretty_print(children, force=True)

        # Create JourneyItemProgress instance
        journey_item_progress = JourneyItemProgress(
            id=item.id,
            parent_id=item.parent_id,
            journey_id=item.journey_id,
            journey_item_id=item.journey_item_id,
            journey_item_parent_id=item.journey_item_parent_id,
            item_type=JourneyItemType(item.item_type),
            user_id=item.user_id,
            assigned_at=item.assigned_at,
            started_at=item.started_at,
            completed_at=item.completed_at,
            end_of_day=item.end_of_day or 0,
            due_at=item.due_at,
            length_in_days=item.length_in_days,
            test_results=item.test_results,
            extras=item.extras,
            chat_id=item.chat_id,
            children=children,
            disabled=item.disabled or False,
            removed=item.removed or False,
        )

        # Store in session_state
        get_journey_cache()[cache_key] = journey_item_progress

        return journey_item_progress

    @classmethod
    def get(
        cls,
        progress_item: JourneyProgressDataTable = None,
        progress_id: str = None,
        journey_item_id: str = None,
        user_id: str = None,
        reset=False,
    ) -> "JourneyItemProgress":
        if reset:
            del get_journey_cache()[progress_id]

        if progress_item is not None:
            progress_id = progress_item.id

        if journey_item_id is not None and progress_id is None:
            if user_id is None:
                raise ValueError("User id is required with journey_item_id")
            journey_progress = JourneyProgressDataTable.load_from_db(
                journey_item_id=journey_item_id, user_id=user_id
            )
            progress_id = journey_progress.id

        if progress_id is not None:
            if progress_id not in get_journey_cache().keys():
                journey_progress = cls.from_db(id=progress_id)
                get_journey_cache()[progress_id] = journey_progress

            return cls.reinit(get_journey_cache()[progress_id], progress_id)

        raise ValueError("Either progress_item or progress_id must be defined")

    def get_relations(self, reset=False) -> Dict[str, str]:
        if self.cache.get("relations") and not reset:
            return self.cache["relations"]

        relations = {}
        if self.children is not None:
            for child in self.children:
                relations[child.id] = self.id

            for child in self.children:
                relations.update(child.get_relations(reset))

        self.cache["relations"] = relations
        return relations

    def get_ancestry(
        self, root: "JourneyItemProgress" = None, reset=False
    ) -> List[str]:
        if self == root:
            return [self.id]

        if self.cache.get("ancestry") and not reset:
            return self.cache["ancestry"]

        if root is None:
            root = self.get_root()

        all_children = root.all_children_by_id()

        journey_relations = root.get_relations(reset)
        # print(
        #     f"{self.item_type=} {self.id=} {root.item_type=} {root.id=} {self.journey_item_parent_id=}"
        # )
        # pretty_print(journey_relations, "Journey Relations", force=True)
        # pretty_print(all_children.keys(), "All children IDs", force=True)
        parent = journey_relations[self.id] if self.id != root.id else None
        if parent is not None:
            ancestry: list[str] = [parent]
        while parent is not None:
            # add parent to the beginning of the list and check its parent
            if parent in journey_relations.keys():
                parent = journey_relations[parent]
            elif parent is not None:
                parent = None
            if parent is not None and parent not in ancestry:
                ancestry.insert(0, parent)

        ancestry.append(self.id)
        self.cache["ancestry"] = ancestry
        return ancestry

    def save_to_db(
        self, session=None, commit=True, include_children=True
    ) -> tuple[bool, "JourneyProgressDataTable"]:
        session = session or user_db_get_session()
        changes, result = JourneyProgressDataTable.from_journey_item_progress(
            self, session=session, commit=commit, include_children=include_children
        )
        if changes:
            self.reset_cache()

        return changes, result

    def get_ident(self):
        cache_key = "ident"
        if cache_key in self.cache:
            return self.cache[cache_key]

        parent_part = self.parent_id.split("-")[-1] if self.parent_id else "root"
        id_part = self.id.split("-")[-1]

        # Get the journey item
        journey_item = JourneyItem.get(journey_id=self.journey_item_id)
        journey_title = (
            f"{journey_item.get_index()} {journey_item.title}"
            if journey_item
            else "Unknown"
        ).ljust(25)[:25]

        # Get the user's email
        user = get_db_user(id=self.user_id)  # Assuming this method is defined
        ident = (
            f" ({user.email if user else ""}) {parent_part}-{id_part}:{journey_title}"
        )
        self.cache[cache_key] = ident
        return ident

    def get_progress(self) -> float:
        if not self.children:
            if self.get_state() == JourneyItemProgressState.COMPLETED:
                return 1.0
            else:
                return 0.0
        else:
            total_progress = 0.0
            # total_length = 0.0
            child_len = len(self.children)
            for child in self.children:
                child_progress = child.get_progress()
                total_progress += child_progress / child_len  # child.length_in_days
                # total_length += child.length_in_days
            # print(f"{"\n\n" if self.parent_id is None else ""}{self.get_ident()} - {child_len=} {total_progress=}")
            return total_progress
            # return total_progress / total_length if total_length > 0 else 0.0

    def flatten(
        self,
        type_filter: Optional[JourneyItemType] = None,
        state_filter: Optional[JourneyItemProgressState] = None,
        reset=False,
    ) -> List[str]:
        cache_key = (
            "flattened"
            + ("_" + type_filter.value if type_filter else "")
            + ("_" + state_filter.value if state_filter else "")
        )
        # print(f"{self.id=} {len(self.children)=}")
        if self.cache.get(cache_key) and not reset:
            return self.cache.get(cache_key)
        items: list[str] = []
        if (not type_filter or type_filter == self.item_type) and (
            not state_filter or state_filter <= self.get_state()
        ):
            items.append(self.id)
        # elif state_filter and state_filter > self.get_state():
        #     pretty_print(
        #         self,
        #         f"Filtered out by {state_filter}, is: {self.get_state()}, type {self.item_type=}",
        #         force=True,
        #     )
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

        journey_item_parent_id = None
        if stick_with_one_parent:
            if len(incomplete_modules) > 0:
                journey_item_parent_id = incomplete_modules[0].journey_item_parent_id
            else:
                return []

            incomplete_modules = [
                module
                for module in incomplete_modules
                if (
                    journey_item_parent_id == None
                    or module.journey_item_parent_id == journey_item_parent_id
                )
            ]

        return incomplete_modules[:amount]

    def get_by_journey_item(self, journey_item: JourneyItem) -> "JourneyItemProgress":
        item_type = journey_item.item_type
        all_items = self.all_children_by_id()
        filtered_items = self.flatten(type_filter=item_type)
        items = [
            all_items[item]
            for item in filtered_items
            if all_items[item].journey_item_id == journey_item.id
        ]

        return items[0] if len(items) > 0 else None

    @classmethod
    def get_overall_progress(
        cls,
        user_id: str,
        journey_id: str = None,
        journey_item_id: str = None,
        session=None,
        reset=False,
    ) -> Dict[str, Union[float, int]]:
        cache_key = f"overall_progress_{user_id}_{journey_id}_{journey_item_id}"

        if cache_key not in get_journey_cache() or reset:
            if session is None:
                session = user_db_get_session()

            db_journey_item_progresss = JourneyProgressDataTable.load_all_from_db(
                user_id=user_id,
                journey_id=journey_id,
                journey_item_id=journey_item_id,
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

            get_journey_cache()[cache_key] = {
                "overall_progress": overall_progress,
                "state_counts": state_counts,
            }

        return get_journey_cache()[cache_key]
