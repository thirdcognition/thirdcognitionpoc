from datetime import datetime
from enum import Enum
from functools import cache
import json
from typing import Any, Dict, List, Optional, Union
import uuid
from fuzzywuzzy import fuzz
from pydantic import BaseModel, Field
import sqlalchemy as sqla
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import Session
from lib.db.sqlite import Base
from lib.load_env import SETTINGS
from lib.models.user import user_db_get_session
from lib.prompts.journey import JourneyPrompts
from lib.models.reference import Reference


def get_journey_data_from_db(id: str, session=None) -> "JourneyDataTable":
    # if reset:
    #     get_journey_data_from_db.cache_clear()
    if session is None:
        session = user_db_get_session()

    db_journey_item: JourneyDataTable = (
        session.query(JourneyDataTable).filter(JourneyDataTable.id == id)
    ).first()

    return db_journey_item


def get_all_journeys_from_db(session=None) -> list["JourneyDataTable"]:
    # if reset:
    #     get_journey_data_from_db.cache_clear()
    if session is None:
        session = user_db_get_session()

    db_journey_items: JourneyDataTable = list(
        (
            session.query(JourneyDataTable).filter(
                JourneyDataTable.item_type == JourneyItemType.JOURNEY.value
            )
        ).all()
    )

    print(f"{db_journey_items=}")

    if db_journey_items is None or len(db_journey_items) == 0:
        raise ValueError(f"Journey Items not found in the database.")

    return db_journey_items


class JourneyDataTable(Base):
    __tablename__ = SETTINGS.journey_tablename

    id = sqla.Column(sqla.String, primary_key=True)
    journey_name = sqla.Column(sqla.String)
    journey_template_id = sqla.Column(sqla.String, default=None)
    parent_id = sqla.Column(sqla.String, default=None)
    after_id = sqla.Column(sqla.String, default=None)
    references = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    children = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    use_guide = sqla.Column(sqla.Text, default=None)
    title = sqla.Column(sqla.String, default=None)
    icon = sqla.Column(sqla.String, default=None)
    intro = sqla.Column(sqla.Text, default=None)
    summary = sqla.Column(sqla.Text, default=None)
    description = sqla.Column(sqla.Text, default=None)
    content_instructions = sqla.Column(sqla.PickleType, default=None)
    content = sqla.Column(sqla.Text, default=None)
    test = sqla.Column(sqla.Text, default=None)
    action = sqla.Column(sqla.Text, default=None)
    end_of_day = sqla.Column(sqla.Integer, default=None)
    item_type = sqla.Column(sqla.String)
    chroma_collections = sqla.Column(
        MutableList.as_mutable(sqla.PickleType), default=[]
    )
    prompts = sqla.Column(
        sqla.PickleType, default=[]
    )  # JourneyPrompts = Field(default=JourneyPrompts())
    template = sqla.Column(
        sqla.PickleType, default=[]
    )  # JourneyPrompts = Field(default=JourneyPrompts())
    disabled = sqla.Column(sqla.Boolean, default=False)
    complete = sqla.Column(sqla.Boolean, default=False)
    last_updated = sqla.Column(sqla.DateTime, default=None)

    @classmethod
    def load_from_db(cls, id: str, session=None):
        if session is None:
            session = user_db_get_session()
        db_journey_item: JourneyDataTable = (
            session.query(JourneyDataTable).filter(JourneyDataTable.id == id)
        ).first()

        if db_journey_item is None:
            raise ValueError(f"Journey Item {id} not found in the database.")

        return db_journey_item

    @classmethod
    def create_from_journey_item(
        cls, journey_item: "JourneyItem", session: Session = None, commit=True
    ) -> "JourneyDataTable":
        if session is None:
            session = user_db_get_session()

        # Create a new JourneyDataTable item from the JourneyItem
        db_journey_item: JourneyDataTable = get_journey_data_from_db(
            journey_item.id, session
        )
        children = (
            ([child.id for child in journey_item.children])
            if journey_item.children is not None
            else (db_journey_item.children if db_journey_item is not None else None)
        )

        if db_journey_item is None:
            # print("Create new db item", journey_item.template_id, journey_item.id)
            db_journey_item = JourneyDataTable(
                id=journey_item.id,
                journey_name=journey_item.title,
                journey_template_id=journey_item.template_id,
                parent_id=journey_item.parent_id,
                after_id=journey_item.after_id,
                references=journey_item.references,
                children=children,
                use_guide=journey_item.use_guide,
                title=journey_item.title,
                icon=journey_item.icon,
                intro=journey_item.intro,
                summary=journey_item.summary,
                description=journey_item.description,
                content_instructions=(
                    journey_item.content_instructions.to_json()
                    if journey_item.content_instructions is not None
                    else None
                ),
                content=journey_item.content,
                test=journey_item.test,
                action=journey_item.action,
                end_of_day=journey_item.end_of_day,
                item_type=journey_item.item_type.value,
                # Assuming that prompts and template are not present in JourneyItem
                prompts=[],
                template=[],
                disabled=False,
                complete=False,
                last_updated=datetime.now(),
            )
            session.add(db_journey_item)
        else:
            # print("Update existing db item", journey_item.id)
            db_journey_item.id = journey_item.id
            db_journey_item.journey_name = journey_item.title
            db_journey_item.journey_template_id = journey_item.template_id
            db_journey_item.parent_id = journey_item.parent_id
            db_journey_item.after_id = journey_item.after_id
            db_journey_item.references = journey_item.references
            db_journey_item.children = children
            db_journey_item.use_guide = journey_item.use_guide
            db_journey_item.title = journey_item.title
            db_journey_item.icon = journey_item.icon
            db_journey_item.intro = journey_item.intro
            db_journey_item.summary = journey_item.summary
            db_journey_item.description = journey_item.description
            db_journey_item.content_instructions = (
                journey_item.content_instructions.to_json()
                if journey_item.content_instructions is not None
                else None
            )
            db_journey_item.content = journey_item.content
            db_journey_item.test = journey_item.test
            db_journey_item.action = journey_item.action
            db_journey_item.end_of_day = journey_item.end_of_day
            db_journey_item.item_type = journey_item.item_type.value
            db_journey_item.disabled = False
            db_journey_item.complete = False
            db_journey_item.last_updated = datetime.now()

        if journey_item.children and len(journey_item.children) > 0:
            for child in journey_item.children:
                JourneyDataTable.create_from_journey_item(child, session, commit=False)

        # Add the new JourneyDataTable item to the session and commit the changes
        if commit:
            print("Save to db", journey_item.id)
            # get_journey_data_from_db.cache_clear()
            session.commit()

        return db_journey_item

    def to_journey_item(self) -> "JourneyItem":
        # Create a new JourneyItem object from the data in the JourneyDataTable
        journey_item = JourneyItem(
            id=self.id,
            template_id=self.journey_template_id,
            parent_id=self.parent_id,
            after_id=self.after_id,
            references=self.references,
            children=[],  # self.children,
            use_guide=self.use_guide,
            title=self.title,
            icon=self.icon,
            intro=self.intro,
            summary=self.summary,
            description=self.description,
            content_instructions=ContentInstructions.from_json(
                self.content_instructions
            ),
            content=self.content,
            test=self.test,
            action=self.action,
            end_of_day=self.end_of_day,
            item_type=JourneyItemType(self.item_type),
        )

        if self.children and len(self.children) > 0:
            db_children: list[JourneyDataTable] = []
            children: list[JourneyItem] = []
            for child_id in self.children:
                db_children.append(JourneyDataTable.load_from_db(child_id))
            for db_child in db_children:
                children.append(db_child.to_journey_item())
            journey_item.children = children
            journey_item.sort_children()

        return journey_item

    def update_from_journey_item(self, session: Session, journey_item: "JourneyItem"):
        if session is None:
            session = user_db_get_session()

        # Update the fields of the JourneyDataTable object with the data from the JourneyItem object
        self.journey_name = journey_item.title
        self.journey_template_id = journey_item.template_id
        self.parent_id = journey_item.parent_id
        self.after_id = journey_item.after_id
        self.references = journey_item.references
        self.children = (
            [child.id for child in journey_item.children]
            if journey_item.children
            else self.children
        )
        self.use_guide = journey_item.use_guide
        self.title = journey_item.title
        self.intro = journey_item.intro
        self.summary = journey_item.summary
        self.description = journey_item.description
        self.content_instructions = journey_item.content_instructions.to_json()
        self.content = journey_item.content
        self.test = journey_item.test
        self.action = journey_item.action
        self.end_of_day = journey_item.end_of_day
        self.item_type = journey_item.item_type.value
        self.last_updated = datetime.now()

        if journey_item.children and len(journey_item.children) > 0:
            for child in journey_item.children:
                JourneyDataTable.create_from_journey_item(child, session, False)

        # Commit the changes to the database
        # get_journey_data_from_db.cache_clear()
        session.commit()


class JourneyItemType(Enum):
    JOURNEY = "journey"
    SECTION = "section"
    MODULE = "module"
    ACTION = "action"

    def __eq__(self, other):
        if isinstance(other, Enum):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return False

    def __gt__(self, other):
        order = ["journey", "section", "module", "action"]
        if isinstance(other, Enum):
            return order.index(self.value) > order.index(other.value)
        if isinstance(other, str):
            return order.index(self.value) > order.index(other)
        return NotImplemented

    def __lt__(self, other):
        order = ["journey", "section", "module", "action"]
        if isinstance(other, Enum):
            return order.index(self.value) < order.index(other.value)
        if isinstance(other, str):
            return order.index(self.value) < order.index(other)
        return NotImplemented


class ContentInstructions(BaseModel):
    # content_role: From which roles point of view the content is generated from
    role: Optional[str] = Field(
        default="",
        title="Role",
        description="The role or perspective from which the content is generated.",
    )
    # topic: The topic for the generated content
    topic: Optional[str] = Field(
        default="",
        title="Topic",
        description="The main topic or subject of the generated content.",
    )
    # instructions: Any instructions for the content generation
    instructions: Optional[str] = Field(
        default="",
        title="Instructions",
        description="Detailed instructions or guidelines for generating the content.",
    )

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ContentInstructions":
        if data is None:
            return None
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, ContentInstructions) or type(data).__name__ == ContentInstructions.__name__:
            data = data.to_json()

        return cls(
            role=data.get("role") if data is not None else "",
            topic=data.get("topic") if data is not None else "",
            instructions=data.get("instructions") if data is not None else "",
        )

    def to_json(self) -> Dict[str, Any]:
        data = {
            "role": self.role,
            "topic": self.topic,
            "instructions": self.instructions,
        }
        return {k: v for k, v in data.items() if v is not None}


# Global journey cache
# journey_cache = {}


# def get_journey_from_cache(id: str) -> Optional["JourneyItem"]:
#     return journey_cache.get(id)


# def add_journey_to_cache(journey: "JourneyItem"):
#     journey_cache[journey.id] = journey


# def get_available_journeys():
#     return journey_cache.keys()


class JourneyItem(BaseModel):
    id: str = Field(
        default=None,
        title="Unique Identifier",
        description="Unique identifier for the journey item.",
    )
    template_id: Optional[str] = Field(
        default=None,
        title="Template ID",
        description="Identifier of the template item used for this journey item.",
    )
    parent_id: Optional[str] = Field(
        default=None,
        title="Parent ID",
        description="Identifier of the parent item for this item.",
    )
    after_id: Optional[str] = Field(
        default=None,
        title="After ID",
        description="Identifier of the sibling that precedes this item.",
    )
    children: Optional[List["JourneyItem"]] = Field(
        default_factory=list,
        title="Children",
        description="List of child items for this item.",
    )
    references: Optional[List[Reference]] = Field(
        default=None,
        title="References",
        description="List of references related to the journey item.",
    )
    use_guide: Optional[str] = Field(
        default=None,
        title="Usage guidance",
        description="Specific instructions or guidance for using the journey item.",
    )
    title: str = Field(
        default=None, title="Title", description="Title or name of the journey item."
    )
    icon: Optional[str] = Field(
        default=None,
        title="Icon",
        description="Icon or image associated with the journey item.",
    )
    intro: Optional[str] = Field(
        default=None,
        title="Introduction",
        description="Introduction or opening statement related to the journey item.",
    )
    summary: Optional[str] = Field(
        default=None,
        title="Summary",
        description="Brief overview or summary of the journey item.",
    )
    description: Optional[str] = Field(
        default=None,
        title="Description",
        description="Additional description or explanation of the journey item.",
    )
    content_instructions: Optional[ContentInstructions] = Field(
        default=None,
        title="Content instructions",
        description="Detailed instructions in how to generate content for the item.",
    )
    content: Optional[str] = Field(
        default=None,
        title="Content",
        description="Main content or details of the journey item.",
    )
    test: Optional[str] = Field(
        default=None,
        title="Test",
        description="Description of a test or evaluation related to the journey item.",
    )
    action: Optional[str] = Field(
        default=None,
        title="Action",
        description="Description of an action connected to the journey item.",
    )
    end_of_day: Optional[int] = Field(
        default=None,
        title="Done by end of day #",
        description="Number of days after the start of the journey that the item should be completed.",
    )
    item_type: JourneyItemType = Field(
        default=JourneyItemType.JOURNEY,
        title="Item Type",
        description="Type of the journey item. Can be 'subject', 'module' or 'action'.",
    )

    cache: Dict[str, Any] = Field(
        default_factory=dict,
        title="Cache",
        description="Dictionary to store cached data related to the journey item. This field is ignored during serialization.",
        exclude=True,
    )

    def __str__(self):
        return self.title

    def reset_cache(self):
        self.cache = {}
        if self.children and len(self.children) > 0:
            for child in self.children:
                child.reset_cache()

    @classmethod
    def create_new(cls, data: Dict[str, Any]) -> "JourneyItem":
        id = data.get("id", str(uuid.uuid4()))
        item_type = JourneyItemType(data.get("item_type", JourneyItemType.JOURNEY))
        template_id = data.get("template_id", id)
        parent_id = data.get("parent_id")
        after_id = data.get("after_id")
        children = data.get("children", [])
        references = data.get("references", [])
        use_guide = data.get("use_guide", "")
        title = data.get("title", "New journey " + item_type.name.capitalize())
        icon = data.get("icon", "icon-1")
        intro = data.get("intro", "")
        summary = data.get("summary", "")
        description = data.get("description", "")
        content_instructions = (
            data.get("content_instructions")
            if isinstance(data.get("content_instructions"), ContentInstructions)
            else ContentInstructions.from_json(
                data.get(
                    "content_instructions",
                    {"role": "instructor", "topic": "Overview of provided content."},
                )
            )
        )
        content = data.get("content", "")
        test = data.get("test", "")
        action = data.get("action", "")
        end_of_day = data.get("end_of_day", 0)

        children_items = []
        if children:
            for child_data in children:
                child_item = cls.create_new(child_data)
                children_items.append(child_item)

        return cls(
            id=id,
            template_id=template_id,
            parent_id=parent_id,
            after_id=after_id,
            children=children_items,
            references=references,
            use_guide=use_guide,
            title=title,
            icon=icon,
            intro=intro,
            summary=summary,
            description=description,
            content_instructions=content_instructions,
            content=content,
            test=test,
            action=action,
            end_of_day=end_of_day,
            item_type=item_type,
        )

    def add_child(self, item:"JourneyItem", index=0):
        if index < 0 or index > len(self.children):
            raise IndexError("Index out of range")

        if index == len(self.children):
            # Adding item at the end of children list
            if self.children:
                self.children[-1].after_id = item.id
        elif index > 0:
            # Adding item in the middle of children list
            next_item = self.children[index]
            prev_item = self.children[index - 1]
            item.after_id = prev_item.id
            next_item.after_id = item.id

        self.children.insert(index, item)
        item.parent_id = self.id
        self.reset_cache()

    def remove_child(self, item: "JourneyItem"):
        if item in self.children:
            index = self.children.index(item)
            self.children.remove(item)
            item.parent_id = None
            item.after_id = None

            # Update after_id of next_child
            if index < len(self.children):
                next_child = self.children[index]
                next_child.after_id = self.children[index - 1].id if index > 0 else None

            self.reset_cache()
            self.sort_children()

    @classmethod
    def from_json(cls, data: Dict[str, Any], from_template=False) -> "JourneyItem":
        if data is None:
            return None
        if isinstance(data, str):
            data = json.loads(data)
        children = None
        template_id = data.get("template_id", data.get("id") if from_template else None)

        if from_template and data.get("id") == template_id:
            # template_id = data.get("id")
            data["id"] = str(
                uuid.uuid4()  # uuid.NAMESPACE_DNS, data.get("id", "journey_template"))
            )

        if "children" in data:
            children = []
            prev_id = None
            for child in data.get("children", []):
                child_template_id = child.get(
                    "template_id", child.get("id") if from_template else None
                )
                if from_template and child.get("id") == child_template_id:
                    child_template_id = child.get("id")
                    child["id"] = str(
                        uuid.uuid4()
                        # uuid.uuid5(
                        #     uuid.NAMESPACE_DNS,
                        #     child.get("id", "journey_child_template"),
                        # )
                    )
                after_id = prev_id
                children.append(
                    cls.from_json(
                        {
                            **child,
                            "parent_id": data["id"],
                            "after_id": after_id,
                            "template_id": child_template_id,
                        },
                        from_template,
                    )
                )
                prev_id = child["id"]
        return cls(
            id=data["id"],
            template_id=template_id,
            parent_id=data.get("parent_id"),
            after_id=data.get("after_id"),
            children=children,
            references=data.get("references"),
            use_guide=data.get("use_guide"),
            title=data.get("title"),
            icon=data.get("icon"),
            intro=data.get("intro"),
            summary=data.get("summary"),
            description=data.get("description"),
            content_instructions=(
                ContentInstructions.from_json(data.get("content_instructions"))
                if data.get("content_instructions")
                else None
            ),
            content=data.get("content"),
            test=data.get("test"),
            action=data.get("action"),
            end_of_day=data.get("end_of_day"),
            item_type=JourneyItemType(data["type"]),
        )

    def to_json(self) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "template_id": self.template_id,
            "parent_id": self.parent_id,
            "after_id": self.after_id,
            "references": [ref.model_dump(mode="json") for ref in self.references],
            "children": [child.to_json() for child in self.children],
            "use_guide": self.use_guide,
            "title": self.title,
            "icon": self.icon,
            "intro": self.intro,
            "summary": self.summary,
            "description": self.description,
            "content_instructions": (
                self.content_instructions.to_json()
                if self.content_instructions
                else None
            ),
            "content": self.content,
            "test": self.test,
            "action": self.action,
            "end_of_day": self.end_of_day,
            "type": self.item_type,
        }
        return {k: v for k, v in data.items() if v is not None}

    def flatten(
        self, type_filter: Optional[JourneyItemType] = None, reset=False
    ) -> List[str]:
        cache_key = "flattened" + ("_" + type_filter.value if type_filter else "")
        if self.cache.get(cache_key) and not reset:
            return self.cache.get(cache_key)
        items: list[str] = []
        if not type_filter or type_filter == self.item_type:
            items.append(self.id)
        if self.children:
            for child in self.children:
                items.extend(child.flatten(type_filter, reset))

        # if type_filter:
        #     items = [item for item in items if item.item_type == type_filter]

        self.cache[cache_key] = items
        return items

    def search_children_with_token(self, search_token, item_type:JourneyItemType=None):
        results = []

        # Fuzzy match search_token against self title
        if item_type is None or item_type == self.item_type:
            # Check if the search token contains whitespace
            if ' ' not in search_token:
                # Split the search token into individual words
                # search_words = search_token.lower().split()

                # Split the title into individual words
                lc_search_token = search_token.lower()
                title_words = self.title.lower().split()

                # Check if all search words are present in the title
                for word in title_words:
                        # Calculate the match ratio using the fuzz.ratio function
                        match_ratio = fuzz.ratio(word.lower(), lc_search_token)
                        print("Match ratio:", match_ratio, word)
                        if match_ratio > 70:  # You can adjust this threshold as needed
                            results.append(self.id)
                            break  # Break the loop if a match is found
            else:
                # Calculate the match ratio using the fuzz.ratio function
                match_ratio = fuzz.ratio(search_token.lower(), self.title.lower())
                print("Match ratio:", match_ratio, self.title)
                if match_ratio > 50:  # You can adjust this threshold as needed
                    results.append(self.id)

        # Iterate children with same arguments
        for child in self.children:
            results.extend(child.search_children_with_token(search_token, item_type))

        return results


    def get_child_by_id(self, id: str) -> Optional["JourneyItem"]:
        if self.id == id:
            return self
        # if self.children:
        #     for child in self.children:
        #         item = child.get_child_by_id(id)
        #         if item:
        #             return item
        children_by_id = self.all_children_by_id()
        if id in children_by_id.keys():
            return children_by_id[id]

        return None

    def all_children_by_id(self, reset=False) -> dict[str, "JourneyItem"]:
        cache_key = "all_children"
        if self.cache.get(cache_key) and not reset:
            return self.cache.get(cache_key)

        all_children_by_id = {}  # [child for child in self.children]
        if self.children and len(self.children) > 0:
            for child in self.children:
                all_children_by_id[child.id] = child
                all_children_by_id.update(child.all_children_by_id())

        self.cache[cache_key] = all_children_by_id
        return all_children_by_id

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

    def get_ancestry(self, root: "JourneyItem", reset=False) -> List[str]:
        if self == root:
            return [self]

        if self.cache.get("ancestry") and not reset:
            return self.cache["ancestry"]

        journey_relations = root.get_relations(reset)
        parent = journey_relations[self.id]
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

    def get_index(self, root: "JourneyItem", reset=False, as_str=True):
        ancestry = self.get_ancestry(root, reset)
        all_items = root.all_children_by_id(reset)

        indexes = []

        for i, ancestor in enumerate(ancestry):
            if (i + 1) < len(ancestry):
                if ancestor in all_items:
                    indexes.append(
                        str(
                            all_items[ancestor].children.index(
                                all_items[ancestry[i + 1]]
                            )
                            + 1
                        )
                    )
                else:
                    indexes.append(
                        str(root.children.index(all_items[ancestry[i + 1]]) + 1)
                    )

        if as_str:
            return ".".join(indexes)

        return indexes

    def move(self, target: "JourneyItem", journey: "JourneyItem"):
        print(
            "Move to/after", (target.title if target is not None else "first at parent")
        )
        all_children = journey.all_children_by_id()

        if target is not None and self.item_type != target.item_type:
            while (
                target.item_type != self.item_type
                and target.children
                and len(target.children) > 0
            ):
                target = target.children[0]
            target = all_children[target.parent_id]

            print("Move to/after (children?)", target.title)

        ancestry = self.get_ancestry(
            journey
        )  # get a list of all parents of self, 0 being the journey and last self being theself
        if target is not None:
            target_ancestry = target.get_ancestry(journey)
        # journey.get_child_by_id - get any child by id
        # journey.flatten - get list of all children in a list (also children of children and so on)
        # journey.get_relations return dict with self id as key and parent self as value

        # get parent ofself
        # check if self has siblings
        # if self has siblings and one sibling is after self set said sibling after_id to self after_id

        # get parent of target
        # check if target has siblings
        # if target has siblings and one sibling is after target set said sibling after_id to self id

        # if self parent id is not same as target parent id remove self from self parent children and add to target parent children

        # set self parent_id to target parent_id
        # set self after_id to target id

        # Get parent of self
        parent = (
            all_children[ancestry[-2]]
            if len(ancestry) > 1 and ancestry[-2] in all_children
            else journey
        )
        target_parent = None
        # Get siblings ofself
        siblings = [child for child in parent.children if child != self]
        # If self has siblings and one sibling is after self, set that sibling's after_id to self's after_id
        for sibling in siblings:
            if sibling.after_id == self.id:
                sibling.after_id = self.after_id
                break

        if target is not None and self.item_type == target.item_type:
            # Get parent of target

            target_parent = (
                all_children[target_ancestry[-2]]
                if len(target_ancestry) > 1 and target_ancestry[-2] in all_children
                else journey
            )
            # Get siblings of target
            target_siblings = [
                child for child in target_parent.children if child != target
            ]
            # If target has siblings and one sibling is after target, set that sibling's after_id to self's id
            for sibling in target_siblings:
                if sibling.after_id == target.id:
                    sibling.after_id = self.id
                    break

            # If self's parent is not the same as target's parent, remove self from self's parent's children and add it to target's parent's children
        if target is not None and parent != target_parent:
            if self in parent.children:
                parent.remove_child(self) #children.remove(self)
            if target_parent is not None:
                target_parent.add_child(self) #.children.append(self)

        # Set self's parent_id to target's parent_id
        if target_parent is not None:
            self.parent_id = target_parent.id

        # Set self's after_id to target's id
        if target is not None and self.item_type == target.item_type:
            self.after_id = target.id
        elif target is not None:
            if target_parent is not None:
                if self in target_parent.children:
                    target_parent.remove_child(self) #children.remove(self)
            if self not in target.children:
                target.add_child(self) #children.append(self)
            # self.after_id = None
            self.parent_id = target.id
        else:
            self.after_id = None
            for sibling in siblings:
                if sibling.after_id is None:
                    sibling.after_id = self.id

        journey.reset_cache()
        journey.sort_children()

    def sort_children(self):
        if self.children:
            sibling_order = []
            younger_siblings = []
            original_order = []
            children_by_id = {child.id: child for child in self.children}
            for child in self.children:
                original_order.append(child.id)
                if child.after_id is None:
                    sibling_order.append(child.id)
                else:
                    younger_siblings.append(child.id)

            for sibling_id in younger_siblings:
                sibling = children_by_id[sibling_id]
                if sibling.after_id in sibling_order:
                    index = sibling_order.index(sibling.after_id)
                    sibling_order.insert(index + 1, sibling_id)
                else:
                    sibling_order.append(sibling_id)

            # self.children.sort(key=lambda x: x.after_id)
            if original_order != sibling_order:
                # print("Sorting children of", self.title)
                # print("Original order:", original_order)
                # print("New order:", sibling_order)
                self.children = [children_by_id[child_id] for child_id in sibling_order]
            for child in self.children:
                child.sort_children()

    def update_eod(self, new_eod=0):

        if self.children and len(self.children) > 0:
            new_eod = max(*[child.update_eod() for child in self.children], new_eod)

        if new_eod != 0:
            self.end_of_day = new_eod

        return new_eod

    def save_to_db(self) -> JourneyDataTable:
        session = user_db_get_session()
        self.reset_cache()

        print("Save to db", self.id)

        return JourneyDataTable.create_from_journey_item(self, session)

    @classmethod
    def load_from_db(cls, id: str, session=None) -> "JourneyItem":
        db_journey_item = get_journey_data_from_db(id, session)
        return db_journey_item.to_journey_item()


# The rest of the code remains the same as it is not part of the requested edit.
class ActionModel(BaseModel):
    id: str = Field(
        default=None,
        description="Unique identifier for the action. If available use the existing ID.",
        title="ID",
    )
    after_id: str = Field(
        default=None,
        description="Identifier of the item preseeds this subject.",
        title="After ID",
    )
    parent_id: str = Field(
        default=None,
        description="Identifier of the parent subject for this action.",
        title="Parent ID",
    )
    title: str = Field(description="Title for the content", title="Title")
    description: str = Field(
        description="Objective, or target for the content", title="Description"
    )
    references: List[Reference] = Field(
        description="List of references to topics, concepts or sources to help with the specifics for the content.",
        title="References",
    )
    use_guide: str = Field(
        description="Detailed instructions on how to use the content with the user",
        title="Usage guidance",
    )
    test: str = Field(
        description="Description on how to do a test to verify that the student has succeeded in learning the contents.",
        title="Test",
    )
    end_of_day: int = Field(
        description="Number of days after the start of the journey that the action should be completed.",
        title="Done by end of day #",
    )

    def to_journey_item(self, journey_item: JourneyItem = None) -> JourneyItem:
        """
        Converts the ActionModel instance to a JourneyItem instance.
        If a JourneyItem instance is provided, the new instance will use the values
        from the old instance as defaults.
        """
        if journey_item:
            return JourneyItem(
                id=self.id or journey_item.id,
                template_id=journey_item.template_id,
                after_id=self.after_id or journey_item.after_id,
                parent_id=self.parent_id or journey_item.parent_id,
                title=self.title or journey_item.title,
                description=self.description or journey_item.description,
                references=self.references or journey_item.references,
                use_guide=self.use_guide or journey_item.use_guide,
                test=self.test or journey_item.test,
                end_of_day=self.end_of_day or journey_item.end_of_day,
                item_type=JourneyItemType.ACTION,
            )

        return JourneyItem(
            id=self.id,
            after_id=self.after_id,
            parent_id=self.parent_id,
            title=self.title,
            description=self.description,
            references=self.references,
            use_guide=self.use_guide,
            test=self.test,
            end_of_day=self.end_of_day,
            item_type=JourneyItemType.ACTION,
        )


class ModuleStructure(BaseModel):
    id: str = Field(
        default=None,
        description="Unique identifier for the module. If available use the existing ID.",
        title="ID",
    )
    after_id: str = Field(
        default=None,
        description="Identifier of the item preseeds this subject.",
        title="After ID",
    )
    parent_id: str = Field(
        default=None,
        description="Identifier of the parent subject for this subject.",
        title="Parent ID",
    )
    title: str = Field(description="The title or name of this subject.", title="Title")
    icon: str = Field(
        description="Icon to represent this subject. Use an icon from the provided options.",
        title="Icon",
    )
    subject: str = Field(description="Section of this subject", title="Section")
    intro: str = Field(description="Introduction to this subject", title="Intro")
    content: str = Field(
        description="Detailed content of this subject", title="Content"
    )
    references: List[Reference] = Field(
        description="List of references to topics, concepts or sources to help with the specifics for the content.",
        title="References",
    )
    children: List[Union[ActionModel, "ModuleStructure"]] = Field(
        description="List of child subjects and actions for this subject.",
        title="children",
    )
    end_of_day: int = Field(
        description="Number of days after the start of the journey that the module should be completed.",
        title="Done by end of day #",
    )

    def to_journey_item(
        self, journey_item: JourneyItem = None, after_id: str = None
    ) -> JourneyItem:
        """
        Converts the ModuleStructure instance to a JourneyItem instance.
        If a JourneyItem instance is provided, the new instance will use the values
        from the old instance as defaults.
        """
        children: List[JourneyItem] = []
        if journey_item:
            if self.children is not None:
                last_child_id = None
                for child in self.children:
                    if isinstance(child, ModuleStructure):
                        existing_child = next(
                            (c for c in journey_item.children if c.id == child.id), None
                        )
                        if existing_child:
                            children.append(
                                child.to_journey_item(
                                    existing_child, after_id=last_child_id
                                )
                            )
                        else:
                            children.append(
                                child.to_journey_item(after_id=last_child_id)
                            )
                        last_child_id = children[-1].id
                    else:
                        children.append(child)
                        last_child_id = child.id
            return JourneyItem(
                id=self.id or journey_item.id,
                template_id=journey_item.template_id,
                after_id=self.after_id or after_id or journey_item.after_id,
                parent_id=self.parent_id or journey_item.parent_id,
                title=self.title or journey_item.title,
                icon=self.icon or journey_item.icon,
                content_instructions=journey_item.content_instructions,
                intro=self.intro or journey_item.intro,
                content=self.content or journey_item.content,
                references=self.references or journey_item.references,
                children=children or journey_item.children,
                item_type=JourneyItemType.MODULE,
            )

        if self.children is not None:
            for i, child in enumerate(self.children):
                if isinstance(child, ModuleStructure):
                    after_id = children[-1].id if i > 0 else None
                    children.append(child.to_journey_item(after_id=after_id))
                else:
                    children.append(child)
        return JourneyItem(
            id=self.id,
            after_id=self.after_id or after_id,
            parent_id=self.parent_id,
            title=self.title,
            icon=self.icon,
            intro=self.intro,
            content=self.content,
            references=self.references,
            children=children,
            item_type=JourneyItemType.MODULE,
        )
