from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
import sqlalchemy as sqla
from sqlalchemy.ext.mutable import MutableList
from lib.db.sqlite import Base
from lib.load_env import SETTINGS
from lib.prompts.journey import JourneyPrompts
from lib.models.reference import Reference


class JourneyDataTable(Base):
    __tablename__ = SETTINGS.journey_tablename

    id = sqla.Column(sqla.Integer, primary_key=True)
    journey_name = sqla.Column(sqla.String)
    journey_template_id = sqla.Column(sqla.String, default=None)
    references = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    children = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
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
    title = sqla.Column(sqla.String, default=None)
    summary = sqla.Column(sqla.Text, default=None)
    instructions = sqla.Column(sqla.Text, default=None)
    complete = sqla.Column(sqla.Boolean, default=False)
    last_updated = sqla.Column(sqla.DateTime, default=None)

    def create_from_journey_item(self, session, journey_item):
        # Create a new JourneyDataTable item from the JourneyItem
        new_journey = JourneyDataTable(
            id=journey_item.id,
            journey_name=journey_item.title,
            journey_template_id=journey_item.template_id,
            references=journey_item.references,
            children=journey_item.children,
            # Assuming that prompts and template are not present in JourneyItem
            prompts=[],
            template=[],
            disabled=False,
            title=journey_item.title,
            summary=journey_item.summary,
            instructions=journey_item.instructions,
            complete=False,
            last_updated=datetime.now()
        )

        # Add the new JourneyDataTable item to the session and commit the changes
        session.add(new_journey)
        session.commit()

        return new_journey

    def to_journey_item(self):
        # Create a new JourneyItem object from the data in the JourneyDataTable
        journey_item = JourneyItem(
            id=self.id,
            template_id=self.journey_template_id,
            title=self.title,
            summary=self.summary,
            references=self.references,
            children=self.children,
            instructions=self.instructions,
            item_type="journey"
        )

        return journey_item


    def update_from_journey_item(self, session, journey_item):
        # Update the fields of the JourneyDataTable object with the data from the JourneyItem object
        self.journey_name = journey_item.title
        self.journey_template_id = journey_item.template_id
        self.references = journey_item.references
        self.children = journey_item.children
        self.title = journey_item.title
        self.summary = journey_item.summary
        self.instructions = journey_item.instructions
        self.last_updated = datetime.now()

        # Commit the changes to the database
        session.commit()


from typing import Any, Dict, List, Union
from enum import Enum
from pydantic import BaseModel, Field


class JourneyItemType(Enum):
    JOURNEY = "journey"
    SECTION = "section"
    MODULE = "module"
    ACTION = "action"


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
    after_id: Optional[str] = Field(
        default=None,
        title="After ID",
        description="Identifier of the sibling that precedes this item.",
    )
    parent_id: Optional[str] = Field(
        default=None,
        title="Parent ID",
        description="Identifier of the parent item for this item.",
    )
    title: str = Field(
        default=None, title="Title", description="Title or name of the journey item."
    )
    summary: Optional[str] = Field(
        default=None,
        title="Summary",
        description="Brief overview or summary of the journey item.",
    )
    references: Optional[List[Reference]] = Field(
        default=None,
        title="References",
        description="List of references related to the journey item.",
    )
    children: Optional[List["JourneyItem"]] = Field(
        default_factory=list,
        title="Children",
        description="List of child items for this item.",
    )
    instructions: Optional[str] = Field(
        default=None,
        title="Instructions",
        description="Detailed instructions or guidelines related to the journey item.",
    )
    intro: Optional[str] = Field(
        default=None,
        title="Introduction",
        description="Introduction or opening statement related to the journey item.",
    )
    content: Optional[str] = Field(
        default=None,
        title="Content",
        description="Main content or details of the journey item.",
    )
    description: Optional[str] = Field(
        default=None,
        title="Description",
        description="Additional description or explanation of the journey item.",
    )
    instruct: Optional[str] = Field(
        default=None,
        title="Instruct",
        description="Specific instructions or guidance for using the journey item.",
    )
    test: Optional[str] = Field(
        default=None,
        title="Test",
        description="Description of a test or evaluation related to the journey item.",
    )
    end_of_day: Optional[int] = Field(
        default=None,
        title="Done by end of day #",
        description="Number of days after the start of the journey that the item should be completed.",
    )
    item_type: JourneyItemType = Field(
        ...,
        title="Item Type",
        description="Type of the journey item. Can be 'subject', 'module' or 'action'.",
    )

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "JourneyItem":
        children = None
        if "children" in data:
            children = []
            prev_id = None
            for child in data.get("children", []):
                after_id = prev_id
                children.append(
                    cls.from_json(
                        {**child, "parent_id": data["id"], "after_id": after_id}
                    )
                )
                prev_id = child["id"]
        return cls(
            id=data["id"],
            template_id=data.get("template_id"),
            after_id=data.get("after_id"),
            parent_id=data.get("parent_id"),
            title=data.get("title"),
            summary=data.get("summary"),
            references=data.get("references"),
            children=children,
            instructions=data.get("instructions"),
            intro=data.get("intro"),
            content=data.get("content"),
            description=data.get("description"),
            instruct=data.get("instruct"),
            test=data.get("test"),
            end_of_day=data.get("end_of_day"),
            item_type=JourneyItemType(data["type"]),
        )

    def to_json(self) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "template_id": self.template_id,
            "after_id": self.after_id,
            "parent_id": self.parent_id,
            "title": self.title,
            "summary": self.summary,
            "references": [ref.model_dump(mode='json') for ref in self.references],
            "children": [child.to_json() for child in self.children],
            "instructions": self.instructions,
            "intro": self.intro,
            "content": self.content,
            "description": self.description,
            "instruct": self.instruct,
            "test": self.test,
            "end_of_day": self.end_of_day,
            "type": self.item_type.value,
        }
        return {k: v for k, v in data.items() if v is not None}


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
    instruct: str = Field(
        description="Detailed instructions on how to use the content with the user",
        title="Instruct",
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
                instruct=self.instruct or journey_item.instruct,
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
            instruct=self.instruct,
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
                instructions=journey_item.instructions,
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
            intro=self.intro,
            content=self.content,
            references=self.references,
            children=children,
            item_type=JourneyItemType.MODULE,
        )
