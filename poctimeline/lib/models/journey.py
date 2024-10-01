from typing import List
from pydantic import BaseModel, Field
import sqlalchemy as sqla
from sqlalchemy.ext.mutable import MutableList
from lib.db.sqlite import Base
from lib.load_env import SETTINGS
from lib.models.concepts import ConceptData
from lib.prompts.journey import JourneyPrompts
from lib.models.reference import Reference

class JourneyDataTable(Base):
    __tablename__ = SETTINGS.journey_tablename

    id = sqla.Column(sqla.Integer, primary_key=True)
    journey_name = sqla.Column(sqla.String)
    journey_template_id = sqla.Column(sqla.String, default=None)
    references = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    subjects = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    chroma_collections = sqla.Column(MutableList.as_mutable(sqla.PickleType), default=[])
    disabled = sqla.Column(sqla.Boolean, default=False)
    title = sqla.Column(sqla.String, default=None)
    summary = sqla.Column(sqla.Text, default=None)
    instructions = sqla.Column(sqla.Text, default=None)
    complete = sqla.Column(sqla.Boolean, default=False)
    last_updated = sqla.Column(sqla.DateTime, default=None)


class TaskStructure(BaseModel):
    title: str = Field(description="Title for the content", title="Title")
    description: str = Field(
        description="Objective, or target for the content", title="Description"
    )
    references: List[Reference] = Field(
        description="List of references to topics, concepts or sources to help with the specifics for the content.",
        title="References",
    )
    instruct: str = Field(
        description="Detailed instructions on how to use the content with the user", title="Instruct"
    )
    test: str = Field(
        description="Description on how to do a test to verify that the student has succeeded in learning the contents.",
        title="Test",
    )

class StepStructure(BaseModel):
    title: str = Field(description="Title of the class", title="Title")
    subject: str = Field(description="Subject of the class", title="Subject")
    intro: str = Field(description="Introduction to the class", title="Intro")
    content: str = Field(description="Detailed content of the class", title="Content")
    tasks: List[TaskStructure] = Field(
        description="List of content for the subject.",
        title="Content",
    )


class StepModel(BaseModel):
    title: str = Field(default=None)
    subject: str = Field(default=None)
    content: str = Field(default=None)
    concepts: List[ConceptData] = Field(default=None)
    intro: str = Field(default=None)
    summary: str = Field(default=None)
    tasks: str = Field(default=None)
    structured: StepStructure = Field(
        default=StepStructure(title="", subject="", intro="", content="", tasks=[])
    )
    instructions: str = Field(default=None)


class SubjectModel(BaseModel):
    title: str = Field(default=None)
    summary: str = Field(default=None)
    plan: List[StepModel] = Field(default_factory=list)
    prompts: JourneyPrompts = Field(default=JourneyPrompts())
    instructions: str = Field(default=None)
    steps_amount: int = Field(default=None)
    task_amount: int = Field(default=None)
    concepts: List[ConceptData] = Field(default=None)



