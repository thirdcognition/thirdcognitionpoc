from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from models.sqlite_tables import SourceData
from prompts.journey import JourneyPrompts

class ResourceStructure(BaseModel):
    title: str = Field(
        description="Title for the content to help in the task.", title="Title"
    )
    summary: str = Field(
        description="Most important parts of the document for the step", title="Summary"
    )
    reference: str = Field(
        description="Name of the resource, references or link if available.",
        title="Reference",
    )

class ActionStructure(BaseModel):
    title: str = Field(description="Title for the step", title="Title")
    description: str = Field(
        description="Objective, task, or target for what to do on the step",
        title="Description",
    )
    resources: List[ResourceStructure] = Field(
        description="List of content to help the Teacher to perform the step.",
        title="Resources",
    )
    test: str = Field(
        description="Description on how to do a test to verify that the student has succeeded in learning the contents for the step.",
        title="Test",
    )

class SubjectStructure(BaseModel):
    title: str = Field(description="Title of the class", title="Title")
    subject: str = Field(description="Subject of the class", title="Subject")
    intro: str = Field(description="Introduction to the class", title="Intro")
    content: str = Field(description="Detailed content of the class", title="Content")
    actions: List[ActionStructure] = Field(
        description="List steps for the teacher to take within the class to teach the subject.",
        title="Actions",
    )


class StepModel(BaseModel):
    title: str = Field(default=None)
    subject: str = Field(default=None)
    content: str = Field(default=None)
    intro: str = Field(default=None)
    actions: str = Field(default=None)
    structured: SubjectStructure = Field(
        default=SubjectStructure(title="", subject="", intro="", content="", actions=[])
    )


class SubjectModel(BaseModel):
    title: str = Field(default=None)
    summary: str = Field(default=None)
    steps: List[StepModel] = Field(default_factory=list)
    prompts: JourneyPrompts = Field(default=JourneyPrompts())
    instructions: str = Field(default=None)
    step_amount: int = Field(default=None)
    action_amount: int = Field(default=None)
    files: List[str] = Field(default_factory=list)
    db_sources: Dict[str, SourceData] = Field(default_factory=dict)


class JourneyModel(BaseModel):
    journeyname: Optional[str] = None
    files: List[str] = Field(default_factory=list)
    db_sources: Dict[str, SourceData] = Field(default_factory=dict)
    subjects: List[SubjectModel] = Field(default_factory=list)
    chroma_collection: List[str] = Field(default_factory=list)
    disabled: Optional[bool] = False
    title: Optional[str] = None
    summary: Optional[str] = None
    instructions: Optional[str] = None
    complete: bool = False
