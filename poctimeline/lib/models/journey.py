from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from lib.models.sqlite_tables import SourceConcept, SourceData
from lib.prompts.journey import JourneyPrompts

class ResourceStructure(BaseModel):
    title: str = Field(
        description="Title for the content to help in the task.", title="Title"
    )
    summary: str = Field(
        description="Most important parts of the document", title="Summary"
    )
    reference: str = Field(
        description="Name of the resource, references or link if available.",
        title="Reference",
    )

class TaskStructure(BaseModel):
    title: str = Field(description="Title for the content", title="Title")
    description: str = Field(
        description="Objective, or target for the content", title="Description"
    )
    resources: List[ResourceStructure] = Field(
        description="List of references to documents or resources to help with the specifics for the content.",
        title="Resources",
    )
    guide: str = Field(
        description="Detailed guide on how to use the content with the user", title="Guide"
    )
    test: str = Field(
        description="Description on how to do a test to verify that the student has succeeded in learning the contents.",
        title="Test",
    )

class SubjectStructure(BaseModel):
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
    concepts: List[SourceConcept] = Field(default=None)
    intro: str = Field(default=None)
    summary: str = Field(default=None)
    tasks: str = Field(default=None)
    structured: SubjectStructure = Field(
        default=SubjectStructure(title="", subject="", intro="", content="", tasks=[])
    )
    instructions: str = Field(default=None)


class SubjectModel(BaseModel):
    title: str = Field(default=None)
    summary: str = Field(default=None)
    plan: List[StepModel] = Field(default_factory=list)
    prompts: JourneyPrompts = Field(default=JourneyPrompts())
    instructions: str = Field(default=None)
    step_amount: int = Field(default=None)
    task_amount: int = Field(default=None)
    concepts: List[SourceConcept] = Field(default=None)
    # files: List[str] = Field(default_factory=list)
    # db_sources: Dict[str, SourceData] = Field(default_factory=dict)


class JourneyModel(BaseModel):
    journeyname: Optional[str] = None
    journey_template_id: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    db_sources: Dict[str, SourceData] = Field(default_factory=dict)
    subjects: List[SubjectModel] = Field(default_factory=list)
    chroma_collection: List[str] = Field(default_factory=list)
    disabled: Optional[bool] = False
    title: Optional[str] = None
    summary: Optional[str] = None
    instructions: Optional[str] = None
    complete: bool = False
