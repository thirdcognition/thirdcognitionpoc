from mailbox import BabylMessage
from typing import Annotated, List
from pydantic import BaseModel, Field

from lib.models.journey import TaskStructure, SubsubjectStructure

class UserData(BaseModel):
    name: str
    age: int
    # Add more fields as needed

class TeachingItem(BaseModel):
    """Teaching item"""

    purpose: str = Field(description="Purpose of the teaching item")
    content: str = Field(description="Content of the teaching item")
    test: str = Field(description="Question for user to verify that they have learned the item")
    test_verification: str = Field(description="What to verify from the user test answer")


class TeachingItemPlan(BaseModel):
    """Plan for teaching"""

    plan: List[TeachingItem] = Field(
        description="different teaching items to follow",
    )

class TeachingTask(BaseModel):
    """Teaching task"""

    parent_subject: SubsubjectStructure = Field(description="Parent subject the teaching task is part of")
    parent_task: TaskStructure = Field(description="Parent task the teaching task is part of")
    plan: TeachingItemPlan = Field(description="Plan for teaching")
    messages: List[BabylMessage]


