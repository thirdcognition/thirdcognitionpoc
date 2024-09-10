from pydantic import BaseModel, Field


class CustomPrompt(BaseModel):
    system: str = Field(default=None)
    user: str = Field(default=None)


class TitledSummary(BaseModel):
    title: str = Field(description="Title for the content")
    summary: str = Field(description="Summary of the content")
