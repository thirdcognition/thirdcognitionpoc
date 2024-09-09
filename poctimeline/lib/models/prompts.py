from pydantic import BaseModel, Field


class CustomPrompt(BaseModel):
    system: str = Field(default=None)
    user: str = Field(default=None)
