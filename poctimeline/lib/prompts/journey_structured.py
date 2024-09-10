import textwrap
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from lib.models.journey import StepStructure
from lib.models.sqlite_tables import CategoryTag
from lib.prompts.actions import structured
from lib.prompts.base import (
    ACTOR_INTRODUCTIONS,
    MAINTAIN_CONTENT_AND_USER_LANGUAGE,
    PromptFormatter,
)


class Step(BaseModel):
    title: str = Field(description="Title for the subject", title="Title")
    subject: str = Field(
        description="Describes the subject in one sentence", title="Subject"
    )
    summary: str = Field(
        description="Summary of what the subject is about and concepts it uses",
        title="Summary",
    )
    concept_ids: List[str] = Field(
        description="List of concepts id used by the subject", title="Concepts IDs"
    )
    concept_tags: List[CategoryTag] = Field(
        description="List of concepts tags used by the subject", title="Concept Tags"
    )


class Plan(BaseModel):
    plan: List[Step] = Field(description="List of subjects", title="Subjects")


plan = PromptFormatter(
    system=textwrap.dedent(
        f"""
        {ACTOR_INTRODUCTIONS}
        Act as a teacher who is planning a curriculum.
        Using the content between context start and end write a list
        with the specified format structure.
        If instructions are provided follow them exactly.
        Only use the information available within the context.
        If there's a history with previous titles or subjects,
        use them to make sure you don't repeat the same subjects.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        Return only the properly formatted JSON object with the formatted data.
        """
    ),
    user=textwrap.dedent(  # Use get_journey_format_example instead
        """
        instuctions start
        {journey_instructions}
        {subject_instructions}
        instructions end

        context start
        {context}
        context end

        ----------------
        format structure start
        {format_instructions}
        format structure end
        ----------------

        Create a list of {amount} subjects.
        If instructions are provided, follow them exactly. If instructions specify
        a topic or subject, make sure the list includes only items which fall within
        within that topic. Make sure the list has exactly {amount} items.
        Format the context data using the format structure.
        Do not add any information to the context or come up with subjects
        not defined within the context.
        If there's a history with previous titles or subjects,
        use them to make sure you don't repeat the same subjects.
        Return only the properly formatted JSON object with the formatted data.
        """
    ),
)
plan.parser = PydanticOutputParser(pydantic_object=Plan)

step_structured = structured.customize()
step_structured.parser = PydanticOutputParser(pydantic_object=StepStructure)
