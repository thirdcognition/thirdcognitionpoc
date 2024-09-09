import textwrap
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from lib.chains.prompt_generator import CustomPromptContainer
from lib.models.prompts import CustomPrompt
from lib.models.sqlite_tables import CategoryTag
from lib.prompts.base import (
    ACTOR_INTRODUCTIONS,
    KEEP_PRE_THINK_TOGETHER,
    MAINTAIN_CONTENT_AND_USER_LANGUAGE,
    PRE_THINK_INSTRUCT,
    PromptFormatter,
    TagsParser,
)


class JourneyStep(BaseModel):
    title: str = Field(description="Title for the subject", title="Title")
    subject: str = Field(
        description="Describes the subject in one sentence", title="Subject"
    )
    summary: str = Field(
        description="Summary of what the subject is about and concepts it uses", title="Summary"
    )
    concept_ids: List[str] = Field(
        description="List of concepts id used by the subject", title="Concepts IDs"
    )
    concept_tags: List[CategoryTag] = Field(
        description="List of concepts tags used by the subject", title="Concept Tags"
    )

class JourneyStepList(BaseModel):
    steps: List[JourneyStep] = Field(description="List of subjects", title="Subjects")


journey_steps = PromptFormatter(
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
        {instructions}
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
journey_steps.parser = PydanticOutputParser(pydantic_object=JourneyStepList)

journey_step_content = PromptFormatter(
    system=textwrap.dedent(
        f"""
        {ACTOR_INTRODUCTIONS}
        Act as a teacher who is writing study material for a class with a specific subject.
        Use blog content style and structure starting with an introduction and synopsis and
        continuing with clearly sectioned content. Use markdown syntax for formatting.
        Your student is a business graduate who is interested in learning about the subject.
        You only have one student you're tutoring and you are making study materials for them.
        {PRE_THINK_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
        Create the study material for the student with the following information between context start and end.
        Only use the information available within the context. Do not add or remove information from the context.
        If there's a history with previous titles or subjects, use them to make sure you don't repeat the same subjects.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        If instructions are provided follow them exactly.
        """
    ),
    user=textwrap.dedent(  # Use get_journey_format_example instead
        """
        instuctions start
        {journey_instructions}
        {instructions}
        instructions end

        context start
        {context}
        context end

        Subject:
        {subject}


        Create study materials for the student defined by the subject.
        Use blog content style and structure starting with an introduction and synopsis and continuing with
        clearly sectioned content. Don't include any other content outside of the subject.
        Only use the information available within the context. Do not add or remove information from the context.
        If instructions are provided, follow them exactly. If instructions specify a topic or subject, make sure the
        output includes only items which fall within them.
        The study materials should be exhaustive, detailed and generated from the context.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        If instructions are provided follow them exactly.
        The generated material should follow a descriptive tutorial and blog style with a clear structure using only
        the available content.
        """
    ),
)
journey_step_content.parser = TagsParser(min_len=10)

journey_step_intro = PromptFormatter(
    system=textwrap.dedent(
        f"""
        {ACTOR_INTRODUCTIONS}
        Act as a teacher who is writing a brief introduction and a synopsis to a specific subject
        for the student. Use an informal style and 3 sentences maximum.
        Do not use code, lists, or any markup, markdown or html. Just use natural spoken language.
        Your student is a business graduate who is interested in learning about the subject.
        You only have one student you're tutoring so don't have to address more than one person.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
        If there's a history with previous titles or subjects,
        use them to make sure you don't repeat the same subjects.
        Use the content for the class available between content start and end.
        """
    ),
    user=textwrap.dedent(  # Use get_journey_format_example instead
        """
        instuctions start
        {journey_instructions}
        {instructions}
        instructions end

        Subject:
        {subject}

        content start
        {context}
        content end

        Write an introduction and a synopsis to the subject for the student using only natural language.
        If instructions are provided, follow them exactly. If instructions specify
        a topic or subject, make sure the list includes only items which fall within
        within that topic.
        Don't add anything new to the content, just explain it with 3 sentences maximum.
        """
    ),
)
journey_step_intro.parser = TagsParser(min_len=10)

journey_step_actions = PromptFormatter(
    system=textwrap.dedent(
        f"""
        {ACTOR_INTRODUCTIONS}
        Act as a teacher who planning sections of content for teaching the student a specific subject.
        You only have one student you're tutoring so don't have to address more than one person. Also add references to support documents
        with their summary and material to use when teaching the student about the subject.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
        If there's a history with previous titles or subjects,
        use them to make sure you don't repeat the same subjects.
        Use the content for the class available between content start and end.
        """
    ),
    user=textwrap.dedent(  # Use get_journey_format_example instead
        """
        instuctions start
        {journey_instructions}
        {instructions}
        instructions end

        Amount:
        {amount}

        Subject:
        {subject}

        content start
        {context}
        content end

        Write a list of sections with their content to teach the subject to the student.
        Prepare also a list of reference documents and their summary to use with each section when teaching the student about the subject.
        If instructions are provided, follow them exactly. If instructions specify a topic or subject, make sure the list includes only
        items which fall within within that topic. Create at maximum the specified amount of items.
        If there's a history with previous titles or subjects, use them to make sure you don't repeat the same subjects.
        """
    ),
)
journey_step_actions.parser = TagsParser(min_len=10)

journey_step_action_details = PromptFormatter(
    system=textwrap.dedent(
        f"""
        {ACTOR_INTRODUCTIONS}
        Act as a teacher who is creating resources and content to support teaching in class
        to use as a base for the discussion and lesson with the student.
        {MAINTAIN_CONTENT_AND_USER_LANGUAGE}
        {PRE_THINK_INSTRUCT}
        {KEEP_PRE_THINK_TOGETHER}
        If there's a history with previous titles or subjects, use them to make sure you don't repeat the same subjects.
        Use the provided resource description and the content available between content start and end to create the resources.
        """
    ),
    user=textwrap.dedent(  # Use get_journey_format_example instead
        """
        instuctions start
        {journey_instructions}
        {instructions}
        instructions end

        Resource description:
        {resource}

        content start
        {context}
        content end

        Prepare max 10 sentences of material as the resource described to be used while teaching a student.
        If instructions are provided, follow them exactly. If instructions specify a topic or subject, make sure the list includes only
        items which fall within within that topic.
        If there's a history with previous titles or subjects,
        use them to make sure you don't repeat the same subjects.
        """
    ),
)
journey_step_action_details.parser = TagsParser(min_len=20)


class JourneyPrompts(BaseModel):
    steps: CustomPrompt = Field(
        default=CustomPrompt(system=journey_steps.system, user=journey_steps.user)
    )
    step_content: CustomPrompt = Field(
        default=CustomPrompt(
            system=journey_step_content.system, user=journey_step_content.user
        )
    )
    step_intro: CustomPrompt = Field(
        default=CustomPrompt(
            system=journey_step_intro.system, user=journey_step_intro.user
        )
    )
    step_actions: CustomPrompt = Field(
        default=CustomPrompt(
            system=journey_step_actions.system, user=journey_step_actions.user
        )
    )
    step_action_details: CustomPrompt = Field(
        default=CustomPrompt(
            system=journey_step_action_details.system,
            user=journey_step_action_details.user,
        )
    )

def convert_to_journey_prompts(container: CustomPromptContainer) -> JourneyPrompts:
    return JourneyPrompts(
        steps=container.steps,
        step_content=container.step_content,
        step_intro=container.step_intro,
        step_actions=container.step_actions,
        step_action_details=container.step_action_details
    )