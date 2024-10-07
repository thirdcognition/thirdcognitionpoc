import json
import os
import textwrap
from typing import List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from lib.helpers.journey import get_available_journey_template_roles
from lib.chains.prompt_generator import CustomPromptContainer
from lib.models.prompts import CustomPrompt
from lib.models.taxonomy import Taxonomy
from lib.prompts.base import (
    ACTOR_INTRODUCTIONS,
    KEEP_PRE_THINK_TOGETHER,
    MAINTAIN_CONTENT_AND_USER_LANGUAGE,
    PRE_THINK_INSTRUCT,
    PromptFormatter,
    TagsParser,
)
from lib.load_env import SETTINGS


class Module(BaseModel):
    title: str = Field(description="Title for the subject", title="Title")
    subject: str = Field(
        description="Describes the subject in one sentence", title="Section"
    )
    summary: str = Field(
        description="Summary of what the subject is about and concepts it uses",
        title="Summary",
    )
    concept_ids: List[str] = Field(
        description="List of concepts id used by the subject", title="Concepts IDs"
    )
    concept_tags: List[Taxonomy] = Field(
        description="List of concepts tags used by the subject", title="Concept Tags"
    )


class Plan(BaseModel):
    plan: List[Module] = Field(description="List of subjects", title="Sections")


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
        {journey_content_instructions}
        {section_content_instructions}
        instructions end

        context start
        {context}
        context end

        ----------------
        format structure start
        {format_content_instructions}
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

module_content = PromptFormatter(
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
        {journey_content_instructions}
        {section_content_instructions}
        {module_content_instructions}
        instructions end

        context start
        {context}
        context end

        Section:
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
module_content.parser = TagsParser(min_len=10)

module_intro = PromptFormatter(
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
        {journey_content_instructions}
        {section_content_instructions}
        {module_content_instructions}
        instructions end

        Section:
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
module_intro.parser = TagsParser(min_len=10)

module_actions = PromptFormatter(
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
        {journey_content_instructions}
        {section_content_instructions}
        {module_content_instructions}
        instructions end

        Amount:
        {amount}

        Section:
        {subject}

        content start
        {context}
        content end

        Write a list of subjects with their content to teach the subject to the student.
        Prepare also a list of reference documents and their summary to use with each subject when teaching the student about the subject.
        If instructions are provided, follow them exactly. If instructions specify a topic or subject, make sure the list includes only
        items which fall within within that topic. Create at maximum the specified amount of items.
        If there's a history with previous titles or subjects, use them to make sure you don't repeat the same subjects.
        """
    ),
)
module_actions.parser = TagsParser(min_len=10)

action_details = PromptFormatter(
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
        {journey_content_instructions}
        {section_content_instructions}
        {module_content_instructions}
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
action_details.parser = TagsParser(min_len=20)


class JourneyPrompts(BaseModel):
    plan: CustomPrompt = Field(default=CustomPrompt(system=plan.system, user=plan.user))
    module_content: CustomPrompt = Field(
        default=CustomPrompt(system=module_content.system, user=module_content.user)
    )
    module_intro: CustomPrompt = Field(
        default=CustomPrompt(system=module_intro.system, user=module_intro.user)
    )
    module_actions: CustomPrompt = Field(
        default=CustomPrompt(system=module_actions.system, user=module_actions.user)
    )
    action_details: CustomPrompt = Field(
        default=CustomPrompt(
            system=action_details.system,
            user=action_details.user,
        )
    )


def convert_to_journey_prompts(container: CustomPromptContainer) -> JourneyPrompts:
    return JourneyPrompts(
        plan=container.plan,
        module_content=container.module_content,
        module_intro=container.module_intro,
        module_actions=container.module_actions,
        action_details=container.action_details,
    )


journey_template_dir = os.path.join(
    SETTINGS.file_repository_path, "journey_structures_json"
)

with open(os.path.join(journey_template_dir, "matched_descriptions.json"), "r") as f:
    matched_descriptions: dict = json.load(f)

selected_roles = [
    "Technical Program Manager",
    "Strategy Consultant",
    "Data Scientist",
    "Product Manager",
    "Software Engineer",
]

matched_descriptions = [
    (k, v) for k, v in matched_descriptions.items() if k in selected_roles
]
matched_descriptions_str = "\n".join(
    [
        f"Example {i+1}:\n\nJob description:\n{v}\n\nOutput:\n<role>{v}</role>"
        for i, (k, v) in enumerate(matched_descriptions)
    ]
)

all_available_roles = get_available_journey_template_roles(as_str=True)
print(all_available_roles)
journey_template_selector = PromptFormatter(
    system=textwrap.dedent(
        f"""
        Act as a role assigner who is defining role title based on job description.
        {PRE_THINK_INSTRUCT}
        Use <thinking>-tag to identify the best possible match for the title.
        Explain your reasoning using <reflect>-tag.

        Output your selected role within <role>-tag.

        <role>-tag is required and it can only contain the selected role.
        {KEEP_PRE_THINK_TOGETHER}

        Use the following roles and return one of them based on the job description.

        Available roles by category:
        {textwrap.indent(all_available_roles, "        ")}

        Use following examples to understand the job description and the role title:

        {textwrap.indent(matched_descriptions_str, "        ")}
        """
    ),
    user=textwrap.dedent(
        """
        Job description:
        {job_description}

        Output:
        """
    ),
)
journey_template_selector.parser = TagsParser(
    min_len=10,
    return_tag=True,
    tags=["role", "thinking", "reflect"],
    required_output_tag="role",
    required_output_values=list(
        set(
            value
            for sublist in get_available_journey_template_roles().values()
            for value in sublist
        )
    ),
)
