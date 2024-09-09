import json
import os
import textwrap
from pydantic import BaseModel, Field
import yaml
from langchain_core.output_parsers import PydanticOutputParser

from lib.prompts.base import PromptFormatter


current_dir = os.path.dirname(os.path.abspath(__file__))


class CustomPrompt(BaseModel):
    system: str = Field(default=None, description="System message to set the behavior of the assistant.")
    user: str = Field(default=None, description="User message for the LLM generation")


class CustomPromptContainer(BaseModel):
    steps: CustomPrompt = Field(description="Description of the steps involved in the scenario.")
    step_content: CustomPrompt = Field(description="Detailed explanation of each step in the scenario.")
    step_intro: CustomPrompt = Field(description="Introduction to the scenario, setting the context.")
    step_actions: CustomPrompt = Field(description="Actions that should be taken in each step of the scenario.")
    step_action_details: CustomPrompt = Field(description="Specific details about the actions to be taken in each step.")

def read_and_load_yaml(file_path):
    with open(file_path, "r") as file:
        content = file.read().replace('\t', '    ')
        data = yaml.safe_load(content)
    return data

hr_rep_values = read_and_load_yaml(os.path.join(current_dir, "prompt_templates/hr_rep-values.yaml"))
hr_rep_values_instruct = {
    "actor": "HR Representative",
    "target": "Explain the company's core values to a new employee.",
}

sales_rep_products = read_and_load_yaml(os.path.join(current_dir, "prompt_templates/sales_rep-products.yaml"))
sales_rep_products_instruct = {
    "actor": "Sales Representative",
    "target": "Pitch a product to a potential customer.",
}

teacher_class_curriculum = read_and_load_yaml(os.path.join(current_dir, "prompt_templates/teacher-class_curriculum.yaml"))
teacher_class_curriculum_instruct = {
    "actor": "Teacher",
    "target": "Create a class curriculum for students.",
}

journey_prompts = PromptFormatter(
    system=textwrap.dedent(
        f"""
        You are an expert Prompt Writer for Large Language Models.

        Use following examples as a guide on how to write prompts for different actors and targets.

        Export prompts in the define format using JSON.

        Example 1:
        Actor:
        {hr_rep_values_instruct["actor"]}
        Target:
        {hr_rep_values_instruct["target"]}
        Output
        {json.dumps(hr_rep_values, indent=4)}

        Example 2:
        Actor:
        {sales_rep_products_instruct["actor"]}
        Target:
        {sales_rep_products_instruct["target"]}
        Output
        {json.dumps(sales_rep_products, indent=4)}

        Example 3:
        Actor:
        {teacher_class_curriculum_instruct["actor"]}
        Target:
        {teacher_class_curriculum_instruct["target"]}
        Output
        {json.dumps(teacher_class_curriculum, indent=4)}
        """
    ),
    user=textwrap.dedent(  # Use get_journey_format_example instead
        """
        Actor:
        {actor}
        Target:
        {target}
        Output:
        """
    ),
)
journey_prompts.parser = PydanticOutputParser(pydantic_object=CustomPromptContainer)
