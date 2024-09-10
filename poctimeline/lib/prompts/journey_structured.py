from langchain_core.output_parsers import PydanticOutputParser
from lib.models.journey import StepStructure
from lib.prompts.actions import structured

step_structured = structured.customize()
step_structured.parser = PydanticOutputParser(pydantic_object=StepStructure)
