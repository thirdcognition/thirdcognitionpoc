from langchain_core.output_parsers import PydanticOutputParser
from lib.models.journey import StepStructure
from lib.prompts.actions import structured


journey_structured = structured.customize()
journey_structured.parser = PydanticOutputParser(pydantic_object=StepStructure)