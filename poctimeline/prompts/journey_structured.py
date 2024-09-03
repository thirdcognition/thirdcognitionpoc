from langchain_core.output_parsers import PydanticOutputParser
from models.journey import SubjectStructure
from prompts.actions import structured


journey_structured = structured.customize()
journey_structured.parser = PydanticOutputParser(pydantic_object=SubjectStructure)