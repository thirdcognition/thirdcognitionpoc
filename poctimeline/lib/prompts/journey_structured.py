from langchain_core.output_parsers import PydanticOutputParser
from lib.models.journey import SubsubjectStructure
from lib.prompts.actions import structured

subsubject_structured = structured.customize()
subsubject_structured.parser = PydanticOutputParser(pydantic_object=SubsubjectStructure)
