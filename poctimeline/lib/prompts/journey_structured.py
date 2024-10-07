from langchain_core.output_parsers import PydanticOutputParser
from lib.models.journey import ModuleStructure
from lib.prompts.actions import structured

module_structured = structured.customize()
module_structured.parser = PydanticOutputParser(pydantic_object=ModuleStructure)
