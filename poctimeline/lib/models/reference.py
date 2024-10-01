from enum import Enum
import json
from typing import List, Optional
from pydantic import BaseModel, Field


class ReferenceType(Enum):
    source = "source"
    concept = "concept"
    topic = "topic"
    taxonomy = "taxonomy"

    @classmethod
    def from_str(cls, value: str):
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"{value} is not a valid ReferenceType")


class Reference(BaseModel):
    id: str = Field(description="The id of the reference", title="Reference ID")
    type: ReferenceType = Field(
        description="The type of the reference", title="Reference Type"
    )
    index: int = Field(
        description="The index, or e.g. page number of the reference if applicable",
        title="Reference Index",
    )

    @classmethod
    def from_str(cls, value: str):
        value = (
            value.replace("Reference(id(", "")
            .replace("),type(", ",")
            .replace("),index(", ",")
            .rstrip(")")
        )
        id, type, index = value.split(",")
        return cls(id=id, type=ReferenceType.from_str(type), index=int(index))

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        return cls.model_validate(data)

    def __str__(self):
        return f"Reference(id({self.id}),type({self.type.value}),index({self.index}))"

    def to_json(self):
        return self.model_dump(mode='json')

def type_in_references(
    id: str, references: list[Reference], type: ReferenceType = ReferenceType.source
) -> bool:
    for reference in references:
        if reference.type == type and reference.id == id:
            return True
    return False


def unique_references(references: list[Reference]) -> list[Reference]:
    return [Reference.from_str(item) for item in set([str(ref) for ref in references])]


def is_in_references(
    id: str,
    type: ReferenceType,
    references: List[Reference],
    index: Optional[int] = None,
) -> bool:
    for ref in references:
        if ref.id == id and ref.type == type:
            if index is None or ref.index == index:
                return True
    return False