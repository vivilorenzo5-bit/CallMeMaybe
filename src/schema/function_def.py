from typing import Literal
from pydantic import BaseModel, Field


ParameterType = Literal["number", "integer", "float", "string", "boolean"]


class ParameterProperty(BaseModel):
    type: ParameterType
    description: str | None = None


class ReturnProperty(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ParameterProperty] = Field(default_factory=dict)
    returns: ReturnProperty | None = None

    def get_parameter_names(self) -> list[str]:
        return list(self.parameters.keys())

    def get_parameter_type(self, param_name: str) -> ParameterType | None:
        prop = self.parameters.get(param_name)
        return prop.type if prop else None
