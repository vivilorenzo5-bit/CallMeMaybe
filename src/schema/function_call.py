from typing import Any
from pydantic import BaseModel, Field


class FunctionCallResult(BaseModel):
    prompt: str = Field(min_length=1)
    name: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
