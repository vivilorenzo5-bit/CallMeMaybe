from pydantic import BaseModel, Field


class TestPrompt(BaseModel):
    prompt: str = Field(min_length=1)
