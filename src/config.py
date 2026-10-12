import argparse
from pathlib import Path
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    functions_definition_path: Path = Field(
        default=Path("data/input/functions_definition.json")
    )
    input_path: Path = Field(
        default=Path("data/input/function_calling_tests.json")
    )
    output_path: Path = Field(
        default=Path("data/output/function_calling_results.json")
    )

    @classmethod
    def from_cli_args(cls, args: list[str] | None = None) -> "AppConfig":
        parser = argparse.ArgumentParser(
            description="Call Me Maybe - LLM Function Calling System"
        )
        parser.add_argument(
            "--functions_definition",
            type=Path,
            default=Path("data/input/functions_definition.json"),
            help="Path to functions definition JSON file.",
        )
        parser.add_argument(
            "--input",
            type=Path,
            default=Path("data/input/function_calling_tests.json"),
            help="Path to input test prompts JSON file.",
        )
        parser.add_argument(
            "--output",
            type=Path,
            default=Path("data/output/function_calling_results.json"),
            help="Path to output results JSON file.",
        )

        parsed = parser.parse_args(args)
        return cls(
            functions_definition_path=parsed.functions_definition,
            input_path=parsed.input,
            output_path=parsed.output,
        )
