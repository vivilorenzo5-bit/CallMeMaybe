"""Main entry point for running the call-me-maybe application."""

import json
import sys
from pathlib import Path
from pydantic import ValidationError
from src.config import AppConfig
from src.schema.function_def import FunctionDefinition
from src.schema.test_prompt import TestPrompt


def load_json_file(path: Path) -> list[dict[str, object]]:
    """Safely load and parse a JSON array from a given file path.

    Args:
        path: Path to the target JSON file.

    Returns:
        Parsed list of JSON objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON is invalid or root is not an array.
    """
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        try:
            content = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in file '{path}': {exc}") from exc

    if not isinstance(content, list):
        raise ValueError(f"Expected root array in JSON file '{path}', got {type(content).__name__}")

    return content  # type: ignore[no-any-return]


def main() -> int:
    """Execute the main initialization and validation pipeline.

    Returns:
        Status code (0 for success, 1 for handled error).
    """
    try:
        config = AppConfig.from_cli_args()

        raw_functions = load_json_file(config.functions_definition_path)
        function_definitions = [
            FunctionDefinition.model_validate(fn) for fn in raw_functions
        ]

        raw_prompts = load_json_file(config.input_path)
        test_prompts = [
            TestPrompt.model_validate(prompt) for prompt in raw_prompts
        ]

        # Ensure output directory exists without committing it
        config.output_path.parent.mkdir(parents=True, exist_ok=True)

        print(
            f"[OK] Initialized successfully.\n"
            f"     - Loaded {len(function_definitions)} functions.\n"
            f"     - Loaded {len(test_prompts)} prompts.\n"
            f"     - Target output: {config.output_path}"
        )
        return 0

    except (FileNotFoundError, ValueError) as err:
        sys.stderr.write(f"Configuration/IO Error: {err}\n")
        return 1
    except ValidationError as err:
        sys.stderr.write(f"Schema Validation Error:\n{err}\n")
        return 1
    except Exception as err:
        sys.stderr.write(f"Unexpected Fatal Error: {err}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())