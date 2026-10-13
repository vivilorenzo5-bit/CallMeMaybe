import json
import sys
from pathlib import Path
from pydantic import ValidationError
from llm_sdk import Small_LLM_Model
from src.config import AppConfig
from src.schema.function_def import FunctionDefinition
from src.schema.test_prompt import TestPrompt
from src.vocab.manager import VocabManager


def load_json_file(path: Path) -> list[dict[str, object]]:
    """Safely load and parse a JSON array from a given file path.

    Args:
        path: Path to the target JSON file.

    Returns:
        Parsed list of JSON objects.
    """
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        try:
            content = json.load(stream)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in file '{path}': {exc}"
            raise ValueError(msg) from exc

    if not isinstance(content, list):
        msg = (
            f"Expected root array in JSON file '{path}', "
            f"got {type(content).__name__}"
        )
        raise ValueError(msg)

    return content


def main() -> int:
    """Execute the main initialization and validation pipeline.

    Returns:
        Status code (0 for success, 1 for handled error).
    """
    try:
        config = AppConfig.from_cli_args()

        raw_functions = load_json_file(config.functions_definition_path)
        function_definitions = [
            FunctionDefinition.model_validate(fn)
            for fn in raw_functions
        ]

        raw_prompts = load_json_file(config.input_path)
        test_prompts = [
            TestPrompt.model_validate(prompt)
            for prompt in raw_prompts
        ]

        config.output_path.parent.mkdir(parents=True, exist_ok=True)

        print(
            f"[OK] Schemas loaded: {len(function_definitions)} functions, "
            f"{len(test_prompts)} prompts."
        )

        print("[OK] Initializing model and vocabulary indexing...")
        model = Small_LLM_Model()
        vocab_file = Path(model.get_path_to_vocab_file())
        vocab_mgr = VocabManager.from_vocab_file(vocab_file)

        print(
            f"[OK] Vocabulary indexed:\n"
            f"     - Total tokens: {vocab_mgr.vocab_size}\n"
            f"     - Numeric tokens: {len(vocab_mgr.number_tokens)}\n"
            f"     - Boolean tokens: {len(vocab_mgr.boolean_tokens)}"
        )

        test_fn_name = function_definitions[0].name
        prefixes = vocab_mgr.trie.get_tokens_that_are_prefixes(
            test_fn_name
        )
        decoded_prefixes = [
            vocab_mgr.decode_token_id(t) for t in prefixes[:5]
        ]
        print(
            f"[OK] Trie prefix test for '{test_fn_name}': "
            f"{decoded_prefixes}"
        )

        return 0

    except ValidationError as err:
        sys.stderr.write(f"Schema Validation Error:\n{err}\n")
        return 1
    except (FileNotFoundError, ValueError) as err:
        sys.stderr.write(f"Configuration/IO Error: {err}\n")
        return 1
    except Exception as err:
        sys.stderr.write(f"Unexpected Fatal Error: {err}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
