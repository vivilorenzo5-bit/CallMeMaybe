import json
import sys
import traceback
from pathlib import Path
from pydantic import ValidationError
from llm_sdk import Small_LLM_Model
from src.config import AppConfig
from src.generator.engine import ConstrainedEngine
from src.schema.function_def import FunctionDefinition
from src.schema.test_prompt import TestPrompt
from src.vocab.manager import VocabManager


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
            content: object = json.load(stream)
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
        fn_map = {fn.name: fn for fn in function_definitions}

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

        print("[OK] Initializing Constrained Decoding Engine...")
        engine = ConstrainedEngine(
            model=model,
            vocab_mgr=vocab_mgr,
            function_defs=fn_map,
        )

        results: list[dict[str, object]] = []
        total = len(test_prompts)

        print(f"\n[RUN] Generating function calls for {total} prompts...")

        for idx, item in enumerate(test_prompts, start=1):
            print(f"[{idx:02d}/{total:02d}] Processing: \"{item.prompt}\"")
            generated = engine.generate_function_call(item.prompt)

            entry: dict[str, object] = {
                "prompt": item.prompt,
                "name": generated["name"],
                "parameters": generated["parameters"],
            }
            results.append(entry)

        with config.output_path.open("w", encoding="utf-8") as out_file:
            json.dump(results, out_file, indent=2, ensure_ascii=False)

        print(f"\n[OK] Successfully wrote {len(results)} calls to:")
        print(f"     -> {config.output_path}")

        return 0

    except ValidationError as err:
        sys.stderr.write(f"Schema Validation Error:\n{err}\n")
        return 1
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        sys.stderr.write(f"Engine/IO Error: {err}\n")
        return 1
    except Exception as err:
        sys.stderr.write(f"Unexpected Fatal Error: {err}\n")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
