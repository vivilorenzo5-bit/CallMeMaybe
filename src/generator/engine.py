import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from llm_sdk import Small_LLM_Model
from src.fsm.json_fsm import JsonFSM
from src.generator.masker import LogitsMasker
from src.schema.function_def import FunctionDefinition
from src.vocab.manager import VocabManager


class ConstrainedEngine(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model
    vocab_mgr: VocabManager
    function_defs: dict[str, FunctionDefinition]
    max_new_tokens: int = Field(default=256)

    def _build_prompt(self, user_prompt: str) -> str:
        tools_lines: list[str] = []
        for fn in self.function_defs.values():
            params_str = ", ".join(
                f"{p_name}: {p_info.type}"
                for p_name, p_info in fn.parameters.items()
            )
            tools_lines.append(
                f"- {fn.name}({params_str}): {fn.description}"
            )
        tools_desc = "\n".join(tools_lines)

        return (
            f"<|im_start|>system\n"
            f"You are a helpful assistant that calls functions. "
            f"Select the appropriate tool and extract arguments exactly "
            f"from the user query without adding extra text.\n\n"
            f"Tools:\n{tools_desc}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def generate_function_call(
            self, prompt_text: str
    ) -> dict[str, object]:
        formatted_prompt = self._build_prompt(prompt_text)
        encoded = np.asarray(
            self.model.encode(formatted_prompt)).flatten()
        input_ids: list[int] = [int(tok) for tok in encoded]

        fsm = JsonFSM(
            function_defs=self.function_defs,
            vocab_mgr=self.vocab_mgr
        )
        initial_len = len(input_ids)

        while not fsm.is_done():
            if len(input_ids) - initial_len >= self.max_new_tokens:
                msg = f"Generation exceeded max tokens ({self.max_new_tokens})"
                raise RuntimeError(msg)

            raw_logits = self.model.get_logits_from_input_ids(input_ids)
            allowed_tokens = fsm.get_allowed_tokens()

            if not allowed_tokens:
                msg = f"FSM dead-end at state: {fsm.context.current_state}"
                raise RuntimeError(msg)

            masked_logits = LogitsMasker.apply_mask(
                raw_logits, allowed_tokens
            )
            next_token = LogitsMasker.select_best_token(masked_logits)

            # token_str = self.vocab_mgr.decode_token_id(next_token)
            # step_idx = len(input_ids) - initial_len
            # print(
            #     f"[DEBUG Step {step_idx:02d}] "
            #     f"Token: {repr(token_str):<12} "
            #     f"(ID {next_token}) | "
            #     f"State: {fsm.context.current_state}"
            # )
            fsm.consume_token(next_token)
            input_ids.append(next_token)

        fn_name: str = ""
        if fsm.context.selected_function is not None:
            fn_name = fsm.context.selected_function.name

        return {
            "name": fn_name,
            "parameters": fsm.context.parsed_parameters
        }
