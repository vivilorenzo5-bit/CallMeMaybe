from typing import Optional
from pydantic import BaseModel, Field
from src.fsm.state import FSMContext, FSMStateType
from src.schema.function_def import FunctionDefinition
from src.vocab.manager import VocabManager


class JsonFSM(BaseModel):
    function_defs: dict[str, FunctionDefinition]
    vocab_mgr: VocabManager
    context: FSMContext = Field(default_factory=FSMContext)

    def is_done(self) -> bool:
        return self.context.current_state == FSMStateType.DONE

    def get_allowed_tokens(self) -> list[int]:
        state = self.context.current_state

        if state in (
            FSMStateType.START,
            FSMStateType.AFTER_FUNCTION_NAME,
            FSMStateType.COLON
        ):
            return self._get_static_buffer_tokens()

        if state == FSMStateType.FUNCTION_NAME:
            return self._get_function_name_tokens()

        if state == FSMStateType.PARAM_KEY:
            return self._get_param_key_tokens()

        if state == FSMStateType.PARAM_VALUE:
            return self._get_param_value_tokens()

        if state == FSMStateType.PARAM_DELIMITER:
            return self._get_static_buffer_tokens()

        return []

    def consume_token(self, token_id: int) -> None:
        token_str = self.vocab_mgr.decode_token_id(token_id)
        state = self.context.current_state

        if state in (
            FSMStateType.START,
            FSMStateType.AFTER_FUNCTION_NAME,
            FSMStateType.COLON,
            FSMStateType.PARAM_DELIMITER
        ):
            self._consume_static_buffer(token_str)
            return

        if state == FSMStateType.FUNCTION_NAME:
            self._consume_function_name(token_str)
            return

        if state == FSMStateType.PARAM_KEY:
            self._consume_param_key(token_str)
            return

        if state == FSMStateType.PARAM_VALUE:
            self._consume_param_value(token_str)
            return

    def _get_static_buffer_tokens(self) -> list[int]:
        buffer = self.context.static_buffer
        if not buffer:
            return []
        return self.vocab_mgr.trie.get_tokens_that_are_prefixes(buffer)

    def _consume_static_buffer(self, token_str: str) -> None:
        buf = self.context.static_buffer
        if buf.startswith(token_str):
            self.context.static_buffer = buf[len(token_str):]
        else:
            self.context.static_buffer = ""

        if not self.context.static_buffer:
            self._advance_from_static_state()

    def _advance_from_static_state(self) -> None:
        state = self.context.current_state

        if state == FSMStateType.START:
            self.context.current_state = FSMStateType.FUNCTION_NAME
            self.context.accumulated_text = ""

        elif state == FSMStateType.AFTER_FUNCTION_NAME:
            fn = self.context.selected_function
            if fn and fn.parameters:
                self.context.remaining_params = list(fn.parameters.keys())
                self.context.current_state = FSMStateType.PARAM_KEY
                self.context.accumulated_text = ""
            else:
                self.context.static_buffer = "}}"
                self.context.current_state = FSMStateType.PARAM_DELIMITER

            