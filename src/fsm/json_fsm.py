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

        elif state == FSMStateType.COLON:
            self.context.current_state = FSMStateType.PARAM_VALUE
            self.context.accumulated_text = ""
            self.context.in_string_literal = False

        elif state == FSMStateType.PARAM_DELIMITER:
            if self.context.remaining_params:
                self.context.current_state = FSMStateType.PARAM_KEY
                self.context.accumulated_text = ""
            else:
                self.context.current_state = FSMStateType.DONE

    def _get_function_name_tokens(self) -> list[int]:
        acc = self.context.accumulated_text
        valid_tokens: set[int] = set()

        for fn_name in self.function_defs:
            if fn_name.startswith(acc):
                remainder = fn_name[len(acc):]
                if remainder:
                    matches = (
                        self.vocab_mgr.trie.get_tokens_that_are_prefixes(
                            remainder
                        )
                    )
                    valid_tokens.update(matches)

        return list(valid_tokens)

    def _consume_function_name(self, token_str: str) -> None:
        self.context.accumulated_text += token_str
        acc = self.context.accumulated_text

        if acc in self.function_defs:
            self.context.selected_function = self.function_defs[acc]
            self.context.current_state = FSMStateType.AFTER_FUNCTION_NAME
            self.context.static_buffer = '", "parameters": {'
            self.context.accumulated_text = ""

    def _get_param_key_tokens(self) -> list[int]:
        acc = self.context.accumulated_text
        valid_tokens: set[int] = set()

        for param in self.context.remaining_params:
            target = f'"{param}"'
            if target.startswith(acc):
                remainder = target[len(acc):]
                if remainder:
                    matches = (
                        self.vocab_mgr.trie.get_tokens_that_are_prefixes(
                            remainder
                        )
                    )
                    valid_tokens.update(matches)

        return list(valid_tokens)

    def _consume_param_key(self, token_str: str) -> None:
        self.context.accumulated_text += token_str
        acc = self.context.accumulated_text

        for param in self.context.remaining_params:
            if acc == f'"{param}"':
                self.context.current_param_name = param
                fn = self.context.selected_function
                if fn:
                    self.context.current_param_type = fn.get_parameter_type(
                        param
                    )
                self.context.remaining_params.remove(param)
                self.context.current_state = FSMStateType.COLON
                self.context.static_buffer = ": "
                self.context.accumulated_text = ""
                return

    def _get_param_value_tokens(self) -> list[int]:
        param_type = self.context.current_param_type

        if param_type in ("number", "integer", "float"):
            allowed = list(self.vocab_mgr.number_tokens)
            if self.context.accumulated_text:
                if self.context.remaining_params:
                    comma_tokens = (
                        self.vocab_mgr.trie.get_tokens_that_are_prefixes(",")
                    )
                    allowed.extend(comma_tokens)
                else:
                    brace_tokens = (
                        self.vocab_mgr.trie.get_tokens_that_are_prefixes("}")
                    )
                    allowed.extend(brace_tokens)
            return allowed

        if param_type == "boolean":
            return list(self.vocab_mgr.boolean_tokens)

        if not self.context.in_string_literal:
            return [self.vocab_mgr.quote_token_id]

        closing_tokens: list[int] = [self.vocab_mgr.quote_token_id]
        if self.context.remaining_params:
            closing_tokens.extend(
                self.vocab_mgr.trie.get_tokens_that_are_prefixes('",')
            )
        else:
            closing_tokens.extend(
                self.vocab_mgr.trie.get_tokens_that_are_prefixes('"}')
            )
            closing_tokens.extend(
                self.vocab_mgr.trie.get_tokens_that_are_prefixes('"}}')
            )

        if len(self.context.accumulated_text) >= 60:
            return closing_tokens

        allowed_str = list(self.vocab_mgr.safe_string_tokens)
        backslash_tokens = (
            self.vocab_mgr.trie.get_tokens_that_are_prefixes("\\")
        )
        allowed_str.extend(backslash_tokens)
        allowed_str.extend(closing_tokens)

        return allowed_str

    def _consume_param_value(self, token_str: str) -> None:
        param_type = self.context.current_param_type

        if param_type in ("number", "integer", "float"):
            if "," in token_str:
                raw_num = self.context.accumulated_text.strip()
                self._record_parsed_value(raw_num)
                self.context.accumulated_text = ""
                self.context.static_buffer = " "
                self.context.current_state = FSMStateType.PARAM_DELIMITER
                return

            if "}" in token_str:
                raw_num = self.context.accumulated_text.strip()
                self._record_parsed_value(raw_num)
                self.context.accumulated_text = ""
                self.context.static_buffer = "}"
                self.context.current_state = FSMStateType.PARAM_DELIMITER
                return

            self.context.accumulated_text += token_str

        elif param_type == "boolean":
            self.context.accumulated_text += token_str
            acc = self.context.accumulated_text
            if acc in ("true", "false"):
                self._record_parsed_value(acc)
                self._transition_after_param_value()

        elif param_type == "string":
            if not self.context.in_string_literal:
                if '"' in token_str:
                    self.context.in_string_literal = True
                    self.context.accumulated_text = ""
                return

            is_escaped = (
                self.context.accumulated_text.endswith("\\")
                and not self.context.accumulated_text.endswith("\\\\")
            )
            if '"' in token_str and is_escaped:
                self.context.accumulated_text += token_str
                return

            if '"' in token_str:
                before_quote = token_str.split('"', 1)[0]
                after_quote = token_str.split('"', 1)[1]
                self.context.accumulated_text += before_quote

                clean_str = self.context.accumulated_text.strip()
                clean_str = clean_str.replace("\\\\", "\\")
                clean_str = clean_str.replace('\\"', '"')

                self._record_parsed_value(clean_str)
                self._transition_after_param_value()

                if (
                    "," in after_quote
                    and self.context.static_buffer.startswith(",")
                ):
                    self.context.static_buffer = self.context.static_buffer[1:]
                elif "}" in after_quote:
                    num_braces = after_quote.count("}")
                    for _ in range(num_braces):
                        if self.context.static_buffer.startswith("}"):
                            self.context.static_buffer = (
                                self.context.static_buffer[1:]
                            )
                    if not self.context.static_buffer:
                        self.context.current_state = FSMStateType.DONE
                return

            self.context.accumulated_text += token_str

    def _record_parsed_value(self, raw_value: str) -> None:
        param_name = self.context.current_param_name
        param_type = self.context.current_param_type
        if not param_name:
            return

        if param_type == "integer":
            try:
                val: object = int(float(raw_value))
            except ValueError:
                val = 0
            self.context.parsed_parameters[param_name] = val
        elif param_type in ("number", "float"):
            try:
                val = float(raw_value)
            except ValueError:
                val = 0.0
            self.context.parsed_parameters[param_name] = val
        elif param_type == "boolean":
            self.context.parsed_parameters[param_name] = raw_value == "true"
        else:
            self.context.parsed_parameters[param_name] = raw_value

    def _transition_after_param_value(self) -> None:
        self.context.accumulated_text = ""
        self.context.in_string_literal = False

        if self.context.remaining_params:
            self.context.static_buffer = ", "
            self.context.current_state = FSMStateType.PARAM_DELIMITER
        else:
            self.context.static_buffer = "}}"
            self.context.current_state = FSMStateType.PARAM_DELIMITER
