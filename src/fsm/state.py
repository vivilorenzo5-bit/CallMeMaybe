from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from src.schema.function_def import FunctionDefinition, ParameterType


class FSMStateType(str, Enum):
    START = "START"
    FUNCTION_NAME = "FUNCTION_NAME"
    AFTER_FUNCTION_NAME = "AFTER_FUNCTION_NAME"
    PARAM_KEY = "PARAM_KEY"
    COLON = "COLON"
    PARAM_VALUE = "PARAM_VALUE"
    PARAM_DELIMITER = "PARAM_DELIMITER"
    DONE = "DONE"


class FSMContext(BaseModel):
    """Runtime tracking context for the active function call generation.

    Attributes:
        current_state: Active state in the state machine.
        selected_function: Function definition once chosen by the model.
        static_buffer: Remaining static syntax characters to emit.
        accumulated_text: Text accumulated in the current dynamic state.
        remaining_params: Set of parameter names yet to be generated.
        current_param_name: The parameter currently being generated.
        current_param_type: Expected data type of the current parameter.
        in_string_literal: Flag indicating if a string value is open.
        parsed_parameters: Collected parameter key-value pairs.
    """

    current_state: FSMStateType = Field(default=FSMStateType.START)
    selected_function: Optional[FunctionDefinition] = None
    static_buffer: str = Field(default='{"name": "')
    accumulated_text: str = Field(default="")
    remaining_params: list[str] = Field(default_factory=list)
    current_param_name: Optional[str] = None
    current_param_type: Optional[ParameterType] = None
    in_string_literal: bool = Field(default=False)
    parsed_parameters: dict[str, object] = Field(default_factory=dict)
