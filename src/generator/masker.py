from typing import Sequence, Union
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class LogitsMasker(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def apply_mask(
        logits: Union[Sequence[float], NDArray[np.float32]],
        allowed_token_ids: Sequence[int]
    ) -> NDArray[np.float32]:
        if not allowed_token_ids:
            msg = "Cannot apply mask with zero allowed token IDs."
            raise ValueError(msg)

        raw_arr: NDArray[np.float32] = np.asarray(
            logits, dtype=np.float32
        )
        flat_logits: NDArray[np.float32] = np.squeeze(raw_arr)
        if flat_logits.ndim != 1:
            flat_logits = flat_logits[-1]

        masked: NDArray[np.float32] = np.full(
            flat_logits.shape, -np.inf, dtype=np.float32
        )

        valid_indices = list(allowed_token_ids)
        masked[valid_indices] = flat_logits[valid_indices]

        return masked

    @staticmethod
    def select_best_token(masked_logits: NDArray[np.float32]) -> int:
        return int(np.argmax(masked_logits))
