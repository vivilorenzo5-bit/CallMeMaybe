import json
from pathlib import Path
from pydantic import BaseModel, Field
from src.vocab.trie import Trie


def _bytes_to_unicode_mapping() -> dict[str, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    forward_map = {b: chr(c) for b, c in zip(bs, cs)}
    return {v: chr(k) for k, v in forward_map.items()}


class VocabManager(BaseModel):
    vocab_size: int = Field(default=0)
    id_to_token: dict[int, str] = Field(default_factory=dict)
    token_to_id: dict[str, int] = Field(default_factory=dict)
    trie: Trie = Field(default_factory=Trie)
    number_tokens: list[int] = Field(default_factory=list)
    boolean_tokens: list[int] = Field(default_factory=list)
    quote_token_ids: list[int] = Field(default_factory=list)

    @classmethod
    def from_vocab_file(cls, vocab_file_path: Path) -> "VocabManager":
        if not vocab_file_path.is_file():
            msg = f"Vocab file not found: {vocab_file_path}"
            raise FileNotFoundError(msg)
        with vocab_file_path.open("r", encoding="utf-8") as stream:
            try:
                raw_content: object = json.load(stream)
            except json.JSONDecodeError as exc:
                msg = f"Corrupted vocab file '{vocab_file_path}': {exc}"
                raise ValueError(msg) from exc
        if not isinstance(raw_content, dict):
            msg = (
                f"Expected dict in vocab file '{vocab_file_path}', "
                f"got {type(raw_content).__name__}"
            )
            raise ValueError(msg)

        raw_vocab: dict[str, int] = raw_content
        reverse_map = _bytes_to_unicode_mapping()
        id_to_token: dict[int, str] = {}
        token_to_id: dict[str, int] = {}
        trie = Trie()
        number_tokens: list[int] = []
        boolean_tokens: list[int] = []
        quote_token_ids: list[int] = []

        allowed_number_chars: set[str] = set("0123456789.-+eE")

        for raw_token, token_id in raw_vocab.items():
            decoded_chars = [reverse_map.get(ch, ch) for ch in raw_token]
            decoded_token = "".join(decoded_chars)

            id_to_token[token_id] = decoded_token
            if decoded_token not in token_to_id:
                token_to_id[decoded_token] = token_id

            trie.insert(decoded_token, token_id)

            if decoded_token and all(
                c in allowed_number_chars for c in decoded_token
            ):
                number_tokens.append(token_id)
            if (
                "true".startswith(decoded_token)
                or "false".startswith(decoded_token)
            ):
                if decoded_token:
                    boolean_tokens.append(token_id)
            if '"' in decoded_token:
                quote_token_ids.append(token_id)

        return cls(
            vocab_size=len(id_to_token),
            id_to_token=id_to_token,
            token_to_id=token_to_id,
            trie=trie,
            number_tokens=number_tokens,
            boolean_tokens=boolean_tokens,
            quote_token_ids=quote_token_ids
        )

    def decode_token_id(self, token_id: int) -> str:
        return self.id_to_token.get(token_id, "")
