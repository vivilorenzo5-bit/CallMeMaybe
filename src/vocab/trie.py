from pydantic import BaseModel, Field


class TrieNode(BaseModel):
    """Represents a node in the prefix tree."""
    children: dict[str, "TrieNode"] = Field(default_factory=dict)
    token_ids: list[int] = Field(default_factory=list)


class Trie(BaseModel):
    root: TrieNode = Field(default_factory=TrieNode)

    def insert(self, text: str, token_id: int) -> None:
        if not text:
            return
        current: TrieNode = self.root
        for char in text:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.token_ids.append(token_id)

    def get_tokens_that_are_prefixes(self, target: str) -> list[int]:
        valid_ids: list[int] = []
        current: TrieNode = self.root

        for char in target:
            if char not in current.children:
                break
            current = current.children[char]
            if current.token_ids:
                valid_ids.extend(current.token_ids)
        return valid_ids

    def get_tokens_matching_prefix(self, prefix: str) -> list[int]:
        current: TrieNode = self.root
        for char in prefix:
            if char not in current.children:
                return []
            current = current.children[char]

        results: list[int] = []
        stack: list[TrieNode] = [current]
        while stack:
            node = stack.pop()
            if node.token_ids:
                results.extend(node.token_ids)
            stack.extend(node.children.values())
        return results
