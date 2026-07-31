"""Archived prototype prefix index; production ownership is in KV Block Manager."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _TrieNode:
    children: dict[int, "_TrieNode"] = field(default_factory=dict)
    resident_slots: set[int] = field(default_factory=set)


class PrefixIndex:
    def __init__(self) -> None:
        self._root = _TrieNode()
        self._slot_tokens: dict[int, tuple[int, ...]] = {}

    def upsert(self, slot_id: int, tokens: tuple[int, ...]) -> None:
        self.remove(slot_id)
        self._slot_tokens[slot_id] = tokens
        node = self._root
        for token in tokens:
            node = node.children.setdefault(token, _TrieNode())
            node.resident_slots.add(slot_id)

    def remove(self, slot_id: int) -> None:
        tokens = self._slot_tokens.pop(slot_id, None)
        if tokens is None:
            return
        path: list[tuple[_TrieNode, int, _TrieNode]] = []
        node = self._root
        for token in tokens:
            child = node.children[token]
            path.append((node, token, child))
            child.resident_slots.discard(slot_id)
            node = child
        for parent, token, child in reversed(path):
            if child.resident_slots or child.children:
                break
            del parent.children[token]

    def match_lengths(self, tokens: tuple[int, ...]) -> dict[int, int]:
        matches: dict[int, int] = {}
        node = self._root
        for depth, token in enumerate(tokens, start=1):
            child = node.children.get(token)
            if child is None:
                break
            for slot_id in child.resident_slots:
                matches[slot_id] = depth
            node = child
        return matches

    def tokens_for(self, slot_id: int) -> tuple[int, ...]:
        return self._slot_tokens.get(slot_id, ())

    def __len__(self) -> int:
        return len(self._slot_tokens)
