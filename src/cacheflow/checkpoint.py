from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class CheckpointEntry:
    conversation_id: str
    model: str
    filename: str
    size_bytes: int
    tokens: tuple[int, ...]
    last_access: float


class CheckpointStore:
    def __init__(self, byte_budget: int) -> None:
        if byte_budget < 0:
            raise ValueError("checkpoint byte budget must be non-negative")
        self.byte_budget = byte_budget
        self._entries: dict[tuple[str, str], CheckpointEntry] = {}
        self._used_bytes = 0

    @staticmethod
    def filename_for(model: str, conversation_id: str) -> str:
        digest = hashlib.sha256(f"{model}\0{conversation_id}".encode()).hexdigest()[:24]
        return f"cacheflow-{digest}.bin"

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    def get(self, model: str, conversation_id: str, now: float) -> CheckpointEntry | None:
        entry = self._entries.get((model, conversation_id))
        if entry is not None:
            entry.last_access = now
        return entry

    def register(self, entry: CheckpointEntry) -> list[CheckpointEntry]:
        key = (entry.model, entry.conversation_id)
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._used_bytes -= previous.size_bytes
        self._entries[key] = entry
        self._used_bytes += entry.size_bytes

        evicted: list[CheckpointEntry] = []
        while self._used_bytes > self.byte_budget and self._entries:
            victim_key, victim = min(
                self._entries.items(), key=lambda item: item[1].last_access
            )
            del self._entries[victim_key]
            self._used_bytes -= victim.size_bytes
            evicted.append(victim)
        return evicted

    def remove(self, model: str, conversation_id: str) -> CheckpointEntry | None:
        entry = self._entries.pop((model, conversation_id), None)
        if entry is not None:
            self._used_bytes -= entry.size_bytes
        return entry

    def snapshot(self) -> dict:
        return {
            "byte_budget": self.byte_budget,
            "used_bytes": self._used_bytes,
            "entries": [
                {
                    "conversation_id": entry.conversation_id,
                    "model": entry.model,
                    "filename": entry.filename,
                    "size_bytes": entry.size_bytes,
                    "tokens": len(entry.tokens),
                    "last_access": entry.last_access,
                }
                for entry in sorted(
                    self._entries.values(), key=lambda item: item.last_access, reverse=True
                )
            ],
        }
