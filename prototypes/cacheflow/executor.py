"""Archived prototype executor; production execution is owned by the C++ engine."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ExecutorResult:
    response: dict[str, Any]
    prompt_tokens_processed: int
    prompt_ms: float
    decode_ms: float
    cached_tokens: tuple[int, ...]


class Executor(Protocol):
    def tokenize_messages(self, messages: list[dict[str, str]]) -> tuple[int, ...]: ...

    def complete(
        self, payload: dict[str, Any], slot_id: int, prompt_tokens: tuple[int, ...]
    ) -> ExecutorResult: ...

    def save_slot(self, slot_id: int, filename: str) -> tuple[int, int]: ...

    def restore_slot(self, slot_id: int, filename: str) -> int: ...

    def erase_slot(self, slot_id: int) -> int: ...

    def delete_checkpoint(self, filename: str) -> None: ...


class LlamaCppExecutor:
    def __init__(
        self,
        base_url: str,
        *,
        checkpoint_directory: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.checkpoint_directory = checkpoint_directory
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama.cpp {path} failed ({exc.code}): {body}") from exc

    def tokenize_messages(self, messages: list[dict[str, str]]) -> tuple[int, ...]:
        templated = self._post("/apply-template", {"messages": messages})["prompt"]
        tokenized = self._post(
            "/tokenize", {"content": templated, "add_special": True}
        )
        return tuple(int(token) for token in tokenized["tokens"])

    def complete(
        self, payload: dict[str, Any], slot_id: int, prompt_tokens: tuple[int, ...]
    ) -> ExecutorResult:
        request_payload = dict(payload)
        request_payload.update({"id_slot": slot_id, "stream": False, "cache_prompt": True})
        response = self._post("/v1/chat/completions", request_payload)
        timings = response.get("timings", {})
        prompt_ms = float(timings.get("prompt_ms", 0.0))
        decode_ms = float(timings.get("predicted_ms", 0.0))
        processed = int(timings.get("prompt_n", len(prompt_tokens)))
        return ExecutorResult(response, processed, prompt_ms, decode_ms, prompt_tokens)

    def save_slot(self, slot_id: int, filename: str) -> tuple[int, int]:
        response = self._post(f"/slots/{slot_id}?action=save", {"filename": filename})
        return int(response["n_saved"]), int(response["n_written"])

    def restore_slot(self, slot_id: int, filename: str) -> int:
        response = self._post(f"/slots/{slot_id}?action=restore", {"filename": filename})
        return int(response["n_restored"])

    def erase_slot(self, slot_id: int) -> int:
        response = self._post(f"/slots/{slot_id}?action=erase", {})
        return int(response["n_erased"])

    def delete_checkpoint(self, filename: str) -> None:
        if self.checkpoint_directory is None:
            return
        from pathlib import Path

        root = Path(self.checkpoint_directory).resolve()
        target = (root / filename).resolve()
        if target.parent != root:
            raise ValueError("checkpoint path escaped configured directory")
        target.unlink(missing_ok=True)
