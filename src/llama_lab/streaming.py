from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Iterable
from typing import Any


def parse_sse_lines(lines: Iterable[bytes]) -> Iterable[dict[str, Any]]:
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        if payload:
            yield json.loads(payload)


def stream_chat(
    base_url: str,
    prompt: str,
    model: str = "local-model",
    max_tokens: int = 64,
    timeout: float = 120.0,
    seed: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
    }
    if seed is not None:
        payload["seed"] = seed
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    chunks: list[str] = []
    usage: dict[str, int] = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for event in parse_sse_lines(response):
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            content = choices[0].get("delta", {}).get("content")
            if content:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                chunks.append(content)
    finished = time.perf_counter()
    if first_token_at is None:
        first_token_at = finished
    completion_tokens = int(usage.get("completion_tokens", 0))
    decode_seconds = max(finished - first_token_at, 1e-9)
    tokens_after_first = max(completion_tokens - 1, 0)
    return {
        "prompt": prompt,
        "text": "".join(chunks),
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": completion_tokens,
        "ttft_ms": (first_token_at - started) * 1000,
        "total_ms": (finished - started) * 1000,
        "tpot_ms": (
            decode_seconds * 1000 / tokens_after_first
            if tokens_after_first
            else 0.0
        ),
        "output_tps": tokens_after_first / decode_seconds,
    }
