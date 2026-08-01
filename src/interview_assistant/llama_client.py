from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator


class LlamaUnavailable(RuntimeError):
    pass


class LlamaClient:
    def __init__(self, base_url: str, api_key: str, model: str = "local-model", timeout: int = 180) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def healthy(self) -> bool:
        request = urllib.request.Request(f"{self.base_url}/health")
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def stream(self, messages: list[dict[str, str]], max_tokens: int = 512) -> Iterator[str]:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": 0.25,
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        return
                    event = json.loads(data)
                    choices = event.get("choices") or []
                    if choices:
                        content = choices[0].get("delta", {}).get("content")
                        if content:
                            yield content
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise LlamaUnavailable(f"model service unavailable: {exc}") from exc

