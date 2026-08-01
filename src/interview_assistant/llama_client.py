from __future__ import annotations

import json
import ipaddress
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlsplit

from .domain import ChatMessage


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


class LlamaUnavailable(RuntimeError):
    pass


class LlamaClient:
    def __init__(self, base_url: str, api_key: str, model: str = "local-model", timeout: int = 180) -> None:
        self.base_url = self._validate_base_url(base_url)
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._opener = urllib.request.build_opener(_RejectRedirects())

    @staticmethod
    def _validate_base_url(base_url: str) -> str:
        parsed = urlsplit(base_url)
        try:
            address = ipaddress.ip_address(parsed.hostname or "")
            port = parsed.port
        except ValueError as exc:
            raise ValueError("llama URL must use a loopback IP address and explicit port") from exc
        if parsed.scheme != "http" or not address.is_loopback or port is None:
            raise ValueError("llama URL must be http://<loopback-ip>:<port>")
        if parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("llama URL must not contain credentials, path, query, or fragment")
        return base_url.rstrip("/")

    def healthy(self) -> bool:
        request = urllib.request.Request(f"{self.base_url}/health")
        try:
            with self._opener.open(request, timeout=3) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def stream(self, messages: list[ChatMessage], max_tokens: int = 512) -> Iterator[str]:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": 0.0,
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
            with self._opener.open(request, timeout=self.timeout) as response:
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
