from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .domain import QueueFullError
from .service import CacheFlowEngine


MAX_BODY_BYTES = 2 * 1024 * 1024


class CacheFlowApiServer:
    """Small dependency-free HTTP gateway for the CacheFlow control plane."""

    def __init__(
        self,
        engine: CacheFlowEngine,
        host: str = "127.0.0.1",
        port: int = 8088,
    ) -> None:
        self.engine = engine
        self._server = ThreadingHTTPServer((host, port), self._handler_type())
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def _handler_type(self):
        engine = self.engine

        class Handler(BaseHTTPRequestHandler):
            server_version = "CacheFlow/0.1"

            def log_message(self, _format: str, *args: object) -> None:
                return

            def _json(self, status: int, body: dict[str, Any], **headers: str) -> None:
                encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                for name, value in headers.items():
                    self.send_header(name.replace("_", "-"), value)
                self.end_headers()
                self.wfile.write(encoded)

            def _error(self, status: int, message: str, kind: str) -> None:
                self._json(status, {"error": {"message": message, "type": kind}})

            def _read_json(self) -> dict[str, Any]:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise ValueError("invalid Content-Length") from exc
                if length <= 0 or length > MAX_BODY_BYTES:
                    raise ValueError(f"body size must be between 1 and {MAX_BODY_BYTES} bytes")
                try:
                    value = json.loads(self.rfile.read(length))
                except json.JSONDecodeError as exc:
                    raise ValueError("request body is not valid JSON") from exc
                if not isinstance(value, dict):
                    raise ValueError("request body must be a JSON object")
                return value

            def do_GET(self) -> None:
                if self.path == "/health":
                    self._json(HTTPStatus.OK, {"status": "ok"})
                    return
                if self.path == "/debug/state":
                    self._json(HTTPStatus.OK, engine.snapshot())
                    return
                if self.path == "/metrics":
                    encoded = engine.metrics.prometheus().encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                    return
                self._error(HTTPStatus.NOT_FOUND, "route not found", "not_found")

            def do_POST(self) -> None:
                if self.path.startswith("/v1/requests/") and self.path.endswith("/cancel"):
                    request_id = self.path[len("/v1/requests/") : -len("/cancel")].strip("/")
                    if not request_id:
                        self._error(HTTPStatus.BAD_REQUEST, "missing request id", "invalid_request")
                    elif engine.cancel(request_id):
                        self._json(HTTPStatus.OK, {"id": request_id, "cancelled": True})
                    else:
                        self._error(
                            HTTPStatus.CONFLICT,
                            "request is unknown or no longer queued",
                            "not_cancellable",
                        )
                    return
                if self.path != "/v1/chat/completions":
                    self._error(HTTPStatus.NOT_FOUND, "route not found", "not_found")
                    return
                try:
                    payload = self._read_json()
                    if payload.get("stream", False):
                        raise ValueError("stream=true is not supported by this gateway yet")
                    options = payload.pop("cacheflow", {})
                    if not isinstance(options, dict):
                        raise ValueError("cacheflow options must be an object")
                    conversation_id = str(
                        self.headers.get("X-Conversation-ID")
                        or options.get("conversation_id")
                        or payload.get("user")
                        or f"anonymous-{uuid.uuid4().hex}"
                    )
                    handle = engine.submit(
                        payload,
                        conversation_id=conversation_id,
                        quality_floor=float(options.get("quality_floor", 0.0)),
                        latency_slo_ms=float(options.get("latency_slo_ms", 2000.0)),
                        timeout_ms=float(options.get("queue_timeout_ms", 30_000.0)),
                        available_vram_mib=float(options.get("available_vram_mib", float("inf"))),
                    )
                    result = handle.result(float(options.get("wait_timeout_s", 180.0)))
                    response = dict(result.response)
                    response["cacheflow"] = {
                        "request_id": result.request_id,
                        "model": result.model,
                        "slot_id": result.slot_id,
                        "queue_ms": result.queue_ms,
                        "total_ms": result.total_ms,
                        "prompt_tokens_processed": result.prompt_tokens_processed,
                        "checkpoint_restored": result.checkpoint_restored,
                        "routing": asdict(result.routing),
                    }
                    self._json(
                        HTTPStatus.OK,
                        response,
                        X_CacheFlow_Request_Id=result.request_id,
                        X_CacheFlow_Model=result.model,
                        X_Conversation_Id=conversation_id,
                    )
                except QueueFullError as exc:
                    self._error(HTTPStatus.TOO_MANY_REQUESTS, str(exc), "queue_full")
                except TimeoutError as exc:
                    self._error(HTTPStatus.GATEWAY_TIMEOUT, str(exc), "timeout")
                except ValueError as exc:
                    self._error(HTTPStatus.BAD_REQUEST, str(exc), "invalid_request")
                except RuntimeError as exc:
                    self._error(HTTPStatus.BAD_GATEWAY, str(exc), "backend_error")
                except Exception as exc:
                    self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc), "internal_error")

        return Handler

    def start(self) -> "CacheFlowApiServer":
        if self._thread is not None:
            raise RuntimeError("API server already started")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="cacheflow-api",
            daemon=True,
        )
        self._thread.start()
        return self

    def shutdown(self) -> None:
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=5)
            self._thread = None
        self._server.server_close()

    def __enter__(self) -> "CacheFlowApiServer":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.shutdown()
