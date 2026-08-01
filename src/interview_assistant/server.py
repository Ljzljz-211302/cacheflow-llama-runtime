from __future__ import annotations

import json
import mimetypes
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .llama_client import LlamaUnavailable
from .service import InterviewService


SESSION_MESSAGES_RE = re.compile(r"^/api/sessions/([a-f0-9]{32})/messages$")
STATIC_ROOT = Path(__file__).with_name("static")


class ApplicationServer(ThreadingHTTPServer):
    daemon_threads = False

    def __init__(self, address: tuple[str, int], service: InterviewService, max_concurrent: int) -> None:
        super().__init__(address, ApplicationHandler)
        self.service = service
        self.generation_slots = threading.BoundedSemaphore(max_concurrent)

    def server_close(self) -> None:
        super().server_close()
        self.service.store.close()


class ApplicationHandler(BaseHTTPRequestHandler):
    server: ApplicationServer

    def log_message(self, format: str, *args: object) -> None:
        super().log_message(format, *args)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            model_available = self.server.service.model.healthy()
            self._json({
                "status": "ok" if model_available else "degraded",
                "model_available": model_available,
                "knowledge_documents": self.server.service.knowledge.document_count,
                "knowledge_chunks": len(self.server.service.knowledge.chunks),
            })
            return
        if path == "/api/sessions":
            self._json({"sessions": self.server.service.store.list_sessions()})
            return
        match = SESSION_MESSAGES_RE.match(path)
        if match:
            try:
                messages = self.server.service.store.messages(match.group(1))
            except KeyError:
                self._error(HTTPStatus.NOT_FOUND, "session not found")
                return
            self._json({"messages": messages})
            return
        self._static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._read_json()
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if path == "/api/sessions":
            session = self.server.service.store.create_session(str(body.get("title", "")))
            self._json(session, HTTPStatus.CREATED)
            return
        match = SESSION_MESSAGES_RE.match(path)
        if not match:
            self._error(HTTPStatus.NOT_FOUND, "route not found")
            return
        if not self.server.generation_slots.acquire(blocking=False):
            self._error(HTTPStatus.TOO_MANY_REQUESTS, "all model generation slots are busy")
            return
        stream = None
        try:
            stream = self.server.service.answer(match.group(1), str(body.get("content", "")))
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            for event in stream:
                encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self.wfile.write(b"data: " + encoded + b"\n\n")
                self.wfile.flush()
        except KeyError:
            self._stream_error("session not found")
        except (ValueError, RuntimeError, LlamaUnavailable) as exc:
            self._stream_error(str(exc))
        except (BrokenPipeError, ConnectionResetError):
            return
        finally:
            if stream is not None:
                stream.close()
            self.server.generation_slots.release()

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            raise ValueError("request body must contain at most 1 MB")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status)

    def _stream_error(self, message: str) -> None:
        try:
            encoded = json.dumps({"type": "error", "error": message}, ensure_ascii=False).encode("utf-8")
            self.wfile.write(b"data: " + encoded + b"\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _static(self, path: str) -> None:
        relative = "index.html" if path == "/" else path.removeprefix("/")
        candidate = (STATIC_ROOT / relative).resolve()
        try:
            candidate.relative_to(STATIC_ROOT.resolve())
        except ValueError:
            self._error(HTTPStatus.NOT_FOUND, "asset not found")
            return
        if not candidate.is_file():
            self._error(HTTPStatus.NOT_FOUND, "asset not found")
            return
        content = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'")
        self.end_headers()
        self.wfile.write(content)


def create_server(host: str, port: int, service: InterviewService, max_concurrent: int = 8) -> ApplicationServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("interview assistant may only bind to loopback")
    if not 1 <= max_concurrent <= 128:
        raise ValueError("max_concurrent must be between 1 and 128")
    return ApplicationServer((host, port), service, max_concurrent)
