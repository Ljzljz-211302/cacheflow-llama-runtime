#!/usr/bin/env python3
"""Run a real multi-session user journey through the interview application."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interview_assistant.knowledge import KnowledgeIndex  # noqa: E402
from interview_assistant.llama_client import LlamaClient  # noqa: E402
from interview_assistant.server import create_server  # noqa: E402
from interview_assistant.service import InterviewService  # noqa: E402
from interview_assistant.store import ConversationStore  # noqa: E402
from llama_lab.server_bench import wait_until_ready  # noqa: E402


LLAMA_PORT = 19730
APP_PORT = 19731
LLAMA_URL = f"http://127.0.0.1:{LLAMA_PORT}"
APP_URL = f"http://127.0.0.1:{APP_PORT}"
API_KEY = "cacheflow-user-application-acceptance"
DB = ROOT / "results/raw/interview-assistant-journey.db"
CHECKPOINT = ROOT / "results/raw/interview-assistant-benefit.json"
LOG = ROOT / "results/raw/interview-assistant-llama.log"
OUTPUT = ROOT / "results/user-application-journey.json"
DEFAULT_USER_KNOWLEDGE = Path(r"D:\exam\tuimian-monitor\docs\study")
KNOWLEDGE_ROOT = Path(os.environ.get(
    "CACHEFLOW_APPLICATION_KNOWLEDGE",
    str(DEFAULT_USER_KNOWLEDGE if DEFAULT_USER_KNOWLEDGE.is_dir() else ROOT / "docs"),
))


def start_llama(log: object) -> subprocess.Popen[bytes]:
    executable = ROOT / "build/patched-cuda-ninja3/bin/llama-server.exe"
    model = ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    if not executable.is_file() or not model.is_file():
        raise FileNotFoundError("patched CUDA server and Qwen model are required")
    command = [
        str(executable), "-m", str(model), "--host", "127.0.0.1", "--port", str(LLAMA_PORT),
        "-c", "8192", "-np", "4", "-b", "512", "-ub", "512", "-ngl", "99", "-t", "8",
        "--api-key", API_KEY, "--no-webui", "--metrics", "--scheduler-policy", "cacheflow",
        "--benefit-policy", "learned", "--benefit-checkpoint", str(CHECKPOINT),
        "--benefit-checkpoint-key", "interview-assistant-qwen2.5-0.5b-cuda-v1",
        "--benefit-checkpoint-interval", "1", "--kv-block-runtime", "--kv-block-size", "16",
    ]
    environment = os.environ.copy()
    cuda_bin = ROOT / "runtime/cuda-dev/Library/bin"
    environment["PATH"] = str(cuda_bin) + os.pathsep + str(executable.parent) + os.pathsep + environment.get("PATH", "")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        wait_until_ready(LLAMA_URL, process=process, log_path=LOG)
    except BaseException:
        stop_process(process)
        raise
    return process


def stop_process(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def start_application() -> tuple[Any, threading.Thread]:
    knowledge = KnowledgeIndex([KNOWLEDGE_ROOT])
    service = InterviewService(
        ConversationStore(DB),
        knowledge,
        LlamaClient(LLAMA_URL, API_KEY, timeout=180),
    )
    server = create_server("127.0.0.1", APP_PORT, service, max_concurrent=4)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def stop_application(server: Any, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def request_json(path: str, body: dict[str, str] | None = None) -> dict[str, Any]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        APP_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        return json.load(response)


def ask(session_id: str, question: str) -> dict[str, Any]:
    request = urllib.request.Request(
        APP_URL + f"/api/sessions/{session_id}/messages",
        data=json.dumps({"content": question}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    citations: list[dict[str, Any]] = []
    answer = ""
    done = False
    with urllib.request.urlopen(request, timeout=240) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["type"] == "citations":
                citations = event["citations"]
            elif event["type"] == "delta":
                answer += event["content"]
            elif event["type"] == "done":
                done = True
            elif event["type"] == "error":
                raise RuntimeError(event["error"])
    if not done or not answer.strip() or not citations:
        raise AssertionError("user answer must complete with non-empty text and citations")
    return {"answer_chars": len(answer), "citations": citations}


def cleanup() -> None:
    for path in (DB, Path(str(DB) + "-wal"), Path(str(DB) + "-shm"), CHECKPOINT, Path(str(CHECKPOINT) + ".tmp")):
        path.unlink(missing_ok=True)


def main() -> None:
    if not KNOWLEDGE_ROOT.is_dir():
        raise FileNotFoundError(f"real study document directory is missing: {KNOWLEDGE_ROOT}")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    cleanup()
    evidence: dict[str, Any] = {"scenario": "multi-session interview study assistant"}
    with LOG.open("wb") as log:
        llama = start_llama(log)
        try:
            app, app_thread = start_application()
            try:
                with urllib.request.urlopen(APP_URL + "/", timeout=30) as response:
                    page = response.read().decode("utf-8")
                if "面试学习助手" not in page:
                    raise AssertionError("user-facing application did not render")
                health = request_json("/api/health")
                if health["status"] != "ok" or health["knowledge_documents"] < 5:
                    raise AssertionError(f"application is not ready: {health}")
                first = request_json("/api/sessions", {"title": "机器学习模拟面试"})
                first_answer = ask(first["id"], "请用面试回答格式解释偏差与方差，并给出一个追问。")
                evidence.update({
                    "ui_rendered": True,
                    "knowledge_documents": health["knowledge_documents"],
                    "knowledge_chunks": health["knowledge_chunks"],
                    "first_answer_chars": first_answer["answer_chars"],
                    "first_citation": first_answer["citations"][0]["source"],
                })
            finally:
                stop_application(app, app_thread)

            app, app_thread = start_application()
            try:
                history = request_json(f"/api/sessions/{first['id']}/messages")["messages"]
                if len(history) != 2 or history[-1]["role"] != "assistant":
                    raise AssertionError("conversation did not survive application restart")
                second = request_json("/api/sessions", {"title": "408 模拟面试"})
                questions = [
                    (first["id"], "沿着上一个回答继续追问：过拟合时偏差和方差通常如何变化？"),
                    (second["id"], "请比较进程与线程，并给出常见面试追问。"),
                ]
                started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(lambda item: ask(*item), questions))
                evidence.update({
                    "session_restart_preserved_messages": len(history),
                    "concurrent_users": 2,
                    "concurrent_elapsed_seconds": round(time.perf_counter() - started, 3),
                    "followup_answer_chars": results[0]["answer_chars"],
                    "second_session_answer_chars": results[1]["answer_chars"],
                    "sessions_persisted": len(request_json("/api/sessions")["sessions"]),
                })
            finally:
                stop_application(app, app_thread)
        finally:
            stop_process(llama)
    evidence["passed"] = True
    evidence["llama_log"] = str(LOG.relative_to(ROOT))
    OUTPUT.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False))
    cleanup()


if __name__ == "__main__":
    main()
