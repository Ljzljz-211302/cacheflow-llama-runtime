#!/usr/bin/env python3
"""Run a real multi-session user journey through the interview application."""

from __future__ import annotations

import json
import http.client
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llama_lab.server_bench import wait_until_ready  # noqa: E402


LLAMA_PORT = 19730
APP_PORT = 19731
LLAMA_URL = f"http://127.0.0.1:{LLAMA_PORT}"
APP_URL = f"http://127.0.0.1:{APP_PORT}"
API_KEY = "cacheflow-user-application-acceptance"
DB = ROOT / "results/raw/interview-assistant-journey.db"
CHECKPOINT = ROOT / "results/raw/interview-assistant-benefit.json"
LOG = ROOT / "results/raw/interview-assistant-llama.log"
APP_LOG = ROOT / "results/raw/interview-assistant-app.log"
API_KEY_FILE = ROOT / "results/raw/interview-assistant-api-key.txt"
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
        "--benefit-min-observations", "1", "--benefit-exploration-interval", "1",
        "--benefit-confidence-beta", "0.1", "--benefit-safety-margin-ms", "0.05",
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


def start_application(log: object, max_concurrent: int) -> subprocess.Popen[bytes]:
    command = [
        sys.executable, "-m", "interview_assistant",
        "--knowledge-root", str(KNOWLEDGE_ROOT), "--db", str(DB),
        "--llama-url", LLAMA_URL, "--api-key-file", str(API_KEY_FILE),
        "--host", "127.0.0.1", "--port", str(APP_PORT),
        "--max-concurrent", str(max_concurrent),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"application exited with code {process.returncode}; inspect {APP_LOG}")
        try:
            health = request_json("/api/health")
            if health.get("model_available"):
                return process
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.1)
    stop_process(process)
    raise TimeoutError(f"application did not become ready; inspect {APP_LOG}")


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
    return {"answer": answer, "answer_chars": len(answer), "citations": citations}


def cancel_answer(session_id: str, question: str) -> None:
    connection = http.client.HTTPConnection("127.0.0.1", APP_PORT, timeout=30)
    body = json.dumps({"content": question}, ensure_ascii=False).encode("utf-8")
    connection.request(
        "POST",
        f"/api/sessions/{session_id}/messages",
        body=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
    )
    response = connection.getresponse()
    if response.status != 200 or not response.readline().startswith(b"data: "):
        raise AssertionError("cancel journey did not begin an SSE response")
    if connection.sock is not None:
        connection.sock.shutdown(2)
    connection.close()


def expect_backpressure(session_id: str) -> None:
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(
            ask,
            session_id,
            "请详细解释操作系统虚拟内存、页表、TLB和缺页异常，并连续给出十个追问。",
        )
        time.sleep(0.05)
        deadline = time.monotonic() + 3
        rejected = False
        while time.monotonic() < deadline and not running.done():
            try:
                request_json(
                    f"/api/sessions/{session_id}/messages",
                    {"content": "解释进程与线程"},
                )
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    rejected = True
                    break
                raise
            time.sleep(0.02)
        running.result(timeout=240)
    if not rejected:
        raise AssertionError("bounded application did not reject excess generation work")


def prometheus_text() -> str:
    request = urllib.request.Request(
        LLAMA_URL + "/metrics",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def metric_sum(text: str, name: str) -> float:
    pattern = re.compile(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$")
    return sum(float(match.group(1)) for line in text.splitlines() if (match := pattern.match(line)))


def metric_exact(text: str, sample: str) -> float:
    prefix = sample + " "
    line = next((row for row in text.splitlines() if row.startswith(prefix)), None)
    if line is None:
        raise AssertionError(f"missing Prometheus sample: {sample}")
    return float(line[len(prefix):])


def cleanup() -> None:
    for path in (DB, Path(str(DB) + "-wal"), Path(str(DB) + "-shm"), CHECKPOINT,
                 Path(str(CHECKPOINT) + ".tmp"), API_KEY_FILE):
        path.unlink(missing_ok=True)


def main() -> None:
    if not KNOWLEDGE_ROOT.is_dir():
        raise FileNotFoundError(f"real study document directory is missing: {KNOWLEDGE_ROOT}")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    cleanup()
    API_KEY_FILE.write_text(API_KEY + "\n", encoding="utf-8")
    evidence: dict[str, Any] = {"scenario": "multi-session interview study assistant"}
    with LOG.open("wb") as log, APP_LOG.open("ab") as app_log:
        llama = start_llama(log)
        try:
            app = start_application(app_log, max_concurrent=1)
            try:
                with urllib.request.urlopen(APP_URL + "/", timeout=30) as response:
                    page = response.read().decode("utf-8")
                with urllib.request.urlopen(APP_URL + "/app.js", timeout=30) as response:
                    javascript = response.read().decode("utf-8")
                if "面试学习助手" not in page:
                    raise AssertionError("user-facing application did not render")
                if "getReader()" not in javascript or "type === 'delta'" not in javascript:
                    raise AssertionError("browser SSE contract is missing")
                health = request_json("/api/health")
                if health["status"] != "ok" or health["knowledge_documents"] < 5:
                    raise AssertionError(f"application is not ready: {health}")
                first = request_json("/api/sessions", {"title": "机器学习模拟面试"})
                first_answer = ask(first["id"], "请用面试回答格式解释偏差与方差，并给出一个追问。")
                if "偏差" not in first_answer["answer"] or "方差" not in first_answer["answer"]:
                    raise AssertionError("model answer failed the minimum grounded topic contract")
                cancelled = request_json("/api/sessions", {"title": "取消场景"})
                cancel_answer(cancelled["id"], "请详细解释数据库索引并给出十个追问。")
                time.sleep(0.5)
                cancelled_history = request_json(f"/api/sessions/{cancelled['id']}/messages")["messages"]
                if [message["role"] for message in cancelled_history] != ["user"]:
                    raise AssertionError("cancelled stream persisted a partial assistant answer")
                expect_backpressure(first["id"])
                evidence.update({
                    "ui_rendered": True,
                    "browser_sse_contract": True,
                    "knowledge_documents": health["knowledge_documents"],
                    "knowledge_chunks": health["knowledge_chunks"],
                    "first_answer_chars": first_answer["answer_chars"],
                    "first_citation": first_answer["citations"][0]["source"],
                    "cancel_left_partial_assistant": False,
                    "backpressure_rejected": True,
                })
            finally:
                stop_process(app)

            app = start_application(app_log, max_concurrent=4)
            try:
                history = request_json(f"/api/sessions/{first['id']}/messages")["messages"]
                if len(history) < 2 or history[0]["role"] != "user" or history[1]["role"] != "assistant":
                    raise AssertionError("conversation did not survive application restart")
                second = request_json("/api/sessions", {"title": "408 模拟面试"})
                questions = [
                    (first["id"], "沿着上一个回答继续追问：过拟合时偏差和方差通常如何变化？"),
                    (second["id"], "请比较进程与线程，并给出常见面试追问。"),
                ]
                started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(lambda item: ask(*item), questions))
                if "进程" not in results[1]["answer"] or "线程" not in results[1]["answer"]:
                    raise AssertionError("second user answer failed the minimum grounded topic contract")
                evidence.update({
                    "session_restart_preserved_messages": len(history),
                    "concurrent_users": 2,
                    "concurrent_elapsed_seconds": round(time.perf_counter() - started, 3),
                    "followup_answer_chars": results[0]["answer_chars"],
                    "second_session_answer_chars": results[1]["answer_chars"],
                    "sessions_persisted": len(request_json("/api/sessions")["sessions"]),
                })
            finally:
                stop_process(app)

            metrics = prometheus_text()
            required_positive = {
                "scheduler_iterations": metric_sum(metrics, "llamacpp:scheduler_iterations_total"),
                "prefill_chunks": metric_sum(metrics, "llamacpp:prefill_chunks_scheduled_total"),
                "cached_prompt_tokens": metric_sum(metrics, "llamacpp:prompt_tokens_cached_total"),
                "cuda_kv_kernel_launches": metric_sum(metrics, "llamacpp:cuda_kv_kernel_launches_total"),
                "cuda_benefit_decisions": sum(
                    float(match.group(1))
                    for line in metrics.splitlines()
                    if (match := re.match(
                        r'^llamacpp:benefit_decisions_total\{backend="cuda",action="[^"]+"\}\s+([-+0-9.eE]+)$',
                        line,
                    ))
                ),
                "checkpoint_saves": metric_exact(
                    metrics,
                    'llamacpp:benefit_checkpoint_save_total{result="completed"}',
                ),
            }
            missing = [name for name, value in required_positive.items() if value <= 0]
            if missing:
                raise AssertionError(f"application traffic did not exercise infra metrics: {missing}")
            if metric_sum(metrics, "llamacpp:cacheflow_scheduler_policy") != 1:
                raise AssertionError("application did not run with CacheFlow scheduler")
            if metric_sum(metrics, "llamacpp:n_busy_slots_per_decode") <= 1:
                raise AssertionError("concurrent application traffic was not batched")
            if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size == 0:
                raise AssertionError("application traffic produced no durable online-policy checkpoint")
            log.flush()
            log_text = LOG.read_text(encoding="utf-8", errors="replace")
            for marker in ("CUDA0", "CacheFlow policies: scheduler=cacheflow", "shared "):
                if marker not in log_text:
                    raise AssertionError(f"missing native application-path evidence: {marker}")
            evidence["infra_metrics"] = required_positive
            evidence["busy_slots_per_decode"] = metric_sum(metrics, "llamacpp:n_busy_slots_per_decode")
            evidence["native_shared_prefix_log"] = True
        finally:
            stop_process(llama)
    evidence["passed"] = True
    evidence["llama_log"] = str(LOG.relative_to(ROOT))
    evidence["application_log"] = str(APP_LOG.relative_to(ROOT))
    OUTPUT.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False))
    cleanup()


if __name__ == "__main__":
    main()
