from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from production_journey import ROOT, cuda_environment, get_text, metric, request_json, wait_ready


CROSS_PAGE_PROMPT = " one" * 17

def stop_process(process: subprocess.Popen[bytes], launcher: list[str] | None) -> None:
    if not launcher:
        process.terminate()
        return
    # Profilers need the target to exit first so they can flush a complete
    # report. Terminating the profiler wrapper would truncate the evidence.
    inventory = subprocess.run(
        [
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name | ConvertTo-Json -Compress",
        ],
        text=True, capture_output=True, check=True,
    )
    entries = json.loads(inventory.stdout)
    if isinstance(entries, dict):
        entries = [entries]
    children: dict[int, list[dict[str, Any]]] = {}
    for entry in entries:
        children.setdefault(int(entry["ParentProcessId"]), []).append(entry)
    pending = [process.pid]
    descendants: list[dict[str, Any]] = []
    while pending:
        for child in children.get(pending.pop(), []):
            descendants.append(child)
            pending.append(int(child["ProcessId"]))
    targets = [entry for entry in descendants if str(entry["Name"]).lower() == "llama-server.exe"]
    if len(targets) != 1:
        raise RuntimeError(f"expected one profiled llama-server child, found {targets}")
    subprocess.run(
        ["taskkill", "/F", "/PID", str(targets[0]["ProcessId"])],
        text=True, capture_output=True, check=True,
    )


def run_mode(
    server: Path,
    model: Path,
    port: int,
    action: str,
    label: str = "journey",
    launcher: list[str] | None = None,
    flash_attention: bool = True,
    prompt: str = CROSS_PAGE_PROMPT,
    n_probs: int = 0,
    launcher_manages_lifetime: bool = False,
    environment_overrides: dict[str, str] | None = None,
    warm_requests: int = 1,
    n_predict: int = 1,
    measured_requests: int = 1,
) -> tuple[dict[str, Any], str, Path]:
    log_path = ROOT / f"results/raw/production-{action}-{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{port}"
    command = (launcher or []) + [
        str(server.resolve()), "-m", str(model.resolve()),
        "--host", "127.0.0.1", "--port", str(port),
        "-c", "512", "-np", "1", "-t", "8", "-ngl", "99",
        "--flash-attn", "on" if flash_attention else "off", "--no-warmup", "--metrics", "--slots",
        "--kv-block-runtime", "--kv-block-size", "16", "--kv-paged-decode",
        "--kv-action-policy", "analytical", "--kv-action-override", action,
        "-lv", "4",
    ]
    environment = cuda_environment()
    if environment_overrides:
        environment.update(environment_overrides)
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "seed": 20260808,
        "cache_prompt": True,
    }
    if n_probs > 0:
        payload["n_probs"] = n_probs
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            wait_ready(base_url, process, log_path, attempts=120)
            for _ in range(warm_requests):
                request_json(f"{base_url}/completion", payload)
            elapsed = []
            for _ in range(measured_requests):
                started = time.perf_counter_ns()
                _, result = request_json(f"{base_url}/completion", payload)
                elapsed.append((time.perf_counter_ns() - started) / 1.e6)
            result["_client_elapsed_ms"] = sum(elapsed) / len(elapsed)
            metrics = get_text(f"{base_url}/metrics")
        finally:
            if launcher_manages_lifetime:
                process.wait(timeout=30)
            else:
                stop_process(process, launcher)
            try:
                process.wait(timeout=30 if launcher else 10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    if "error" in result or not result.get("content"):
        raise AssertionError(f"{action} user request failed: {result}")
    return result, metrics, log_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path,
                        default=ROOT / "build/patched-cuda-ninja3/bin/llama-server.exe")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    parser.add_argument("--port", type=int, default=8130)
    parser.add_argument("--memcheck", action="store_true",
                        help="run the real Paged request under Compute Sanitizer")
    args = parser.parse_args()

    if args.memcheck:
        sanitizer = ROOT / "runtime/cuda-dev/Library/compute-sanitizer/compute-sanitizer.exe"
        if not sanitizer.exists():
            raise FileNotFoundError(f"Compute Sanitizer is missing: {sanitizer}")
        result, _, log_path = run_mode(
            args.server, args.model, args.port, "paged", "production-memcheck",
            launcher=[str(sanitizer), "--tool", "memcheck", "--error-exitcode", "99"],
        )
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "ERROR SUMMARY: 0 errors" not in log_text:
            raise AssertionError(f"Compute Sanitizer did not report a clean request: {log_path}")
        print(json.dumps({
            "content": result["content"],
            "compute_sanitizer_error_summary": 0,
            "log": str(log_path),
        }, ensure_ascii=False))
        return

    direct, direct_metrics, direct_log = run_mode(
        args.server, args.model, args.port, "direct", n_probs=64,
    )
    direct_reference, _, reference_log = run_mode(
        args.server, args.model, args.port + 1, "direct", "nonflash-reference",
        flash_attention=False, n_probs=64,
    )
    paged, paged_metrics, paged_log = run_mode(
        args.server, args.model, args.port + 2, "paged", n_probs=64,
        environment_overrides={"LLAMA_CACHEFLOW_PAGED_CONTIGUOUS_FASTPATH": "0"},
    )
    long_prompt = "This prompt deliberately contains enough distinct words to cross the production paged decode short context envelope safely. " * 2
    fallback, fallback_metrics, fallback_log = run_mode(
        args.server, args.model, args.port + 3, "paged", "capability-fallback",
        prompt=long_prompt,
    )
    if direct["content"] != paged["content"]:
        raise AssertionError("production Paged output differs from deterministic Direct output")
    paged_context_tokens = int(paged["timings"]["cache_n"]) + int(
        paged["timings"]["prompt_n"]
    )
    if paged_context_tokens != 17:
        raise AssertionError(
            f"production Paged request did not cross the page-16 boundary: {paged_context_tokens}"
        )
    if direct["content"] != direct_reference["content"]:
        raise AssertionError("native Flash and non-Flash references select different top-1 tokens")
    direct_probs = direct.get("completion_probabilities", [{}])[0].get("top_logprobs", [])
    reference_probs = direct_reference.get("completion_probabilities", [{}])[0].get("top_logprobs", [])
    paged_probs = paged.get("completion_probabilities", [{}])[0].get("top_logprobs", [])
    direct_by_id = {int(item["id"]): float(item["logprob"]) for item in direct_probs}
    reference_by_id = {int(item["id"]): float(item["logprob"]) for item in reference_probs}
    paged_by_id = {int(item["id"]): float(item["logprob"]) for item in paged_probs}
    paged_common = direct_by_id.keys() & paged_by_id.keys()
    reference_common = direct_by_id.keys() & reference_by_id.keys()
    max_logprob_error = max(
        abs(direct_by_id[token] - paged_by_id[token]) for token in paged_common
    )
    native_backend_max_error = max(
        abs(direct_by_id[token] - reference_by_id[token]) for token in reference_common
    )
    if len(paged_common) < len(reference_common) or max_logprob_error > native_backend_max_error + 0.01:
        raise AssertionError(
            "production Paged distribution exceeded the native Flash/non-Flash backend envelope: "
            f"paged overlap={len(paged_common)}, native overlap={len(reference_common)}, "
            f"paged max={max_logprob_error}, native max={native_backend_max_error}"
        )
    paged_calls = metric(paged_metrics, "llamacpp:paged_decode_calls_total ")
    paged_fallbacks = metric(paged_metrics, "llamacpp:paged_decode_fallbacks_total ")
    fallback_reason = metric(paged_metrics, "llamacpp:paged_decode_last_fallback_reason ")
    paged_decisions = metric(paged_metrics, 'llamacpp:kv_action_decisions_total{action="paged"}')
    paged_reason_decisions = sum(
        float(line.split()[-1])
        for line in paged_metrics.splitlines()
        if line.startswith('llamacpp:kv_action_decisions_by_reason_total{action="paged",reason=')
    )
    if paged_calls <= 0:
        decision_rows = [line for line in paged_metrics.splitlines()
                         if line.startswith("llamacpp:kv_action_decisions_total")]
        raise AssertionError(
            "real user request did not enter the production paged graph: "
            f"decisions={paged_decisions}, fallbacks={paged_fallbacks}, reason={fallback_reason}, "
            f"all={decision_rows}"
        )
    if paged_decisions <= 0:
        raise AssertionError("adaptive policy did not select Paged")
    if paged_reason_decisions != paged_decisions:
        raise AssertionError(
            "Paged decisions are not attributable to action/reason metrics: "
            f"actions={paged_decisions}, joint={paged_reason_decisions}"
        )
    if metric(direct_metrics, 'llamacpp:kv_action_decisions_total{action="direct"}') <= 0:
        raise AssertionError("controlled Direct path was not selected")
    fallback_recompute = metric(
        fallback_metrics, 'llamacpp:kv_action_decisions_total{action="recompute"}'
    )
    if metric(fallback_metrics, "llamacpp:paged_decode_calls_total ") != 0 or fallback_recompute <= 0:
        raise AssertionError("out-of-envelope Paged request did not fail closed through Recompute")
    if "error" in fallback or not fallback.get("content"):
        raise AssertionError(f"out-of-envelope fallback request failed: {fallback}")
    print(json.dumps({
        "direct_content": direct["content"],
        "paged_content": paged["content"],
        "paged_calls": paged_calls,
        "paged_fallbacks": paged_fallbacks,
        "paged_reason_decisions": paged_reason_decisions,
        "paged_context_tokens": paged_context_tokens,
        "top_logprob_count": len(direct_by_id),
        "paged_top_logprob_overlap": len(paged_common),
        "max_logprob_error": max_logprob_error,
        "native_backend_top_logprob_overlap": len(reference_common),
        "native_backend_max_logprob_error": native_backend_max_error,
        "out_of_envelope_recompute_decisions": fallback_recompute,
        "direct_log": str(direct_log),
        "native_nonflash_log": str(reference_log),
        "paged_log": str(paged_log),
        "fallback_log": str(fallback_log),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
