from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llama_lab.public_external_evidence import analyze_public_external  # noqa: E402
from llama_lab.public_workloads import longbench_qa_f1  # noqa: E402
from production_journey import cuda_environment, get_text, request_json, terminate_process, wait_ready  # noqa: E402
from run_batched_paged_performance import RUNTIME_FILES, metric, vendor_diff_sha256  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def execution_plan(protocol: dict[str, Any]) -> list[tuple[int, int, str]]:
    blocks = int(protocol["matched_process_blocks"])
    first = ["direct"] * (blocks // 2) + ["paged"] * (blocks - blocks // 2)
    random.Random(int(protocol["random_seed"])).shuffle(first)
    return [
        (block, order, action)
        for block, first_action in enumerate(first, 1)
        for order, action in enumerate((first_action, "paged" if first_action == "direct" else "direct"), 1)
    ]


def observed_binding(protocol_path: Path, server: Path, model: Path, workload: Path) -> dict[str, Any]:
    return {
        "protocol_sha256": sha256(protocol_path), "workload_sha256": sha256(workload),
        "runner_revision": git("rev-parse", "HEAD"),
        "vendor_revision": git("-C", "vendor/llama.cpp", "rev-parse", "HEAD"),
        "vendor_diff_sha256": vendor_diff_sha256(), "model_sha256": sha256(model),
        "runtime_sha256": {name: sha256(server.parent / name) for name in RUNTIME_FILES},
    }


def verify_binding(protocol: dict[str, Any], observed: dict[str, Any]) -> None:
    frozen = protocol["artifact_binding"]
    for field in ("vendor_revision", "vendor_diff_sha256", "model_sha256", "runtime_sha256"):
        if observed[field] != frozen[field]:
            raise RuntimeError(f"public external {field} differs from preregistration")
    if observed["workload_sha256"] != protocol["workload_sha256"]:
        raise RuntimeError("public workload differs from preregistration")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", frozen["outer_revision"], observed["runner_revision"]],
        cwd=ROOT,
    )


def metric_delta(before: str, after: str, name: str) -> float:
    return metric(after, name) - metric(before, name)


def route_delta(before: str, after: str) -> dict[str, float]:
    return {
        field: metric_delta(before, after, metric_name) for field, metric_name in {
            "paged_contiguous_fastpath_calls": "llamacpp:paged_decode_contiguous_fastpath_calls_total",
            "paged_contiguous_fastpath_sequences": "llamacpp:paged_decode_contiguous_fastpath_sequences_total",
            "paged_calls": "llamacpp:paged_decode_calls_total",
            "cuda_dispatches": "llamacpp:paged_decode_cuda_dispatches_total",
            "paged_fallbacks": "llamacpp:paged_decode_fallbacks_total",
        }.items()
    }


def normalize_response(response: dict[str, Any], workload: dict[str, Any], latency_ms: float,
                       scheduled_ms: float, started_ms: float) -> dict[str, Any]:
    if "error" in response or not response.get("tokens"):
        raise AssertionError(f"public replay request failed: {response}")
    probabilities = []
    for token in response.get("completion_probabilities", []):
        probabilities.append([
            {"id": int(item["id"]), "logprob": float(item["logprob"])}
            for item in token.get("top_logprobs", [])
        ])
    return {
        "trace_row": int(workload["trace_row"]), "prompt_id": workload["prompt_id"],
        "source_arrival_seconds": float(workload["source_arrival_seconds"]),
        "scheduled_arrival_ms": scheduled_ms, "actual_start_ms": started_ms,
        "latency_ms": latency_ms, "output_token_ids": [int(value) for value in response["tokens"]],
        "content": str(response.get("content", "")), "top_logprobs": probabilities,
        "cache_tokens": int(response["timings"]["cache_n"]) + int(response["timings"]["prompt_n"]),
        "actual_local_input_tokens": int(workload["actual_local_input_tokens"]),
    }


def replay_trace(base_url: str, rows: list[dict[str, Any]], protocol: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    span = max(float(row["source_arrival_seconds"]) for row in rows)
    target_span = float(protocol["replay"]["target_arrival_span_seconds"])
    scale = target_span / span if span > 0 else 0.0
    origin = time.perf_counter()

    def issue(row: dict[str, Any]) -> dict[str, Any]:
        due = float(row["source_arrival_seconds"]) * scale
        remaining = origin + due - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        started = time.perf_counter()
        status, response = request_json(f"{base_url}/completion", {
            "prompt": row["prompt"], "n_predict": int(row["requested_output_tokens"]),
            "temperature": 0, "seed": int(protocol["random_seed"]) + int(row["trace_row"]),
            "cache_prompt": False, "n_probs": int(protocol["request"]["top_probabilities"]),
            "return_tokens": True,
        })
        ended = time.perf_counter()
        if status != 200 or not isinstance(response, dict):
            raise AssertionError(f"public replay HTTP failure: {status} {response}")
        return normalize_response(response, row, (ended - started) * 1000.0,
                                  due * 1000.0, (started - origin) * 1000.0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=int(protocol["service"]["parallel_slots"])) as pool:
        responses = list(pool.map(issue, rows))
    elapsed_ms = (time.perf_counter() - origin) * 1000.0
    responses.sort(key=lambda row: row["trace_row"])
    for response in responses:
        if response["cache_tokens"] != response["actual_local_input_tokens"]:
            raise AssertionError("runtime cache length differs from the frozen public prompt")
    return elapsed_ms, responses


def run_quality(base_url: str, cases: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for case in cases:
        status, response = request_json(f"{base_url}/completion", {
            "prompt": case["prompt"], "n_predict": int(protocol["quality"]["maximum_output_tokens"]),
            "temperature": 0, "seed": int(protocol["random_seed"]), "cache_prompt": False,
            "return_tokens": True,
        })
        if status != 200 or "error" in response:
            raise AssertionError(f"LongBench request failed: {status} {response}")
        prediction = str(response.get("content", ""))
        results.append({
            "dataset": case["dataset"], "record_id": case["record_id"],
            "prediction": prediction, "answers": case["answers"],
            "score": max(longbench_qa_f1(prediction, answer) for answer in case["answers"]),
            "output_token_ids": [int(value) for value in response["tokens"]],
        })
    return results


def collect_arm(protocol: dict[str, Any], workloads: dict[str, Any], server: Path, model: Path,
                output: Path, block: int, order: int, action: str, port: int) -> list[dict[str, Any]]:
    arm_dir = output / "raw" / f"block-{block:02d}-{order}-{action}"
    arm_dir.mkdir(parents=True)
    log_path = arm_dir / "server.log"
    service = protocol["service"]
    command = [
        str(server.resolve()), "-m", str(model.resolve()), "--host", "127.0.0.1", "--port", str(port),
        "-c", str(service["context_size"]), "-np", str(service["parallel_slots"]), "-b", "512", "-ub", "512",
        "-t", str(service["threads"]), "-ngl", str(service["gpu_layers"]), "--flash-attn", "on",
        "--no-warmup", "--metrics", "--no-cache-idle-slots", "--cache-ram", "0",
        "--kv-block-runtime", "--kv-block-size", str(service["kv_block_size_tokens"]),
        "--kv-paged-decode", "--kv-action-policy", "analytical", "--kv-action-override", action, "-lv", "4",
    ]
    base_url = f"http://127.0.0.1:{port}"
    collected = []
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            command, cwd=ROOT, env=cuda_environment({
                "LLAMA_CACHEFLOW_PAGED_KERNEL": service["paged_kernel_variant"],
                "LLAMA_CACHEFLOW_PAGED_CONTIGUOUS_FASTPATH": "1",
            }), stdout=log_file, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            wait_ready(base_url, process, log_path, attempts=180)
            request_json(f"{base_url}/completion", {"prompt": "Public replay warmup.", "n_predict": 1,
                                                    "temperature": 0, "cache_prompt": False})
            quality = run_quality(base_url, workloads["quality_cases"], protocol) if block == 1 else []
            for index, (trace_source, replay) in enumerate(workloads["performance_replays"].items()):
                before = get_text(f"{base_url}/metrics")
                elapsed_ms, responses = replay_trace(base_url, replay, protocol)
                after = get_text(f"{base_url}/metrics")
                row = {
                    "block": block, "order_in_block": order, "action": action,
                    "trace_source": trace_source, "elapsed_ms": elapsed_ms, "requests": responses,
                    "route": route_delta(before, after), "quality": quality if index == 0 else [],
                }
                (arm_dir / f"{trace_source}.json").write_text(
                    json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                )
                collected.append(row)
        finally:
            terminate_process(process)
    (arm_dir / "arm.json").write_text(json.dumps({
        "block": block, "order_in_block": order, "action": action,
        "server_log_sha256": sha256(log_path), "traces": len(collected),
    }, indent=2) + "\n", encoding="utf-8")
    return collected


def render_report(summary: dict[str, Any]) -> str:
    ci = summary["block_cluster_bootstrap_95_percent"]
    return "\n".join([
        "# Official public workload Direct/Paged result", "",
        f"- Promotion: **{'PASS' if summary['promotion_passed'] else 'FAIL'}**",
        f"- Trace sources: {', '.join(summary['trace_sources'])}",
        f"- Median throughput gain: {summary['throughput_gain_percent']['median']:+.2f}%",
        f"- Matched-process-block bootstrap 95% interval: [{ci[0]:+.2f}%, {ci[1]:+.2f}%]",
        f"- Request P95 Direct/Paged: {summary['direct_request_latency_p95_ms']:.3f}/{summary['paged_request_latency_p95_ms']:.3f} ms",
        f"- P95 latency regression: {summary['p95_latency_regression_percent']:+.2f}%",
        f"- Exact trace outputs: {summary['correctness']['token_matches']}/{summary['correctness']['token_comparisons']}",
        f"- LongBench paired outputs: {summary['quality']['token_matches']}/{summary['quality']['comparisons']}",
        f"- Maximum Direct/Paged LongBench score delta: {summary['quality']['maximum_score_delta']:.6f}", "",
        "BurstGPT/Azure arrival traces and LongBench text are separately sourced and then matched for replay. "
        "This is trace-driven public-content synthetic replay, not a claim about their joint production distribution.", "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8360)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    workload_path = ROOT / protocol["workload_file"]
    workloads = json.loads(workload_path.read_text(encoding="utf-8"))
    binding = observed_binding(args.protocol, args.server, args.model, workload_path)
    verify_binding(protocol, binding)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for block, order, action in execution_plan(protocol):
        rows.extend(collect_arm(protocol, workloads, args.server, args.model, args.output,
                                block, order, action, args.port))
    summary = analyze_public_external(protocol, rows)
    (args.output / "trials.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render_report(summary), encoding="utf-8")
    hashes = {str(path.relative_to(args.output)).replace("\\", "/"): sha256(path)
              for path in sorted((args.output / "raw").rglob("*")) if path.is_file()}
    (args.output / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "binding": binding, "raw_sha256": hashes,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
