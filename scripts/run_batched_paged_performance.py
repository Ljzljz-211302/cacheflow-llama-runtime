from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llama_lab.batched_paged_performance import analyze, experiment_plan  # noqa: E402
from llama_lab.gpu_memory import GpuMemorySampler  # noqa: E402
from production_journey import cuda_environment, get_text, request_json, terminate_process, wait_ready  # noqa: E402
from run_production_paged_journey import metric  # noqa: E402


RUNTIME_FILES = (
    "llama-server.exe", "llama-server-impl.dll", "llama.dll", "ggml.dll",
    "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def vendor_diff_sha256() -> str:
    value = subprocess.check_output(
        ["git", "-C", "vendor/llama.cpp", "diff", "--binary", "--no-ext-diff"], cwd=ROOT,
    ).replace(b"\r\n", b"\n")
    return hashlib.sha256(value).hexdigest()


def binding(protocol_path: Path, server: Path, model: Path) -> dict[str, Any]:
    return {
        "protocol_sha256": sha256(protocol_path),
        "runner_revision": git("rev-parse", "HEAD"),
        "vendor_revision": git("-C", "vendor/llama.cpp", "rev-parse", "HEAD"),
        "vendor_diff_sha256": vendor_diff_sha256(),
        "model_sha256": sha256(model),
        "runtime_sha256": {name: sha256(server.parent / name) for name in RUNTIME_FILES},
    }


def load_inputs(protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    corpus_path = ROOT / protocol["workload_file"]
    if sha256(corpus_path) != protocol["workload_sha256"]:
        raise ValueError("batched workload differs from the preregistered hash")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if corpus["target_context_tokens"] != protocol["matrix"]["context_tokens"]:
        raise ValueError("batched workload contexts differ from the protocol")
    if int(corpus["streams_per_context"]) < max(map(int, protocol["matrix"]["batch_sizes"])):
        raise ValueError("batched workload has too few independent prompt streams")
    for source in corpus["sources"]:
        if sha256(ROOT / source["path"]) != source["sha256"]:
            raise ValueError(f"batched workload source changed: {source['path']}")
    return protocol, corpus


def verify_binding(protocol: dict[str, Any], observed: dict[str, Any]) -> None:
    frozen = protocol["artifact_binding"]
    for field in ("vendor_revision", "vendor_diff_sha256", "model_sha256", "runtime_sha256"):
        if observed[field] != frozen[field]:
            raise RuntimeError(f"batched performance {field} differs from preregistration")
    if not git("merge-base", "--is-ancestor", frozen["outer_revision"], observed["runner_revision"]) == "":
        raise RuntimeError("unreachable")


def erase_slots(port: int, count: int) -> None:
    for slot in range(count):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/slots/{slot}?action=erase", data=b"", method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise AssertionError(f"failed to erase slot {slot}")


def issue_wave(base_url: str, payloads: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    common = {key: value for key, value in payloads[0].items() if key not in ("prompt", "seed")}
    batched_payload = {
        **common,
        "prompt": [payload["prompt"] for payload in payloads],
        "seed": int(payloads[0]["seed"]),
    }
    started = time.perf_counter_ns()
    status, body = request_json(f"{base_url}/completion", batched_payload)
    elapsed = (time.perf_counter_ns() - started) / 1.e6
    responses = body if isinstance(body, list) else [body]
    if (status != 200 or len(responses) != len(payloads) or any(
            "error" in response or len(response.get("tokens", [])) != 1 for response in responses)):
        raise AssertionError(f"controlled batched request failed: {status} {body}")
    return elapsed, responses


def metric_delta(before: str, after: str, name: str) -> float:
    return metric(after, name) - metric(before, name)


def normalize_response(response: dict[str, Any]) -> tuple[list[int], int, list[dict[str, float]]]:
    probabilities = response.get("completion_probabilities", [{}])[0].get("top_logprobs", [])
    return (
        [int(token) for token in response["tokens"]],
        int(response["timings"]["cache_n"]) + int(response["timings"]["prompt_n"]),
        [{"id": int(item["id"]), "logprob": float(item["logprob"])} for item in probabilities],
    )


def collect_arm(
    protocol: dict[str, Any], corpus: dict[str, Any], server: Path, model: Path,
    output: Path, block: int, order: int, action: str, port: int,
) -> list[dict[str, Any]]:
    arm_dir = output / "raw" / f"block-{block:02d}-{order}-{action}"
    arm_dir.mkdir(parents=True)
    slot_state = arm_dir / "slot-state"
    slot_state.mkdir()
    log_path = arm_dir / "server.log"
    service = protocol["service"]
    command = [
        str(server.resolve()), "-m", str(model.resolve()), "--host", "127.0.0.1", "--port", str(port),
        "-c", str(service["context_size"]), "-np", str(service["parallel_slots"]),
        "-b", "512", "-ub", "512", "-t", str(service["threads"]), "-ngl", str(service["gpu_layers"]),
        "--flash-attn", "on", "--no-warmup", "--metrics", "--slots", "--no-cache-idle-slots",
        "--cache-ram", "0",
        "--slot-save-path", str(slot_state.resolve()),
        "--kv-block-runtime", "--kv-block-size", str(service["kv_block_size_tokens"]),
        "--kv-paged-decode", "--kv-action-policy", "analytical", "--kv-action-override", action, "-lv", "4",
    ]
    workloads = {int(row["context_tokens"]): row for row in corpus["workloads"]}
    base_url = f"http://127.0.0.1:{port}"
    rows = []
    with log_path.open("wb") as log_file:
        environment = cuda_environment({"LLAMA_CACHEFLOW_PAGED_KERNEL": service["paged_kernel_variant"]})
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=log_file, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            wait_ready(base_url, process, log_path, attempts=180)
            cells = [row for row in experiment_plan(protocol) if row["block"] == block and row["action"] == action]
            for cell in cells:
                batch, context = int(cell["batch_size"]), int(cell["context_tokens"])
                erase_slots(port, int(service["parallel_slots"]))
                prompt_rows = workloads[context]["prompts"][:batch]
                payloads = [{
                    "prompt": row["prompt"], "n_predict": int(protocol["request"]["predicted_tokens"]),
                    "temperature": float(protocol["request"]["temperature"]),
                    "seed": int(protocol["random_seed"]) + int(row["stream"]),
                    "cache_prompt": True, "n_probs": int(protocol["request"]["top_probabilities"]),
                    "return_tokens": True,
                } for row in prompt_rows]
                for payload, prompt_row in zip(payloads, prompt_rows):
                    status, tokenized = request_json(f"{base_url}/tokenize", {"content": payload["prompt"]})
                    if status != 200 or len(tokenized.get("tokens", [])) != prompt_row["actual_prompt_tokens"]:
                        raise AssertionError("runtime tokenizer differs from frozen batched input")
                for _ in range(int(protocol["measurement"]["warm_waves_per_cell"])):
                    issue_wave(base_url, payloads)
                before = get_text(f"{base_url}/metrics")
                wave_times, responses = [], []
                with GpuMemorySampler(interval_seconds=0.05) as memory:
                    for _ in range(int(protocol["measurement"]["waves_per_cell"])):
                        wave, wave_responses = issue_wave(base_url, payloads)
                        wave_times.append(wave)
                        responses.extend(wave_responses)
                after = get_text(f"{base_url}/metrics")
                normalized = [normalize_response(response) for response in responses]
                row = {
                    "block": block, "order_in_block": order, "action": action,
                    "batch_size": batch, "context_tokens": context,
                    "prompt_sha256": [prompt["prompt_sha256"] for prompt in prompt_rows],
                    "wave_elapsed_ms": wave_times,
                    "output_token_ids": [item[0] for item in normalized],
                    "cache_tokens": [item[1] for item in normalized],
                    "top_logprobs": [item[2] for item in normalized],
                    "paged_calls": metric_delta(before, after, "llamacpp:paged_decode_calls_total"),
                    "paged_sequences": metric_delta(before, after, "llamacpp:paged_decode_sequences_total"),
                    "paged_fallbacks": metric_delta(before, after, "llamacpp:paged_decode_fallbacks_total"),
                    "cuda_dispatches": metric_delta(before, after, "llamacpp:paged_decode_cuda_dispatches_total"),
                    "cuda_sequences": metric_delta(before, after, "llamacpp:paged_decode_cuda_sequences_total"),
                    "action_decisions": metric_delta(before, after, f'llamacpp:kv_action_decisions_total{{action="{action}"}}'),
                    "peak_gpu_memory_mib": memory.peak_mib,
                }
                cell_path = arm_dir / f"batch-{batch}-context-{context}.json"
                cell_path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                rows.append(row)
        finally:
            terminate_process(process)
    (arm_dir / "arm.json").write_text(json.dumps({
        "block": block, "order_in_block": order, "action": action,
        "server_log_sha256": sha256(log_path), "cells": len(rows),
    }, indent=2) + "\n", encoding="utf-8")
    return rows


def render_report(summary: dict[str, Any]) -> str:
    interval = summary["primary_block_cluster_bootstrap_95_percent"]
    lines = [
        "# Batched Paged Decode objective performance result", "",
        f"- Promotion: **{'PASS' if summary['promotion_passed'] else 'FAIL'}**",
        f"- Batch-{summary['primary_batch_size']} throughput gain median: {summary['primary_throughput_gain_percent']['median']:+.2f}%",
        f"- Matched-process-block bootstrap 95% interval: [{interval[0]:+.2f}%, {interval[1]:+.2f}%]",
        f"- Batch-{summary['primary_batch_size']} P95 batched-wave latency regression: {summary['primary_p95_wave_latency_regression_percent']:+.2f}%",
        f"- Worst cell median batched-wave latency regression: {summary['worst_cell_median_wave_latency_regression_percent']:+.2f}%", "",
        f"- Exact output-token matches: {summary['correctness']['output_token_matches']}/{summary['correctness']['output_token_comparisons']}",
        f"- Top-64 minimum overlap / maximum common logprob error: {summary['correctness']['minimum_top64_overlap']} / {summary['correctness']['maximum_common_logprob_error']:.6f}", "",
        f"- Probability rows compared / incomplete: {summary['correctness']['probability_rows_compared']} / {summary['correctness']['incomplete_probability_rows']}", "",
        "## Throughput by batch", "", "| Batch | Median gain | 95% interval |", "|---:|---:|---:|",
    ]
    for batch, row in summary["throughput_by_batch"].items():
        ci = row["block_cluster_bootstrap_95_percent"]
        lines.append(f"| {batch} | {row['throughput_gain_percent']['median']:+.2f}% | [{ci[0]:+.2f}%, {ci[1]:+.2f}%] |")
    lines.extend(["", "GPU memory is descriptive only: Direct and Paged share the same allocator, so this experiment does not claim a capacity or fragmentation advantage.", ""])
    return "\n".join(lines)


def render_chart(summary: dict[str, Any]) -> str:
    batches = sorted(map(int, summary["throughput_by_batch"]))
    gains = [summary["throughput_by_batch"][str(batch)]["throughput_gain_percent"]["median"] for batch in batches]
    maximum = max(10.0, max(map(abs, gains)) * 1.2)
    bars = []
    for index, (batch, gain) in enumerate(zip(batches, gains)):
        x = 100 + index * 150
        y0 = 210
        height = 140 * abs(gain) / maximum
        y = y0 - height if gain >= 0 else y0
        color = "#13795b" if gain >= 0 else "#b42318"
        bars.append(f'<rect x="{x}" y="{y:.1f}" width="70" height="{height:.1f}" fill="{color}"/><text x="{x + 35}" y="250" text-anchor="middle">B={batch}</text><text x="{x + 35}" y="{y - 8 if gain >= 0 else y + height + 18:.1f}" text-anchor="middle">{gain:+.1f}%</text>')
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="290" viewBox="0 0 720 290">'
        '<rect width="720" height="290" fill="white"/><text x="24" y="32" font-size="18">Paged vs Direct throughput gain by batch</text>'
        '<line x1="60" y1="210" x2="680" y2="210" stroke="#555"/>' + "".join(bars) + '</svg>\n'
    )


def artifact_hashes(output: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output)).replace("\\", "/"): sha256(path)
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"
    }


def validate_artifact(protocol_path: Path, output: Path, server: Path, model: Path) -> dict[str, Any]:
    protocol, _ = load_inputs(protocol_path)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != sha256(protocol_path) or manifest["files"] != artifact_hashes(output):
        raise AssertionError("batched performance manifest differs from the artifact tree")
    observed = binding(protocol_path, server, model)
    verify_binding(protocol, observed)
    run_binding = json.loads((output / "execution-start-binding.json").read_text(encoding="utf-8"))
    verify_binding(protocol, run_binding)
    if any(run_binding[field] != observed[field] for field in (
            "protocol_sha256", "vendor_revision", "vendor_diff_sha256", "model_sha256", "runtime_sha256")):
        raise AssertionError("batched performance execution binding differs")
    rows = json.loads((output / "trials.json").read_text(encoding="utf-8"))
    raw_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in output.glob("raw/block-*/batch-*-context-*.json")
    ]
    row_key = lambda row: (int(row["block"]), int(row["order_in_block"]), row["action"], int(row["batch_size"]), int(row["context_tokens"]))
    if sorted(rows, key=row_key) != sorted(raw_rows, key=row_key):
        raise AssertionError("batched performance trials differ from raw cell evidence")
    for arm_path in output.glob("raw/block-*/arm.json"):
        arm = json.loads(arm_path.read_text(encoding="utf-8"))
        if arm["server_log_sha256"] != sha256(arm_path.parent / "server.log"):
            raise AssertionError("batched performance server log hash differs")
    summary = analyze(protocol, rows)
    if summary != json.loads((output / "summary.json").read_text(encoding="utf-8")):
        raise AssertionError("batched performance summary differs from raw trials")
    if (output / "report.md").read_text(encoding="utf-8") != render_report(summary):
        raise AssertionError("batched performance report differs from raw trials")
    if (output / "comparison.svg").read_text(encoding="utf-8") != render_chart(summary):
        raise AssertionError("batched performance chart differs from raw trials")
    return summary


def finalize_artifact(protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, _ = load_inputs(protocol_path)
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in output.glob("raw/block-*/batch-*-context-*.json")
    ]
    summary = analyze(protocol, rows)
    (output / "trials.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render_report(summary), encoding="utf-8")
    (output / "comparison.svg").write_text(render_chart(summary), encoding="utf-8")
    manifest = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(protocol_path), "analysis_revision": git("rev-parse", "HEAD"),
        "files": artifact_hashes(output),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "config/batched_paged_performance_protocol_v5.json")
    parser.add_argument("--server", type=Path, default=ROOT / "build/patched-cuda-ninja3/bin/llama-server.exe")
    parser.add_argument("--model", type=Path, default=ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    parser.add_argument("--output", type=Path, default=ROOT / "results/research/h19-production-batched-paged-v5.0.0")
    parser.add_argument("--port", type=int, default=8350)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        summary = validate_artifact(args.protocol, args.output, args.server, args.model)
        print(json.dumps({"validated": str(args.output), "promotion_passed": summary["promotion_passed"]}))
        return
    if args.finalize_existing:
        if (args.output / "manifest.json").exists():
            raise FileExistsError("completed batched performance artifact is immutable")
        summary = finalize_artifact(args.protocol, args.output)
        validate_artifact(args.protocol, args.output, args.server, args.model)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    protocol, corpus = load_inputs(args.protocol)
    if args.output.exists():
        raise FileExistsError("batched performance artifact is immutable")
    observed = binding(args.protocol, args.server, args.model)
    verify_binding(protocol, observed)
    args.output.mkdir(parents=True)
    (args.output / "execution-start-binding.json").write_text(json.dumps(observed, indent=2) + "\n", encoding="utf-8")
    rows = []
    arms = []
    for row in experiment_plan(protocol):
        identity = (row["block"], row["order_in_block"], row["action"])
        if identity not in arms:
            arms.append(identity)
    for block, order, action in arms:
        rows.extend(collect_arm(protocol, corpus, args.server, args.model, args.output, block, order, action, args.port))
    summary = finalize_artifact(args.protocol, args.output)
    validate_artifact(args.protocol, args.output, args.server, args.model)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
