from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llama_lab.objective_paged_benchmark import (  # noqa: E402
    analyze,
    arm_plan,
    file_sha256,
    load_definition,
    workload_order,
)
from production_journey import cuda_environment, get_text, request_json, terminate_process, wait_ready  # noqa: E402
from run_k2_production_experiment import erase_slot, metric_delta  # noqa: E402
from run_production_paged_experiment import action_reason_total, device_identity, git_output  # noqa: E402
from run_production_paged_journey import metric  # noqa: E402


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def render_report(summary: dict, corpus: dict) -> str:
    primary = summary["primary_paired_regression_percent"]
    interval = summary["primary_pair_cluster_bootstrap_95_percent"]
    rows = []
    for workload_id, result in summary["per_workload"].items():
        effect = result["paired_regression_percent"]
        rows.append(
            f"| {workload_id} | {result['category']} | {result['actual_context_tokens']} | "
            f"{result['actual_page_count']} | {effect['median']:+.2f}% | {effect['p95']:+.2f}% |"
        )
    decision = "PASS" if summary["promotion_passed"] else "FAIL"
    gate_lines = [
        f"- Preregistered upper bound: +{summary['promotion_limit_percent']:.2f}% → **{decision}**"
    ]
    if "primary_p95_limit_percent" in summary:
        gate_lines = [
            f"- Preregistered median-CI upper bound: +{summary['promotion_limit_percent']:.2f}%",
            f"- P95 regression / limit: {primary['p95']:+.2f}% / +{summary['primary_p95_limit_percent']:.2f}%",
            f"- Worst workload median / limit: {summary['worst_workload_median_regression_percent']:+.2f}% / +{summary['worst_workload_limit_percent']:.2f}%",
            f"- Required page coverage passed: {summary['page_coverage_passed']} → **{decision}**",
        ]
    return "\n".join([
        "# Objective Paged-vs-Direct prompt-matrix report", "",
        f"- Frozen corpus: `{corpus['corpus_version']}`; {summary['workload_count']} workloads",
        f"- Design: {summary['paired_trials']} process pairs × Direct/Paged × every workload",
        f"- Raw workload-arm observations: {summary['observations']}",
        f"- Primary median paired regression: {primary['median']:+.2f}%",
        f"- Pair-cluster bootstrap 95% interval: [{interval[0]:+.2f}%, {interval[1]:+.2f}%]",
        *gate_lines, "",
        "| Workload | Category | Actual tokens | Pages | Median regression | P95 regression |",
        "|---|---|---:|---:|---:|---:|", *rows, "",
        "Positive values mean Paged is slower. Results are stratified rather than inferred from one synthetic prompt.", "",
    ])


def render_chart(summary: dict) -> str:
    items = list(summary["per_workload"].items())
    width, height = 1100, 120 + 55 * len(items)
    zero_x, scale = 540, 5.0
    bars = []
    for index, (name, result) in enumerate(items):
        value = float(result["paired_regression_percent"]["median"])
        y = 85 + index * 55
        x = zero_x if value >= 0 else zero_x + value * scale
        bar_width = max(1.0, abs(value) * scale)
        color = "#dc2626" if value > 0 else "#16a34a"
        bars.append(
            f'<text x="20" y="{y + 17}">{name} ({result["actual_context_tokens"]} tok)</text>'
            f'<rect x="{x:.1f}" y="{y}" width="{bar_width:.1f}" height="24" fill="{color}"/>'
            f'<text x="{zero_x + value * scale + (8 if value >= 0 else -8):.1f}" y="{y + 17}" text-anchor="{"start" if value >= 0 else "end"}">{value:+.2f}%</text>'
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><style>text{{font-family:Segoe UI,Arial,sans-serif;fill:#172033;font-size:14px}}.title{{font-size:20px;font-weight:700}}</style><rect width="100%" height="100%" fill="#f8fafc"/><text x="20" y="32" class="title">Paged vs Direct by frozen prompt workload</text><text x="20" y="55">Median paired regression; negative is faster, positive is slower</text><line x1="{zero_x}" y1="70" x2="{zero_x}" y2="{height - 20}" stroke="#64748b" stroke-width="2"/>{''.join(bars)}</svg>'''


def validate_artifact(root: Path, protocol_path: Path, output: Path) -> dict:
    protocol, corpus = load_definition(root, protocol_path)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    actual_files = {
        str(path.relative_to(output).as_posix()): file_sha256(path)
        for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"
    }
    if manifest["files"] != actual_files:
        raise AssertionError("objective artifact manifest differs from exact file tree")
    if manifest["protocol_sha256"] != file_sha256(protocol_path):
        raise AssertionError("objective artifact protocol hash differs")
    corpus_path = root / protocol["workload_file"]
    if manifest["corpus_sha256"] != file_sha256(corpus_path):
        raise AssertionError("objective artifact corpus hash differs")
    rows = json.loads((output / "trials.json").read_text(encoding="utf-8"))
    expected = analyze(protocol, corpus, rows)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    binding = protocol.get("artifact_binding")
    if binding and (
        summary.get("server_sha256") != binding["server_sha256"]
        or summary.get("vendor_revision") != binding["vendor_revision"]
    ):
        raise AssertionError("objective artifact binary/source binding differs")
    for field, value in expected.items():
        if summary.get(field) != value:
            raise AssertionError(f"objective summary field is not derived from raw rows: {field}")
    if (output / "report.md").read_text(encoding="utf-8") != render_report(summary, corpus):
        raise AssertionError("objective report differs from validated summary")
    if (output / "comparison.svg").read_text(encoding="utf-8") != render_chart(summary):
        raise AssertionError("objective chart differs from validated summary")
    return summary


def normalize_cell(protocol: dict, workload: dict, cell: dict, cell_path: Path, output: Path) -> dict:
    pair, order, action = int(cell["pair"]), int(cell["order_in_pair"]), cell["action"]
    expected_payload = {
        "prompt": workload["prompt"],
        "n_predict": int(protocol["request"]["predicted_tokens"]),
        "temperature": float(protocol["request"]["temperature"]),
        "seed": int(protocol["random_seed"]),
        "cache_prompt": bool(protocol["request"]["cache_prompt"]),
    }
    responses = cell["responses"]
    elapsed_values = cell["client_elapsed_ms"]
    if cell["workload_id"] != workload["id"] or cell["payload"] != expected_payload:
        raise ValueError("completed objective arm differs from the frozen workload")
    if len(responses) != int(protocol["request"]["measured_requests_per_workload_arm"]):
        raise ValueError("completed objective arm has incomplete responses")
    before, after = cell["metrics_before"], cell["metrics_after"]
    return {
        "pair": pair, "order_in_pair": order, "action": action,
        "workload_id": workload["id"], "category": workload["category"],
        "prompt_sha256": prompt_sha256(workload["prompt"]),
        "client_elapsed_ms": elapsed_values,
        "actual_context_tokens": [int(row["timings"]["cache_n"]) + int(row["timings"]["prompt_n"]) for row in responses],
        "contents": [row["content"] for row in responses],
        "paged_calls": metric_delta(before, after, "llamacpp:paged_decode_calls_total "),
        "paged_fallbacks": metric_delta(before, after, "llamacpp:paged_decode_fallbacks_total "),
        "action_decisions": metric_delta(before, after, f'llamacpp:kv_action_decisions_total{{action="{action}"}}'),
        "action_reason_decisions": action_reason_total(after, action) - action_reason_total(before, action),
        "action_observations": metric_delta(before, after, f'llamacpp:kv_action_observations_total{{action="{action}"}}'),
        "raw": str(cell_path.relative_to(output).as_posix()),
    }


def load_completed_arm(protocol: dict, corpus: dict, output: Path,
                       pair: int, order: int, action: str) -> list[dict] | None:
    arm_dir = output / "raw" / f"pair-{pair:02d}-{order}-{action}"
    if not arm_dir.exists():
        return None
    workloads = {row["id"]: row for row in corpus["workloads"]}
    expected = {f"{workload_id}.json" for workload_id in workloads}
    actual = {path.name for path in arm_dir.glob("*.json")}
    if actual != expected or not (arm_dir / "server.log").is_file():
        raise ValueError(f"partial objective arm cannot be resumed: {arm_dir}")
    result = []
    for workload_id in workload_order(protocol, corpus, pair, action):
        cell_path = arm_dir / f"{workload_id}.json"
        cell = json.loads(cell_path.read_text(encoding="utf-8"))
        if (int(cell["pair"]), int(cell["order_in_pair"]), cell["action"]) != (pair, order, action):
            raise ValueError("completed objective arm identity differs from seeded plan")
        result.append(normalize_cell(protocol, workloads[workload_id], cell, cell_path, output))
    return result


def collect_arm(protocol: dict, corpus: dict, server: Path, model: Path, output: Path,
                pair: int, order: int, action: str, port: int) -> list[dict]:
    arm_dir = output / "raw" / f"pair-{pair:02d}-{order}-{action}"
    arm_dir.mkdir(parents=True)
    slot_save_path = arm_dir / "slot-state"
    slot_save_path.mkdir()
    log_path = arm_dir / "server.log"
    service = protocol["service"]
    command = [
        str(server.resolve()), "-m", str(model.resolve()), "--host", "127.0.0.1",
        "--port", str(port), "-c", str(service["context_size"]), "-np", str(service["parallel_slots"]),
        "-t", str(service["threads"]), "-ngl", str(service["gpu_layers"]), "--flash-attn", "on",
        "--no-warmup", "--metrics", "--slots", "--slot-save-path", str(slot_save_path.resolve()),
        "--kv-block-runtime", "--kv-block-size",
        str(service["kv_block_size_tokens"]), "--kv-paged-decode", "--kv-action-policy", "analytical",
        "--kv-action-override", action, "-lv", "4",
    ]
    workloads = {row["id"]: row for row in corpus["workloads"]}
    payload_base = protocol["request"]
    result_rows = []
    base_url = f"http://127.0.0.1:{port}"
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=cuda_environment(), stdout=log,
                                   stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            wait_ready(base_url, process, log_path, attempts=120)
            for workload_id in workload_order(protocol, corpus, pair, action):
                workload = workloads[workload_id]
                erase_slot(port)
                payload = {
                    "prompt": workload["prompt"],
                    "n_predict": int(payload_base["predicted_tokens"]),
                    "temperature": float(payload_base["temperature"]),
                    "seed": int(protocol["random_seed"]),
                    "cache_prompt": bool(payload_base["cache_prompt"]),
                }
                before = get_text(f"{base_url}/metrics")
                for _ in range(int(payload_base["warm_requests_per_workload_arm"])):
                    request_json(f"{base_url}/completion", payload)
                elapsed_values, responses = [], []
                for _ in range(int(payload_base["measured_requests_per_workload_arm"])):
                    started = time.perf_counter_ns()
                    status, response = request_json(f"{base_url}/completion", payload)
                    elapsed_values.append((time.perf_counter_ns() - started) / 1.e6)
                    if status != 200 or "error" in response or not response.get("content"):
                        raise AssertionError(f"objective request failed: {response}")
                    responses.append(response)
                after = get_text(f"{base_url}/metrics")
                cell = {
                    "pair": pair, "order_in_pair": order, "action": action,
                    "workload_id": workload_id, "payload": payload,
                    "responses": responses, "client_elapsed_ms": elapsed_values,
                    "metrics_before": before, "metrics_after": after,
                }
                cell_path = arm_dir / f"{workload_id}.json"
                cell_path.write_text(json.dumps(cell, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                result_rows.append(normalize_cell(protocol, workload, cell, cell_path, output))
        finally:
            terminate_process(process)
            process.wait(timeout=15)
    return result_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "config/production_paged_objective_protocol_v1.json")
    parser.add_argument("--server", type=Path, default=ROOT / "build/patched-cuda-ninja3/bin/llama-server.exe")
    parser.add_argument("--model", type=Path, default=ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    parser.add_argument("--output", type=Path, default=ROOT / "results/research/h9-objective-paged-v1.0.0")
    parser.add_argument("--port", type=int, default=8320)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_artifact(ROOT, args.protocol, args.output), ensure_ascii=False))
        return
    protocol, corpus = load_definition(ROOT, args.protocol)
    if (args.output / "manifest.json").exists():
        raise FileExistsError("completed objective artifact is immutable; use --validate-only")
    if device_identity()["name"] != protocol["device"]["name"]:
        raise RuntimeError("objective benchmark device differs from protocol")
    binding = protocol.get("artifact_binding")
    if binding and (
        file_sha256(args.server) != binding["server_sha256"]
        or git_output("-C", "vendor/llama.cpp", "rev-parse", "HEAD") != binding["vendor_revision"]
    ):
        raise RuntimeError("objective benchmark binary/source binding differs from protocol")
    clean = not bool(git_output("status", "--porcelain", "--untracked-files=no"))
    if not clean:
        raise RuntimeError("objective benchmark requires a clean tracked worktree")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair, order, action in arm_plan(protocol):
        completed = load_completed_arm(protocol, corpus, args.output, pair, order, action)
        rows.extend(completed if completed is not None else collect_arm(
            protocol, corpus, args.server, args.model, args.output, pair, order, action, args.port
        ))
        print(json.dumps({"pair": pair, "order": order, "action": action}, ensure_ascii=False), flush=True)
    summary = analyze(protocol, corpus, rows)
    summary.update({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_output("rev-parse", "HEAD"),
        "vendor_revision": git_output("-C", "vendor/llama.cpp", "rev-parse", "HEAD"),
        "server_sha256": file_sha256(args.server), "model_sha256": file_sha256(args.model),
        "device": device_identity(), "worktree_clean_before_run": clean,
    })
    (args.output / "trials.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render_report(summary, corpus), encoding="utf-8")
    (args.output / "comparison.svg").write_text(render_chart(summary), encoding="utf-8")
    corpus_path = ROOT / protocol["workload_file"]
    files = {str(path.relative_to(args.output).as_posix()): file_sha256(path)
             for path in args.output.rglob("*") if path.is_file()}
    manifest = {"schema_version": 1, "protocol_sha256": file_sha256(args.protocol),
                "corpus_sha256": file_sha256(corpus_path), "files": files}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validate_artifact(ROOT, args.protocol, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
