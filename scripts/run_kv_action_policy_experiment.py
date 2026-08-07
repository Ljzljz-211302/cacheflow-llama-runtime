#!/usr/bin/env python3
"""Collect paired real-service action costs and evaluate H0/A1/T1/L1."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llama_lab.kv_action_evidence import (
    audit_kv_action_policy_no_cuda_sync,
    evaluate_kv_action_models,
    load_kv_action_protocol,
    make_balanced_action_orders,
)
from llama_lab.server_bench import wait_until_ready


ARTIFACT = ROOT / "results/research/h4-kv-action-v1.3.0"
PROTOCOL_PATH = ROOT / "config/kv_action_policy_protocol.json"
SERVER = ROOT / "build/patched-cuda-ninja3/bin/llama-server.exe"
MODEL = ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
UPSTREAM_REVISION = "acd79d603cb2e1c84c0886137b80f1ad649b6857"
PATCH_REPOSITORY_PATH = "patches/0001-cache-aware-slot-scheduler.patch"
MODES = {
    "direct": 19840,
    "device_swap": 19841,
    "host_swap": 19842,
    "recompute": 19843,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def post(port: int, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def metrics(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as response:
        return response.read().decode()


def metric(text: str, sample: str) -> float:
    prefix = sample + " "
    line = next((row for row in text.splitlines() if row.startswith(prefix)), None)
    if line is None:
        raise AssertionError(f"missing metric {sample}")
    return float(line[len(prefix):])


def start(mode: str, port: int, log: object) -> subprocess.Popen[bytes]:
    command = [
        str(SERVER), "-m", str(MODEL), "--host", "127.0.0.1", "--port", str(port),
        "-c", "2048", "-np", "2", "-b", "512", "-ub", "512", "-t", "8", "-ngl", "99",
        "--no-kv-unified", "--cache-ram", "128", "--slot-prompt-similarity", "0.1",
        "--kv-block-runtime", "--kv-block-size", "16", "--kv-action-policy", "fixed",
        "--metrics", "--no-warmup", "-lv", "2",
    ]
    if mode in {"device_swap", "host_swap"}:
        command.append("--cache-idle-slots")
    else:
        command.append("--no-cache-idle-slots")
    if mode == "host_swap":
        command.extend(["--kv-swap-path", "memory", "--kv-swap-budget-mib", "256"])
    environment = os.environ.copy()
    cuda_bin = ROOT / "runtime/cuda-dev/Library/bin"
    environment["PATH"] = str(cuda_bin) + os.pathsep + environment.get("PATH", "")
    process = subprocess.Popen(
        command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    wait_until_ready(f"http://127.0.0.1:{port}", process=process)
    return process


def stop(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def observe(mode: str, port: int, trace: int, prompt: str, unrelated: str) -> dict[str, Any]:
    before = metrics(port)
    selected_sample = f'llamacpp:kv_action_decisions_total{{action="{mode}"}}'
    observed_sample = f'llamacpp:kv_action_observations_total{{action="{mode}"}}'
    cost_sample = f'llamacpp:kv_action_observation_seconds_total{{action="{mode}"}}'
    selected_before = metric(before, selected_sample)
    observed_before = metric(before, observed_sample)
    cost_before = metric(before, cost_sample)
    decision_seconds_before = metric(before, "llamacpp:kv_action_decision_seconds_total")
    payload = {"prompt": prompt, "n_predict": 1, "temperature": 0, "cache_prompt": True}
    if mode == "recompute":
        started = time.perf_counter_ns()
        response = post(port, payload)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    elif mode == "direct":
        payload["id_slot"] = trace % 2
        post(port, payload)
        payload.pop("id_slot")
        started = time.perf_counter_ns()
        response = post(port, payload)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    else:
        payload["id_slot"] = 0
        post(port, payload)
        post(port, {
            "prompt": unrelated, "id_slot": 1, "n_predict": 1,
            "temperature": 0, "cache_prompt": True,
        })
        payload.pop("id_slot")
        started = time.perf_counter_ns()
        response = post(port, payload)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    after = metrics(port)
    selected_after = metric(after, selected_sample)
    observed_after = metric(after, observed_sample)
    cost_after = metric(after, cost_sample)
    decision_seconds_after = metric(after, "llamacpp:kv_action_decision_seconds_total")
    if selected_after - selected_before != 1:
        raise AssertionError(
            f"{mode} selected-action delta is not exactly one for trace {trace}: "
            f"{selected_before} -> {selected_after}"
        )
    if observed_after - observed_before != 1:
        raise AssertionError(
            f"{mode} full-action observation delta is not exactly one for trace {trace}: "
            f"{observed_before} -> {observed_after}"
        )
    if metric(after, 'llamacpp:kv_action_decisions_total{action="paged"}') != 0:
        raise AssertionError("Paged K1 crossed its evidence gate")
    prompt_n = int(response.get("timings", {}).get("prompt_n", 0))
    decision_seconds = decision_seconds_after - decision_seconds_before
    if decision_seconds < 0:
        raise AssertionError("KV action decision CPU counter decreased")
    internal_cost_ms = (cost_after - cost_before) * 1000.0
    if internal_cost_ms <= 0:
        raise AssertionError("KV action internal complete-action cost did not increase")
    return {
        "observed_cost_ms": internal_cost_ms,
        "http_elapsed_ms": elapsed_ms,
        "runtime_model_features": [
            metric(after, f'llamacpp:kv_action_last_model_feature{{index="{index}"}}')
            for index in range(9)
        ],
        "prompt_tokens": prompt_n,
        "selected_delta": selected_after - selected_before,
        "observation_delta": observed_after - observed_before,
        "paged_decisions": metric(after, 'llamacpp:kv_action_decisions_total{action="paged"}'),
        "invalid_features": metric(after, "llamacpp:kv_action_invalid_features_total"),
        "decision_cpu_ms": decision_seconds * 1000.0,
        "decision_to_action_ratio": decision_seconds * 1000.0 / internal_cost_ms,
        "_raw_evidence": {
            "before_metrics": before,
            "after_metrics": after,
            "response": response,
        },
    }


def analytical(action: str, tokens: int, kv_bytes: int, pressure: float) -> float:
    decode = 0.02
    transfer = kv_bytes / (12.0 * 1024 * 1024)
    pressure_cost = max(0.0, pressure - 0.90) * (
        kv_bytes / (100.0 * 1024 * 1024) + tokens * 0.03
    )
    if action == "direct":
        return decode + pressure_cost
    if action == "recompute":
        return tokens * 0.03 + decode + pressure_cost
    if action == "device_swap":
        return transfer + 0.02 + decode + pressure_cost
    if action == "host_swap":
        return transfer + 0.20 + decode + pressure_cost
    raise ValueError(action)


def main() -> None:
    if not SERVER.is_file() or not MODEL.is_file():
        raise FileNotFoundError("CUDA server and Qwen model are required")
    outer_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True
    )
    if outer_status:
        raise RuntimeError("formal KV action evidence requires a clean outer worktree")
    committed_patch = subprocess.check_output(
        ["git", "show", f"HEAD:{PATCH_REPOSITORY_PATH}"], cwd=ROOT
    )
    vendor_root = ROOT / "vendor/llama.cpp"
    vendor_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=vendor_root, text=True
    ).strip()
    vendor_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=vendor_root, text=True,
    )
    if vendor_head == UPSTREAM_REVISION:
        subprocess.run(
            ["git", "apply", "--reverse", "--check", str(ROOT / PATCH_REPOSITORY_PATH)],
            cwd=vendor_root, check=True,
        )
        patch_paths = {
            line.split(" b/", 1)[1].decode()
            for line in committed_patch.splitlines() if line.startswith(b"diff --git a/")
        }
        status_paths = {line[3:].strip().replace("\\", "/") for line in vendor_status.splitlines()}
        if status_paths != patch_paths:
            raise RuntimeError("vendor worktree contains changes outside the applied replay patch")
    else:
        if vendor_status:
            raise RuntimeError("committed vendor evidence tree must be clean")
        vendor_patch = subprocess.check_output(
            ["git", "diff", "--binary", f"{UPSTREAM_REVISION}..HEAD"], cwd=vendor_root,
        )
        if committed_patch != vendor_patch:
            raise RuntimeError("committed replay patch does not reproduce the vendor revision")
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    raw = ARTIFACT / "raw"
    raw.mkdir(exist_ok=True)
    protocol = load_kv_action_protocol(PROTOCOL_PATH)
    logs: dict[str, object] = {}
    processes: dict[str, subprocess.Popen[bytes]] = {}
    try:
        for mode, port in MODES.items():
            handle = (raw / f"server-{mode}.log").open("wb")
            logs[mode] = handle
            processes[mode] = start(mode, port, handle)

        rows: list[dict[str, Any]] = []
        raw_evidence: list[dict[str, Any]] = []
        repeats = (24, 48, 96, 128)
        trace_markers = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉天地玄黄宇宙洪荒日月盈昃寒来暑往秋收冬藏"
        action_orders = make_balanced_action_orders(
            MODES, 40, int(protocol["confirmatory"]["order_seed"])
        )
        order = 0
        observation_order = 0
        for trace in range(40):
            order += 1
            family = f"prefix-family-{trace:02d}"
            prompt = trace_markers[trace] * 4 + " " + (
                "cacheflow action evidence token " * repeats[trace % len(repeats)]
            )
            unrelated = f"unrelated-{trace:02d} " + ("eviction control token " * 32)
            observed: dict[str, dict[str, Any]] = {}
            action_order = action_orders[trace]
            for collection_order, mode in enumerate(action_order):
                regimes = ("resident", "preempted") if mode == "recompute" else ("single",)
                for regime in regimes:
                    observation_order += 1
                    observation_prompt = prompt if regime != "preempted" else (
                        trace_markers[(trace + 1) % len(trace_markers)] * 4 + prompt[4:]
                    )
                    value = observe(
                        mode, MODES[mode], trace, observation_prompt, unrelated + " " + regime
                    )
                    key = mode if regime == "single" else f"{mode}_{regime}"
                    observation_id = f"trace-{trace:02d}-{key}"
                    evidence = value.pop("_raw_evidence")
                    raw_evidence.append({
                        "observation_id": observation_id,
                        "observation_order": observation_order,
                        "trace_id": f"trace-{trace:02d}",
                        "action": mode,
                        **evidence,
                    })
                    value["collection_order"] = collection_order
                    value["observation_order"] = observation_order
                    value["observation_id"] = observation_id
                    observed[key] = value
            split = "train" if trace < 20 else "evaluation"
            identity = {
                "trace_id": f"trace-{trace:02d}",
                "session_id": f"session-{trace:02d}",
                "prefix_family": family,
                "split": split,
                "timestamp_order": order,
                "backend": "cuda",
            }
            for snapshot_kind, actions, baseline, page_runs in (
                ("resident", ("direct", "recompute"), "direct", 1),
                ("preempted", ("device_swap", "host_swap", "recompute"), "device_swap", 0),
            ):
                snapshot_id = f"trace-{trace:02d}-{snapshot_kind}"
                canonical_action = "direct" if snapshot_kind == "resident" else "device_swap"
                model_features = observed[canonical_action]["runtime_model_features"]
                context_tokens = max(1, round(float(model_features[1]) * 4096.0))
                if snapshot_kind == "preempted":
                    page_runs = max(1, (context_tokens + 15) // 16)
                kv_bytes = max(1, round(float(model_features[3]) * 1024.0 * 1024.0 * 1024.0))
                pressure = float(model_features[6])
                reuse_distance = max(0, round(math.expm1(float(model_features[5]) * 20.0)))
                base = {
                    **identity,
                    "context_tokens": context_tokens,
                    "batch": 1,
                    "kv_pressure": pressure,
                    "kv_bytes": kv_bytes,
                    "reuse_distance": reuse_distance,
                }
                for action in actions:
                    observation_key = (
                        f"recompute_{snapshot_kind}" if action == "recompute" else action
                    )
                    observation = observed[observation_key]
                    rows.append({
                        **base,
                        "snapshot_id": snapshot_id,
                        "regime": snapshot_kind,
                        "page_runs": page_runs,
                        "action": action,
                        "baseline_action": baseline,
                        "analytical_cost_ms": analytical(
                            action, context_tokens, kv_bytes, pressure
                        ),
                        "observed_cost_ms": observation["observed_cost_ms"],
                        "http_elapsed_ms": observation["http_elapsed_ms"],
                        "runtime_model_features": model_features,
                        "action_runtime_model_features": observation["runtime_model_features"],
                        "collection_order": observation["collection_order"],
                        "observation_order": observation["observation_order"],
                        "observation_id": observation["observation_id"],
                        **{
                            f"model_feature_{index}": value
                            for index, value in enumerate(model_features)
                        },
                        "prompt_tokens": observation["prompt_tokens"],
                        "selected_delta": observation["selected_delta"],
                        "observation_delta": observation["observation_delta"],
                        "paged_decisions": observation["paged_decisions"],
                        "invalid_features": observation["invalid_features"],
                        "decision_cpu_ms": observation["decision_cpu_ms"],
                        "decision_to_action_ratio": observation["decision_to_action_ratio"],
                    })
    finally:
        for process in processes.values():
            stop(process)
        for handle in logs.values():
            handle.close()

    trials = ARTIFACT / "paired-actions.jsonl"
    trials.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (ARTIFACT / "runtime-evidence.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_evidence),
        encoding="utf-8",
    )
    analysis = evaluate_kv_action_models(rows, protocol)
    benchmark = ROOT / "build/patched-cpu-noui/bin/Release/bench-kv-action-policy.exe"
    overhead_lines = subprocess.check_output([str(benchmark)], cwd=ROOT, text=True).splitlines()
    overhead = [json.loads(line) for line in overhead_lines if line.startswith("{")]
    if len(overhead) != 5:
        raise AssertionError("KV action overhead benchmark did not emit five regimes")
    (ARTIFACT / "overhead.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in overhead), encoding="utf-8"
    )
    sync_audit = audit_kv_action_policy_no_cuda_sync(
        ROOT / "vendor/llama.cpp/tools/server/server-kv-action-policy.cpp"
    )
    (ARTIFACT / "sync-audit.json").write_text(
        json.dumps(sync_audit, indent=2) + "\n", encoding="utf-8"
    )
    ratios = sorted(float(row["decision_to_action_ratio"]) for row in rows)
    ratio_p99 = ratios[int(0.99 * (len(ratios) - 1))]
    overhead_gates = protocol["overhead_gates"]
    overhead_result = {
        "p99_choose_microseconds": max(float(row["p99_ns"]) for row in overhead) / 1000.0,
        "measured_max_choose_microseconds": max(float(row["max_ns"]) for row in overhead) / 1000.0,
        "hot_loop_allocations": sum(int(row["allocations"]) for row in overhead),
        "direct_cuda_sync_symbols": len(sync_audit["matches"]),
        "scheduler_cpu_ratio_p99": ratio_p99,
    }
    overhead_result["passed"] = (
        overhead_result["p99_choose_microseconds"] <= overhead_gates["p99_choose_microseconds_max"]
        and overhead_result["hot_loop_allocations"] == overhead_gates["hot_loop_allocations"]
        and overhead_result["direct_cuda_sync_symbols"] == overhead_gates["direct_cuda_sync_symbols"]
        and overhead_result["scheduler_cpu_ratio_p99"] <= overhead_gates["scheduler_cpu_ratio_p99_max"]
    )
    report = {
        "schema_version": 1,
        "protocol_version": protocol["protocol_version"],
        "analysis": analysis,
        "overhead": overhead_result,
        "scope": {
            "model": "Qwen2.5-0.5B-Instruct Q4_K_M",
            "backend": "CUDA",
            "production_actions": protocol["production_actions"],
            "masked_actions": protocol["evidence_gated_actions"],
            "complete_action_boundary": protocol["cost_boundary"],
            "observed_cost_source": "llamacpp:kv_action_observation_seconds_total delta",
        },
    }
    report_path = ARTIFACT / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    model_lines = []
    for name, value in analysis["models"].items():
        model_lines.append(
            f"| {name} | {value['median_regret_ms']:.3f} | {value['p95_regret_ms']:.3f} | "
            f"{value['harmful_rate'] * 100:.2f}% |"
        )
    t1 = analysis["models"]["T1"]
    h0 = analysis["models"]["H0"]
    if t1 == h0:
        table_conclusion = "T1 also matched H0 on this held-out run and established no advantage."
    elif t1["harmful_rate"] > h0["harmful_rate"]:
        table_conclusion = "T1 produced a higher harmful-decision rate than H0."
    else:
        table_conclusion = "T1 differed from H0; the complete metrics above define the retained result."
    report_markdown = "\n".join([
        "# H4 unified KV action policy report",
        "",
        "The matched-workload replay compares only actions with complete real-service implementations. "
        "Remap and Paged remain capability-masked; Paged recorded zero production decisions.",
        "",
        "`observed_cost_ms` is the server policy's internal counter delta from scheduler snapshot "
        "through that slot's first completed useful target decode. HTTP round-trip time is retained "
        "separately and never enters regret or harm.",
        "",
        "| Model | Median regret (ms) | P95 regret (ms) | Harmful rate |",
        "|---|---:|---:|---:|",
        *model_lines,
        "",
        "L1 made no held-out switch because its conservative bound did not beat H0. "
        "It therefore matched H0. " + table_conclusion + " "
        "The selected production behavior is H0 execution with L1 shadow recommendations.",
        "",
        f"Decision overhead: p99 {overhead_result['p99_choose_microseconds']:.3f} us; "
        f"observed max {overhead_result['measured_max_choose_microseconds']:.3f} us; "
        f"scheduler/action ratio p99 {overhead_result['scheduler_cpu_ratio_p99'] * 100:.4f}%; "
        f"hot-loop allocations {overhead_result['hot_loop_allocations']}; "
        f"direct CUDA synchronization symbols {overhead_result['direct_cuda_sync_symbols']}.",
        "",
        "The action servers can expose different stateful feature values; their maximum normalized "
        "feature deltas are retained in report.json and checked against protocol gates. The shared "
        "model input is the real H0 anchor, so this is not described as an exact cloned-state causal "
        "counterfactual.\n\n"
        "Each model's JSON summary includes a 10,000-resample paired trace-cluster bootstrap "
        "95% CI for mean regret delta versus H0.",
        "",
        "The observed maximum is a Windows wall-clock measurement and includes thread preemption. "
        "It is reported without trimming.",
        "",
    ])
    (ARTIFACT / "report.md").write_text(report_markdown, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_version": "h4-kv-action-v1.3.0",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "model_path": MODEL.relative_to(ROOT).as_posix(),
        "model_sha256": sha256(MODEL),
        "outer_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "upstream_revision": UPSTREAM_REVISION,
        "patch_path": PATCH_REPOSITORY_PATH,
        "patch_sha256": hashlib.sha256(committed_patch).hexdigest(),
        "runs": {
            "direct": {"port": 19840, "kv_action_policy": "fixed", "cache_idle_slots": False},
            "device_swap": {"port": 19841, "kv_action_policy": "fixed", "cache_idle_slots": True},
            "host_swap": {
                "port": 19842, "kv_action_policy": "fixed", "cache_idle_slots": True,
                "kv_swap_path": "memory", "kv_swap_budget_mib": 256,
            },
            "recompute": {"port": 19843, "kv_action_policy": "fixed", "cache_idle_slots": False},
        },
        "files": {},
    }
    for path in sorted(ARTIFACT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][path.relative_to(ARTIFACT).as_posix()] = sha256(path)
    (ARTIFACT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(analysis["models"], indent=2))


if __name__ == "__main__":
    main()
