from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_production_paged_experiment import (
    ROOT,
    action_reason_total,
    bootstrap_median_interval,
    device_identity,
    discover_nsys,
    git_output,
    sha256,
    summarize,
)
from production_journey import (
    cuda_environment,
    get_text,
    request_json,
    terminate_process,
    wait_ready,
)
from run_production_paged_journey import metric, run_mode

import sys

sys.path.insert(0, str(ROOT / "src"))
from llama_lab.cuda_profile_evidence import parse_nsys_sqlite  # noqa: E402


VARIANTS = {
    "k1": ("K1", "cacheflow_paged_decode_fattn_k1"),
    "k2": ("K2", "cacheflow_paged_decode_fattn_k2_t2"),
}


def metric_delta(before: str, after: str, name: str) -> float:
    return metric(after, name) - metric(before, name)


def erase_slot(port: int) -> None:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/slots/0?action=erase", data=b"", method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise AssertionError("failed to erase the single controlled slot")


def randomized_arm_plan(protocol: dict[str, Any]) -> list[tuple[int, int, str]]:
    generator = random.Random(int(protocol["random_seed"]))
    plan: list[tuple[int, int, str]] = []
    for pair in range(1, int(protocol["paired_trials"]) + 1):
        variants = list(VARIANTS)
        generator.shuffle(variants)
        plan.extend((pair, order, variant) for order, variant in enumerate(variants, 1))
    return plan


def parse_structured_rows(protocol: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, (pair, order_in_pair, variant) in enumerate(randomized_arm_plan(protocol), 1):
        directory = output / f"raw/arm-{arm:03d}-{variant}"
        response = json.loads((directory / "response.json").read_text(encoding="utf-8"))
        timing = json.loads((directory / "client-timing.json").read_text(encoding="utf-8"))
        before = (directory / "metrics-before.prom").read_text(encoding="utf-8")
        after = (directory / "metrics-after.prom").read_text(encoding="utf-8")
        if timing != {
            "pair": pair,
            "order_in_pair": order_in_pair,
            "variant": variant,
            "client_elapsed_ns": timing.get("client_elapsed_ns"),
        } or not isinstance(timing["client_elapsed_ns"], int) or timing["client_elapsed_ns"] <= 0:
            raise AssertionError("raw client timing does not match the seeded arm plan")
        rows.append({
            "pair": pair,
            "order_in_pair": order_in_pair,
            "variant": variant,
            "content": response["content"],
            "client_elapsed_ms": timing["client_elapsed_ns"] / 1.e6,
            "prompt_ms": float(response["timings"]["prompt_ms"]),
            "action_decisions": metric_delta(
                before, after, 'llamacpp:kv_action_decisions_total{action="paged"}'),
            "action_reason_decisions": (
                action_reason_total(after, "paged") - action_reason_total(before, "paged")),
            "action_observations": metric_delta(
                before, after, 'llamacpp:kv_action_observations_total{action="paged"}'),
            "action_cost_seconds": metric_delta(
                before, after, 'llamacpp:kv_action_observation_seconds_total{action="paged"}'),
            "paged_calls": metric_delta(before, after, "llamacpp:paged_decode_calls_total "),
            "paged_fallbacks": metric_delta(
                before, after, "llamacpp:paged_decode_fallbacks_total "),
            "raw": str(directory.relative_to(output).as_posix()),
        })
    return rows


def collect_single_process_rows(
    protocol: dict[str, Any], server: Path, model: Path, output: Path, port: int,
) -> list[dict[str, Any]]:
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=False)
    log_path = raw / "server.log"
    control_file = ROOT / "results/raw/k2-kernel-control.txt"
    slot_save_path = ROOT / "results/raw/k2-slot-state"
    control_file.parent.mkdir(parents=True, exist_ok=True)
    slot_save_path.mkdir(parents=True, exist_ok=True)
    control_file.write_bytes(b"K1")
    command = [
        str(server.resolve()), "-m", str(model.resolve()),
        "--host", "127.0.0.1", "--port", str(port),
        "-c", "512", "-np", "1", "-t", "8", "-ngl", "99",
        "--flash-attn", "on", "--no-warmup", "--metrics", "--slots",
        "--slot-save-path", str(slot_save_path.resolve()),
        "--kv-block-runtime", "--kv-block-size", "16", "--kv-paged-decode",
        "--kv-action-policy", "analytical", "--kv-action-override", "paged", "-lv", "4",
    ]
    environment = cuda_environment({
        "LLAMA_CACHEFLOW_PAGED_KERNEL_CONTROL_FILE": str(control_file.resolve()),
    })
    payload = {
        "prompt": str(protocol["request"]["prompt"]),
        "n_predict": 1,
        "temperature": 0,
        "seed": 20260808,
        "cache_prompt": True,
    }
    base_url = f"http://127.0.0.1:{port}"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            wait_ready(base_url, process, log_path, attempts=120)
            for arm, (pair, order_in_pair, variant) in enumerate(
                randomized_arm_plan(protocol), 1,
            ):
                control_file.write_bytes(VARIANTS[variant][0].encode("ascii"))
                erase_slot(port)
                before = get_text(f"{base_url}/metrics")
                for _ in range(int(protocol["request"]["warm_requests_before_measurement"])):
                    request_json(f"{base_url}/completion", payload)
                started = time.perf_counter_ns()
                status, response = request_json(f"{base_url}/completion", payload)
                elapsed = time.perf_counter_ns() - started
                after = get_text(f"{base_url}/metrics")
                if status != 200 or "error" in response or not response.get("content"):
                    raise AssertionError(f"{variant} controlled request failed: {response}")
                directory = raw / f"arm-{arm:03d}-{variant}"
                directory.mkdir()
                (directory / "response.json").write_text(
                    json.dumps(response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                (directory / "client-timing.json").write_text(json.dumps({
                    "pair": pair,
                    "order_in_pair": order_in_pair,
                    "variant": variant,
                    "client_elapsed_ns": elapsed,
                }, indent=2) + "\n", encoding="utf-8")
                (directory / "metrics-before.prom").write_text(before, encoding="utf-8")
                (directory / "metrics-after.prom").write_text(after, encoding="utf-8")
        finally:
            terminate_process(process)
            control_file.unlink(missing_ok=True)
    return parse_structured_rows(protocol, output)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def expected_summary(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = int(protocol["paired_trials"])
    if protocol.get("execution_scope") == "single_server":
        expected_plan = randomized_arm_plan(protocol)
    else:
        expected_plan = [
            (pair, order, variant)
            for pair in range(1, pairs + 1)
            for order, variant in enumerate(
                ["k1", "k2"] if pair % 2 else ["k2", "k1"], 1)
        ]
    actual_plan = [
        (int(row["pair"]), int(row["order_in_pair"]), str(row["variant"])) for row in rows
    ]
    if actual_plan != expected_plan:
        raise AssertionError("trial order does not match the preregistered arm plan")
    by_pair: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(int(row["pair"]), {})[str(row["variant"])] = row
    if set(by_pair) != set(range(1, pairs + 1)) or any(
        set(pair_rows) != set(VARIANTS) for pair_rows in by_pair.values()
    ):
        raise AssertionError("incomplete K1/K2 pair set")
    if any(pair_rows["k1"]["content"] != pair_rows["k2"]["content"]
           for pair_rows in by_pair.values()):
        raise AssertionError("K1/K2 differential output mismatch")
    k1_client = [by_pair[p]["k1"]["client_elapsed_ms"] for p in sorted(by_pair)]
    k2_client = [by_pair[p]["k2"]["client_elapsed_ms"] for p in sorted(by_pair)]
    k1_prompt = [by_pair[p]["k1"]["prompt_ms"] for p in sorted(by_pair)]
    k2_prompt = [by_pair[p]["k2"]["prompt_ms"] for p in sorted(by_pair)]
    client_effects = [k2 - k1 for k1, k2 in zip(k1_client, k2_client)]
    prompt_effects = [k2 - k1 for k1, k2 in zip(k1_prompt, k2_prompt)]
    k1_summary = summarize(k1_client)
    k2_summary = summarize(k2_client)
    p95_regression = (k2_summary["p95"] / k1_summary["p95"] - 1.0) * 100.0
    limit = float(protocol["acceptance"]["p95_maximum_regression_percent"])
    client_median_not_slower = summarize(client_effects)["median"] <= 0.0
    prompt_median_not_slower = summarize(prompt_effects)["median"] <= 0.0
    acceptance_metric = protocol["acceptance"].get(
        "paired_median_metric", "client_elapsed_ms")
    if acceptance_metric == "client_elapsed_ms":
        paired_median_passed = client_median_not_slower
    elif acceptance_metric == "server_prompt_ms":
        paired_median_passed = prompt_median_not_slower
    else:
        raise ValueError(f"unsupported paired_median_metric: {acceptance_metric}")
    result = {
        "paired_trials": pairs,
        "k1_client_elapsed_ms": k1_summary,
        "k2_client_elapsed_ms": k2_summary,
        "paired_k2_minus_k1_client_ms": summarize(client_effects),
        "k1_prompt_ms": summarize(k1_prompt),
        "k2_prompt_ms": summarize(k2_prompt),
        "paired_k2_minus_k1_prompt_ms": summarize(prompt_effects),
        "paired_client_median_bootstrap_95_ms": bootstrap_median_interval(
            client_effects, int(protocol["random_seed"]), resamples=10000,
        ),
        "p95_regression_percent": p95_regression,
        "promotion_limit_percent": limit,
        "promotion_passed": p95_regression <= limit and paired_median_passed,
        "correctness_passed": True,
        "k1_paged_graph_entries": int(sum(r["paged_calls"] for r in rows if r["variant"] == "k1")),
        "k2_paged_graph_entries": int(sum(r["paged_calls"] for r in rows if r["variant"] == "k2")),
        "paged_fallbacks": int(sum(r["paged_fallbacks"] for r in rows)),
    }
    if "paired_median_metric" in protocol["acceptance"]:
        result["paired_median_acceptance_metric"] = acceptance_metric
        result["paired_median_acceptance_passed"] = paired_median_passed
    return result


def profile_variant(
    protocol: dict[str, Any], server: Path, model: Path, output: Path,
    port: int, variant: str,
) -> dict[str, Any]:
    selector, kernel_pattern = VARIANTS[variant]
    nsys = discover_nsys()
    profile_dir = output / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    raw_prefix = ROOT / f"results/raw/k2-production-{variant}-nsys"
    launcher = [
        str(nsys), "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
        "--duration=6", "--kill=true", "--force-overwrite=true", f"--output={raw_prefix}",
    ]
    result, _, log = run_mode(
        server, model, port, "paged", f"formal-{variant}-nsys",
        launcher=launcher, launcher_manages_lifetime=True,
        prompt=str(protocol["request"]["prompt"]),
        environment_overrides={"LLAMA_CACHEFLOW_PAGED_KERNEL": selector},
        warm_requests=int(protocol["request"].get("warm_requests_before_measurement", 1)),
    )
    report = profile_dir / f"{variant}.nsys-rep"
    sqlite = profile_dir / f"{variant}.sqlite"
    server_log = profile_dir / f"{variant}-server.log"
    shutil.copy2(raw_prefix.with_suffix(".nsys-rep"), report)
    shutil.copy2(log, server_log)
    exported = subprocess.run(
        [str(nsys), "export", "--type=sqlite", "--force-overwrite=true",
         f"--output={sqlite}", str(report)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if exported.returncode:
        raise RuntimeError(exported.stderr)
    mechanism = parse_nsys_sqlite(sqlite, kernel_patterns=(kernel_pattern,))
    if mechanism["kernel_launches"] != 24:
        raise AssertionError(f"{variant} expected 24 layer launches: {mechanism}")
    mechanism.update({
        "variant": variant,
        "selector": selector,
        "kernel_pattern": kernel_pattern,
        "request_content": result["content"],
        "timing_role": (
            "mechanism only; unprofiled server prompt is primary and client P95 is the guardrail"
            if protocol["acceptance"].get("paired_median_metric") == "server_prompt_ms"
            else "mechanism only; no-profiler paired client latency is primary"
        ),
    })
    return mechanism


def render_report(summary: dict[str, Any], mechanisms: dict[str, Any]) -> str:
    effect = summary["paired_k2_minus_k1_client_ms"]
    prompt = summary["paired_k2_minus_k1_prompt_ms"]
    gate_metric = summary.get("paired_median_acceptance_metric", "client_elapsed_ms")
    gate_label = "服务内部 prompt" if gate_metric == "server_prompt_ms" else "客户端"
    return "\n".join([
        "# K2 相对生产 K1 的正式晋级实验",
        "",
        f"- 配对服务样本：{summary['paired_trials']} 组；每个 arm 均进入真实 Paged 图且零 fallback。",
        f"- 客户端配对中位差（K2-K1）：{effect['median']:.3f} ms；bootstrap 95% 区间 "
        f"[{summary['paired_client_median_bootstrap_95_ms'][0]:.3f}, "
        f"{summary['paired_client_median_bootstrap_95_ms'][1]:.3f}] ms。",
        f"- 客户端 P95：K1 {summary['k1_client_elapsed_ms']['p95']:.3f} ms，"
        f"K2 {summary['k2_client_elapsed_ms']['p95']:.3f} ms，变化 "
        f"{summary['p95_regression_percent']:.2f}%。",
        f"- 服务内部 prompt 配对中位差：{prompt['median']:.3f} ms。",
        f"- NSYS：K1 {mechanisms['k1']['kernel_launches']} 次 / "
        f"{mechanisms['k1']['kernel_duration_ms']:.3f} ms；K2 "
        f"{mechanisms['k2']['kernel_launches']} 次 / "
        f"{mechanisms['k2']['kernel_duration_ms']:.3f} ms（仅机制证据）。",
        f"- 生产晋级：{'通过' if summary['promotion_passed'] else '未通过'}。门槛为 "
        f"客户端 P95 回退不超过 {summary['promotion_limit_percent']:.1f}% 且"
        f"{gate_label}配对中位数不慢。",
        "",
        "该结论只回答 K2 能否替代同一 Paged 路径中的 K1；不把它改写成 Paged 已优于 Direct。",
        "",
    ])


def validate_artifact(protocol_path: Path, output: Path) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    rows = json.loads((output / "trials.json").read_text(encoding="utf-8"))
    if protocol.get("raw_schema") == "structured-arm-v1":
        reconstructed = parse_structured_rows(protocol, output)
        if rows != reconstructed:
            raise AssertionError("trials are not exactly reconstructed from structured raw arms")
    expected = expected_summary(protocol, rows)
    for key, value in expected.items():
        if summary.get(key) != value:
            raise AssertionError(f"summary field is not derived from trials: {key}")
    if summary["protocol_sha256"] != sha256(protocol_path):
        raise AssertionError("protocol hash mismatch")
    if not summary["worktree_clean_before_run"]:
        raise AssertionError("formal artifact did not start from a clean worktree")
    expected_entries = int(protocol["request"].get("warm_requests_before_measurement", 1))
    for row in rows:
        if protocol.get("raw_schema") != "structured-arm-v1":
            log = output / row["log"]
            if row["log_sha256"] != sha256(log):
                raise AssertionError("raw service log hash mismatch")
        if row["paged_calls"] != expected_entries or row["paged_fallbacks"] != 0:
            raise AssertionError("trial did not execute the preregistered Paged warm/measured entries")
        if row["action_observations"] != expected_entries or row["action_decisions"] != expected_entries:
            raise AssertionError("trial lacks the preregistered complete production observations")
    mechanisms = json.loads((output / "mechanisms.json").read_text(encoding="utf-8"))
    for variant, (selector, pattern) in VARIANTS.items():
        reparsed = parse_nsys_sqlite(
            output / f"profile/{variant}.sqlite", kernel_patterns=(pattern,),
        )
        for key, value in reparsed.items():
            if mechanisms[variant].get(key) != value:
                raise AssertionError(f"{variant} mechanism is not derived from SQLite: {key}")
        if mechanisms[variant]["variant"] != variant or \
                mechanisms[variant]["selector"] != selector or \
                mechanisms[variant]["kernel_pattern"] != pattern:
            raise AssertionError(f"{variant} mechanism metadata mismatch")
    if protocol.get("raw_schema") == "structured-arm-v1" and \
            (output / "report.md").read_text(encoding="utf-8") != render_report(summary, mechanisms):
        raise AssertionError("report is not rendered from the validated summary and mechanisms")
    if "artifact_binding" in protocol:
        binding = protocol["artifact_binding"]
        for key in ("server_sha256", "model_sha256", "vendor_revision", "build_command"):
            if summary.get(key) != binding[key]:
                raise AssertionError(f"formal artifact binding mismatch: {key}")
        if summary.get("device") != protocol["device"] or not summary.get("vendor_clean_before_run"):
            raise AssertionError("formal artifact device/vendor provenance mismatch")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    actual = {
        str(path.relative_to(output).as_posix()): sha256(path)
        for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"
    }
    if manifest["files"] != actual:
        raise AssertionError("manifest does not bind the exact artifact tree")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--server", type=Path, default=ROOT / "build/patched-cuda-ninja3/bin/llama-server.exe")
    parser.add_argument("--model", type=Path, default=ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port-base", type=int, default=8240)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_artifact(args.protocol, args.output)
        print(json.dumps({"validated": str(args.output)}, ensure_ascii=False))
        return

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    device = device_identity()
    if device != protocol["device"]:
        raise RuntimeError("device differs from preregistration")
    clean = not bool(git_output("status", "--porcelain"))
    vendor = ROOT / "vendor/llama.cpp"
    vendor_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=vendor, text=True).strip()
    vendor_clean = not bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=vendor, text=True).strip())
    if not clean or not vendor_clean:
        raise RuntimeError("formal experiment requires clean outer and vendor worktrees")
    binding = protocol.get("artifact_binding")
    if binding:
        actual_binding = {
            "server_sha256": sha256(args.server),
            "model_sha256": sha256(args.model),
            "vendor_revision": vendor_revision,
            "build_command": binding["build_command"],
        }
        if actual_binding != binding:
            raise RuntimeError(f"binary/model/vendor binding differs from preregistration: {actual_binding}")
    if args.output.exists():
        raise FileExistsError(f"formal output already exists: {args.output}")
    if protocol.get("execution_scope") == "single_server":
        rows = collect_single_process_rows(
            protocol, args.server, args.model, args.output, args.port_base)
    else:
        raw = args.output / "raw"
        raw.mkdir(parents=True)
        rows = []
        for pair in range(1, int(protocol["paired_trials"]) + 1):
            order = ["k1", "k2"] if pair % 2 else ["k2", "k1"]
            for order_in_pair, variant in enumerate(order, 1):
                selector, _ = VARIANTS[variant]
                result, metrics, log = run_mode(
                    args.server, args.model,
                    args.port_base + (pair - 1) * 2 + order_in_pair - 1,
                    "paged", f"k2-pair-{pair:02d}-{variant}",
                    prompt=str(protocol["request"]["prompt"]),
                    environment_overrides={"LLAMA_CACHEFLOW_PAGED_KERNEL": selector},
                    warm_requests=int(protocol["request"].get("warm_requests_before_measurement", 1)),
                )
                artifact_log = raw / f"{variant}-pair-{pair:02d}-{order_in_pair}.log"
                shutil.copy2(log, artifact_log)
                rows.append({
                    "pair": pair, "order_in_pair": order_in_pair, "variant": variant,
                    "content": result["content"],
                    "client_elapsed_ms": float(result["_client_elapsed_ms"]),
                    "prompt_ms": float(result["timings"]["prompt_ms"]),
                    "action_decisions": metric(metrics, 'llamacpp:kv_action_decisions_total{action="paged"}'),
                    "action_reason_decisions": action_reason_total(metrics, "paged"),
                    "action_observations": metric(metrics, 'llamacpp:kv_action_observations_total{action="paged"}'),
                    "action_cost_seconds": metric(metrics, 'llamacpp:kv_action_observation_seconds_total{action="paged"}'),
                    "paged_calls": metric(metrics, "llamacpp:paged_decode_calls_total "),
                    "paged_fallbacks": metric(metrics, "llamacpp:paged_decode_fallbacks_total "),
                    "log": str(artifact_log.relative_to(args.output).as_posix()),
                    "log_sha256": sha256(artifact_log),
                })

    derived = expected_summary(protocol, rows)
    mechanisms = {
        variant: profile_variant(
            protocol, args.server, args.model, args.output,
            args.port_base + int(protocol["paired_trials"]) * 2 + index, variant,
        ) for index, variant in enumerate(VARIANTS)
    }
    summary = {
        "schema_version": 1,
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256(args.protocol),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_output("rev-parse", "HEAD"),
        "vendor_revision": vendor_revision,
        "worktree_clean_before_run": clean,
        "vendor_clean_before_run": vendor_clean,
        "device": device,
        **(binding or {}),
        **derived,
    }
    (args.output / "trials.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (args.output / "mechanisms.json").write_text(json.dumps(mechanisms, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render_report(summary, mechanisms), encoding="utf-8")
    files = {
        str(path.relative_to(args.output).as_posix()): sha256(path)
        for path in args.output.rglob("*") if path.is_file() and path.name != "manifest.json"
    }
    (args.output / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": files}, indent=2) + "\n", encoding="utf-8",
    )
    validate_artifact(args.protocol, args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
