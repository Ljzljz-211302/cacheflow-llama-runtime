from __future__ import annotations

import argparse
import json
import shutil
import subprocess
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
from run_production_paged_journey import metric, run_mode

import sys

sys.path.insert(0, str(ROOT / "src"))
from llama_lab.cuda_profile_evidence import parse_nsys_sqlite  # noqa: E402


VARIANTS = {
    "k1": ("K1", "cacheflow_paged_decode_fattn_k1"),
    "k2": ("K2", "cacheflow_paged_decode_fattn_k2_t2"),
}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def expected_summary(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = int(protocol["paired_trials"])
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
        log = output / row["log"]
        if row["log_sha256"] != sha256(log):
            raise AssertionError("raw service log hash mismatch")
        if row["paged_calls"] != expected_entries or row["paged_fallbacks"] != 0:
            raise AssertionError("trial did not execute the preregistered Paged warm/measured entries")
        if row["action_observations"] != expected_entries or row["action_decisions"] != expected_entries:
            raise AssertionError("trial lacks the preregistered complete production observations")
    mechanisms = json.loads((output / "mechanisms.json").read_text(encoding="utf-8"))
    for variant, (_, pattern) in VARIANTS.items():
        reparsed = parse_nsys_sqlite(
            output / f"profile/{variant}.sqlite", kernel_patterns=(pattern,),
        )
        for key, value in reparsed.items():
            if mechanisms[variant].get(key) != value:
                raise AssertionError(f"{variant} mechanism is not derived from SQLite: {key}")
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
    if device_identity()["name"] != protocol["device"]["name"]:
        raise RuntimeError("device differs from preregistration")
    clean = not bool(git_output("status", "--porcelain"))
    raw = args.output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
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
            row = {
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
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

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
        "vendor_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT / "vendor/llama.cpp", text=True,
        ).strip(),
        "worktree_clean_before_run": clean,
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
