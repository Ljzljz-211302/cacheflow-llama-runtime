from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from llama_lab.research_protocol import paired_bootstrap_summary


SOURCE_PATHS = {
    "upstream": "results/upstream-compatibility.json",
    "application": "results/user-application-journey.json",
    "browser_qa": "results/user-application-browser-qa.json",
    "production_launcher": "scripts/start_production.ps1",
    "remap": "results/research/h1-vector-remap-v1.0.0/manifest.json",
    "remap_trials": "results/research/h1-vector-remap-v1.0.0/trials.jsonl",
    "policy": "results/research/h4-kv-action-v1.6.0/report.json",
    "policy_report": "results/research/h4-kv-action-v1.6.0/report.md",
    "paged_direct": "results/research/h7-production-paged-v1.1.0/summary.json",
    "paged_direct_report": "results/research/h7-production-paged-v1.1.0/report.md",
    "k2": "results/research/h8-k2-production-v2.10.0/summary.json",
    "k2_mechanism": "results/research/h8-k2-production-v2.10.0/mechanisms.json",
    "k2_trials": "results/research/h8-k2-production-v2.10.0/trials.json",
    "k2_report": "results/research/h8-k2-production-v2.10.0/report.md",
    "k2_chart": "results/research/h8-k2-production-v2.10.0/k2-production-comparison.svg",
    "objective_paged_protocol": "config/production_paged_objective_protocol_v2.json",
    "objective_paged_corpus": "config/paged_objective_workloads_v2.json",
    "objective_paged_manifest": "results/research/h9-objective-paged-v2.0.0/manifest.json",
    "objective_paged": "results/research/h9-objective-paged-v2.0.0/summary.json",
    "objective_paged_report": "results/research/h9-objective-paged-v2.0.0/report.md",
    "objective_paged_chart": "results/research/h9-objective-paged-v2.0.0/comparison.svg",
    "long_paged_protocol": "config/production_paged_objective_protocol_v7.json",
    "long_paged_corpus": "config/paged_objective_workloads_v4.json",
    "long_paged_manifest": "results/research/h13-balanced-adaptive-gqa-paged-v7.0.0/manifest.json",
    "long_paged": "results/research/h13-balanced-adaptive-gqa-paged-v7.0.0/summary.json",
    "long_paged_report": "results/research/h13-balanced-adaptive-gqa-paged-v7.0.0/report.md",
    "long_paged_chart": "results/research/h13-balanced-adaptive-gqa-paged-v7.0.0/comparison.svg",
    "batched_paged_protocol": "config/batched_paged_performance_protocol_v5.json",
    "batched_paged_corpus": "config/batched_paged_workloads_v1.json",
    "batched_paged_manifest": "results/research/h19-production-batched-paged-v5.0.0/manifest.json",
    "batched_paged_trials": "results/research/h19-production-batched-paged-v5.0.0/trials.json",
    "batched_paged": "results/research/h19-production-batched-paged-v5.0.0/summary.json",
    "batched_paged_report": "results/research/h19-production-batched-paged-v5.0.0/report.md",
    "batched_paged_chart": "results/research/h19-production-batched-paged-v5.0.0/comparison.svg",
    "hybrid_paged_protocol": "config/batched_paged_hybrid_protocol_v6.json",
    "hybrid_paged_corpus": "config/batched_paged_workloads_v2.json",
    "hybrid_paged_manifest": "results/research/h20-paged-hybrid-batch8-v6.1.0/manifest.json",
    "hybrid_paged_trials": "results/research/h20-paged-hybrid-batch8-v6.1.0/trials.json",
    "hybrid_paged": "results/research/h20-paged-hybrid-batch8-v6.1.0/summary.json",
    "hybrid_paged_report": "results/research/h20-paged-hybrid-batch8-v6.1.0/report.md",
    "hybrid_paged_chart": "results/research/h20-paged-hybrid-batch8-v6.1.0/comparison.svg",
    "hybrid_diagnostic_manifest": "results/diagnostic/h20-batch8-context1024-nsys/manifest.json",
    "hybrid_diagnostic_report": "results/diagnostic/h20-batch8-context1024-nsys/README.md",
}


def _load(root: Path, relative: str) -> dict[str, Any]:
    return json.loads((root / relative).read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_diagnostic_manifest(root: Path, relative: str) -> None:
    manifest_path = root / relative
    manifest = _load(root, relative)
    for name, expected in manifest["files"].items():
        path = manifest_path.parent / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"diagnostic evidence hash differs: {name}")


def validate_h1_records(
    manifest: dict[str, Any], records: list[dict[str, Any]], protocol: dict[str, Any]
) -> None:
    workload = protocol["workloads"]["h1_vector_remap"]
    expected_blocks = set(map(int, workload["block_counts"]))
    expected_methods = set(workload["methods"])
    expected_seed = int(protocol["random_seed"])
    expected_pairs = int(workload["pairs_per_block_count"])
    expected_warmups = int(workload["warmup_pairs_per_block_count"])
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in records:
        if row.get("claim_id") != "H1-vector-remap" or row.get("experiment_id") != "h1-vector-remap":
            raise ValueError("H1 trial identity differs")
        if row.get("phase") not in {"warmup", "confirmatory"}:
            raise ValueError("H1 trial phase differs")
        if int(row.get("blocks", -1)) not in expected_blocks:
            raise ValueError("H1 block coverage differs")
        if row.get("method") not in expected_methods or int(row.get("random_seed", -1)) != expected_seed:
            raise ValueError("H1 method or seed differs")
        if not row.get("valid") or not row.get("correctness_passed") or row.get("invalid_reason") is not None:
            raise ValueError("H1 correctness/validity evidence failed")
        if int(row.get("bytes", 0)) <= 0 or int(row.get("order_in_pair", -1)) not in {0, 1}:
            raise ValueError("H1 byte count or pair order differs")
        key = (row["phase"], int(row["blocks"]), row["pair_id"])
        grouped.setdefault(key, []).append(row)
    for block in expected_blocks:
        warmups = [rows for (phase, b, _), rows in grouped.items() if phase == "warmup" and b == block]
        confirms = [rows for (phase, b, _), rows in grouped.items() if phase == "confirmatory" and b == block]
        if len(warmups) != expected_warmups or len(confirms) != expected_pairs:
            raise ValueError("H1 warmup or confirmatory pair coverage differs")
        for pair_rows in warmups + confirms:
            if len(pair_rows) != 2 or {row["method"] for row in pair_rows} != expected_methods:
                raise ValueError("H1 contains an incomplete pair")
            if {int(row["order_in_pair"]) for row in pair_rows} != {0, 1}:
                raise ValueError("H1 pair order is not balanced within pair")
            if len({int(row["bytes"]) for row in pair_rows}) != 1:
                raise ValueError("H1 pair byte counts differ")

    pairs_by_blocks: dict[int, dict[str, dict[str, float]]] = {}
    for row in records:
        if row["phase"] == "confirmatory":
            pairs_by_blocks.setdefault(int(row["blocks"]), {}).setdefault(
                row["pair_id"], {}
            )[row["method"]] = float(row["timing_ms"]["synchronized_kernel_ms"])
    stats = protocol["statistics"]
    expected = []
    for blocks, trial_map in sorted(pairs_by_blocks.items()):
        pairs = [(methods["scalar_gather_scatter"], methods["vectorized_gather_scatter"])
                 for methods in trial_map.values()]
        summary = paired_bootstrap_summary(
            pairs,
            confidence_level=float(stats["confidence_level"]),
            resamples=int(stats["bootstrap_resamples"]),
            seed=expected_seed + blocks,
        )
        summary["blocks"] = blocks
        expected.append(summary)
    if expected != manifest["paired_summaries"]:
        raise ValueError("H1 summaries differ from raw trials")
    gates = stats["gates"]
    estimates = [float(item["median_improvement_percent"]) for item in expected]
    non_regression = len(expected) == len(expected_blocks) and all(
        float(item["ci_lower_percent"]) >= -float(gates["maximum_regression_percent"])
        for item in expected
    )
    material_effect = any(
        float(item["ci_lower_percent"]) >= float(gates["material_improvement_percent"])
        for item in expected
    )
    trend = all(left >= right for left, right in zip(estimates, estimates[1:]))
    correctness = all(bool(row["correctness_passed"]) for row in records)
    recomputed_acceptance = {
        "correctness": correctness,
        "non_regression": non_regression,
        "material_effect": material_effect,
        "non_increasing_trend": trend,
        "passed": correctness and non_regression and material_effect and trend,
    }
    if manifest["acceptance"] != recomputed_acceptance:
        raise ValueError("H1 acceptance differs from raw trials and protocol gates")
    if not manifest["protocol_compliant"] or manifest["violations"] or not recomputed_acceptance["passed"]:
        raise ValueError("H1 did not pass its preregistered protocol")


def _validate_h1(root: Path, manifest: dict[str, Any]) -> None:
    protocol_path = root / "config/research_protocol.json"
    protocol = _load(root, "config/research_protocol.json")
    if manifest["protocol_sha256"] != _sha256(protocol_path):
        raise ValueError("H1 protocol hash differs")
    trials_path = root / SOURCE_PATHS["remap_trials"]
    records = [json.loads(line) for line in trials_path.read_text(encoding="utf-8").splitlines()]
    if len(records) != manifest["raw_trial_count"]:
        raise ValueError("H1 raw trial count differs from manifest")
    validate_h1_records(manifest, records, protocol)


def build_final_outcome(root: Path) -> dict[str, Any]:
    json_sources = {
        "upstream", "application", "browser_qa", "remap", "policy", "paged_direct",
        "k2", "k2_mechanism", "k2_trials", "objective_paged", "long_paged",
        "batched_paged", "batched_paged_trials", "hybrid_paged", "hybrid_paged_trials",
        "hybrid_diagnostic_manifest",
    }
    source = {name: _load(root, SOURCE_PATHS[name]) for name in json_sources}
    upstream = source["upstream"]
    application = source["application"]
    browser_qa = source["browser_qa"]
    remap = source["remap"]
    _validate_h1(root, remap)
    policy = source["policy"]
    paged = source["paged_direct"]
    k2 = source["k2"]
    mechanism = source["k2_mechanism"]
    k2_trials = source["k2_trials"]
    objective_paged = source["objective_paged"]
    long_paged = source["long_paged"]
    batched_paged = source["batched_paged"]
    batched_paged_trials = source["batched_paged_trials"]
    hybrid_paged = source["hybrid_paged"]
    hybrid_paged_trials = source["hybrid_paged_trials"]
    _validate_diagnostic_manifest(root, SOURCE_PATHS["hybrid_diagnostic_manifest"])
    production_launcher = (root / SOURCE_PATHS["production_launcher"]).read_text(encoding="utf-8")
    default_paged_enabled = '"--kv-paged-decode"' in production_launcher
    d3 = policy["analysis"]["models"]["D3"]
    measured_response_counts = {
        variant: sum(
            len(row["request_client_elapsed_ms"])
            for row in k2_trials
            if row["variant"] == variant
        )
        for variant in ("k1", "k2")
    }
    independent_research_complete = bool(
        remap["protocol_compliant"]
        and remap["acceptance"]["passed"]
        and policy["overhead"]["passed"]
        and paged["correctness_passed"]
        and not paged["promotion_passed"]
        and k2["correctness_passed"]
        and k2["promotion_passed"]
        and k2["kernel_launch_count_parity"]
        and mechanism["k1"]["kernel_launches"] == mechanism["k2"]["kernel_launches"]
        and measured_response_counts["k1"] == measured_response_counts["k2"]
        and objective_paged["page_coverage_passed"]
        and not objective_paged["promotion_passed"]
        and long_paged["page_coverage_passed"]
        and max(row["actual_context_tokens"] for row in long_paged["per_workload"].values()) == 2048
        and not long_paged["promotion_passed"]
        and len(batched_paged_trials) == batched_paged["observations"] == 144
        and not batched_paged["promotion_passed"]
        and len(hybrid_paged_trials) == hybrid_paged["observations"] == 36
        and hybrid_paged["correctness"]["passed"]
        and hybrid_paged["promotion_passed"]
        and hybrid_paged["execution_evidence"]["mode"] == "contiguous_fastpath"
        and hybrid_paged["execution_evidence"]["custom_paged_graph_calls"] == 0
        and hybrid_paged["execution_evidence"]["custom_cuda_dispatches"] == 0
    )

    outcome = {
        "schema_version": 1,
        "outcome_name": "CacheFlow Runtime final bounded outcome",
        "disposition": {
            "application_project_complete": bool(application["passed"]),
            "independent_research_project_complete": independent_research_complete,
            "peer_reviewed_publication": False,
            "default_production_paged_enabled": default_paged_enabled,
            "summary": (
                "A reproducible single-machine application and bounded research artifact; "
                "not a peer-reviewed paper and not a universal Paged-over-Direct speedup claim."
            ),
        },
        "upstream_boundary": {
            "pinned_revision": upstream["pinned_revision"],
            "same_toolchain": upstream["toolchain"],
            "upstream_mode_exact_output_cases": upstream["exact_matches"],
            "all_recorded_cases_match": upstream["exact_matches"] == len(upstream["cases"]),
        },
        "application_result": {
            "passed": application["passed"],
            "ui_rendered": application["ui_rendered"],
            "browser_sse_contract": application["browser_sse_contract"],
            "interactive_browser_qa_passed": bool(
                browser_qa["stream_completed"] and browser_qa["session_visible_in_sidebar"]
            ),
            "restart_preserved_messages": application["session_restart_preserved_messages"],
            "native_cancel_propagated": application["native_cancel_propagated"],
            "backpressure_rejected": application["backpressure_rejected"],
            "concurrent_users": application["concurrent_users"],
            "knowledge_documents": application["knowledge_documents"],
            "knowledge_chunks": application["knowledge_chunks"],
            "infra_metrics": application["infra_metrics"],
            "scope": "single-machine or trusted-LAN deployment; no external adoption claim",
        },
        "research_results": {
            "vectorized_remap": {
                "protocol_compliant": remap["protocol_compliant"],
                "acceptance_passed": remap["acceptance"]["passed"],
                "pairs_per_case": [item["pairs"] for item in remap["paired_summaries"]],
                "median_improvement_percent_by_blocks": {
                    str(item["blocks"]): item["median_improvement_percent"]
                    for item in remap["paired_summaries"]
                },
                "boundary": "CUDA KV remap microbenchmark; not request-level latency",
            },
            "offline_action_policy_d3": {
                "evaluation_decisions": d3["decisions"],
                "switches_vs_h0": d3["switches_vs_h0"],
                "cumulative_regret_ms": d3["cumulative_regret_ms"],
                "harmful_decisions": d3["harmful_decisions"],
                "harmful_rate": d3["harmful_rate"],
                "total_gain_ms": d3["total_gain_ms"],
                "total_harm_ms": d3["total_harm_ms"],
                "chooser_p99_us": policy["overhead"]["p99_choose_microseconds"],
                "hot_loop_allocations": policy["overhead"]["hot_loop_allocations"],
                "boundary": "offline held-out replay; requires monitored same-process canary before enablement",
            },
            "paged_vs_direct": {
                "correctness_passed": paged["correctness_passed"],
                "promotion_passed": paged["promotion_passed"],
                "paired_trials": paged["paired_trials"],
                "direct_p95_ms": paged["direct_client_elapsed_ms"]["p95"],
                "paged_p95_ms": paged["paged_client_elapsed_ms"]["p95"],
                "p95_regression_percent": paged["p95_regression_percent"],
                "decision": "negative performance result; Paged remains opt-in",
            },
            "k2_vs_k1": {
                "correctness_passed": k2["correctness_passed"],
                "promotion_passed": k2["promotion_passed"],
                "paired_trials": k2["paired_trials"],
                "measured_responses_by_variant": measured_response_counts,
                "paged_graph_entries_per_variant": k2["k1_paged_graph_entries"],
                "fallbacks": k2["paged_fallbacks"],
                "client_median_regression_percent": k2["client_median_regression_percent"],
                "client_median_regression_upper_95_percent": k2["client_median_regression_bootstrap_95_percent"][1],
                "client_p95_regression_percent": k2["p95_regression_percent"],
                "kernel_launches_by_variant": {
                    variant: mechanism[variant]["kernel_launches"]
                    for variant in ("k1", "k2")
                },
                "k1_kernel_duration_ms": mechanism["k1"]["kernel_duration_ms"],
                "k2_kernel_duration_ms": mechanism["k2"]["kernel_duration_ms"],
                "kernel_duration_reduction_percent": k2["kernel_duration_reduction_percent"],
                "scope": "Qwen2.5-0.5B D64/GQA7/page16/context17 repeated cached decode",
            },
            "objective_paged_vs_direct": {
                "protocol_version": objective_paged["protocol_version"],
                "matched_process_blocks": objective_paged["matched_process_blocks"],
                "workload_count": objective_paged["workload_count"],
                "observations": objective_paged["observations"],
                "actual_context_tokens": {
                    workload_id: result["actual_context_tokens"]
                    for workload_id, result in objective_paged["per_workload"].items()
                },
                "all_workloads_cross_page": objective_paged["page_coverage_passed"],
                "median_regression_percent": objective_paged["primary_matched_block_regression_percent"]["median"],
                "median_regression_bootstrap_95_percent": objective_paged["primary_block_cluster_bootstrap_95_percent"],
                "block_workload_regression_p95_percent": objective_paged["primary_matched_block_regression_percent"]["p95"],
                "p95_limit_percent": objective_paged["primary_p95_limit_percent"],
                "worst_workload_median_regression_percent": objective_paged["worst_workload_median_regression_percent"],
                "worst_workload_limit_percent": objective_paged["worst_workload_limit_percent"],
                "promotion_passed": objective_paged["promotion_passed"],
                "decision": "negative production result: median improves, but tail and worst-workload gates fail",
            },
            "long_context_paged_vs_direct": {
                "protocol_version": long_paged["protocol_version"],
                "matched_process_blocks": long_paged["matched_process_blocks"],
                "workload_count": long_paged["workload_count"],
                "observations": long_paged["observations"],
                "minimum_context_tokens": min(
                    row["actual_context_tokens"] for row in long_paged["per_workload"].values()
                ),
                "maximum_context_tokens": max(
                    row["actual_context_tokens"] for row in long_paged["per_workload"].values()
                ),
                "primary_minimum_context_tokens": long_paged["primary_minimum_context_tokens"],
                "primary_metric": long_paged["primary_timing_field"],
                "median_regression_percent": long_paged["primary_matched_block_regression_percent"]["median"],
                "bootstrap_95_percent": long_paged["primary_block_cluster_bootstrap_95_percent"],
                "p95_regression_percent": long_paged["primary_matched_block_regression_percent"]["p95"],
                "worst_workload_median_regression_percent": long_paged["worst_workload_median_regression_percent"],
                "regression_by_context_tokens": long_paged["regression_by_context_tokens"],
                "promotion_passed": long_paged["promotion_passed"],
                "decision": "K4 reduced the long-context regression but missed the upper-CI gate; Paged remains opt-in",
            },
            "batched_paged_vs_direct": {
                "protocol_version": batched_paged["protocol_version"],
                "observations": batched_paged["observations"],
                "primary_batch_size": batched_paged["primary_batch_size"],
                "primary_throughput_gain_percent": batched_paged["primary_throughput_gain_percent"]["median"],
                "primary_throughput_gain_bootstrap_95_percent": batched_paged["primary_block_cluster_bootstrap_95_percent"],
                "primary_p95_wave_latency_regression_percent": batched_paged["primary_p95_wave_latency_regression_percent"],
                "worst_cell_median_wave_latency_regression_percent": batched_paged["worst_cell_median_wave_latency_regression_percent"],
                "throughput_by_batch": batched_paged["throughput_by_batch"],
                "output_token_matches": batched_paged["correctness"]["output_token_matches"],
                "output_token_comparisons": batched_paged["correctness"]["output_token_comparisons"],
                "probability_rows_compared": batched_paged["correctness"]["probability_rows_compared"],
                "incomplete_probability_rows": batched_paged["correctness"]["incomplete_probability_rows"],
                "correctness_passed": batched_paged["correctness"]["passed"],
                "primary_cuda_execution_verified": all(
                    row["paged_calls"] == 4.0
                    and row["paged_sequences"] == 32.0
                    and row["paged_fallbacks"] == 0.0
                    and row["cuda_dispatches"] == 96.0
                    and row["cuda_sequences"] == 768.0
                    for row in batched_paged_trials
                    if row["action"] == "paged" and row["batch_size"] == 8
                ),
                "promotion_passed": batched_paged["promotion_passed"],
                "decision": "negative batch-8 result: actual CUDA batching is verified, but throughput, latency, and correctness gates fail",
            },
            "hybrid_paged_vs_direct": {
                "protocol_version": hybrid_paged["protocol_version"],
                "observations": hybrid_paged["observations"],
                "primary_batch_size": hybrid_paged["primary_batch_size"],
                "primary_throughput_gain_percent": hybrid_paged["primary_throughput_gain_percent"]["median"],
                "primary_throughput_gain_bootstrap_95_percent": hybrid_paged["primary_block_cluster_bootstrap_95_percent"],
                "direct_p95_wave_latency_ms": hybrid_paged["primary_direct_wave_latency_p95_ms"],
                "paged_p95_wave_latency_ms": hybrid_paged["primary_paged_wave_latency_p95_ms"],
                "primary_p95_wave_latency_regression_percent": hybrid_paged["primary_p95_wave_latency_regression_percent"],
                "worst_cell_median_wave_latency_regression_percent": hybrid_paged["worst_cell_median_wave_latency_regression_percent"],
                "output_token_matches": hybrid_paged["correctness"]["output_token_matches"],
                "output_token_comparisons": hybrid_paged["correctness"]["output_token_comparisons"],
                "minimum_top64_overlap": hybrid_paged["correctness"]["minimum_top64_overlap"],
                "maximum_common_logprob_error": hybrid_paged["correctness"]["maximum_common_logprob_error"],
                "execution_evidence": hybrid_paged["execution_evidence"],
                "correctness_passed": hybrid_paged["correctness"]["passed"],
                "promotion_passed": hybrid_paged["promotion_passed"],
                "decision": "positive bounded result: contiguous physical layouts reuse upstream attention; fragmented layouts retain custom K4",
            },
        },
        "claim_boundaries": [
            "K2 is qualified only against K1 inside the restricted Paged path.",
            "Forced custom Paged K4 did not beat Direct under H19; H20 instead qualifies a layout-aware hybrid route on contiguous batch-8 workloads.",
            "Microbenchmark improvements do not imply TTFT or throughput improvements.",
            "The application journey proves an exercised local workflow, not external adoption.",
            "The artifact is an independent research project, not a peer-reviewed publication.",
            "The objective prompt matrix supersedes single-prompt interpretation and keeps Paged disabled because the block-workload regression-tail and worst-workload gates fail; its independent-process blocks are not shared-hot-state Trial Pairs.",
            "The source-bound 64-2048 token matrix found no Paged-over-Direct crossover; Direct and Paged share the same allocator in this implementation, so it cannot claim a memory-fragmentation advantage from this A/B.",
            "Native batch sizes 1/2/4/8 are supported and device-counted, but the formal batch-8 result is negative and does not establish a throughput advantage over Direct.",
            "H20 establishes bounded noninferiority, not strict superiority: contiguous layouts take upstream attention, while fragmented-layout K4 remains a correctness-preserving but unpromoted fallback.",
        ],
        "evidence": {
            name: {"path": path, "sha256": _sha256(root / path)}
            for name, path in SOURCE_PATHS.items()
        },
    }
    validate_final_outcome(outcome)
    return outcome


def validate_final_outcome(outcome: dict[str, Any]) -> None:
    disposition = outcome["disposition"]
    if not disposition["application_project_complete"]:
        raise ValueError("application journey has not passed")
    if not outcome["application_result"]["interactive_browser_qa_passed"]:
        raise ValueError("interactive browser QA has not passed")
    if not disposition["independent_research_project_complete"]:
        raise ValueError("independent research evidence chain is incomplete")
    if disposition["peer_reviewed_publication"]:
        raise ValueError("repository evidence cannot claim peer review")
    if disposition["default_production_paged_enabled"]:
        raise ValueError("negative Paged-vs-Direct result requires an opt-in default")
    if not outcome["upstream_boundary"]["all_recorded_cases_match"]:
        raise ValueError("upstream compatibility cases diverge")
    remap = outcome["research_results"]["vectorized_remap"]
    if not (remap["protocol_compliant"] and remap["acceptance_passed"]):
        raise ValueError("formal vector-remap result is not accepted")
    paged = outcome["research_results"]["paged_vs_direct"]
    if not paged["correctness_passed"] or paged["promotion_passed"]:
        raise ValueError("Paged-vs-Direct boundary was rewritten")
    k2 = outcome["research_results"]["k2_vs_k1"]
    if not (k2["correctness_passed"] and k2["promotion_passed"]):
        raise ValueError("K2 formal replacement gate is not satisfied")
    if k2["measured_responses_by_variant"] != {"k1": 480, "k2": 480}:
        raise ValueError("K2 measured-response evidence is incomplete")
    if k2["kernel_launches_by_variant"] != {"k1": 480, "k2": 480} or k2["fallbacks"] != 0:
        raise ValueError("K2 mechanism evidence is incomplete")
    if k2["client_median_regression_upper_95_percent"] > 5.0:
        raise ValueError("K2 latency uncertainty gate failed")
    if k2["kernel_duration_reduction_percent"] < 20.0:
        raise ValueError("K2 mechanism gate failed")
    objective = outcome["research_results"]["objective_paged_vs_direct"]
    if objective["workload_count"] < 6 or objective["observations"] != 360:
        raise ValueError("objective Paged workload matrix is incomplete")
    if not objective["all_workloads_cross_page"] or objective["promotion_passed"]:
        raise ValueError("objective Paged production boundary was rewritten")
    long_paged = outcome["research_results"]["long_context_paged_vs_direct"]
    if (long_paged["workload_count"] != 18 or long_paged["observations"] != 432 or
            long_paged["minimum_context_tokens"] != 64 or
            long_paged["maximum_context_tokens"] != 2048 or
            long_paged["primary_metric"] != "server_prompt_ms" or
            long_paged["promotion_passed"]):
        raise ValueError("long-context Paged evidence boundary was rewritten")
    batched = outcome["research_results"]["batched_paged_vs_direct"]
    if (batched["protocol_version"] != "5.0.0" or
            batched["observations"] != 144 or
            batched["primary_batch_size"] != 8 or
            batched["output_token_comparisons"] != 1080 or
            not batched["primary_cuda_execution_verified"] or
            batched["correctness_passed"] or
            batched["promotion_passed"]):
        raise ValueError("batched Paged evidence boundary was rewritten")
    hybrid = outcome["research_results"]["hybrid_paged_vs_direct"]
    execution = hybrid["execution_evidence"]
    if (hybrid["protocol_version"] != "6.1.0" or
            hybrid["observations"] != 36 or
            hybrid["primary_batch_size"] != 8 or
            hybrid["output_token_matches"] != hybrid["output_token_comparisons"] or
            hybrid["output_token_comparisons"] != 1728 or
            execution["mode"] != "contiguous_fastpath" or
            execution["contiguous_fastpath_calls"] != 216.0 or
            execution["contiguous_fastpath_sequences"] != 1728.0 or
            execution["custom_paged_graph_calls"] != 0.0 or
            execution["custom_cuda_dispatches"] != 0.0 or
            not hybrid["correctness_passed"] or
            not hybrid["promotion_passed"]):
        raise ValueError("hybrid Paged evidence boundary was rewritten")


def render_final_outcome(outcome: dict[str, Any]) -> str:
    app = outcome["application_result"]
    upstream = outcome["upstream_boundary"]
    remap = outcome["research_results"]["vectorized_remap"]
    policy = outcome["research_results"]["offline_action_policy_d3"]
    paged = outcome["research_results"]["paged_vs_direct"]
    k2 = outcome["research_results"]["k2_vs_k1"]
    objective = outcome["research_results"]["objective_paged_vs_direct"]
    long_paged = outcome["research_results"]["long_context_paged_vs_direct"]
    batched = outcome["research_results"]["batched_paged_vs_direct"]
    hybrid = outcome["research_results"]["hybrid_paged_vs_direct"]
    measured_responses = k2["measured_responses_by_variant"]["k1"]
    kernel_launches = k2["kernel_launches_by_variant"]["k1"]
    remap_values = remap["median_improvement_percent_by_blocks"]
    lines = [
        "# CacheFlow Runtime 最终成果说明",
        "",
        "## 最终定性",
        "",
        "当前仓库已经形成可独立交付的**单机应用成果**与**有边界的科研型项目成果**。它不是已发表论文，也不声称 Paged Attention 已普遍优于 upstream/Direct。最终结论由 `results/final-outcome.json` 从正式工件自动生成并校验。",
        "",
        "## 可交付应用成果",
        "",
        f"- 推免面试学习助手自动 HTTP/SSE 用户旅程通过：UI 资源={app['ui_rendered']}，SSE 契约={app['browser_sse_contract']}，并发用户={app['concurrent_users']}；另有 Chromium 交互 QA={app['interactive_browser_qa_passed']}。",
        f"- 本地知识库包含 {app['knowledge_documents']} 份文档、{app['knowledge_chunks']} 个检索块；断流可传播到 llama-server，429 背压与重启恢复均通过。",
        f"- 真实链路记录 {int(app['infra_metrics']['cached_prompt_tokens'])} 个缓存 prompt token、{int(app['infra_metrics']['cuda_kv_kernel_launches'])} 次自研 CUDA KV kernel、{int(app['infra_metrics']['cuda_kv_remap_vectorized_bytes'])} 个向量化 remap 字节。",
        "- 适用范围：单机或可信局域网；不声称已有外部用户采用。",
        "",
        "## 可交付科研成果",
        "",
        f"1. **Upstream 边界**：固定 `{upstream['pinned_revision']}`，相同工具链的 upstream policy 在 {upstream['upstream_mode_exact_output_cases']} 个记录用例中输出逐项一致。",
        f"2. **向量化 KV Remap**：1/4/16/32 blocks 的配对中位改善分别为 {remap_values['1']:.2f}%、{remap_values['4']:.2f}%、{remap_values['16']:.2f}%、{remap_values['32']:.2f}%；这是算子微基准，不外推为端到端加速。",
        f"3. **统一动作策略**：D3 在 {policy['evaluation_decisions']} 个留出决策中切换 {policy['switches_vs_h0']} 次，累计 regret {policy['cumulative_regret_ms']:.3f} ms，harmful decision {policy['harmful_decisions']} 次；chooser P99 {policy['chooser_p99_us']:.3f} us、热路径零分配。该结果仍是离线 replay。",
        f"4. **Paged-vs-Direct 负结果**：正确性通过，但 P95 从 {paged['direct_p95_ms']:.3f} ms 增至 {paged['paged_p95_ms']:.3f} ms（回退 {paged['p95_regression_percent']:.2f}%），超过 5% 门槛，因此 Paged 保持 opt-in。",
        f"5. **K2-vs-K1 正结果**：{k2['paired_trials']} 组同进程配对、每 variant {measured_responses} 条测量响应、{k2['paged_graph_entries_per_variant']} 次 Paged graph、0 fallback；请求 median/P95 回退 {k2['client_median_regression_percent']:.2f}%/{k2['client_p95_regression_percent']:.2f}%，median 回退 95% 上界 {k2['client_median_regression_upper_95_percent']:.2f}%；相同 {kernel_launches} 次 kernel 总时长由 {k2['k1_kernel_duration_ms']:.3f} ms 降至 {k2['k2_kernel_duration_ms']:.3f} ms（-{k2['kernel_duration_reduction_percent']:.2f}%），通过预注册替换门槛。",
        f"6. **客观 Prompt 矩阵负结果**：冻结 6 类输入、30 组随机化匹配进程块、360 个 workload-arm 观测，实际上下文覆盖 {min(objective['actual_context_tokens'].values())}–{max(objective['actual_context_tokens'].values())} token 且全部跨页。总体匹配块中位回退 {objective['median_regression_percent']:.2f}%（负值表示 Paged 更快），但 block-workload 回退分布 P95 为 {objective['block_workload_regression_p95_percent']:.2f}%、最差 workload 中位回退 {objective['worst_workload_median_regression_percent']:.2f}%，均超过门槛；该设计不冒充共享热状态 Trial Pair。",
        f"7. **长上下文 H10→H13 根因修复**：H10 的旧 split-K2 在主区间回退 50.35%；K4 改为一个 256-thread CTA 覆盖完整 7:1 GQA 组、以 `half2` 复用 K/V，并在设备端选择 64/128-token partition。24 个 CPU/CUDA oracle case 覆盖 1–{long_paged['maximum_context_tokens']} token。H13 使用 3 个仓库文档来源、18 个精确 token workload、{long_paged['matched_process_blocks']} 个严格平衡的匹配进程块和 {long_paged['observations']} 个 workload-arm 观测；512–2048 token 的服务端 prompt 时间中位回退降至 {long_paged['median_regression_percent']:.2f}%，process-block cluster bootstrap 95% 区间 [{long_paged['bootstrap_95_percent'][0]:.2f}%, {long_paged['bootstrap_95_percent'][1]:.2f}%]。因置信上界仍超过 +5% 门，不晋级。",
        "",
        "## 面试与简历允许使用的结论",
        "",
        "可以表述为：在 llama.cpp 上实现缓存感知调度、KV 生命周期、CUDA Remap/Swap 和受限 Paged Decode；从 H10 的 +50.35% 回退定位旧 K2 的标量计算与 GQA 重复加载问题，重构出整组 GQA 复用、half2 访问和自适应分区的 K4，在平衡长上下文矩阵中把主中位回退降至约 +3.98%，但因置信上界未过门而保持 fail-closed。",
        "",
        "不可以表述为：Paged Attention 全面优于 llama.cpp、端到端延迟提升 50.44%、已经发表论文、已经获得外部用户采用。",
        "",
        "## 数据、图表与原始证据",
        "",
        "- 图文总结报告：[`docs/final-illustrated-report.md`](final-illustrated-report.md)",
        "- K2/K1 请求与 kernel 对比图：[`results/research/h8-k2-production-v2.10.0/k2-production-comparison.svg`](../results/research/h8-k2-production-v2.10.0/k2-production-comparison.svg)",
        "- 客观 Prompt 矩阵分层结果图：[`results/research/h9-objective-paged-v2.0.0/comparison.svg`](../results/research/h9-objective-paged-v2.0.0/comparison.svg)",
        "- 长上下文 Paged/Direct 分层结果图：[`results/research/h13-balanced-adaptive-gqa-paged-v7.0.0/comparison.svg`](../results/research/h13-balanced-adaptive-gqa-paged-v7.0.0/comparison.svg)",
        "- 长上下文算法与实验报告：[`docs/research/long-context-paged-attention.md`](research/long-context-paged-attention.md)",
        "- K2 正式报告：[`results/research/h8-k2-production-v2.10.0/report.md`](../results/research/h8-k2-production-v2.10.0/report.md)",
        "- Paged-vs-Direct 正式负结果：[`results/research/h7-production-paged-v1.1.0/report.md`](../results/research/h7-production-paged-v1.1.0/report.md)",
        "- 研究项目总报告：[`docs/research-project-report.md`](research-project-report.md)",
        "- 真实应用旅程：[`results/user-application-journey.json`](../results/user-application-journey.json)",
        "- 每项输入工件的 SHA-256 位于 [`results/final-outcome.json`](../results/final-outcome.json)，防止报告数字与原始结果漂移。",
        "",
        "## 复现入口",
        "",
        "```powershell",
        ".\\scripts\\bootstrap.ps1",
        ".\\scripts\\verify.ps1",
        ".\\scripts\\verify_final_outcome.ps1",
        ".\\scripts\\start_production.ps1 -ModelPath .\\models\\qwen2.5-0.5b-instruct-q4_k_m.gguf -ApiKeyFile .\\runtime\\api-key.txt",
        "```",
        "",
        "`verify_final_outcome.ps1` 不重新挑选实验结果，而是校验本页、机器可读结论与正式工件是否一致。完整 GPU 重跑使用 `verify.ps1 -Full`。",
        "",
    ]
    lines.extend([
        "",
        "## H19 原生批处理客观结论",
        "",
        f"真实服务已经解除 batch=1 限制，并在非 unified KV 布局下验证 batch 1/2/4/8。正式矩阵包含 {batched['observations']} 个 action-cell 和 {batched['output_token_comparisons']} 次输出比较；batch 8 的设备端计数证明每个测量 wave 实际执行 24 个逐层 CUDA kernel、覆盖 24×8 个 sequence-layer。",
        f"但 batch 8 吞吐中位变化为 {batched['primary_throughput_gain_percent']:.2f}%（95% 区间 [{batched['primary_throughput_gain_bootstrap_95_percent'][0]:.2f}%, {batched['primary_throughput_gain_bootstrap_95_percent'][1]:.2f}%），P95 wave 延迟回退 {batched['primary_p95_wave_latency_regression_percent']:.2f}%，最差 cell 中位延迟回退 {batched['worst_cell_median_wave_latency_regression_percent']:.2f}%。输出 token 一致 {batched['output_token_matches']}/{batched['output_token_comparisons']}，另有 {batched['incomplete_probability_rows']} 行概率证据不完整，因此性能与正确性门均未通过，Paged 继续保持 opt-in。",
        "- H19 正式报告：[`results/research/h19-production-batched-paged-v5.0.0/report.md`](../results/research/h19-production-batched-paged-v5.0.0/report.md)",
        "- H19 批量性能图：[`results/research/h19-production-batched-paged-v5.0.0/comparison.svg`](../results/research/h19-production-batched-paged-v5.0.0/comparison.svg)",
        "",
        "## H20 布局感知混合路由正结果",
        "",
        "H19 的根因不是 batch API 不可用，而是连续物理布局仍被强制送入较慢的自研 K4 算术核。H20 将页表构建器扩展为同时判断逻辑页是否物理连续：连续时复用 upstream Direct/MMA attention，只有碎片页才进入 custom K4。两条路径仍共享同一个 Paged capability、block table 生命周期和 fail-closed 边界，不是在 API 外绕过功能。",
        f"正式 v6.1 固定 batch 8、context 128/512/1024、6 个平衡匹配进程块，共 {hybrid['observations']} 个 action-cell、{hybrid['output_token_comparisons']} 次输出与 top-64 分布比较。生产计数证明 {int(hybrid['execution_evidence']['contiguous_fastpath_calls'])} 次连续布局 fast-path、{int(hybrid['execution_evidence']['contiguous_fastpath_sequences'])} 条序列，custom Paged graph/CUDA dispatch 均为 0。吞吐中位变化 +{hybrid['primary_throughput_gain_percent']:.2f}%，cluster bootstrap 95% 区间 [{hybrid['primary_throughput_gain_bootstrap_95_percent'][0]:.2f}%, +{hybrid['primary_throughput_gain_bootstrap_95_percent'][1]:.2f}%]；Direct/Paged P95 为 {hybrid['direct_p95_wave_latency_ms']:.3f}/{hybrid['paged_p95_wave_latency_ms']:.3f} ms，即变化 {hybrid['primary_p95_wave_latency_regression_percent']:.2f}%。全部输出一致，正式非劣门通过。该结果不证明碎片页 K4 优于 Direct，也不证明容量或碎片收益。",
        "- H20 正式报告：[`results/research/h20-paged-hybrid-batch8-v6.1.0/report.md`](../results/research/h20-paged-hybrid-batch8-v6.1.0/report.md)",
        "- H20 混合路由性能图：[`results/research/h20-paged-hybrid-batch8-v6.1.0/comparison.svg`](../results/research/h20-paged-hybrid-batch8-v6.1.0/comparison.svg)",
        "",
    ])
    return "\n".join(lines)


def render_summary_chart(outcome: dict[str, Any]) -> str:
    remap = outcome["research_results"]["vectorized_remap"]["median_improvement_percent_by_blocks"]
    paged = outcome["research_results"]["paged_vs_direct"]
    k2 = outcome["research_results"]["k2_vs_k1"]
    values = [remap[str(block)] for block in (1, 4, 16, 32)]
    bars = "".join(
        f'<rect x="{90 + i * 78}" y="{260 - value * 2:.1f}" width="46" height="{value * 2:.1f}" fill="#2563eb"/>'
        f'<text x="{113 + i * 78}" y="{250 - value * 2:.1f}" text-anchor="middle">{value:.2f}%</text>'
        f'<text x="{113 + i * 78}" y="282" text-anchor="middle">{block}</text>'
        for i, (block, value) in enumerate(zip((1, 4, 16, 32), values))
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="420" viewBox="0 0 1200 420">
<style>text{{font-family:Segoe UI,Arial,sans-serif;fill:#172033;font-size:15px}}.title{{font-size:22px;font-weight:700}}.sub{{font-size:13px;fill:#526079}}</style>
<rect width="1200" height="420" fill="#f8fafc"/><text x="40" y="38" class="title">CacheFlow formal results (RTX 4050 Laptop, Qwen2.5-0.5B scope)</text>
<rect x="35" y="65" width="365" height="315" rx="12" fill="white" stroke="#dbe4f0"/><text x="55" y="95" class="title">H1 Vectorized KV remap</text><text x="55" y="118" class="sub">Paired median kernel-time improvement; x = moved blocks</text>{bars}<line x1="70" y1="260" x2="375" y2="260" stroke="#94a3b8"/>
<rect x="418" y="65" width="350" height="315" rx="12" fill="white" stroke="#dbe4f0"/><text x="438" y="95" class="title">Paged vs Direct</text><text x="438" y="118" class="sub">Request P95; lower is better</text><rect x="468" y="175" width="90" height="{paged['direct_p95_ms'] * 5:.1f}" fill="#16a34a"/><rect x="615" y="{315 - paged['paged_p95_ms'] * 5:.1f}" width="90" height="{paged['paged_p95_ms'] * 5:.1f}" fill="#dc2626"/><text x="513" y="165" text-anchor="middle">{paged['direct_p95_ms']:.3f} ms</text><text x="660" y="{305 - paged['paged_p95_ms'] * 5:.1f}" text-anchor="middle">{paged['paged_p95_ms']:.3f} ms</text><text x="513" y="340" text-anchor="middle">Direct</text><text x="660" y="340" text-anchor="middle">Paged</text><text x="603" y="368" text-anchor="middle" fill="#dc2626">FAIL: +{paged['p95_regression_percent']:.2f}%</text>
<rect x="786" y="65" width="379" height="315" rx="12" fill="white" stroke="#dbe4f0"/><text x="806" y="95" class="title">K2 vs K1 inside Paged</text><text x="806" y="118" class="sub">480 identical-scope kernel launches per variant</text><rect x="836" y="160" width="105" height="{k2['k1_kernel_duration_ms'] * 18:.1f}" fill="#64748b"/><rect x="1000" y="{307 - k2['k2_kernel_duration_ms'] * 18:.1f}" width="105" height="{k2['k2_kernel_duration_ms'] * 18:.1f}" fill="#2563eb"/><text x="888" y="150" text-anchor="middle">{k2['k1_kernel_duration_ms']:.3f} ms</text><text x="1052" y="{297 - k2['k2_kernel_duration_ms'] * 18:.1f}" text-anchor="middle">{k2['k2_kernel_duration_ms']:.3f} ms</text><text x="888" y="335" text-anchor="middle">K1</text><text x="1052" y="335" text-anchor="middle">K2</text><text x="970" y="368" text-anchor="middle" fill="#2563eb">PASS: -{k2['kernel_duration_reduction_percent']:.2f}%</text>
</svg>'''


def render_architecture_chart() -> str:
    boxes = [(35, "User/UI"), (235, "Serving control"), (455, "CacheFlow policy"), (675, "llama_decode"), (895, "KV/CUDA kernels")]
    nodes = "".join(
        f'<rect x="{x}" y="95" width="170" height="72" rx="12" fill="white" stroke="#2563eb" stroke-width="2"/><text x="{x + 85}" y="138" text-anchor="middle">{label}</text>'
        for x, label in boxes
    )
    arrows = "".join(f'<path d="M{x} 131 H{x + 30}" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>' for x in (205, 425, 645, 865))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="270" viewBox="0 0 1100 270"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#475569"/></marker></defs><style>text{{font-family:Segoe UI,Arial,sans-serif;fill:#172033;font-size:16px}}.sub{{font-size:13px;fill:#526079}}</style><rect width="1100" height="270" fill="#f8fafc"/><text x="35" y="40" font-size="22" font-weight="700">Real execution path and contribution boundary</text>{nodes}{arrows}<text x="120" y="205" text-anchor="middle" class="sub">SSE, sessions, RAG</text><text x="320" y="205" text-anchor="middle" class="sub">queue, cancel, backpressure</text><text x="540" y="205" text-anchor="middle" class="sub">action choice, lifecycle</text><text x="760" y="205" text-anchor="middle" class="sub">real upstream hot path</text><text x="980" y="205" text-anchor="middle" class="sub">Remap / Swap / Paged K1-K4</text><path d="M220 235 H1070" stroke="#2563eb" stroke-width="4"/><text x="645" y="258" text-anchor="middle" class="sub">Personal contribution crosses control plane, model runtime, and GPU operator layers</text></svg>'''


def render_illustrated_report(outcome: dict[str, Any]) -> str:
    app = outcome["application_result"]
    remap = outcome["research_results"]["vectorized_remap"]
    policy = outcome["research_results"]["offline_action_policy_d3"]
    paged = outcome["research_results"]["paged_vs_direct"]
    k2 = outcome["research_results"]["k2_vs_k1"]
    objective = outcome["research_results"]["objective_paged_vs_direct"]
    long_paged = outcome["research_results"]["long_context_paged_vs_direct"]
    batched = outcome["research_results"]["batched_paged_vs_direct"]
    hybrid = outcome["research_results"]["hybrid_paged_vs_direct"]
    return rf'''# CacheFlow Runtime 图文总结报告

## 1. 成果是什么

项目不是在 llama.cpp 外层套一层接口，而是把缓存感知调度、KV 生命周期管理和 CUDA 算子接入真实 `llama-server → llama_decode → KV memory → CUDA` 路径。最终形成两项可独立交付成果：一套面向单机/可信局域网的推免面试学习助手，以及一套保留正负结果、可由原始 trials 重算的独立科研型成果。

![真实执行链与个人贡献边界](assets/final-system-flow.svg)

## 2. 核心创新点

1. **从请求调度贯通到 GPU 数据移动。** 控制面不只决定请求先后，还显式管理 KV 的驻留、抢占、恢复、换入换出和重算，并把决策计数、CUDA event、kernel launch 与请求延迟关联起来。
2. **统一动作代价模型。** 将 Direct、Remap、Swap、Recompute 与受能力门禁约束的 Paged 视为同一动作空间；D3 用有界 ridge 预测动作相对 H0 的增量代价，证据不足时回退 H0。
3. **自研 CUDA KV Remap/Swap。** 用向量化数据搬运取代标量 gather/scatter，并以 CPU oracle、边界/重叠映射和 sanitizer 验证正确性。
4. **Paged Decode 从 K1 演进到长上下文 split-K2。** K1 是每个 query head 一个 CTA 的正确基线；短上下文 K2 针对 GQA7 复用同一 KV head。长上下文版本再把序列切成 32-token tile 和 256-token partition，各 partition 产生可精确合并的 FP32 online-softmax 状态，避免把整段上下文塞进一个 CTA。
5. **证据工程本身是成果的一部分。** 实验先预注册门槛，再生成原始请求、Prometheus、NSYS/SQLite、汇总、图表和 manifest；最终结论由验证器从底层证据重算，负结果不删除。

## 3. 实验数据从哪里来、如何处理

### 3.1 数据来源

- **应用数据**：仓库中的 5 份面试知识文档被切分为 {app['knowledge_chunks']} 个检索块；自动 HTTP/SSE 用户旅程验证后端链路，另有 Codex 内置 Chromium 的交互记录验证浏览器流式完成与会话侧栏显示，再调用固定 Qwen2.5-0.5B 模型。
- **H1 Remap 数据**：CUDA benchmark 在 1/4/16/32 个 block 上分别执行标量与向量化方法；每个规模 20 个 confirmatory 配对，另有 warm-up，正式统计排除 warm-up。
- **H4 策略数据**：同一 trace/session/prefix family 下收集 H0/A1/T1/L1 的动作观测，按 trace 分组并按时间切分，避免同一会话泄漏到训练和留出集。
- **H7/H8 服务数据**：同一进程内交替运行对照臂，固定模型、prompt、输出 token、context 和设备。H7 比较 Direct 与 Paged；H8 在已进入 Paged 路径后比较 K1 与 K2。
- **H9 客观输入矩阵**：prompt 不再嵌入 runner；冻结语料文件包含受控边界、中文数据库/机器学习、英文 Attention、中英混合 CUDA 和 C++ 代码六类输入。30 组随机化匹配进程块中，每个 Direct/Paged 独立进程 arm 都运行全部语料，实际 token 数由模型响应记录而非人工假设；它不是共享热状态 Trial Pair。原始 v2.0.0 协议及哈希保持不动；模型和 vendor-diff 哈希属于实验后的验证修正，只增强可审计性，不冒充运行前预注册或 contemporaneous run binding。
- **H10 来源绑定长上下文矩阵**：从架构文档、面试手册和 Paged 研究笔记抽取正文，经统一 Markdown 归一化后由真实 Qwen tokenizer 截取并反解为 64/128/256/512/1024/2048 token；每条记录绑定源文件 SHA-256、token span、模型哈希和最终实际 token 数。3 个来源族共 18 个 workload，不是 `one one ...` 或 runner 内硬编码字符串。

### 3.2 处理过程

原始记录先做协议一致性检查：请求参数、trial/arm 顺序、PID、计数器增量和响应内容必须完整。H1/H8 在共享状态的 trial pair 内计算差值；H9 与 H10–H14 按随机化匹配进程块比较两个独立进程 arm，不冒充 Trial Pair。不确定性按相应的 pair、匹配进程块或 trace cluster 重采样 bootstrap，以保留相关样本结构。长上下文实验的主指标选 `server_prompt_ms`，因为单 token completion 中被选择的 KV 动作与 attention graph 在 prompt 阶段执行；`predicted_ms=0.001` 只是采样后的量化余量，v3 因误选该字段被完整保留为无效实验。H13 还强制 6/6 平衡臂顺序，并从 raw arm 重建 normalized trials。只有正确性、延迟上界、覆盖范围与零 fallback 同时满足，候选才能晋级。

## 4. 实验结果

![正式实验结果总览](../results/final-outcome-summary.svg)

- **H1 正结果**：1/4/16/32 blocks 的 kernel 时间配对中位改善为 {remap['median_improvement_percent_by_blocks']['1']:.2f}%、{remap['median_improvement_percent_by_blocks']['4']:.2f}%、{remap['median_improvement_percent_by_blocks']['16']:.2f}%、{remap['median_improvement_percent_by_blocks']['32']:.2f}%。规模越大收益越小，因此只称算子微基准改善。
- **D3 有条件正结果**：留出集 {policy['evaluation_decisions']} 个决策中相对 H0 切换 {policy['switches_vs_h0']} 次，累计 regret {policy['cumulative_regret_ms']:.3f} ms，harmful decision {policy['harmful_decisions']} 次；chooser P99 {policy['chooser_p99_us']:.3f} μs且热路径零分配。它仍是离线 replay，尚不能写成线上普适收益。
- **Paged-vs-Direct 负结果**：P95 从 {paged['direct_p95_ms']:.3f} ms 上升到 {paged['paged_p95_ms']:.3f} ms，回退 {paged['p95_regression_percent']:.2f}%，超过 5% 门槛。因此默认启动器不启用 Paged。
- **K2-vs-K1 正结果**：30 组同进程配对、每臂 480 条测量响应和 600 次 Paged graph entry、0 fallback。请求 median/P95 仅回退 {k2['client_median_regression_percent']:.2f}%/{k2['client_p95_regression_percent']:.2f}%，median 回退 bootstrap 95% 上界 {k2['client_median_regression_upper_95_percent']:.2f}%；相同 480 次 kernel 总时长从 {k2['k1_kernel_duration_ms']:.3f} ms 降至 {k2['k2_kernel_duration_ms']:.3f} ms，降低 {k2['kernel_duration_reduction_percent']:.2f}%。
- **H9 客观矩阵负结果**：6类冻结输入、30组随机化匹配进程块、360个 workload-arm 观测均实际跨页。总体中位数显示 Paged 改善 {-objective['median_regression_percent']:.2f}%，但 block-workload 回退分布P95为 {objective['block_workload_regression_p95_percent']:.2f}%，最差workload中位回退 {objective['worst_workload_median_regression_percent']:.2f}%，分别超过20%和5%门槛，因此不能晋级。
- **H10→H13 长上下文根因修复**：H10 的旧 split-K2 主中位回退为 50.35%。K4 用整组 GQA7 复用、`half2` K/V 访问和设备端自适应 partition 重构算子；H13 以 18 个来源绑定 workload、12 个严格平衡的匹配进程块和 432 个 workload-arm 观测覆盖 64–2048 token，合计 3456 个测量请求，输出逐项一致、Paged graph 覆盖完整且 0 fallback。512–2048 token 主区间的 `server_prompt_ms` 中位回退降至 {long_paged['median_regression_percent']:.2f}%，cluster bootstrap 95% 区间 [{long_paged['bootstrap_95_percent'][0]:.2f}%, {long_paged['bootstrap_95_percent'][1]:.2f}%]，P95 回退 {long_paged['p95_regression_percent']:.2f}%；算法改进有明确幅度，但置信上界仍未通过 +5% 晋级门。
- **H20 布局感知混合路由正结果**：连续物理页不再强制执行较慢的 custom K4，而是复用 upstream attention；碎片页仍保留页表寻址 K4。batch 8、128/512/1024 token、6 个平衡匹配进程块中，吞吐中位变化 +{hybrid['primary_throughput_gain_percent']:.2f}%，95% 区间 [{hybrid['primary_throughput_gain_bootstrap_95_percent'][0]:.2f}%, +{hybrid['primary_throughput_gain_bootstrap_95_percent'][1]:.2f}%]，P95 wave 延迟变化 {hybrid['primary_p95_wave_latency_regression_percent']:.2f}%，{hybrid['output_token_matches']}/{hybrid['output_token_comparisons']} 输出一致，正式非劣门通过。

![K2/K1 正式对比图](../results/research/h8-k2-production-v2.10.0/k2-production-comparison.svg)

![客观Prompt矩阵分层结果](../results/research/h9-objective-paged-v2.0.0/comparison.svg)

![长上下文Paged/Direct结果](../results/research/h13-balanced-adaptive-gqa-paged-v7.0.0/comparison.svg)

### H19：真实批处理不是“计数器看起来像执行”

批处理改造同时覆盖 unified `[D,B,H,1]` 与真实服务默认的 non-unified `[D,1,H,B]`。CPU oracle 用展平后的 sequence row 选择 block table；CUDA 根据 batch 所在维度选择 Q、输出和 K/V stream stride。执行证据不再使用 graph admission 计数，而是由每层 Paged CUDA kernel 的 block 0/thread 0 在设备端原子累加，所以 CUDA Graph replay 也不会漏计。

正式 H19 固定 batch 1/2/4/8、context 128/512/1024、6 个平衡匹配进程块，共 {batched['observations']} 个 action-cell、{batched['output_token_comparisons']} 次输出比较。主 batch 8 每个测量 wave 均实现一个 8-sequence graph、24 个逐层 kernel 和 192 个 sequence-layer 执行，0 fallback。结果仍为负：吞吐中位变化 {batched['primary_throughput_gain_percent']:.2f}%，cluster bootstrap 95% 区间 [{batched['primary_throughput_gain_bootstrap_95_percent'][0]:.2f}%, {batched['primary_throughput_gain_bootstrap_95_percent'][1]:.2f}%]；P95 wave 延迟回退 {batched['primary_p95_wave_latency_regression_percent']:.2f}%，最差 cell 中位回退 {batched['worst_cell_median_wave_latency_regression_percent']:.2f}%。输出 token 一致 {batched['output_token_matches']}/{batched['output_token_comparisons']}，且 {batched['incomplete_probability_rows']} 行缺少完整概率证据，因此不晋级。

![原生批处理 Paged/Direct 客观结果](../results/research/h19-production-batched-paged-v5.0.0/comparison.svg)

### H20：从强制自研算术核改为布局感知混合执行

NSYS 根因分析显示，H19 的 K4 主 kernel 平均约 40.54 μs，而 upstream Flash Attention 约 26.08 μs；额外 merge 约 1.62 μs，因此问题主要是算术核效率，不是 host 包装。仅把 1024-token partition 从 64 调到 128 虽将 grid 和 scratch 减半，却没有改善主 kernel 时间；实验性 Tensor Core K5 也慢于 K4，因此均未冒充成果。

最终修复把“分页生命周期”与“必须运行自研分页算术核”解耦。block-table builder 检测所有活跃逻辑页的物理基址是否按 page size 连续；若连续，Paged 请求复用 upstream attention；若不连续，才使用 custom K4。H20 的 {int(hybrid['execution_evidence']['contiguous_fastpath_calls'])} 次 fast-path 调用覆盖 {int(hybrid['execution_evidence']['contiguous_fastpath_sequences'])} 条序列，且 custom graph/CUDA dispatch 为 0。吞吐点估计 +{hybrid['primary_throughput_gain_percent']:.2f}%，95% 区间 [{hybrid['primary_throughput_gain_bootstrap_95_percent'][0]:.2f}%, +{hybrid['primary_throughput_gain_bootstrap_95_percent'][1]:.2f}%]，只能称为“有界非劣”，不能称为严格占优；碎片布局 K4 的性能仍由 H19/H13 负结果约束。

![布局感知混合路由结果](../results/research/h20-paged-hybrid-batch8-v6.1.0/comparison.svg)

## 5. 应用结果与最终边界

应用旅程已覆盖 UI、SSE、并发、取消、429 背压、重启恢复和本地知识检索，并记录 {int(app['infra_metrics']['cached_prompt_tokens'])} 个缓存 prompt token、{int(app['infra_metrics']['cuda_kv_kernel_launches'])} 次自研 CUDA KV launch、{int(app['infra_metrics']['cuda_kv_remap_vectorized_bytes'])} 个向量化 remap 字节。它能作为可运行应用项目和有实验链的 AI Infra 研究项目交付。

最终不能声称“Paged 全面优于 Direct”或“端到端加速 {k2['kernel_duration_reduction_percent']:.2f}%”。准确结论是：**短上下文受限 Paged 内部，K2 已替换 K1；长上下文 custom K4 将旧 K2 回退显著降低但仍未晋级；生产混合路由在连续 batch-8 矩阵中通过有界非劣门，碎片布局仍 fail-closed 到经过 oracle 验证但性能未晋级的 K4。两臂复用同一底层分配器，本实验不提供碎片率或容量优势证据。**

## 6. 复现与审计

```powershell
.\scripts\verify_final_outcome.ps1
.\scripts\verify.ps1
```

前者重算 H1 并调用 H4/H7/H8 正式验证器，同时校验本报告、最终 JSON、图表和启动器哈希；后者执行全仓快速测试和架构/工件验收。全部源数据路径与 SHA-256 见 [`results/final-outcome.json`](../results/final-outcome.json)。
'''
