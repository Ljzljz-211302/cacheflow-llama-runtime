from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from llama_lab.objective_paged_benchmark import (
    analyze, arm_plan, load_definition, validate_frozen_source_revision,
)


ROOT = Path(__file__).resolve().parents[1]


def rows(protocol, corpus):
    result = []
    for pair, _, action in arm_plan(protocol):
        for workload in corpus["workloads"]:
            base = 10.0 + pair / 100
            row = {
                "pair": pair,
                "action": action,
                "workload_id": workload["id"],
                "category": workload["category"],
                "prompt_sha256": hashlib.sha256(workload["prompt"].encode()).hexdigest(),
                "client_elapsed_ms": [base * (1.01 if action == "paged" else 1)] * 4,
                "server_decode_ms": [base * (1.01 if action == "paged" else 1)] * 4,
                "server_prompt_ms": [base * (1.01 if action == "paged" else 1)] * 4,
                "actual_context_tokens": [17] * 4,
                "contents": [","] * 4,
                "paged_calls": 4 if action == "paged" else 0,
                "paged_fallbacks": 0,
                "action_decisions": 4,
                "action_reason_decisions": 4,
                "action_observations": 4,
            }
            if "paged_kernel_variant" in protocol["service"]:
                row["paged_kernel_variant"] = protocol["service"]["paged_kernel_variant"]
            result.append(row)
    return result


class ObjectivePagedBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.protocol, self.corpus = load_definition(
            ROOT, ROOT / "config/production_paged_objective_protocol_v1.json"
        )

    def test_protocol_uses_external_diverse_frozen_corpus(self):
        self.assertNotIn("prompt", self.protocol["request"])
        self.assertGreaterEqual(len({row["category"] for row in self.corpus["workloads"]}), 4)

    def test_analysis_covers_every_pair_action_and_workload(self):
        summary = analyze(self.protocol, self.corpus, rows(self.protocol, self.corpus))
        self.assertEqual(summary["paired_trials"], 30)
        self.assertEqual(summary["workload_count"], 6)
        self.assertEqual(summary["observations"], 360)
        self.assertTrue(summary["promotion_passed"])

    def test_analysis_rejects_prompt_substitution(self):
        evidence = rows(self.protocol, self.corpus)
        tampered = copy.deepcopy(evidence)
        tampered[0]["prompt_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "frozen corpus"):
            analyze(self.protocol, self.corpus, tampered)

    def test_analysis_rejects_context_length_diverging_from_frozen_corpus(self):
        protocol, corpus = load_definition(
            ROOT, ROOT / "config/production_paged_objective_protocol_v4.json"
        )
        evidence = rows(protocol, corpus)
        evidence[0]["actual_context_tokens"] = [65] * 4
        with self.assertRaisesRegex(ValueError, "frozen tokenizer length"):
            analyze(protocol, corpus, evidence)

    def test_analysis_rejects_missing_workload_cell(self):
        evidence = rows(self.protocol, self.corpus)[:-1]
        with self.assertRaisesRegex(ValueError, "full pair/action/workload matrix"):
            analyze(self.protocol, self.corpus, evidence)

    def test_v2_requires_cross_page_coverage_and_tail_guardrails(self):
        protocol, corpus = load_definition(
            ROOT, ROOT / "config/production_paged_objective_protocol_v2.json"
        )
        amended = copy.deepcopy(protocol)
        amended["matched_process_blocks"] = protocol["paired_trials"]
        evidence = rows(amended, corpus)
        summary = analyze(amended, corpus, evidence)
        self.assertTrue(summary["page_coverage_passed"])
        self.assertIn("primary_p95_limit_percent", summary)
        self.assertIn("worst_workload_median_regression_percent", summary)
        self.assertEqual(summary["matched_process_blocks"], 30)

    def test_repository_v2_retains_negative_tail_result(self):
        summary = json.loads(
            (ROOT / "results/research/h9-objective-paged-v2.0.0/summary.json").read_text(encoding="utf-8")
        )
        self.assertFalse(summary["promotion_passed"])
        self.assertTrue(summary["page_coverage_passed"])
        self.assertGreater(summary["primary_matched_block_regression_percent"]["p95"], 20.0)
        self.assertGreater(summary["worst_workload_median_regression_percent"], 5.0)

    def test_analysis_rejects_misrouted_direct_arm(self):
        protocol, corpus = load_definition(
            ROOT, ROOT / "config/production_paged_objective_protocol_v2.json"
        )
        evidence = rows(protocol, corpus)
        direct = next(row for row in evidence if row["action"] == "direct")
        direct["paged_calls"] = 1
        with self.assertRaisesRegex(ValueError, "Direct row entered"):
            analyze(protocol, corpus, evidence)

    def test_v2_binds_model_and_vendor_overlay(self):
        protocol, _ = load_definition(
            ROOT, ROOT / "config/production_paged_objective_protocol_v2.json"
        )
        amendment = json.loads(
            (ROOT / "config/production_paged_objective_validation_amendment_v2.json").read_text(encoding="utf-8")
        )
        protocol_path = ROOT / amendment["original_protocol_file"]
        self.assertEqual(hashlib.sha256(protocol_path.read_bytes()).hexdigest(), amendment["original_protocol_sha256"])
        self.assertTrue(amendment["terminology"]["not_a_trial_pair"])
        binding = amendment["post_run_evidence_binding"]
        self.assertEqual(len(binding["model_sha256"]), 64)
        overlay = ROOT / binding["vendor_overlay_file"]
        self.assertEqual(hashlib.sha256(overlay.read_bytes()).hexdigest(), binding["vendor_overlay_sha256"])
        self.assertEqual(len(binding["vendor_diff_sha256"]), 64)

    def test_service_and_model_paged_capability_share_context_2048_boundary(self):
        service = (ROOT / "vendor/llama.cpp/tools/server/server-context.cpp").read_text(encoding="utf-8")
        model = (ROOT / "vendor/llama.cpp/src/llama-context.cpp").read_text(encoding="utf-8")
        self.assertIn("resident_prefix <= 2048", service)
        self.assertIn("layout.context_length <= 2048", model)

    def test_v3_is_source_bound_and_uses_long_context_primary(self):
        protocol, corpus = load_definition(
            ROOT, ROOT / "config/production_paged_objective_protocol_v3.json"
        )
        self.assertEqual(protocol["analysis"]["primary_timing_field"], "server_decode_ms")
        self.assertEqual(protocol["analysis"]["primary_minimum_context_tokens"], 512)
        self.assertEqual({row["actual_prompt_tokens"] for row in corpus["workloads"]},
                         {64, 128, 256, 512, 1024, 2048})
        self.assertTrue(all(len(row["source_sha256"]) == 64 for row in corpus["workloads"]))

    def test_v4_source_provenance_uses_measurement_revision_after_docs_evolve(self):
        _, corpus = load_definition(
            ROOT, ROOT / "config/production_paged_objective_protocol_v4.json"
        )
        summary = json.loads((
            ROOT / "results/research/h10-long-context-paged-v4.0.0/summary.json"
        ).read_text(encoding="utf-8"))
        validate_frozen_source_revision(ROOT, corpus, summary["git_revision"])
        with self.assertRaises(ValueError):
            load_definition(
                ROOT, ROOT / "config/production_paged_objective_protocol_v4.json",
                validate_live_sources=True,
            )

    def test_v3_primary_excludes_short_context_worst_case(self):
        protocol, corpus = load_definition(
            ROOT, ROOT / "config/production_paged_objective_protocol_v3.json"
        )
        evidence = rows(protocol, corpus)
        contexts = {row["id"]: row["actual_prompt_tokens"] for row in corpus["workloads"]}
        for row in evidence:
            context = contexts[row["workload_id"]]
            row["actual_context_tokens"] = [context] * 4
            if context < 512 and row["action"] == "paged":
                row["server_decode_ms"] = [100.0] * 4
        summary = analyze(protocol, corpus, evidence)
        self.assertEqual(summary["primary_workload_count"], 9)
        self.assertLess(summary["worst_workload_median_regression_percent"], 2.0)
        self.assertEqual(set(summary["regression_by_context_tokens"]),
                         {"64", "128", "256", "512", "1024", "2048"})

    def test_v5_binds_and_rejects_paged_kernel_substitution(self):
        protocol, corpus = load_definition(
            ROOT, ROOT / "config/production_paged_objective_protocol_v5.json"
        )
        self.assertEqual(protocol["service"]["paged_kernel_variant"], "K4")
        evidence = rows(protocol, corpus)
        evidence[0]["paged_kernel_variant"] = "K2"
        with self.assertRaisesRegex(ValueError, "registered Paged kernel"):
            analyze(protocol, corpus, evidence)

    def test_v6_registers_context_adaptive_k4_and_more_samples(self):
        protocol, _ = load_definition(
            ROOT, ROOT / "config/production_paged_objective_protocol_v6.json"
        )
        self.assertEqual(protocol["service"]["paged_kernel_variant"], "K4")
        self.assertEqual(protocol["request"]["measured_requests_per_workload_arm"], 8)
        self.assertIn("device-side", protocol["operator_design"]["adaptive_partitioning"])

    def test_balanced_arm_plan_has_equal_first_position_counts(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["matched_process_blocks"] = 12
        protocol["balanced_arm_order"] = True
        first = [action for _, order, action in arm_plan(protocol) if order == 1]
        self.assertEqual(first.count("direct"), 6)
        self.assertEqual(first.count("paged"), 6)


if __name__ == "__main__":
    unittest.main()
