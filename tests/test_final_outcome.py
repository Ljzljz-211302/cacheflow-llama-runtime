import copy
import json
import unittest
from pathlib import Path

from llama_lab.final_outcome import (
    build_final_outcome,
    render_final_outcome,
    render_illustrated_report,
    render_summary_chart,
    validate_h1_records,
    validate_final_outcome,
)


ROOT = Path(__file__).resolve().parents[1]


class FinalOutcomeTests(unittest.TestCase):
    def test_formal_sources_build_a_complete_bounded_outcome(self):
        outcome = build_final_outcome(ROOT)
        self.assertTrue(outcome["disposition"]["application_project_complete"])
        self.assertTrue(outcome["disposition"]["independent_research_project_complete"])
        self.assertFalse(outcome["disposition"]["peer_reviewed_publication"])
        self.assertFalse(outcome["disposition"]["default_production_paged_enabled"])
        self.assertFalse(outcome["research_results"]["paged_vs_direct"]["promotion_passed"])
        self.assertTrue(outcome["research_results"]["k2_vs_k1"]["promotion_passed"])

    def test_validator_rejects_rewriting_negative_paged_result(self):
        outcome = build_final_outcome(ROOT)
        tampered = copy.deepcopy(outcome)
        tampered["research_results"]["paged_vs_direct"]["promotion_passed"] = True
        with self.assertRaisesRegex(ValueError, "boundary was rewritten"):
            validate_final_outcome(tampered)

    def test_validator_rejects_incomplete_k2_response_evidence(self):
        outcome = build_final_outcome(ROOT)
        tampered = copy.deepcopy(outcome)
        tampered["research_results"]["k2_vs_k1"]["measured_responses_by_variant"]["k2"] = 479
        with self.assertRaisesRegex(ValueError, "measured-response evidence is incomplete"):
            validate_final_outcome(tampered)

    def test_validator_rejects_missing_browser_qa(self):
        outcome = build_final_outcome(ROOT)
        tampered = copy.deepcopy(outcome)
        tampered["application_result"]["interactive_browser_qa_passed"] = False
        with self.assertRaisesRegex(ValueError, "interactive browser QA has not passed"):
            validate_final_outcome(tampered)

    def test_default_paged_state_is_bound_to_production_launcher(self):
        launcher = (ROOT / "scripts/start_production.ps1").read_text(encoding="utf-8")
        outcome = build_final_outcome(ROOT)
        self.assertEqual(
            outcome["disposition"]["default_production_paged_enabled"],
            '"--kv-paged-decode"' in launcher,
        )

    def test_h1_validator_rejects_protocol_design_tamper(self):
        manifest = json.loads((ROOT / "results/research/h1-vector-remap-v1.0.0/manifest.json").read_text(encoding="utf-8"))
        protocol = json.loads((ROOT / "config/research_protocol.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in (ROOT / "results/research/h1-vector-remap-v1.0.0/trials.jsonl").read_text(encoding="utf-8").splitlines()]
        tampered = copy.deepcopy(records)
        tampered[2]["random_seed"] = 0
        with self.assertRaisesRegex(ValueError, "method or seed differs"):
            validate_h1_records(manifest, tampered, protocol)

    def test_h1_validator_rejects_negative_trials_with_stale_pass_gate(self):
        manifest = json.loads((ROOT / "results/research/h1-vector-remap-v1.0.0/manifest.json").read_text(encoding="utf-8"))
        protocol = json.loads((ROOT / "config/research_protocol.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in (ROOT / "results/research/h1-vector-remap-v1.0.0/trials.jsonl").read_text(encoding="utf-8").splitlines()]
        tampered = copy.deepcopy(records)
        for row in tampered:
            if row["phase"] == "confirmatory":
                row["timing_ms"]["synchronized_kernel_ms"] = 1.0 if row["method"] == "scalar_gather_scatter" else 2.0
        negative_manifest = copy.deepcopy(manifest)
        by_blocks = {}
        for row in tampered:
            if row["phase"] == "confirmatory":
                by_blocks.setdefault(row["blocks"], {}).setdefault(row["pair_id"], {})[row["method"]] = row["timing_ms"]["synchronized_kernel_ms"]
        from llama_lab.research_protocol import paired_bootstrap_summary
        summaries = []
        for blocks, pairs in sorted(by_blocks.items()):
            summary = paired_bootstrap_summary(
                [(p["scalar_gather_scatter"], p["vectorized_gather_scatter"]) for p in pairs.values()],
                confidence_level=0.95, resamples=10000, seed=20260806 + blocks,
            )
            summary["blocks"] = blocks
            summaries.append(summary)
        negative_manifest["paired_summaries"] = summaries
        with self.assertRaisesRegex(ValueError, "acceptance differs"):
            validate_h1_records(negative_manifest, tampered, protocol)

    def test_rendered_report_contains_positive_and_negative_results(self):
        report = render_final_outcome(build_final_outcome(ROOT))
        self.assertIn("Paged-vs-Direct 负结果", report)
        self.assertIn("K2-vs-K1 正结果", report)
        self.assertIn("不是已发表论文", report)

    def test_illustrated_report_explains_innovation_data_and_results(self):
        outcome = build_final_outcome(ROOT)
        report = render_illustrated_report(outcome)
        self.assertIn("核心创新点", report)
        self.assertIn("实验数据从哪里来、如何处理", report)
        self.assertIn("Paged-vs-Direct 负结果", report)
        self.assertIn("K2-vs-K1 正结果", report)
        self.assertIn("final-system-flow.svg", report)
        self.assertIn("final-outcome-summary.svg", report)
        self.assertIn("<svg", render_summary_chart(outcome))
