from __future__ import annotations

import unittest
import hashlib
import json
import tempfile
from pathlib import Path
import sys

from llama_lab.batched_paged_performance import analyze, experiment_plan

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_batched_paged_performance import verify_rederivation_inputs  # noqa: E402


class BatchedPagedPerformanceTest(unittest.TestCase):
    def protocol(self) -> dict:
        return {
            "protocol_version": "1.0.0",
            "random_seed": 17,
            "matched_process_blocks": 2,
            "matrix": {"batch_sizes": [1, 2, 8], "context_tokens": [128]},
            "measurement": {"waves_per_cell": 2, "require_exact_graph_batch_sizes": [8]},
            "statistics": {"bootstrap_resamples": 1000},
            "acceptance": {
                "primary_batch_size": 8,
                "minimum_throughput_gain_lower_95_percent": 0.0,
                "maximum_p95_latency_regression_percent": 5.0,
                "maximum_any_cell_median_latency_regression_percent": 10.0,
                "minimum_top64_overlap": 48,
                "maximum_common_logprob_error": 1.0,
            },
        }

    def rows(self, paged_ms: float = 8.0) -> list[dict]:
        rows = []
        for block in (1, 2):
            for batch in (1, 2, 8):
                for action, elapsed in (("direct", 10.0), ("paged", paged_ms)):
                    rows.append({
                        "block": block, "action": action, "batch_size": batch,
                        "context_tokens": 128, "wave_elapsed_ms": [elapsed, elapsed],
                        "output_token_ids": [[7]] * (batch * 2),
                        "top_logprobs": [[
                            {"id": token, "logprob": -float(token)} for token in range(64)
                        ]] * (batch * 2),
                        "cache_tokens": [128] * (batch * 2),
                        "paged_calls": 0 if action == "direct" else 2,
                        "paged_sequences": 0 if action == "direct" else batch * 2,
                        "paged_fallbacks": 0,
                        "cuda_dispatches": 0 if action == "direct" else 48,
                        "cuda_sequences": 0 if action == "direct" else 48 * batch,
                        "action_decisions": batch * 2,
                        "peak_gpu_memory_mib": 1000.0,
                    })
        return rows

    def test_plan_is_balanced_and_covers_full_matrix(self) -> None:
        plan = experiment_plan(self.protocol())
        self.assertEqual(len(plan), 12)
        self.assertEqual({row["action"] for row in plan}, {"direct", "paged"})
        self.assertEqual({(row["batch_size"], row["context_tokens"]) for row in plan}, {(1, 128), (2, 128), (8, 128)})
        self.assertNotEqual(plan[0]["action"], plan[6]["action"])

    def test_analyzer_requires_real_cuda_batch_and_promotes_clear_win(self) -> None:
        summary = analyze(self.protocol(), self.rows())
        self.assertTrue(summary["promotion_passed"])
        self.assertAlmostEqual(summary["primary_throughput_gain_percent"]["median"], 25.0)
        tampered = self.rows()
        tampered[-1]["cuda_sequences"] = 1
        with self.assertRaisesRegex(ValueError, "CUDA dispatch"):
            analyze(self.protocol(), tampered)

    def test_analyzer_rejects_distribution_divergence_even_when_top1_matches(self) -> None:
        rows = self.rows()
        for row in rows:
            row["top_logprobs"] = [[
                {"id": token, "logprob": -float(token)} for token in range(64)
            ]] * len(row["output_token_ids"])
        paged = next(row for row in rows if row["action"] == "paged")
        paged["top_logprobs"] = [[
            {"id": token, "logprob": -float(token)} for token in range(17, 81)
        ]] * len(paged["output_token_ids"])
        summary = analyze(self.protocol(), rows)
        self.assertFalse(summary["correctness"]["passed"])
        self.assertFalse(summary["promotion_passed"])

    def test_top1_mismatch_is_a_retained_negative_result(self) -> None:
        rows = self.rows()
        paged = next(row for row in rows if row["action"] == "paged")
        paged["output_token_ids"][0] = [8]
        summary = analyze(self.protocol(), rows)
        self.assertEqual(summary["correctness"]["output_token_matches"],
                         summary["correctness"]["output_token_comparisons"] - 1)
        self.assertFalse(summary["promotion_passed"])

    def test_analyzer_retains_negative_result(self) -> None:
        summary = analyze(self.protocol(), self.rows(paged_ms=12.0))
        self.assertFalse(summary["promotion_passed"])
        self.assertLess(summary["primary_throughput_gain_percent"]["median"], 0)

    def test_primary_p95_is_computed_from_raw_wave_samples(self) -> None:
        rows = self.rows()
        primary_paged = [
            row for row in rows
            if row["action"] == "paged" and row["batch_size"] == 8
        ]
        primary_paged[0]["wave_elapsed_ms"] = [8.0, 80.0]
        summary = analyze(self.protocol(), rows)
        self.assertAlmostEqual(summary["primary_direct_wave_latency_p95_ms"], 10.0)
        self.assertAlmostEqual(summary["primary_paged_wave_latency_p95_ms"], 69.2)
        self.assertAlmostEqual(summary["primary_p95_wave_latency_regression_percent"], 592.0)

    def test_secondary_scheduler_fragmentation_is_reported_not_mislabeled(self) -> None:
        rows = self.rows()
        paged_batch_two = next(row for row in rows if row["action"] == "paged" and row["batch_size"] == 2)
        paged_batch_two["paged_calls"] = 4
        paged_batch_two["cuda_dispatches"] = 96
        summary = analyze(self.protocol(), rows)
        cell = summary["per_cell"]["block-1-batch-2-context-128"]
        self.assertEqual(cell["realized_sequences_per_graph"], 1.0)

    def test_contiguous_fastpath_requires_upstream_route_and_zero_custom_cuda(self) -> None:
        protocol = self.protocol()
        protocol["service"] = {"paged_execution_mode": "contiguous_fastpath"}
        protocol["acceptance"]["minimum_throughput_gain_lower_95_percent"] = -5.0
        rows = self.rows(paged_ms=10.0)
        for row in rows:
            row["paged_contiguous_fastpath_calls"] = 0
            row["paged_contiguous_fastpath_sequences"] = 0
            if row["action"] == "paged":
                row["paged_contiguous_fastpath_calls"] = 2
                row["paged_contiguous_fastpath_sequences"] = row["batch_size"] * 2
                row["paged_calls"] = 0
                row["paged_sequences"] = 0
                row["cuda_dispatches"] = 0
                row["cuda_sequences"] = 0
        summary = analyze(protocol, rows)
        self.assertTrue(summary["promotion_passed"])
        self.assertEqual(
            summary["per_cell"]["block-1-batch-8-context-128"]["execution_route"],
            "upstream-contiguous-fastpath",
        )
        tampered = self.rows(paged_ms=10.0)
        for row in tampered:
            row["paged_contiguous_fastpath_calls"] = 0
            row["paged_contiguous_fastpath_sequences"] = 0
        with self.assertRaisesRegex(ValueError, "contiguous fast path"):
            analyze(protocol, tampered)

    def test_rederivation_rejects_modified_raw_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "raw").mkdir()
            (output / "raw/evidence.json").write_text("original", encoding="utf-8")
            (output / "summary.json").write_text("old summary", encoding="utf-8")
            files = {
                str(path.relative_to(output)).replace("\\", "/"):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                for path in output.rglob("*") if path.is_file()
            }
            (output / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")
            verify_rederivation_inputs(output)
            (output / "raw/evidence.json").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "immutable evidence"):
                verify_rederivation_inputs(output)


if __name__ == "__main__":
    unittest.main()
