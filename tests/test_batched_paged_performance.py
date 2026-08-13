from __future__ import annotations

import unittest

from llama_lab.batched_paged_performance import analyze, experiment_plan


class BatchedPagedPerformanceTest(unittest.TestCase):
    def protocol(self) -> dict:
        return {
            "protocol_version": "1.0.0",
            "random_seed": 17,
            "matched_process_blocks": 2,
            "matrix": {"batch_sizes": [1, 8], "context_tokens": [128]},
            "measurement": {"waves_per_cell": 2},
            "statistics": {"bootstrap_resamples": 1000},
            "acceptance": {
                "primary_batch_size": 8,
                "minimum_throughput_gain_lower_95_percent": 0.0,
                "maximum_p95_latency_regression_percent": 5.0,
                "maximum_any_cell_median_latency_regression_percent": 10.0,
            },
        }

    def rows(self, paged_ms: float = 8.0) -> list[dict]:
        rows = []
        for block in (1, 2):
            for batch in (1, 8):
                for action, elapsed in (("direct", 10.0), ("paged", paged_ms)):
                    rows.append({
                        "block": block, "action": action, "batch_size": batch,
                        "context_tokens": 128, "wave_elapsed_ms": [elapsed, elapsed],
                        "request_elapsed_ms": [elapsed] * (batch * 2),
                        "output_token_ids": [[7]] * (batch * 2),
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
        self.assertEqual(len(plan), 8)
        self.assertEqual({row["action"] for row in plan}, {"direct", "paged"})
        self.assertEqual({(row["batch_size"], row["context_tokens"]) for row in plan}, {(1, 128), (8, 128)})
        self.assertNotEqual(plan[0]["action"], plan[4]["action"])

    def test_analyzer_requires_real_cuda_batch_and_promotes_clear_win(self) -> None:
        summary = analyze(self.protocol(), self.rows())
        self.assertTrue(summary["promotion_passed"])
        self.assertAlmostEqual(summary["primary_throughput_gain_percent"]["median"], 25.0)
        tampered = self.rows()
        tampered[-1]["cuda_sequences"] = 1
        with self.assertRaisesRegex(ValueError, "CUDA dispatch"):
            analyze(self.protocol(), tampered)

    def test_analyzer_retains_negative_result(self) -> None:
        summary = analyze(self.protocol(), self.rows(paged_ms=12.0))
        self.assertFalse(summary["promotion_passed"])
        self.assertLess(summary["primary_throughput_gain_percent"]["median"], 0)


if __name__ == "__main__":
    unittest.main()
