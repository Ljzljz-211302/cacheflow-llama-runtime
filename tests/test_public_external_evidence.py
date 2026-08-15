from __future__ import annotations

import unittest

from llama_lab.public_external_evidence import analyze_public_external


def protocol() -> dict:
    return {
        "protocol_version": "1.0.0", "random_seed": 7, "matched_process_blocks": 2,
        "trace_sources": ["burstgpt", "azure-code"],
        "statistics": {"bootstrap_resamples": 200},
        "acceptance": {
            "minimum_throughput_gain_lower_95_percent": -10.0,
            "maximum_p95_latency_regression_percent": 10.0,
            "require_exact_output_tokens": True,
            "maximum_quality_score_delta": 0.0,
        },
    }


def rows(paged_scale: float = 0.98) -> list[dict]:
    result = []
    for block in (1, 2):
        for trace in ("burstgpt", "azure-code"):
            direct_requests = [
                {"trace_row": index, "latency_ms": 100.0 + index,
                 "output_token_ids": [index, index + 10],
                 "top_logprobs": [{"id": index, "logprob": -0.1}], "cache_tokens": 128}
                for index in (1, 2)
            ]
            for action, scale in (("direct", 1.0), ("paged", paged_scale)):
                result.append({
                    "block": block, "trace_source": trace, "action": action,
                    "elapsed_ms": 200.0 * scale,
                    "requests": [{**request, "latency_ms": request["latency_ms"] * scale}
                                 for request in direct_requests],
                    "route": {
                        "paged_contiguous_fastpath_calls": 2 if action == "paged" else 0,
                        "paged_contiguous_fastpath_sequences": 2 if action == "paged" else 0,
                        "paged_calls": 0, "cuda_dispatches": 0, "paged_fallbacks": 0,
                    },
                    "quality": [
                        {"dataset": "triviaqa", "record_id": "q1", "score": 0.5,
                         "output_token_ids": [3, 4]},
                    ] if block == 1 else [],
                })
    return result


class PublicExternalEvidenceTest(unittest.TestCase):
    def test_analyzer_reconstructs_paired_trace_and_quality_gates(self) -> None:
        summary = analyze_public_external(protocol(), rows())
        self.assertTrue(summary["promotion_passed"])
        self.assertEqual(summary["correctness"]["token_matches"], 8)
        self.assertEqual(summary["quality"]["comparisons"], 1)
        self.assertEqual(summary["trace_sources"], ["azure-code", "burstgpt"])

    def test_analyzer_rejects_missing_cell(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete paired matrix"):
            analyze_public_external(protocol(), rows()[:-1])

    def test_analyzer_rejects_direct_paged_output_divergence(self) -> None:
        tampered = rows()
        next(row for row in tampered if row["action"] == "paged")["requests"][0]["output_token_ids"] = [99, 11]
        summary = analyze_public_external(protocol(), tampered)
        self.assertFalse(summary["correctness"]["passed"])
        self.assertFalse(summary["promotion_passed"])

    def test_analyzer_rejects_false_paged_route_evidence(self) -> None:
        tampered = rows()
        next(row for row in tampered if row["action"] == "paged")["route"]["paged_contiguous_fastpath_calls"] = 0
        with self.assertRaisesRegex(ValueError, "Paged route evidence"):
            analyze_public_external(protocol(), tampered)


if __name__ == "__main__":
    unittest.main()
