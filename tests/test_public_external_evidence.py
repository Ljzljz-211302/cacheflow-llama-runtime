from __future__ import annotations

import unittest

from llama_lab.public_external_evidence import analyze_public_external, validate_recorded_workloads


def protocol() -> dict:
    return {
        "protocol_version": "1.0.0", "random_seed": 7, "matched_process_blocks": 2,
        "trace_sources": ["burstgpt", "azure-code"], "request": {"top_probabilities": 1},
        "statistics": {"bootstrap_resamples": 200},
        "acceptance": {
            "minimum_throughput_gain_lower_95_percent": -10.0,
            "maximum_p95_latency_regression_percent": 10.0,
            "maximum_p95_latency_regression_upper_95_percent": 20.0,
            "maximum_arrival_slip_ms": 50.0,
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
                 "scheduled_arrival_ms": float(index * 10),
                 "actual_start_ms": float(index * 10 + 1),
                 "prompt_id": f"p{index}", "prompt_sha256": f"hash{index}",
                 "source_arrival_seconds": float(index), "actual_local_input_tokens": 128,
                 "output_token_ids": [index, index + 10],
                 "top_logprobs": [[{"id": index, "logprob": -0.1}],
                                   [{"id": index + 10, "logprob": -0.1}]],
                 "cache_tokens": 128}
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
                         "prediction": "alpha beta", "answers": ["alpha gamma"],
                         "prompt_sha256": "quality-hash", "output_token_ids": [3, 4]},
                    ] if block == 1 and trace == "burstgpt" else [],
                })
    return result


class PublicExternalEvidenceTest(unittest.TestCase):
    def test_analyzer_reconstructs_paired_trace_and_quality_gates(self) -> None:
        summary = analyze_public_external(protocol(), rows())
        self.assertTrue(summary["promotion_passed"])
        self.assertEqual(summary["correctness"]["token_matches"], 8)
        self.assertEqual(summary["quality"]["comparisons"], 1)
        self.assertEqual(summary["trace_sources"], ["azure-code", "burstgpt"])
        self.assertLessEqual(summary["maximum_arrival_slip_ms"], 50.0)
        self.assertIn("p95_latency_regression_block_bootstrap_95_percent", summary)

    def test_analyzer_rejects_missing_cell(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete paired matrix"):
            analyze_public_external(protocol(), rows()[:-1])

    def test_analyzer_rejects_direct_paged_output_divergence(self) -> None:
        tampered = rows()
        next(row for row in tampered if row["action"] == "paged")["requests"][0]["output_token_ids"] = [99, 11]
        summary = analyze_public_external(protocol(), tampered)
        self.assertFalse(summary["correctness"]["passed"])
        self.assertFalse(summary["promotion_passed"])

    def test_probability_gate_can_accept_near_tie_without_hiding_token_mismatch(self) -> None:
        relaxed = protocol()
        relaxed["acceptance"].update({
            "require_exact_output_tokens": False,
            "minimum_top_probability_overlap": 1,
            "maximum_common_logprob_error": 0.1,
        })
        tampered = rows()
        request = next(row for row in tampered if row["action"] == "paged")["requests"][0]
        request["output_token_ids"] = [99, 11]
        summary = analyze_public_external(relaxed, tampered)
        self.assertTrue(summary["correctness"]["passed"])
        self.assertLess(summary["correctness"]["token_matches"], summary["correctness"]["token_comparisons"])

    def test_analyzer_rejects_false_paged_route_evidence(self) -> None:
        tampered = rows()
        next(row for row in tampered if row["action"] == "paged")["route"]["paged_contiguous_fastpath_calls"] = 0
        with self.assertRaisesRegex(ValueError, "Paged route evidence"):
            analyze_public_external(protocol(), tampered)

    def test_arrival_slip_is_a_promotion_gate(self) -> None:
        tampered = rows()
        tampered[0]["requests"][0]["actual_start_ms"] = 1000.0
        summary = analyze_public_external(protocol(), tampered)
        self.assertGreater(summary["maximum_arrival_slip_ms"], 50.0)
        self.assertFalse(summary["promotion_passed"])

    def test_raw_rows_are_bound_to_frozen_workloads_and_quality_score(self) -> None:
        checked_protocol = protocol()
        checked_protocol["replay"] = {"target_arrival_span_seconds": 0.02}
        checked_protocol["acceptance"]["require_raw_workload_binding"] = True
        workloads = {
            "performance_replays": {
                source: [
                    {"trace_row": index, "prompt_id": f"p{index}",
                     "prompt_sha256": f"hash{index}", "actual_local_input_tokens": 128,
                     "source_arrival_seconds": float(index)}
                    for index in (1, 2)
                ] for source in ("burstgpt", "azure-code")
            },
            "quality_cases": [{"dataset": "triviaqa", "record_id": "q1",
                               "prompt_sha256": "quality-hash", "answers": ["alpha gamma"]}],
        }
        evidence = rows()
        validate_recorded_workloads(checked_protocol, workloads, evidence)
        evidence[0]["quality"][0]["score"] = 999.0
        with self.assertRaisesRegex(ValueError, "score was not reconstructed"):
            validate_recorded_workloads(checked_protocol, workloads, evidence)


if __name__ == "__main__":
    unittest.main()
