from __future__ import annotations

import copy
import hashlib
import unittest
from pathlib import Path

from llama_lab.objective_paged_benchmark import analyze, arm_plan, load_definition


ROOT = Path(__file__).resolve().parents[1]


def rows(protocol, corpus):
    result = []
    for pair, _, action in arm_plan(protocol):
        for workload in corpus["workloads"]:
            base = 10.0 + pair / 100
            result.append({
                "pair": pair,
                "action": action,
                "workload_id": workload["id"],
                "category": workload["category"],
                "prompt_sha256": hashlib.sha256(workload["prompt"].encode()).hexdigest(),
                "client_elapsed_ms": [base * (1.01 if action == "paged" else 1)] * 4,
                "actual_context_tokens": [17] * 4,
                "contents": [","] * 4,
                "paged_calls": 4 if action == "paged" else 0,
                "paged_fallbacks": 0,
            })
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

    def test_analysis_rejects_missing_workload_cell(self):
        evidence = rows(self.protocol, self.corpus)[:-1]
        with self.assertRaisesRegex(ValueError, "full pair/action/workload matrix"):
            analyze(self.protocol, self.corpus, evidence)


if __name__ == "__main__":
    unittest.main()
