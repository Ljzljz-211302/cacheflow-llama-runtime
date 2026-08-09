from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_k2_production_experiment import expected_summary, validate_artifact  # noqa: E402


class K2ProductionEvidenceTests(unittest.TestCase):
    def test_retained_v2_0_artifact_recomputes(self) -> None:
        validate_artifact(
            ROOT / "config/k2_production_protocol_v2.0.json",
            ROOT / "results/research/h8-k2-production-v2.0.0",
        )

    def test_retained_v2_1_artifact_recomputes(self) -> None:
        validate_artifact(
            ROOT / "config/k2_production_protocol_v2.1.json",
            ROOT / "results/research/h8-k2-production-v2.1.0",
        )

    def test_server_prompt_gate_does_not_hide_client_tail_regression(self) -> None:
        protocol = {
            "paired_trials": 2,
            "random_seed": 7,
            "acceptance": {
                "p95_maximum_regression_percent": 5.0,
                "paired_median_metric": "server_prompt_ms",
            },
        }
        rows = []
        for pair, k1_client, k2_client, k1_prompt, k2_prompt in [
            (1, 10.0, 20.0, 4.0, 3.0),
            (2, 10.0, 20.0, 4.0, 3.0),
        ]:
            for variant, client, prompt in [
                ("k1", k1_client, k1_prompt),
                ("k2", k2_client, k2_prompt),
            ]:
                rows.append({
                    "pair": pair,
                    "variant": variant,
                    "content": ",",
                    "client_elapsed_ms": client,
                    "prompt_ms": prompt,
                    "paged_calls": 1,
                    "paged_fallbacks": 0,
                })
        summary = expected_summary(protocol, rows)
        self.assertTrue(summary["paired_median_acceptance_passed"])
        self.assertFalse(summary["promotion_passed"])


if __name__ == "__main__":
    unittest.main()
