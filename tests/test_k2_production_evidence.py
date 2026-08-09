from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
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

    def test_formal_v2_2_artifact_recomputes(self) -> None:
        validate_artifact(
            ROOT / "config/k2_production_protocol_v2.2.json",
            ROOT / "results/research/h8-k2-production-v2.2.0",
        )

    def test_rehashed_false_v2_2_promotion_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "artifact"
            shutil.copytree(ROOT / "results/research/h8-k2-production-v2.2.0", artifact)
            summary_path = artifact / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["promotion_passed"] = False
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            manifest_path = artifact / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["summary.json"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "promotion_passed"):
                validate_artifact(ROOT / "config/k2_production_protocol_v2.2.json", artifact)

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
            variants = [
                ("k1", k1_client, k1_prompt),
                ("k2", k2_client, k2_prompt),
            ] if pair % 2 else [
                ("k2", k2_client, k2_prompt),
                ("k1", k1_client, k1_prompt),
            ]
            for order_in_pair, (variant, client, prompt) in enumerate(variants, 1):
                rows.append({
                    "pair": pair,
                    "order_in_pair": order_in_pair,
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
