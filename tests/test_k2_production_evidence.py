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

from run_k2_production_experiment import (  # noqa: E402
    expected_summary,
    mechanism_acceptance,
    validate_artifact,
)


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

    def test_formal_v2_7_artifact_recomputes(self) -> None:
        validate_artifact(
            ROOT / "config/k2_production_protocol_v2.7.json",
            ROOT / "results/research/h8-k2-production-v2.7.0",
        )

    def test_rehashed_raw_response_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "artifact"
            shutil.copytree(ROOT / "results/research/h8-k2-production-v2.7.0", artifact)
            response_path = artifact / "raw/arm-002-k1/response.json"
            response = json.loads(response_path.read_text(encoding="utf-8"))
            response["timings"]["prompt_ms"] += 1.0
            response_path.write_text(
                json.dumps(response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            manifest_path = artifact / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["raw/arm-002-k1/response.json"] = hashlib.sha256(
                response_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "structured raw arms"):
                validate_artifact(ROOT / "config/k2_production_protocol_v2.7.json", artifact)

    def test_rehashed_chart_claim_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "artifact"
            shutil.copytree(ROOT / "results/research/h8-k2-production-v2.7.0", artifact)
            chart_path = artifact / "k2-production-comparison.svg"
            chart_path.write_text(
                chart_path.read_text(encoding="utf-8").replace("-47.90%", "-97.90%"),
                encoding="utf-8",
            )
            manifest_path = artifact / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["k2-production-comparison.svg"] = hashlib.sha256(
                chart_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "chart"):
                validate_artifact(ROOT / "config/k2_production_protocol_v2.7.json", artifact)

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

    def test_v2_5_requires_latency_noninferiority_and_mechanism_gain(self) -> None:
        protocol = {
            "paired_trials": 1,
            "random_seed": 3,
            "acceptance": {
                "client_median_maximum_regression_percent": 5.0,
                "p95_maximum_regression_percent": 5.0,
                "minimum_kernel_duration_reduction_percent": 20.0,
            },
        }
        rows = [
            {"pair": 1, "order_in_pair": 1, "variant": "k1", "content": ",",
             "client_elapsed_ms": 10.0, "prompt_ms": 4.0, "paged_calls": 1,
             "paged_fallbacks": 0},
            {"pair": 1, "order_in_pair": 2, "variant": "k2", "content": ",",
             "client_elapsed_ms": 10.4, "prompt_ms": 4.1, "paged_calls": 1,
             "paged_fallbacks": 0},
        ]
        latency = expected_summary(protocol, rows)
        mechanism = mechanism_acceptance(protocol, {
            "k1": {"kernel_duration_ms": 1.0, "kernel_launches": 24},
            "k2": {"kernel_duration_ms": 0.85, "kernel_launches": 24},
        })
        self.assertTrue(latency["promotion_passed"])
        self.assertFalse(mechanism["mechanism_acceptance_passed"])
        self.assertFalse(latency["promotion_passed"] and mechanism["mechanism_acceptance_passed"])


if __name__ == "__main__":
    unittest.main()
