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

from run_production_paged_experiment import validate_artifact  # noqa: E402


class ProductionPagedEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = ROOT / "config/production_paged_protocol_v1.1.json"
        self.artifact = ROOT / "results/research/h7-production-paged-v1.1.0"

    @staticmethod
    def _rehash(artifact: Path, relative: str) -> None:
        manifest_path = artifact / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][relative] = hashlib.sha256(
            (artifact / relative).read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def test_repository_artifact_recomputes_all_conclusions(self) -> None:
        validate_artifact(self.protocol, self.artifact)

    def test_rehashed_false_promotion_claim_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "artifact"
            shutil.copytree(self.artifact, artifact)
            summary_path = artifact / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["promotion_passed"] = True
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            self._rehash(artifact, "summary.json")
            with self.assertRaisesRegex(AssertionError, "promotion_passed"):
                validate_artifact(self.protocol, artifact)

    def test_rehashed_mechanism_divergence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "artifact"
            shutil.copytree(self.artifact, artifact)
            mechanism_path = artifact / "mechanism.json"
            mechanism = json.loads(mechanism_path.read_text(encoding="utf-8"))
            mechanism["kernel_duration_ms"] += 1.0
            mechanism_path.write_text(json.dumps(mechanism, indent=2) + "\n", encoding="utf-8")
            self._rehash(artifact, "mechanism.json")
            with self.assertRaisesRegex(AssertionError, "mechanism kernel_duration_ms"):
                validate_artifact(self.protocol, artifact)

    def test_rehashed_trial_order_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "artifact"
            shutil.copytree(self.artifact, artifact)
            trials_path = artifact / "trials.json"
            trials = json.loads(trials_path.read_text(encoding="utf-8"))
            trials[0]["order_in_pair"] = 2
            trials_path.write_text(json.dumps(trials, indent=2) + "\n", encoding="utf-8")
            self._rehash(artifact, "trials.json")
            with self.assertRaisesRegex(AssertionError, "trial order"):
                validate_artifact(self.protocol, artifact)

    def test_rehashed_device_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "artifact"
            shutil.copytree(self.artifact, artifact)
            summary_path = artifact / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["device"]["name"] = "different GPU"
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            self._rehash(artifact, "summary.json")
            with self.assertRaisesRegex(AssertionError, "device"):
                validate_artifact(self.protocol, artifact)

    def test_rehashed_report_conclusion_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "artifact"
            shutil.copytree(self.artifact, artifact)
            report_path = artifact / "report.md"
            report = report_path.read_text(encoding="utf-8")
            report_path.write_text(
                report.replace("did not pass", "passed"), encoding="utf-8"
            )
            self._rehash(artifact, "report.md")
            with self.assertRaisesRegex(AssertionError, "report"):
                validate_artifact(self.protocol, artifact)

    def test_rehashed_raw_log_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "artifact"
            shutil.copytree(self.artifact, artifact)
            trials = json.loads((artifact / "trials.json").read_text(encoding="utf-8"))
            relative = str(trials[0]["log"])
            log_path = artifact / relative
            log_path.write_bytes(log_path.read_bytes() + b"\ntampered\n")
            self._rehash(artifact, relative)
            with self.assertRaisesRegex(AssertionError, "log_sha256"):
                validate_artifact(self.protocol, artifact)


if __name__ == "__main__":
    unittest.main()
