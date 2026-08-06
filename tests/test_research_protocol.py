import csv
import json
import tempfile
import unittest
from pathlib import Path

from llama_lab.research_protocol import (
    ProtocolError,
    load_research_protocol,
    package_cpu_correctness_run,
    package_cuda_remap_trials,
    paired_bootstrap_summary,
)


ROOT = Path(__file__).resolve().parents[1]


class ResearchProtocolTests(unittest.TestCase):
    def test_repository_protocol_locks_timing_statistics_and_cpu_fallback(self) -> None:
        protocol = load_research_protocol(
            ROOT / "config" / "research_protocol.json",
            ROOT / "config" / "research_claims.json",
        )

        self.assertEqual(protocol["schema_version"], 1)
        self.assertEqual(
            set(protocol["timing"]),
            {"host_enqueue_ms", "synchronized_kernel_ms", "end_to_end_ms"},
        )
        self.assertEqual(protocol["outliers"]["outcome_based_deletion"], "forbidden")
        self.assertGreaterEqual(protocol["statistics"]["confidence_level"], 0.95)
        self.assertGreaterEqual(protocol["statistics"]["bootstrap_resamples"], 10000)
        self.assertIsInstance(
            protocol["statistics"]["gates"]["maximum_regression_percent"], float
        )
        self.assertTrue(protocol["workloads"]["h1_vector_remap"]["correctness_runner"])
        self.assertTrue(protocol["cpu_correctness_fallback"]["command"])

    def test_paired_bootstrap_uses_within_pair_effects(self) -> None:
        summary = paired_bootstrap_summary(
            [(10.0, 8.0), (20.0, 16.0), (30.0, 24.0), (40.0, 32.0)],
            confidence_level=0.95,
            resamples=2000,
            seed=7,
        )

        self.assertAlmostEqual(summary["median_improvement_percent"], 20.0)
        self.assertAlmostEqual(summary["ci_lower_percent"], 20.0)
        self.assertAlmostEqual(summary["ci_upper_percent"], 20.0)
        self.assertEqual(summary["pairs"], 4)

    def test_packager_emits_manifest_and_raw_records_with_three_timing_domains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "method",
                        "phase",
                        "blocks",
                        "trial",
                        "order_in_pair",
                        "random_seed",
                        "host_enqueue_ms",
                        "gpu_ms",
                        "end_to_end_ms",
                        "bytes",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "method": "scalar_gather_scatter",
                            "phase": "confirmatory",
                            "blocks": 1,
                            "trial": 0,
                            "order_in_pair": 0,
                            "random_seed": 20260806,
                            "host_enqueue_ms": 0.010,
                            "gpu_ms": 0.030,
                            "end_to_end_ms": 0.040,
                            "bytes": 1024,
                        },
                        {
                            "method": "vectorized_gather_scatter",
                            "phase": "confirmatory",
                            "blocks": 1,
                            "trial": 0,
                            "order_in_pair": 1,
                            "random_seed": 20260806,
                            "host_enqueue_ms": 0.008,
                            "gpu_ms": 0.015,
                            "end_to_end_ms": 0.025,
                            "bytes": 1024,
                        },
                    ]
                )
            environment = root / "environment.json"
            environment.write_text(json.dumps({"gpu": "test-gpu"}), encoding="utf-8")
            output = root / "run"

            artifacts = package_cuda_remap_trials(
                protocol_path=ROOT / "config" / "research_protocol.json",
                claims_path=ROOT / "config" / "research_claims.json",
                source_path=source,
                environment_path=environment,
                output_dir=output,
                command=["bench-kv-block-cuda.exe"],
                code_revision="a" * 40,
                captured_at_utc="2026-08-06T00:00:00Z",
                correctness_evidence={"passed": True, "command": ["oracle"]},
            )

            manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
            records = [
                json.loads(line)
                for line in artifacts.trials.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(manifest["raw_trial_count"], 2)
            self.assertEqual(manifest["code_revision"], "a" * 40)
            self.assertEqual({row["pair_id"] for row in records}, {"blocks-1-trial-0"})
            self.assertEqual(
                set(records[0]["timing_ms"]),
                {"host_enqueue_ms", "synchronized_kernel_ms", "end_to_end_ms"},
            )
            self.assertEqual(records[0]["order_in_pair"], 0)
            self.assertEqual(records[1]["order_in_pair"], 1)
            self.assertEqual(records[0]["random_seed"], 20260806)
            self.assertFalse(manifest["protocol_compliant"])
            self.assertTrue(manifest["acceptance"]["correctness"])

            repackaged = package_cuda_remap_trials(
                protocol_path=ROOT / "config" / "research_protocol.json",
                claims_path=ROOT / "config" / "research_claims.json",
                source_path=source,
                environment_path=environment,
                output_dir=root / "repackaged",
                command=["bench-kv-block-cuda.exe"],
                code_revision="a" * 40,
                captured_at_utc="2026-08-06T00:00:00Z",
            )
            repackaged_manifest = json.loads(
                repackaged.manifest.read_text(encoding="utf-8")
            )
            self.assertFalse(repackaged_manifest["acceptance"]["correctness"])
            self.assertIn(
                "independent correctness preflight did not pass",
                repackaged_manifest["violations"],
            )

    def test_packager_rejects_missing_enqueue_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text(
                "phase,method,blocks,trial,order_in_pair,random_seed,gpu_ms,end_to_end_ms,bytes\n"
                "confirmatory,scalar_gather_scatter,1,0,0,20260806,0.03,0.04,1024\n",
                encoding="utf-8",
            )
            environment = root / "environment.json"
            environment.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "host_enqueue_ms"):
                package_cuda_remap_trials(
                    ROOT / "config" / "research_protocol.json",
                    ROOT / "config" / "research_claims.json",
                    source,
                    environment,
                    root / "run",
                    ["bench"],
                    "a" * 40,
                    "2026-08-06T00:00:00Z",
                )

    def test_cpu_fallback_emits_correctness_evidence_without_cuda_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root / "environment.json"
            environment.write_text(json.dumps({"processor": "test-cpu"}), encoding="utf-8")
            artifacts = package_cpu_correctness_run(
                ROOT / "config" / "research_protocol.json",
                ROOT / "config" / "research_claims.json",
                environment,
                root / "run",
                ["python", "-m", "unittest"],
                "a" * 40,
                "2026-08-06T00:00:00Z",
                exit_code=0,
                elapsed_ms=12.5,
            )
            manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
            record = json.loads(artifacts.trials.read_text(encoding="utf-8"))
            self.assertTrue(record["correctness_passed"])
            self.assertIsNone(record["timing_ms"]["synchronized_kernel_ms"])
            self.assertFalse(manifest["performance_claims_allowed"])
            self.assertFalse(manifest["cuda_claims_allowed"])


if __name__ == "__main__":
    unittest.main()
