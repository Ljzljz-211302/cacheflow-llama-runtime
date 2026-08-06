import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from llama_lab.cuda_profile_evidence import (
    build_ncu_command,
    build_nsys_command,
    characterize_regimes,
    parse_ncu_csv,
    parse_nsys_sqlite,
)


class CudaProfileEvidenceTests(unittest.TestCase):
    def test_parses_nsys_kernel_copy_and_synchronization_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "trace.sqlite"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
                INSERT INTO StringIds VALUES (1, 'llama_kv_remap_gather_vectorized');
                INSERT INTO StringIds VALUES (2, 'unrelated_kernel');
                INSERT INTO StringIds VALUES (3, 'cudaEventSynchronize');
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                    start INTEGER, end INTEGER, demangledName INTEGER
                );
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1000, 51000, 1);
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (60000, 90000, 2);
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
                    start INTEGER, end INTEGER, bytes INTEGER
                );
                INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (100, 1100, 4096);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                    start INTEGER, end INTEGER, nameId INTEGER
                );
                INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (900, 1900, 3);
                """
            )
            connection.commit()
            connection.close()

            parsed = parse_nsys_sqlite(database, kernel_patterns=("llama_kv_remap",))

        self.assertEqual(parsed["kernel_launches"], 1)
        self.assertAlmostEqual(parsed["kernel_duration_ms"], 0.05)
        self.assertEqual(parsed["memcpy_calls"], 1)
        self.assertEqual(parsed["memcpy_bytes"], 4096)
        self.assertEqual(parsed["synchronization_calls"], 1)

    def test_parses_ncu_raw_csv_without_treating_profiler_time_as_latency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ncu.csv"
            rows = [
                ["==PROF== Connected to process 42"],
                ["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"],
                ["1", "llama_kv_remap_gather_vectorized", "gpu__time_duration.sum", "ns", "50000"],
                ["1", "llama_kv_remap_gather_vectorized", "dram__bytes_read.sum", "byte", "8192"],
                ["1", "llama_kv_remap_gather_vectorized", "dram__bytes_write.sum", "byte", "4096"],
                ["1", "llama_kv_remap_gather_vectorized", "dram__throughput.avg.pct_of_peak_sustained_elapsed", "%", "62.5"],
                ["1", "llama_kv_remap_gather_vectorized", "lts__t_sector_hit_rate.pct", "%", "25"],
                ["1", "llama_kv_remap_gather_vectorized", "sm__warps_active.avg.pct_of_peak_sustained_active", "%", "50"],
                ["2", "unrelated_kernel", "gpu__time_duration.sum", "ns", "999999"],
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)

            parsed = parse_ncu_csv(path, kernel_patterns=("llama_kv_remap",))

        self.assertEqual(parsed["profiled_launches"], 1)
        self.assertAlmostEqual(parsed["profiled_kernel_duration_ms"], 0.05)
        self.assertEqual(parsed["dram_bytes"], 12288)
        self.assertEqual(parsed["dram_throughput_pct"], 62.5)
        self.assertEqual(parsed["l2_hit_rate_pct"], 25.0)
        self.assertEqual(parsed["achieved_occupancy_pct"], 50.0)
        self.assertNotIn("end_to_end_ms", parsed)

    def test_characterizes_three_regimes_and_preserves_neutral_counterexample(self) -> None:
        records: list[dict[str, object]] = []
        cases = [
            ("small", 1, 50.0),
            ("transition", 16, 5.0),
            ("large-neutral", 32, 0.0),
            ("misaligned-small", 1, -50.0),
        ]
        for regime_id, blocks, improvement in cases:
            for trial in range(6):
                pair_id = f"blocks-{blocks}-trial-{trial}"
                baseline = 1.0
                variant = baseline * (1.0 - improvement / 100.0)
                for method, value in (
                    ("scalar_gather_scatter", baseline),
                    ("vectorized_gather_scatter", variant),
                ):
                    records.append(
                        {
                            "regime_id": regime_id,
                            "phase": "confirmatory",
                            "blocks": blocks,
                            "pair_id": pair_id,
                            "method": method,
                            "timing_ms": {
                                "synchronized_kernel_ms": value,
                                "end_to_end_ms": value + 0.1,
                            },
                        }
                    )
        regimes = [
            {"id": "small", "blocks": 1},
            {"id": "transition", "blocks": 16},
            {"id": "large-neutral", "blocks": 32},
            {"id": "misaligned-small", "blocks": 1},
        ]
        profiler = {
            regime["id"]: {
                "nsys": {"kernel_launches": 12, "synchronization_calls": 6},
                "ncu": None,
            }
            for regime in regimes
        }

        report = characterize_regimes(
            records,
            regimes,
            profiler,
            confidence_level=0.95,
            bootstrap_resamples=1000,
            seed=7,
            material_improvement_percent=10.0,
            maximum_regression_percent=3.0,
        )

        self.assertEqual(len(report["regimes"]), 4)
        self.assertEqual(report["regimes"][0]["effect_class"], "material-win")
        self.assertEqual(report["regimes"][2]["effect_class"], "neutral")
        self.assertEqual(report["regimes"][3]["effect_class"], "material-loss")
        self.assertTrue(report["contains_neutral_or_loss"])
        self.assertEqual(report["regimes"][0]["mechanism_status"], "unresolved-without-ncu")
        self.assertNotIn("memory-bound", json.dumps(report))

    def test_repository_profile_protocol_locks_claim_boundaries(self) -> None:
        protocol = json.loads(
            Path("config/cuda_profile_protocol.json").read_text(encoding="utf-8")
        )
        self.assertGreaterEqual(len(protocol["regimes"]), 3)
        self.assertTrue(
            any(regime["layout"] == "misaligned" for regime in protocol["regimes"])
        )
        self.assertTrue(protocol["claim_rules"]["memory_bound_requires_ncu_dram_metric"])
        self.assertTrue(protocol["claim_rules"]["occupancy_claim_requires_ncu_occupancy_metric"])
        self.assertTrue(protocol["claim_rules"]["negative_and_neutral_results_are_retained"])

    def test_builds_reproducible_profiler_commands_with_separate_reports(self) -> None:
        executable = Path("bench-kv-block-cuda.exe")
        output = Path("profiles") / "small-vector"
        target = [str(executable), "--profile", "--blocks", "1", "--method", "vectorized"]

        nsys = build_nsys_command(Path("nsys"), output, target, platform_name="linux")
        ncu = build_ncu_command(Path("ncu"), output, target)

        self.assertIn("--trace=cuda,nvtx,osrt", nsys)
        self.assertIn("--capture-range=cudaProfilerApi", nsys)
        self.assertIn("--page=raw", ncu)
        self.assertIn("--csv", ncu)
        self.assertIn("--kernel-name-base=demangled", ncu)
        self.assertEqual(nsys[-len(target):], target)
        self.assertEqual(ncu[-len(target):], target)


if __name__ == "__main__":
    unittest.main()
