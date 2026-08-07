import json
import shutil
import tempfile
import unittest
from pathlib import Path

from llama_lab.paged_decode_evidence import (
    analyze_paged_decode_trials,
    load_paged_decode_protocol,
    validate_paged_decode_artifact,
)
from llama_lab.research_protocol import file_sha256


class PagedDecodeEvidenceTests(unittest.TestCase):
    def test_repository_protocol_locks_model_shapes_and_claim_boundaries(self) -> None:
        protocol = load_paged_decode_protocol(Path("config/paged_decode_protocol.json"))
        self.assertEqual(protocol["shapes"]["qwen2.5-0.5b"]["head_dim"], 64)
        self.assertEqual(protocol["shapes"]["qwen2.5-7b-shape"]["head_dim"], 128)
        self.assertTrue(protocol["claim_rules"]["prototype_is_not_production_integration"])
        self.assertFalse(protocol["claim_rules"]["profiler_latency_is_primary"])

    def test_analysis_preserves_pairs_and_selects_split_k_from_registered_rule(self) -> None:
        protocol = load_paged_decode_protocol(Path("config/paged_decode_protocol.json"))
        rows = []
        for regime in protocol["regimes"]:
            for trial in range(20):
                baseline = 1.0
                regression = 12.0 if regime["id"].endswith("long-b1-fragmented") else 2.0
                paged = baseline * (1.0 + regression / 100.0)
                for method, value, order in (("contiguous", baseline, 0), ("paged", paged, 1)):
                    rows.append({
                        "phase": "confirmatory", "method": method,
                        "shape": regime["shape"], "context": regime["context"],
                        "batch": regime["batch"], "layout": regime["layout"],
                        "trial": trial, "order_in_pair": order,
                        "random_seed": 20260807 + regime["context"] * 17 + regime["batch"] * 101
                            + (1009 if regime["layout"] == "fragmented" else 0)
                            + (10007 if regime["shape"] == "qwen2.5-7b-shape" else 0),
                        "host_enqueue_ms": value, "gpu_ms": value,
                        "end_to_end_ms": value, "max_abs_error": 0.0,
                        "logical_kv_bytes": 4096,
                    })
        report = analyze_paged_decode_trials(rows, protocol)
        self.assertEqual(report["next_kernel_decision"]["selected"], "K3-split-KV")
        self.assertEqual(sum(len(item["raw_pair_ids"]) for item in report["regimes"]), 180)

    def test_committed_artifact_is_self_consistent(self) -> None:
        artifact = Path("results/research/h3-paged-decode-v1.0.0")
        if not artifact.exists():
            self.skipTest("formal artifact is created after implementation commit")
        validated = validate_paged_decode_artifact(
            artifact, Path("config/paged_decode_protocol.json")
        )
        self.assertEqual(len(validated["report"]["analysis"]["regimes"]), 9)

    def test_validator_rejects_semantic_trial_tamper_even_when_rehashed(self) -> None:
        source = Path("results/research/h3-paged-decode-v1.0.0")
        if not source.exists():
            self.skipTest("formal artifact is created after implementation commit")
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact"
            shutil.copytree(source, artifact)
            trials = artifact / "trials.jsonl"
            rows = [json.loads(line) for line in trials.read_text(encoding="utf-8").splitlines()]
            rows[0]["context"] += 1
            trials.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            manifest_path = artifact / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifact_hashes"]["trials.jsonl"] = file_sha256(trials)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "context"):
                validate_paged_decode_artifact(
                    artifact, Path("config/paged_decode_protocol.json")
                )

    def test_validator_rejects_unmanifested_profile_file(self) -> None:
        source = Path("results/research/h3-paged-decode-v1.0.0")
        if not source.exists():
            self.skipTest("formal artifact is created after implementation commit")
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact"
            shutil.copytree(source, artifact)
            (artifact / "profiles" / "untracked.txt").write_text("not evidence", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file tree"):
                validate_paged_decode_artifact(
                    artifact, Path("config/paged_decode_protocol.json")
                )


if __name__ == "__main__":
    unittest.main()
