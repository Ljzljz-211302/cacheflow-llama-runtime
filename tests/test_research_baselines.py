import json
import tempfile
import unittest
from pathlib import Path

from llama_lab.research_baselines import BaselineManifestError, load_baseline_manifest


ROOT = Path(__file__).resolve().parents[1]


class ResearchBaselineTests(unittest.TestCase):
    def test_repository_manifest_locks_required_runnable_baselines(self) -> None:
        manifest = load_baseline_manifest(ROOT / "config" / "research_baselines.json")

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(
            manifest["upstream_revision"],
            "acd79d603cb2e1c84c0886137b80f1ad649b6857",
        )
        self.assertEqual(
            manifest["scope"]["gpu"], "NVIDIA GeForce RTX 4050 Laptop GPU"
        )

        baselines = {item["id"]: item for item in manifest["baselines"]}
        self.assertLessEqual(
            {"upstream", "direct-copy", "scalar-remap", "vector-remap", "fixed-rule"},
            baselines.keys(),
        )
        for baseline_id in (
            "upstream",
            "direct-copy",
            "scalar-remap",
            "vector-remap",
            "fixed-rule",
        ):
            baseline = baselines[baseline_id]
            self.assertEqual(baseline["comparison_class"], "quantitative")
            self.assertIs(baseline["runnable"], True)
            self.assertTrue(baseline["commands"])
            self.assertEqual(baseline["license"], "MIT")

    def test_related_work_cannot_masquerade_as_local_quantitative_baseline(self) -> None:
        manifest = {
            "schema_version": 1,
            "upstream_revision": "a" * 40,
            "scope": {"gpu": "test", "model": "test", "kv_layout": "test"},
            "baselines": [
                {
                    "id": "external-system",
                    "kind": "external",
                    "comparison_class": "quantitative",
                    "runnable": False,
                    "source": "https://example.invalid",
                    "license": "MIT",
                    "commands": [],
                    "comparability_limits": ["different runtime"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                BaselineManifestError, "quantitative baseline must be runnable"
            ):
                load_baseline_manifest(path)


if __name__ == "__main__":
    unittest.main()
