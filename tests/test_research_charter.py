import json
import tempfile
import unittest
from pathlib import Path

from llama_lab.research_charter import CharterError, load_research_charter


ROOT = Path(__file__).resolve().parents[1]


class ResearchCharterTests(unittest.TestCase):
    def test_repository_charter_makes_every_claim_falsifiable(self) -> None:
        charter = load_research_charter(
            ROOT / "config" / "research_claims.json",
            ROOT / "config" / "research_baselines.json",
        )

        self.assertEqual(charter["schema_version"], 1)
        self.assertGreaterEqual(len(charter["claims"]), 4)
        statuses = {claim["status"] for claim in charter["claims"]}
        self.assertLessEqual(
            statuses, {"prospective", "existing-evidence", "limited-evidence"}
        )
        self.assertIn("existing-evidence", statuses)
        self.assertIn("limited-evidence", statuses)
        for claim in charter["claims"]:
            self.assertTrue(claim["independent_variables"])
            self.assertTrue(claim["dependent_metrics"])
            self.assertTrue(claim["confounders"])
            self.assertTrue(claim["baselines"])
            self.assertTrue(claim["mechanism"])
            self.assertTrue(claim["falsification"])
            self.assertTrue(claim["evidence_sources"])
            self.assertTrue(claim["scope_limits"])

    def test_claim_without_falsification_is_rejected(self) -> None:
        claim = {
            "id": "H-bad",
            "question": "Is the new path faster?",
            "hypothesis": "It is always faster.",
            "status": "prospective",
            "independent_variables": ["path"],
            "dependent_metrics": ["latency"],
            "confounders": ["GPU clocks"],
            "baselines": ["upstream"],
            "mechanism": "Unknown",
            "falsification": [],
            "evidence_sources": ["future experiment"],
            "observed_results": [],
            "scope_limits": ["single GPU"],
            "negative_result_policy": "retain",
        }
        payload = {"schema_version": 1, "charter_version": "1.0.0", "claims": [claim]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "claims.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(CharterError, "falsification"):
                load_research_charter(path)


if __name__ == "__main__":
    unittest.main()
