import csv
import json
import tempfile
import unittest
from pathlib import Path

from llama_lab.benefit_experiment import (
    BenefitSnapshot,
    LongLivedAcceptance,
    PhaseEvidence,
    evaluate_long_lived,
    validate_long_lived_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


def metrics(
    *,
    upstream: int,
    cacheflow: int,
    exploration: int,
    positive: int,
    drift: int,
    cooldown: int,
    safety: int = 0,
    predicted_benefit: float = 0.0,
    uncertainty: float = 0.0,
) -> str:
    return "\n".join(
        (
            f'llamacpp:benefit_decisions_total{{backend="cuda",action="upstream"}} {upstream}',
            f'llamacpp:benefit_decisions_total{{backend="cuda",action="cacheflow"}} {cacheflow}',
            f'llamacpp:benefit_exploration_total{{backend="cuda"}} {exploration}',
            f'llamacpp:benefit_reason_total{{backend="cuda",reason="positive_lower_bound"}} {positive}',
            f'llamacpp:benefit_drift_total{{backend="cuda"}} {drift}',
            f'llamacpp:benefit_safety_fallback_total{{backend="cuda"}} {safety}',
            f'llamacpp:benefit_cooldown_remaining{{backend="cuda"}} {cooldown}',
            f'llamacpp:benefit_predicted_benefit_ms{{backend="cuda"}} {predicted_benefit}',
            f'llamacpp:benefit_uncertainty_ms{{backend="cuda"}} {uncertainty}',
        )
    )


class BenefitSnapshotTests(unittest.TestCase):
    def test_parses_labeled_backend_metrics_and_deltas_counters_only(self) -> None:
        before = BenefitSnapshot.from_prometheus(
            metrics(
                upstream=5,
                cacheflow=2,
                exploration=2,
                positive=0,
                drift=0,
                cooldown=0,
            ),
            "cuda",
        )
        after = BenefitSnapshot.from_prometheus(
            metrics(
                upstream=9,
                cacheflow=8,
                exploration=5,
                positive=3,
                drift=1,
                cooldown=7,
                predicted_benefit=4.5,
                uncertainty=1.25,
            ),
            "cuda",
        )
        delta = after.delta(before)
        self.assertEqual(delta.upstream_decisions, 4)
        self.assertEqual(delta.cacheflow_decisions, 6)
        self.assertEqual(delta.positive_decisions, 3)
        self.assertEqual(delta.drift_events, 1)
        self.assertEqual(delta.cooldown_remaining, 7)
        self.assertEqual(delta.predicted_benefit_ms, 4.5)
        self.assertEqual(delta.uncertainty_ms, 1.25)

    def test_rejects_counter_regression(self) -> None:
        newer = BenefitSnapshot.from_prometheus(
            metrics(upstream=1, cacheflow=0, exploration=0, positive=0, drift=0, cooldown=0),
            "cuda",
        )
        older = BenefitSnapshot.from_prometheus(
            metrics(upstream=2, cacheflow=0, exploration=0, positive=0, drift=0, cooldown=0),
            "cuda",
        )
        with self.assertRaisesRegex(ValueError, "counter regressed"):
            newer.delta(older)


class LongLivedAcceptanceTests(unittest.TestCase):
    def test_repository_artifact_recomputes_terminal_acceptance(self) -> None:
        summary = validate_long_lived_artifact(
            ROOT / "results/long_lived_benefit_cuda_waves.csv",
            ROOT / "results/long_lived_benefit_cuda_summary.json",
        )
        stable = next(item for item in summary["phases"] if item["phase"] == "stable_reuse")
        self.assertGreaterEqual(stable["terminal_consecutive_positive_waves"], 3)
        self.assertTrue(summary["acceptance"]["passed"])

    def test_artifact_validator_rejects_wave_counter_tamper(self) -> None:
        source_csv = ROOT / "results/long_lived_benefit_cuda_waves.csv"
        source_summary = ROOT / "results/long_lived_benefit_cuda_summary.json"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with source_csv.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            stable_rows = [row for row in rows if row["phase"] == "stable_reuse"]
            stable_rows[-1]["positive_decisions"] = "0"
            csv_path = directory / "waves.csv"
            with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summary_path = directory / "summary.json"
            summary_path.write_text(source_summary.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "phase evidence"):
                validate_long_lived_artifact(csv_path, summary_path)

    def test_artifact_validator_rejects_copied_acceptance_tamper(self) -> None:
        source_csv = ROOT / "results/long_lived_benefit_cuda_waves.csv"
        source_summary = ROOT / "results/long_lived_benefit_cuda_summary.json"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            csv_path = directory / "waves.csv"
            csv_path.write_bytes(source_csv.read_bytes())
            summary = json.loads(source_summary.read_text(encoding="utf-8"))
            summary["acceptance"] = {"passed": False, "violations": ["fabricated"]}
            summary_path = directory / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "acceptance"):
                validate_long_lived_artifact(csv_path, summary_path)

    def test_accepts_learning_then_distribution_shift_fallback(self) -> None:
        phases = [
            PhaseEvidence("cold_start", 12, 0, 3, 0, 0, 0, 120.0, 0.0, 0.0),
            PhaseEvidence("stable_reuse", 24, 9, 1, 8, 0, 0, 78.0, 5.0, 1.2, 8, 5, 5),
            PhaseEvidence("distribution_shift", 15, 0, 0, 0, 1, 6, 82.0, -2.0, 4.0),
        ]
        result = evaluate_long_lived(phases, LongLivedAcceptance())
        self.assertTrue(result.passed)
        self.assertEqual(result.violations, ())

    def test_rejects_exploration_masquerading_as_convergence(self) -> None:
        phases = [
            PhaseEvidence("cold_start", 12, 0, 3, 0, 0, 0, 120.0, 0.0, 0.0),
            PhaseEvidence("stable_reuse", 24, 9, 9, 0, 0, 0, 78.0, 0.0, 5.0),
            PhaseEvidence("distribution_shift", 15, 0, 0, 0, 1, 4, 82.0, -1.0, 4.0),
        ]
        result = evaluate_long_lived(phases, LongLivedAcceptance())
        self.assertFalse(result.passed)
        self.assertTrue(any("positive-lower-bound" in item for item in result.violations))

    def test_rejects_shift_without_drift_or_safety_fallback(self) -> None:
        phases = [
            PhaseEvidence("cold_start", 12, 0, 3, 0, 0, 0, 120.0, 0.0, 0.0),
            PhaseEvidence("stable_reuse", 24, 9, 1, 8, 0, 0, 78.0, 5.0, 1.2, 8, 5, 5),
            PhaseEvidence("distribution_shift", 15, 3, 0, 0, 0, 2, 120.0, -1.0, 4.0),
        ]
        result = evaluate_long_lived(phases, LongLivedAcceptance())
        self.assertFalse(result.passed)
        self.assertTrue(any("continued" in item for item in result.violations))

    def test_rejects_non_persistent_confidence_spike(self) -> None:
        phases = [
            PhaseEvidence("cold_start", 12, 0, 0, 0, 0, 0, 120.0, 0.0, 0.0),
            PhaseEvidence("stable_reuse", 24, 5, 1, 4, 0, 0, 90.0, 5.0, 1.0, 4, 1),
            PhaseEvidence("distribution_shift", 12, 0, 0, 0, 0, 3, 100.0, 0.0, 2.0),
        ]
        result = evaluate_long_lived(phases, LongLivedAcceptance())
        self.assertFalse(result.passed)
        self.assertTrue(any("persistent" in item for item in result.violations))

    def test_rejects_convergence_that_is_not_present_in_terminal_waves(self) -> None:
        phases = [
            PhaseEvidence("cold_start", 12, 0, 0, 0, 0, 0, 120.0, 0.0, 0.0),
            PhaseEvidence("stable_reuse", 24, 8, 1, 7, 0, 0, 90.0, 5.0, 2.0, 7, 5, 0),
            PhaseEvidence("distribution_shift", 12, 0, 0, 0, 0, 3, 100.0, 0.0, 2.0),
        ]
        result = evaluate_long_lived(phases, LongLivedAcceptance())
        self.assertFalse(result.passed)
        self.assertTrue(any("terminal waves" in item for item in result.violations))

    def test_last_value_gauge_cannot_erase_terminal_contextual_confidence(self) -> None:
        phases = [
            PhaseEvidence("cold_start", 12, 0, 0, 0, 0, 0, 120.0, 0.0, 0.0),
            # The final wave has a positive contextual action, followed by an
            # unrelated upstream decision whose last-value gauge has no
            # positive margin. The action counter is the semantic evidence.
            PhaseEvidence("stable_reuse", 24, 8, 1, 7, 0, 0, 90.0, 1.0, 2.0, 7, 5, 3),
            PhaseEvidence("distribution_shift", 12, 0, 0, 0, 0, 3, 100.0, 0.0, 2.0),
        ]
        result = evaluate_long_lived(phases, LongLivedAcceptance())
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
