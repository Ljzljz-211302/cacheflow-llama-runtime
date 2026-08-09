from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from llama_lab.kv_action_evidence import (
    audit_kv_action_policy_no_cuda_sync,
    evaluate_kv_action_models,
    load_kv_action_protocol,
    make_balanced_action_orders,
    validate_kv_action_rows,
    validate_kv_action_artifact,
)


def rows() -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    order = 0
    for split, count in (("train", 80), ("evaluation", 40)):
        for trace in range(count):
            order += 1
            trace_id = f"{split}-{trace}"
            for regime in ("resident", "preempted"):
                snapshot = f"{trace_id}-{regime}"
                for action, observed, analytical in (
                    ("device_swap", 8.0, 5.0),
                    ("recompute", 2.0, 10.0),
                ):
                    model_features = [
                        1.0, 0.25, 1.0 / 512.0, 0.03125, 0.75, 0.05, 0.85, 0.3, 0.026
                    ]
                    output.append({
                        "snapshot_id": snapshot,
                        "trace_id": trace_id,
                        "session_id": f"session-{trace_id}",
                        "prefix_family": f"family-{trace_id}",
                        "split": split,
                        "timestamp_order": order,
                        "backend": "cuda",
                        "regime": regime,
                        "action": action,
                        "baseline_action": "device_swap",
                        "context_tokens": 1024,
                        "batch": 1,
                        "page_runs": 64,
                        "kv_pressure": 0.85,
                        "kv_bytes": 32 * 1024 * 1024,
                        "reuse_distance": 3,
                        "analytical_cost_ms": analytical,
                        "observed_cost_ms": observed,
                        "observation_id": f"observation-{len(output)}",
                        "observation_order": len(output),
                        "prompt_tokens": 1024,
                        "runtime_model_features": model_features,
                        "action_runtime_model_features": model_features,
                        **{f"model_feature_{index}": value for index, value in enumerate(
                            model_features
                        )},
                    })
    return output


def phase_conditioned_delta_rows() -> list[dict[str, object]]:
    output = rows()
    trace_offsets: dict[str, float] = {}
    for row in output:
        trace = str(row["trace_id"])
        trace_offsets.setdefault(trace, 20.0 + 3.0 * len(trace_offsets))
        baseline_cost = trace_offsets[trace]
        if row["action"] == "device_swap":
            row["observed_cost_ms"] = baseline_cost
        elif row["regime"] == "resident":
            row["observed_cost_ms"] = baseline_cost + 2.0
        else:
            row["observed_cost_ms"] = baseline_cost - 2.0
    return output


def interaction_delta_rows() -> list[dict[str, object]]:
    output = phase_conditioned_delta_rows()
    for row in output:
        trace_number = int(str(row["trace_id"]).rsplit("-", 1)[1])
        cached = float((trace_number % 4) // 2)
        prefill = float(trace_number % 2)
        features = list(row["runtime_model_features"])
        features[1] = cached
        features[7] = prefill
        row["runtime_model_features"] = features
        row["action_runtime_model_features"] = list(features)
        row["model_feature_1"] = cached
        row["model_feature_7"] = prefill
        if row["regime"] == "preempted" and row["action"] == "recompute":
            baseline = next(
                candidate for candidate in output
                if candidate["snapshot_id"] == row["snapshot_id"]
                and candidate["action"] == "device_swap"
            )
            delta = -2.0 if cached != prefill else 2.0
            row["observed_cost_ms"] = float(baseline["observed_cost_ms"]) + delta
    return output


class KvActionEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_kv_action_protocol(Path("config/kv_action_policy_protocol.json"))

    def test_same_protocol_distinguishes_model_regret(self) -> None:
        report = evaluate_kv_action_models(rows(), self.protocol)
        self.assertEqual(report["train_traces"], 80)
        self.assertEqual(report["evaluation_traces"], 40)
        self.assertEqual(report["models"]["H0"]["median_regret_ms"], 6.0)
        self.assertEqual(report["models"]["A1"]["median_regret_ms"], 6.0)
        self.assertEqual(report["models"]["T1"]["median_regret_ms"], 0.0)
        self.assertEqual(report["models"]["L1"]["median_regret_ms"], 0.0)
        self.assertEqual(
            report["models"]["L1"]["paired_trace_cluster_mean_regret_delta_vs_h0_ci95_ms"],
            [-6.0, -6.0],
        )

    def test_paired_delta_model_learns_separate_regime_boundaries(self) -> None:
        report = evaluate_kv_action_models(phase_conditioned_delta_rows(), self.protocol)
        decisions = report["decisions"]["D1"]
        resident = [row for row in decisions if row["regime"] == "resident"]
        preempted = [row for row in decisions if row["regime"] == "preempted"]
        self.assertEqual({row["chosen"] for row in resident}, {"device_swap"})
        self.assertEqual({row["chosen"] for row in preempted}, {"recompute"})
        self.assertEqual(report["models"]["D1"]["cumulative_regret_ms"], 0.0)
        self.assertTrue(all(row["reason"] == "negative_upper_bound" for row in preempted))
        self.assertTrue(all(
            float(row["upper_delta_ms"]) + float(row["switch_margin_ms"]) < 0.0
            for row in preempted
        ))
        self.assertTrue(all(row["reason"] == "nonnegative_upper_bound" for row in resident))
        self.assertTrue(report["paired_delta"]["acceptance"]["passed"])

    def test_paired_delta_calibration_blocks_an_optimistic_switch(self) -> None:
        action_rows = phase_conditioned_delta_rows()
        for row in action_rows:
            if (
                row["split"] == "train"
                and int(row["timestamp_order"]) > 19
                and row["regime"] == "preempted"
                and row["action"] == "recompute"
            ):
                baseline = next(
                    candidate for candidate in action_rows
                    if candidate["snapshot_id"] == row["snapshot_id"]
                    and candidate["action"] == "device_swap"
                )
                row["observed_cost_ms"] = float(baseline["observed_cost_ms"]) + 10.0
        protocol = copy.deepcopy(self.protocol)
        protocol["paired_delta"]["confidence_beta"] = 0.0
        report = evaluate_kv_action_models(action_rows, protocol)
        preempted = [
            row for row in report["decisions"]["D1"] if row["regime"] == "preempted"
        ]
        self.assertEqual({row["chosen"] for row in preempted}, {"device_swap"})

    def test_paired_delta_model_uses_preregistered_interaction_features(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["paired_delta"]["ridge_lambda"] = 1e-6
        protocol["paired_delta"]["confidence_beta"] = 0.0
        protocol["paired_delta"]["switch_margin_ms"] = 0.0
        report = evaluate_kv_action_models(interaction_delta_rows(), protocol)
        preempted = [
            row for row in report["decisions"]["D1"] if row["regime"] == "preempted"
        ]
        self.assertTrue(all(row["chosen"] == row["oracle"] for row in preempted))
        self.assertIn(
            "cached_tokens_x_prefill_cost",
            report["paired_delta"]["feature_names"],
        )
        self.assertLess(
            report["models"]["D1"]["cumulative_regret_ms"],
            report["ablations"]["D1-I0-no-interactions"]["cumulative_regret_ms"],
        )

    def test_trace_leakage_is_rejected(self) -> None:
        tampered = copy.deepcopy(rows())
        tampered[-1]["trace_id"] = tampered[0]["trace_id"]
        with self.assertRaisesRegex(ValueError, "trace_id leaks"):
            validate_kv_action_rows(tampered, self.protocol)

    def test_incomplete_trace_cluster_is_rejected(self) -> None:
        tampered = copy.deepcopy(rows())
        source_trace = "evaluation-0"
        target_trace = "evaluation-1"
        for row in tampered:
            if row["trace_id"] == source_trace and row["regime"] == "resident":
                row["trace_id"] = target_trace
        with self.assertRaisesRegex(ValueError, "trace cluster is incomplete"):
            validate_kv_action_rows(tampered, self.protocol)

    def test_paired_feature_drift_is_rejected(self) -> None:
        tampered = copy.deepcopy(rows())
        tampered[1]["kv_bytes"] = int(tampered[1]["kv_bytes"]) + 1
        with self.assertRaisesRegex(ValueError, "changes paired features"):
            validate_kv_action_rows(tampered, self.protocol)

    def test_runtime_model_feature_drift_is_rejected(self) -> None:
        tampered = copy.deepcopy(rows())
        tampered[0]["runtime_model_features"][0] = 2.0
        with self.assertRaisesRegex(ValueError, "differ from runtime evidence"):
            validate_kv_action_rows(tampered, self.protocol)

    def test_canonical_features_must_match_runtime_h0_anchor(self) -> None:
        tampered = copy.deepcopy(rows())
        baseline = next(row for row in tampered if row["snapshot_id"] == tampered[0]["snapshot_id"])
        baseline["action_runtime_model_features"] = list(
            baseline["action_runtime_model_features"]
        )
        baseline["action_runtime_model_features"][0] = 2.0
        with self.assertRaisesRegex(ValueError, "runtime H0 anchor"):
            validate_kv_action_rows(tampered, self.protocol)

    def test_action_feature_divergence_gate_is_enforced(self) -> None:
        tampered = copy.deepcopy(rows())
        tampered[1]["action_runtime_model_features"] = list(
            tampered[1]["action_runtime_model_features"]
        )
        tampered[1]["action_runtime_model_features"][1] = 5.0
        with self.assertRaisesRegex(ValueError, "exceeds divergence gate"):
            validate_kv_action_rows(tampered, self.protocol)

    def test_policy_sync_audit_is_enforceable(self) -> None:
        source = Path("vendor/llama.cpp/tools/server/server-kv-action-policy.cpp")
        audit = audit_kv_action_policy_no_cuda_sync(source)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["matches"], [])

    def test_collection_orders_are_seeded_and_position_balanced(self) -> None:
        actions = self.protocol["production_actions"]
        orders = make_balanced_action_orders(
            actions, 40, self.protocol["confirmatory"]["order_seed"]
        )
        self.assertEqual(orders, make_balanced_action_orders(actions, 40, 6042029))
        for position in range(len(actions)):
            self.assertEqual(
                {action: sum(order[position] == action for order in orders) for action in actions},
                {action: 10 for action in actions},
            )

    def test_bootstrap_decisions_retain_trace_cluster_identity(self) -> None:
        report = evaluate_kv_action_models(rows(), self.protocol)
        for decisions in report["decisions"].values():
            grouped: dict[str, set[str]] = {}
            for decision in decisions:
                grouped.setdefault(decision["trace_id"], set()).add(decision["regime"])
            self.assertTrue(all(regimes == {"resident", "preempted"} for regimes in grouped.values()))

    def test_formal_artifact_recomputes_from_hashed_trials(self) -> None:
        artifact = Path("results/research/h4-kv-action-v1.4.0")
        if not (artifact / "manifest.json").exists():
            self.skipTest("formal H4 artifact has not been generated")
        report = validate_kv_action_artifact(
            artifact, Path("config/kv_action_policy_protocol.json")
        )
        self.assertTrue(report["overhead"]["passed"])

    def _copy_formal_artifact(self, destination: Path) -> Path:
        source = Path("results/research/h4-kv-action-v1.4.0")
        if not (source / "manifest.json").exists():
            self.skipTest("formal H4 artifact has not been generated")
        artifact = destination / source.name
        shutil.copytree(source, artifact)
        return artifact

    @staticmethod
    def _rehash(artifact: Path, relative: str) -> None:
        manifest_path = artifact / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][relative] = hashlib.sha256(
            (artifact / relative).read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def test_formal_artifact_rejects_rehashed_trial_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self._copy_formal_artifact(Path(temporary))
            rows_path = artifact / "paired-actions.jsonl"
            rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
            evaluation_index = next(
                index for index, row in enumerate(rows) if row["split"] == "evaluation"
            )
            rows[evaluation_index]["observed_cost_ms"] = (
                float(rows[evaluation_index]["observed_cost_ms"]) + 1000.0
            )
            rows_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            self._rehash(artifact, "paired-actions.jsonl")
            with self.assertRaisesRegex(ValueError, "raw observation|ratio differs|analysis differs"):
                validate_kv_action_artifact(artifact, Path("config/kv_action_policy_protocol.json"))

    def test_formal_artifact_rejects_rehashed_raw_metric_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self._copy_formal_artifact(Path(temporary))
            evidence_path = artifact / "runtime-evidence.jsonl"
            evidence = [
                json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines()
            ]
            evidence[0]["after_metrics"] = evidence[0]["after_metrics"].replace(
                'llamacpp:kv_action_last_model_feature{index="0"} 1',
                'llamacpp:kv_action_last_model_feature{index="0"} 2',
            )
            evidence_path.write_text(
                "\n".join(json.dumps(row) for row in evidence) + "\n", encoding="utf-8"
            )
            self._rehash(artifact, "runtime-evidence.jsonl")
            with self.assertRaisesRegex(ValueError, "raw runtime feature vector differs"):
                validate_kv_action_artifact(artifact, Path("config/kv_action_policy_protocol.json"))

    def test_formal_artifact_rejects_rehashed_overhead_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self._copy_formal_artifact(Path(temporary))
            overhead_path = artifact / "overhead.jsonl"
            rows = [json.loads(line) for line in overhead_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["p99_ns"] = int(rows[0]["p99_ns"]) + 1000
            overhead_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            self._rehash(artifact, "overhead.jsonl")
            with self.assertRaisesRegex(ValueError, "overhead gates differ"):
                validate_kv_action_artifact(artifact, Path("config/kv_action_policy_protocol.json"))

    def test_formal_artifact_rejects_rehashed_collection_order_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self._copy_formal_artifact(Path(temporary))
            rows_path = artifact / "paired-actions.jsonl"
            action_rows = [
                json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()
            ]
            action_rows[0]["collection_order"] = 99
            rows_path.write_text(
                "\n".join(json.dumps(row) for row in action_rows) + "\n", encoding="utf-8"
            )
            self._rehash(artifact, "paired-actions.jsonl")
            with self.assertRaisesRegex(ValueError, "collection order"):
                validate_kv_action_artifact(artifact, Path("config/kv_action_policy_protocol.json"))

    def test_formal_artifact_rejects_rehashed_sync_audit_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self._copy_formal_artifact(Path(temporary))
            audit_path = artifact / "sync-audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["source_sha256"] = "0" * 64
            audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            self._rehash(artifact, "sync-audit.json")
            with self.assertRaisesRegex(ValueError, "synchronization audit differs"):
                validate_kv_action_artifact(artifact, Path("config/kv_action_policy_protocol.json"))

    def test_formal_artifact_rejects_unmanifested_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self._copy_formal_artifact(Path(temporary))
            (artifact / "untracked.txt").write_text("not provenance-bound\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file tree differs"):
                validate_kv_action_artifact(artifact, Path("config/kv_action_policy_protocol.json"))

    def test_formal_artifact_rejects_unbound_replay_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self._copy_formal_artifact(Path(temporary))
            manifest_path = artifact / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["patch_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "replay-patch hash differs"):
                validate_kv_action_artifact(artifact, Path("config/kv_action_policy_protocol.json"))


if __name__ == "__main__":
    unittest.main()
