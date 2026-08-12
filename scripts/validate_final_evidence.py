from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llama_lab.final_outcome import build_final_outcome  # noqa: E402
from llama_lab.kv_action_evidence import validate_kv_action_artifact  # noqa: E402


def main() -> None:
    build_final_outcome(ROOT)
    validate_kv_action_artifact(
        ROOT / "results/research/h4-kv-action-v1.6.0",
        ROOT / "config/kv_action_policy_protocol.json",
    )
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_production_paged_experiment.py"),
        "--protocol", str(ROOT / "config/production_paged_protocol_v1.1.json"),
        "--output", str(ROOT / "results/research/h7-production-paged-v1.1.0"),
        "--validate-only",
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_objective_paged_benchmark.py"),
        "--protocol", str(ROOT / "config/production_paged_objective_protocol_v2.json"),
        "--output", str(ROOT / "results/research/h9-objective-paged-v2.0.0"),
        "--validate-only",
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_objective_paged_benchmark.py"),
        "--protocol", str(ROOT / "config/production_paged_objective_protocol_v4.json"),
        "--output", str(ROOT / "results/research/h10-long-context-paged-v4.0.0"),
        "--validate-only",
    ], cwd=ROOT, check=True)
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_k2_production_experiment.py"),
        "--protocol", str(ROOT / "config/k2_production_protocol_v2.10.json"),
        "--output", str(ROOT / "results/research/h8-k2-production-v2.10.0"),
        "--validate-only",
    ], cwd=ROOT, check=True)
    print("Final H1/H4/H7/H8/H9/H10 evidence closure passed.")


if __name__ == "__main__":
    main()
