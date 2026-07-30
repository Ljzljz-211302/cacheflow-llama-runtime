from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llama_lab.advisor import ModelArchitecture, recommend_largest_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--available-mib", type=float, required=True)
    parser.add_argument("--slots", type=int, default=1)
    args = parser.parse_args()
    architecture = ModelArchitecture(24, 2, 64, 32768)
    estimate = recommend_largest_context(
        args.model.resolve(), architecture, args.available_mib, args.slots
    )
    if estimate is None:
        raise SystemExit("No candidate context fits the supplied budget")
    print(
        f"recommended_context={estimate.context} "
        f"estimated_total_mib={estimate.total_mib:.1f} "
        f"kv_cache_mib={estimate.kv_cache_mib:.1f}"
    )


if __name__ == "__main__":
    main()
