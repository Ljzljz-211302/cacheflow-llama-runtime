from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llama_lab.server_bench import run_server_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/server_benchmark.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    rows = run_server_benchmark(args.config.resolve(), args.output.resolve())
    print(f"wrote {len(rows)} concurrency summaries")


if __name__ == "__main__":
    main()
