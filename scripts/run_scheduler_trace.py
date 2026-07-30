from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llama_lab.scheduler_sim import generate_conversation_trace, simulate_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/scheduler_trace.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows: list[dict[str, float | str | int]] = []
    for scenario in config["scenarios"]:
        for seed in range(int(config["seeds"])):
            trace = generate_conversation_trace(
                requests=int(config["requests"]),
                revisit_probability=float(scenario["revisit_probability"]),
                active_conversations=int(scenario["active_conversations"]),
                system_tokens=int(config["system_tokens"]),
                turn_tokens_min=int(scenario["turn_tokens_min"]),
                turn_tokens_max=int(scenario["turn_tokens_max"]),
                seed=seed,
            )
            for policy in config["policies"]:
                metrics = simulate_trace(
                    trace,
                    slots_count=int(config["slots"]),
                    policy=policy["policy"],
                    eviction_penalty=float(policy["eviction_penalty"]),
                    system_tokens=int(config["system_tokens"]),
                )
                rows.append(
                    {
                        "scenario": scenario["name"],
                        "seed": seed,
                        "policy": policy["name"],
                        "eviction_penalty": policy["eviction_penalty"],
                        **metrics,
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "scheduler_trace_runs.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: list[dict[str, float | str | int]] = []
    for scenario in config["scenarios"]:
        for policy in config["policies"]:
            group = [
                row
                for row in rows
                if row["scenario"] == scenario["name"] and row["policy"] == policy["name"]
            ]
            summary.append(
                {
                    "scenario": scenario["name"],
                    "policy": policy["name"],
                    "seeds": len(group),
                    "prefill_tokens_total_median": statistics.median(
                        float(row["prefill_tokens_total"]) for row in group
                    ),
                    "prefill_tokens_p95_median": statistics.median(
                        float(row["prefill_tokens_p95"]) for row in group
                    ),
                    "cache_hit_ratio_median": statistics.median(
                        float(row["cache_hit_ratio"]) for row in group
                    ),
                    "evicted_tokens_total_median": statistics.median(
                        float(row["evicted_tokens_total"]) for row in group
                    ),
                }
            )
    with (args.output / "scheduler_trace_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {len(rows)} trace runs and {len(summary)} summaries")


if __name__ == "__main__":
    main()
