from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llama_lab.final_outcome import (  # noqa: E402
    build_final_outcome,
    render_architecture_chart,
    render_final_outcome,
    render_illustrated_report,
    render_summary_chart,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outcome = build_final_outcome(ROOT)
    expected_json = json.dumps(outcome, ensure_ascii=False, indent=2) + "\n"
    expected_markdown = render_final_outcome(outcome)
    targets = {
        ROOT / "results" / "final-outcome.json": expected_json,
        ROOT / "docs" / "final-outcome.md": expected_markdown,
        ROOT / "docs" / "final-illustrated-report.md": render_illustrated_report(outcome),
        ROOT / "docs" / "assets" / "final-system-flow.svg": render_architecture_chart(),
        ROOT / "results" / "final-outcome-summary.svg": render_summary_chart(outcome),
    }
    if args.check:
        stale = [str(path.relative_to(ROOT)) for path, expected in targets.items()
                 if not path.is_file() or path.read_text(encoding="utf-8") != expected]
        if stale:
            raise SystemExit("stale final outcome: " + ", ".join(stale))
        print("Final outcome is evidence-bound and current.")
        return
    for path, content in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    print("Generated docs/final-outcome.md and results/final-outcome.json")


if __name__ == "__main__":
    main()
