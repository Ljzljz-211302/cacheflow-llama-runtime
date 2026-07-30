from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llama_lab.report import render_report


if __name__ == "__main__":
    render_report(ROOT / "results", ROOT / "results/report.md")
    print(ROOT / "results/report.md")
