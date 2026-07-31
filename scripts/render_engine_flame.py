from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "prepare": "#4c78a8",
    "plan": "#f58518",
    "execute": "#e45756",
    "commit": "#72b7b2",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "results/raw/engine-trace-cpu-cacheflow-trial-1.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/engine-flame.svg"
    )
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "results/engine-profile-summary.json"
    )
    args = parser.parse_args()
    events = json.loads(args.input.read_text(encoding="utf-8"))["traceEvents"]
    events = [event for event in events if event.get("ph") == "X" and event.get("dur", 0) > 0]
    if not events:
        raise RuntimeError("engine trace contains no complete spans")

    totals: dict[str, float] = defaultdict(float)
    for event in events:
        totals[event["name"]] += float(event["dur"])
    total = sum(totals.values())
    ordered = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    try:
        source = str(args.input.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        source = str(args.input)
    summary = {
        "source": source,
        "events": len(events),
        "profiled_us": total,
        "phases": [
            {"name": name, "duration_us": duration, "share": duration / total}
            for name, duration in ordered
        ],
        "method": "in-process Chrome trace; WPR sampled stacks require elevated SeSystemProfilePrivilege",
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    width, height = 1200, 240
    x, y, bar_height = 40.0, 90.0, 52.0
    usable = width - 80
    rects: list[str] = []
    for name, duration in ordered:
        bar_width = usable * duration / total
        color = COLORS.get(name, "#999999")
        label = f"{name}: {duration / 1000:.1f} ms ({duration / total * 100:.1f}%)"
        rects.append(
            f'<rect x="{x:.2f}" y="{y}" width="{bar_width:.2f}" height="{bar_height}" '
            f'fill="{color}" stroke="#ffffff"><title>{html.escape(label)}</title></rect>'
        )
        if bar_width > 100:
            rects.append(
                f'<text x="{x + 8:.2f}" y="{y + 32}" fill="white" font-size="16">'
                f'{html.escape(name)} {duration / total * 100:.1f}%</text>'
            )
        x += bar_width
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#111827"/>
<text x="40" y="38" fill="white" font-size="24" font-family="Segoe UI, sans-serif">CacheFlow Inference Engine Flame Chart</text>
<text x="40" y="64" fill="#cbd5e1" font-size="14" font-family="Segoe UI, sans-serif">prepare → plan → execute → commit, aggregated from {len(events)} production spans</text>
{''.join(rects)}
<text x="40" y="185" fill="#cbd5e1" font-size="13" font-family="Segoe UI, sans-serif">Hover a phase for exact duration. Raw trace is Chrome/Perfetto compatible.</text>
</svg>'''
    args.output.write_text(svg, encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
