from __future__ import annotations

import csv
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from llama_lab.research_protocol import paired_bootstrap_summary


NCU_METRICS = (
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
)


def build_nsys_command(
    executable: Path,
    output_prefix: Path,
    target_command: list[str],
    *,
    platform_name: str | None = None,
) -> list[str]:
    platform_name = platform_name or sys.platform
    trace = "cuda,nvtx" if platform_name.startswith("win") else "cuda,nvtx,osrt"
    return [
        str(executable),
        "profile",
        f"--trace={trace}",
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop-shutdown",
        "--force-overwrite=true",
        f"--output={output_prefix}",
        *target_command,
    ]


def build_ncu_command(
    executable: Path, output_prefix: Path, target_command: list[str]
) -> list[str]:
    return [
        str(executable),
        "--target-processes=all",
        "--profile-from-start=off",
        "--kernel-name-base=demangled",
        "--kernel-name=regex:llama_kv_remap_.*",
        "--replay-mode=kernel",
        "--cache-control=all",
        "--clock-control=base",
        f"--metrics={','.join(NCU_METRICS)}",
        "--page=raw",
        "--csv",
        f"--log-file={output_prefix}.csv",
        f"--export={output_prefix}",
        "--force-overwrite",
        *target_command,
    ]


def _table_names(connection: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        )
    ]


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    escaped = table.replace('"', '""')
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{escaped}")')
    }


def _quoted(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _string_ids(connection: sqlite3.Connection, tables: Iterable[str]) -> dict[int, str]:
    if "StringIds" not in tables:
        return {}
    columns = _columns(connection, "StringIds")
    if not {"id", "value"}.issubset(columns):
        return {}
    return {
        int(identifier): str(value)
        for identifier, value in connection.execute("SELECT id, value FROM StringIds")
    }


def _resolve_name(value: object, strings: dict[int, str]) -> str:
    if isinstance(value, int):
        return strings.get(value, str(value))
    if isinstance(value, str) and value.isdigit():
        return strings.get(int(value), value)
    return "" if value is None else str(value)


def _duration_rows(
    connection: sqlite3.Connection,
    table: str,
    selected_columns: list[str],
) -> list[tuple[Any, ...]]:
    query = ", ".join(_quoted(column) for column in selected_columns)
    return list(connection.execute(f"SELECT {query} FROM {_quoted(table)}"))


def parse_nsys_sqlite(
    path: Path, *, kernel_patterns: tuple[str, ...]
) -> dict[str, int | float | list[str]]:
    """Extract only causal timeline facts from an exported Nsight Systems SQLite DB."""
    connection = sqlite3.connect(path)
    try:
        tables = _table_names(connection)
        strings = _string_ids(connection, tables)
        kernel_tables = [
            table for table in tables if table.startswith("CUPTI_ACTIVITY_KIND_KERNEL")
        ]
        kernel_count = 0
        kernel_ns = 0
        kernel_names: set[str] = set()
        for table in kernel_tables:
            columns = _columns(connection, table)
            name_column = next(
                (
                    name
                    for name in ("demangledName", "shortName", "name")
                    if name in columns
                ),
                None,
            )
            if not name_column or not {"start", "end"}.issubset(columns):
                continue
            for start, end, raw_name in _duration_rows(
                connection, table, ["start", "end", name_column]
            ):
                name = _resolve_name(raw_name, strings)
                if not any(pattern in name for pattern in kernel_patterns):
                    continue
                kernel_count += 1
                kernel_ns += max(0, int(end) - int(start))
                kernel_names.add(name)

        memcpy_calls = 0
        memcpy_ns = 0
        memcpy_bytes = 0
        for table in (
            table for table in tables if table.startswith("CUPTI_ACTIVITY_KIND_MEMCPY")
        ):
            columns = _columns(connection, table)
            if not {"start", "end"}.issubset(columns):
                continue
            byte_column = next(
                (name for name in ("bytes", "size", "copySize") if name in columns),
                None,
            )
            selected = ["start", "end"] + ([byte_column] if byte_column else [])
            for row in _duration_rows(connection, table, selected):
                memcpy_calls += 1
                memcpy_ns += max(0, int(row[1]) - int(row[0]))
                if byte_column:
                    memcpy_bytes += int(row[2])

        synchronization_calls = 0
        synchronization_ns = 0
        runtime_tables = [
            table for table in tables if table.startswith("CUPTI_ACTIVITY_KIND_RUNTIME")
        ]
        for table in runtime_tables:
            columns = _columns(connection, table)
            name_column = next(
                (name for name in ("nameId", "name") if name in columns), None
            )
            if not name_column or not {"start", "end"}.issubset(columns):
                continue
            for start, end, raw_name in _duration_rows(
                connection, table, ["start", "end", name_column]
            ):
                name = _resolve_name(raw_name, strings).lower()
                if "synchroniz" not in name or not name.startswith("cuda"):
                    continue
                synchronization_calls += 1
                synchronization_ns += max(0, int(end) - int(start))
        return {
            "kernel_launches": kernel_count,
            "kernel_duration_ms": kernel_ns / 1_000_000.0,
            "kernel_names": sorted(kernel_names),
            "memcpy_calls": memcpy_calls,
            "memcpy_duration_ms": memcpy_ns / 1_000_000.0,
            "memcpy_bytes": memcpy_bytes,
            "synchronization_calls": synchronization_calls,
            "synchronization_duration_ms": synchronization_ns / 1_000_000.0,
        }
    finally:
        connection.close()


def _numeric(value: str) -> float:
    cleaned = value.strip().replace(",", "").replace(" ", "")
    if cleaned in {"", "n/a", "N/A", "nan", "-"}:
        raise ValueError(f"metric value is not numeric: {value!r}")
    return float(cleaned)


def _scaled(value: float, unit: str, kind: str) -> float:
    normalized = unit.strip().lower()
    if kind == "duration_ns":
        scale = {"ns": 1.0, "us": 1_000.0, "µs": 1_000.0, "ms": 1_000_000.0,
                 "s": 1_000_000_000.0}.get(normalized)
    else:
        scale = {"byte": 1.0, "bytes": 1.0, "kbyte": 1_000.0, "kb": 1_000.0,
                 "mbyte": 1_000_000.0, "mb": 1_000_000.0,
                 "gbyte": 1_000_000_000.0, "gb": 1_000_000_000.0}.get(normalized)
    if scale is None:
        raise ValueError(f"unsupported {kind} unit: {unit!r}")
    return value * scale


def parse_ncu_csv(
    path: Path, *, kernel_patterns: tuple[str, ...]
) -> dict[str, int | float | list[str]]:
    """Parse the stable NCU raw-page columns; never expose replay time as latency."""
    rows = list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines()))
    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if "Metric Name" in row and "Kernel Name" in row
        ),
        None,
    )
    if header_index is None:
        raise ValueError(f"NCU CSV has no raw metric header: {path}")
    header = rows[header_index]
    values: dict[str, list[float]] = defaultdict(list)
    launch_ids: set[tuple[str, str]] = set()
    kernel_names: set[str] = set()
    for raw in rows[header_index + 1 :]:
        if len(raw) != len(header):
            continue
        row = dict(zip(header, raw))
        kernel = row.get("Kernel Name", "")
        if not any(pattern in kernel for pattern in kernel_patterns):
            continue
        metric = row.get("Metric Name", "")
        if metric not in NCU_METRICS:
            continue
        try:
            number = _numeric(row.get("Metric Value", ""))
        except ValueError:
            continue
        unit = row.get("Metric Unit", "")
        if metric == "gpu__time_duration.sum":
            number = _scaled(number, unit, "duration_ns")
            launch_ids.add((row.get("ID", ""), kernel))
        elif metric in {"dram__bytes_read.sum", "dram__bytes_write.sum"}:
            number = _scaled(number, unit, "bytes")
        values[metric].append(number)
        kernel_names.add(kernel)

    duration_ns = sum(values["gpu__time_duration.sum"])
    dram_bytes = sum(values["dram__bytes_read.sum"]) + sum(
        values["dram__bytes_write.sum"]
    )

    def median(metric: str) -> float | None:
        return statistics.median(values[metric]) if values[metric] else None

    missing = [metric for metric in NCU_METRICS if not values[metric]]
    result: dict[str, int | float | list[str]] = {
        "profiled_launches": len(launch_ids),
        "profiled_kernel_duration_ms": duration_ns / 1_000_000.0,
        "dram_bytes": int(dram_bytes),
        "kernel_names": sorted(kernel_names),
        "missing_metrics": missing,
    }
    optional = {
        "dram_throughput_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "l2_hit_rate_pct": "lts__t_sector_hit_rate.pct",
        "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    }
    for output_name, metric in optional.items():
        value = median(metric)
        if value is not None:
            result[output_name] = value
    return result


def _pair_timings(
    records: list[dict[str, Any]], blocks: int, timing_name: str, regime_id: str
) -> tuple[list[tuple[float, float]], list[str]]:
    pairs: dict[str, dict[str, float]] = defaultdict(dict)
    for record in records:
        if record.get("phase") != "confirmatory" or int(record.get("blocks", -1)) != blocks:
            continue
        if record.get("regime_id") is not None and record.get("regime_id") != regime_id:
            continue
        method = str(record.get("method"))
        if method not in {"scalar_gather_scatter", "vectorized_gather_scatter"}:
            continue
        pair_id = str(record.get("pair_id"))
        pairs[pair_id][method] = float(record["timing_ms"][timing_name])
    incomplete = [pair_id for pair_id, methods in pairs.items() if len(methods) != 2]
    if incomplete:
        raise ValueError(f"incomplete paired trials for blocks={blocks}: {incomplete}")
    ordered_ids = sorted(pairs)
    return (
        [
            (
                pairs[pair_id]["scalar_gather_scatter"],
                pairs[pair_id]["vectorized_gather_scatter"],
            )
            for pair_id in ordered_ids
        ],
        ordered_ids,
    )


def _effect_class(
    summary: dict[str, float | int], material: float, maximum_regression: float
) -> str:
    lower = float(summary["ci_lower_percent"])
    upper = float(summary["ci_upper_percent"])
    if lower >= material:
        return "material-win"
    if upper < -maximum_regression:
        return "material-loss"
    if lower >= -maximum_regression and upper < material:
        return "neutral"
    return "uncertain"


def characterize_regimes(
    trial_records: list[dict[str, Any]],
    regimes: list[dict[str, Any]],
    profiler_evidence: dict[str, dict[str, Any]],
    *,
    confidence_level: float,
    bootstrap_resamples: int,
    seed: int,
    material_improvement_percent: float,
    maximum_regression_percent: float,
) -> dict[str, Any]:
    if len(regimes) < 3:
        raise ValueError("at least three workload regimes are required")
    results: list[dict[str, Any]] = []
    for index, regime in enumerate(regimes):
        regime_id = str(regime["id"])
        blocks = int(regime["blocks"])
        gpu_pairs, pair_ids = _pair_timings(
            trial_records, blocks, "synchronized_kernel_ms", regime_id
        )
        end_to_end_pairs, end_to_end_ids = _pair_timings(
            trial_records, blocks, "end_to_end_ms", regime_id
        )
        if pair_ids != end_to_end_ids:
            raise ValueError(f"timing domains have different trial IDs for {regime_id}")
        gpu_summary = paired_bootstrap_summary(
            gpu_pairs,
            confidence_level=confidence_level,
            resamples=bootstrap_resamples,
            seed=seed + index * 2,
        )
        end_to_end_summary = paired_bootstrap_summary(
            end_to_end_pairs,
            confidence_level=confidence_level,
            resamples=bootstrap_resamples,
            seed=seed + index * 2 + 1,
        )
        matching_records = [
            record
            for record in trial_records
            if int(record.get("blocks", -1)) == blocks
            and (record.get("regime_id") is None or record.get("regime_id") == regime_id)
        ]
        payload_bytes = int(matching_records[0].get("bytes", 0)) if matching_records else 0
        scalar_gpu_ms = statistics.median(pair[0] for pair in gpu_pairs)
        vector_gpu_ms = statistics.median(pair[1] for pair in gpu_pairs)
        profile = profiler_evidence.get(regime_id)
        if profile is None or profile.get("nsys") is None:
            raise ValueError(f"missing Nsight Systems evidence for regime {regime_id}")
        ncu = profile.get("ncu")
        mechanism = "measured-with-ncu" if ncu else "unresolved-without-ncu"
        results.append(
            {
                "regime_id": regime_id,
                "blocks": blocks,
                "raw_pair_ids": pair_ids,
                "gpu_effect": gpu_summary,
                "end_to_end_effect": end_to_end_summary,
                "median_no_profiler_gpu_ms": {
                    "scalar": scalar_gpu_ms,
                    "vectorized": vector_gpu_ms,
                },
                "effective_payload_gbps": {
                    "definition": "logical payload bytes divided by CUDA-event time; not DRAM bandwidth",
                    "scalar": payload_bytes / (scalar_gpu_ms * 1_000_000.0),
                    "vectorized": payload_bytes / (vector_gpu_ms * 1_000_000.0),
                },
                "effect_class": _effect_class(
                    gpu_summary,
                    material_improvement_percent,
                    maximum_regression_percent,
                ),
                "mechanism_status": mechanism,
                "nsys": profile["nsys"],
                "ncu": ncu,
            }
        )
    return {
        "regimes": results,
        "contains_neutral_or_loss": any(
            result["effect_class"] in {"neutral", "material-loss"}
            for result in results
        ),
        "all_claims_link_raw_to_end_to_end": all(
            result["raw_pair_ids"] and result["end_to_end_effect"] for result in results
        ),
        "measurement_warning": (
            "Profiler measurements diagnose mechanisms; paired end-to-end effects come "
            "from separate no-profiler trials."
        ),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
