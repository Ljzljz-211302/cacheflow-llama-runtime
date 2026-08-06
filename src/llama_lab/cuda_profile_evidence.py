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


def require_complete_nsys_capture(
    parsed: dict[str, dict[str, Any]], *, expected_launches_per_method: int
) -> None:
    for method in ("scalar", "vectorized"):
        evidence = parsed.get(method)
        if not evidence or evidence.get("kernel_launches") != expected_launches_per_method:
            raise ValueError(
                f"NSYS {method} capture must contain exactly "
                f"{expected_launches_per_method} matching kernel launches"
            )
        if not evidence.get("kernel_names"):
            raise ValueError(f"NSYS {method} capture contains no matching kernel names")


def require_complete_ncu_capture(parsed: dict[str, dict[str, Any]]) -> None:
    for method in ("scalar", "vectorized"):
        evidence = parsed.get(method)
        if not evidence or int(evidence.get("profiled_launches", 0)) <= 0:
            raise ValueError(f"NCU {method} capture contains no matching launches")
        missing = evidence.get("missing_metrics", [])
        if missing:
            raise ValueError(f"NCU {method} capture is missing metrics: {', '.join(missing)}")


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


def _windows_global_pid(value: object) -> int:
    return (int(value) >> 24) & 0xFFFFFF


def parse_nsys_sqlite(
    path: Path,
    *,
    kernel_patterns: tuple[str, ...],
    process_ids: set[int] | None = None,
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
            selected = ["start", "end", name_column]
            has_pid = "globalPid" in columns
            if has_pid:
                selected.append("globalPid")
            for row in _duration_rows(connection, table, selected):
                start, end, raw_name = row[:3]
                if process_ids is not None and (
                    not has_pid or _windows_global_pid(row[3]) not in process_ids
                ):
                    continue
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
            has_pid = "globalPid" in columns
            if has_pid:
                selected.append("globalPid")
            for row in _duration_rows(connection, table, selected):
                if process_ids is not None:
                    pid_index = 3 if byte_column else 2
                    if not has_pid or _windows_global_pid(row[pid_index]) not in process_ids:
                        continue
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
            selected = ["start", "end", name_column]
            has_pid = "globalPid" in columns
            if has_pid:
                selected.append("globalPid")
            for row in _duration_rows(connection, table, selected):
                start, end, raw_name = row[:3]
                if process_ids is not None and (
                    not has_pid or _windows_global_pid(row[3]) not in process_ids
                ):
                    continue
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


def validate_profile_artifact(artifact_dir: Path, protocol_path: Path) -> dict[str, Any]:
    """Fail closed when a committed H2 artifact loses provenance or pair completeness."""
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    trials_path = artifact_dir / "trials.jsonl"
    records = load_jsonl(trials_path)
    from llama_lab.research_protocol import file_sha256

    if manifest.get("protocol_sha256") != file_sha256(protocol_path):
        raise ValueError("profile artifact protocol hash does not match")
    expected_hashes = {
        "trials": trials_path,
        "report": artifact_dir / "report.json",
        "report_markdown": artifact_dir / "report.md",
    }
    for name, path in expected_hashes.items():
        if manifest.get("artifact_hashes", {}).get(name) != file_sha256(path):
            raise ValueError(f"profile artifact {name} hash does not match")
    if manifest.get("raw_trial_count") != len(records):
        raise ValueError("profile artifact raw trial count does not match")
    regimes = {str(regime["id"]): regime for regime in protocol["regimes"]}
    expected_pairs = int(protocol["no_profiler"]["paired_trials_per_regime"])
    methods = set(protocol["no_profiler"]["methods"])
    for regime_id in regimes:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if record.get("regime_id") == regime_id:
                grouped[str(record["pair_id"])].append(record)
        if len(grouped) != expected_pairs:
            raise ValueError(f"{regime_id} pair count does not match protocol")
        for pair_id, pair in grouped.items():
            if len(pair) != 2:
                raise ValueError(f"{pair_id} must contain exactly two rows")
            if {str(row["method"]) for row in pair} != methods:
                raise ValueError(f"{pair_id} methods are incomplete")
            if {int(row["order_in_pair"]) for row in pair} != {0, 1}:
                raise ValueError(f"{pair_id} order is not complementary")
            if len({int(row["random_seed"]) for row in pair}) != 1:
                raise ValueError(f"{pair_id} random seed differs within pair")
            expected_seed = int(protocol["random_seed_base"]) + int(
                regimes[regime_id]["blocks"]
            ) + (1000 if regimes[regime_id]["layout"] == "misaligned" else 0)
            if any(int(row["random_seed"]) != expected_seed for row in pair):
                raise ValueError(f"{pair_id} random seed does not match protocol")
            if any(int(row["blocks"]) != int(regimes[regime_id]["blocks"]) for row in pair):
                raise ValueError(f"{pair_id} block count does not match protocol")
            if any(str(row["layout"]) != regimes[regime_id]["layout"] for row in pair):
                raise ValueError(f"{pair_id} layout does not match protocol")
    commands = {str(item["regime_id"]): item for item in manifest["commands"]}
    if set(commands) != set(regimes):
        raise ValueError("profile artifact command coverage is incomplete")
    for regime_id, status in commands.items():
        profile_dir = artifact_dir / "profiles" / regime_id
        report_path = profile_dir / "nsys.nsys-rep"
        sqlite_path = profile_dir / "nsys.sqlite"
        if status["nsys"].get("report_sha256") != file_sha256(report_path):
            raise ValueError(f"{regime_id} NSYS report hash does not match")
        if status["nsys"].get("sqlite_sha256") != file_sha256(sqlite_path):
            raise ValueError(f"{regime_id} NSYS SQLite hash does not match")
        raw_path = artifact_dir / "raw" / f"{regime_id}-no-profiler.csv"
        if status["no_profiler"].get("stdout_sha256") != file_sha256(raw_path):
            raise ValueError(f"{regime_id} no-profiler raw hash does not match")
    if not report.get("contains_neutral_or_loss"):
        raise ValueError("profile artifact omitted required neutral/loss regime")
    if not report.get("all_claims_link_raw_to_end_to_end"):
        raise ValueError("profile claims are not linked to end-to-end effects")
    if not manifest.get("ncu_complete"):
        required = {
            "memory-bound or roofline classification",
            "achieved occupancy explanation",
            "hardware DRAM byte attribution",
        }
        if not required.issubset(set(report.get("prohibited_claims", []))):
            raise ValueError("incomplete NCU artifact does not prohibit counter claims")
        if not all(
            command["ncu"].get("exit_code") != 0
            and (
                command["ncu"].get("permission_denied")
                or command["ncu"].get("driver_incompatible")
            )
            for command in commands.values()
        ):
            raise ValueError("incomplete NCU artifact lacks an auditable failure")
    return {"manifest": manifest, "report": report, "records": records}
