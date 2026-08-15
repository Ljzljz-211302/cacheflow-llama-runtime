from __future__ import annotations

import csv
import json
import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class PublicPrompt:
    dataset: str
    record_id: str
    language: str
    prompt: str
    answers: tuple[str, ...]
    source_length: int
    local_tokens: int


@dataclass(frozen=True)
class TraceRequest:
    source: str
    source_row: int
    timestamp_seconds: float
    arrival_seconds: float
    input_tokens: int
    output_tokens: int
    session_id: str = ""
    model: str = ""
    request_type: str = ""


@dataclass(frozen=True)
class ReplayRequest:
    trace_source: str
    trace_row: int
    prompt_dataset: str
    prompt_id: str
    prompt: str
    arrival_seconds: float
    source_input_tokens: int
    local_input_tokens: int
    output_tokens: int
    provenance: str


def longbench_qa_f1(prediction: str, ground_truth: str) -> float:
    def normalize(value: str) -> list[str]:
        value = value.lower()
        value = "".join(character for character in value if character not in string.punctuation)
        value = re.sub(r"\b(a|an|the)\b", " ", value)
        return " ".join(value.split()).split()

    predicted, expected = normalize(prediction), normalize(ground_truth)
    if not predicted or not expected:
        return float(predicted == expected)
    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


_LONGBENCH_FIELDS = {
    "input", "context", "answers", "length", "dataset", "language", "all_classes", "_id",
}


def load_longbench_records(
    path: Path, *, expected_dataset: str, prompt_template: str = "{input}\n\n{context}",
) -> list[PublicPrompt]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if (not _LONGBENCH_FIELDS.issubset(row) or
                row["dataset"] not in (expected_dataset, f"{expected_dataset}_e")):
            raise ValueError(f"LongBench schema mismatch at line {line_number}")
        answers = row["answers"]
        if not isinstance(answers, list) or not answers or not all(isinstance(value, str) for value in answers):
            raise ValueError(f"LongBench schema mismatch at line {line_number}")
        prompt = prompt_template.format(input=str(row["input"]), context=str(row["context"]))
        records.append(PublicPrompt(
            dataset=expected_dataset,
            record_id=str(row["_id"]),
            language=str(row["language"]),
            prompt=prompt,
            answers=tuple(answers),
            source_length=int(row["length"]),
            local_tokens=0,
        ))
    if not records:
        raise ValueError("LongBench source is empty")
    return sorted(records, key=lambda row: row.record_id)


def select_longbench_prompts(
    prompts: Sequence[PublicPrompt], *, tasks: Sequence[str],
    local_token_buckets: Sequence[tuple[int, int]], per_task_bucket: int, seed: int,
) -> list[PublicPrompt]:
    if per_task_bucket <= 0:
        raise ValueError("per_task_bucket must be positive")
    generator = random.Random(seed)
    selected = []
    for task in tasks:
        task_rows = [row for row in prompts if row.dataset == task]
        for lower, upper in local_token_buckets:
            candidates = [row for row in task_rows if lower <= row.local_tokens <= upper]
            if len(candidates) < per_task_bucket:
                raise ValueError(f"public prompt coverage is incomplete for {task} [{lower}, {upper}]")
            candidates.sort(key=lambda row: row.record_id)
            generator.shuffle(candidates)
            selected.extend(sorted(candidates[:per_task_bucket], key=lambda row: row.record_id))
    return selected


def _positive_int(row: dict[str, str], field: str) -> int:
    try:
        value = int(float(row[field]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"trace field {field} is not an integer") from error
    if value <= 0:
        raise ValueError("trace requires positive token lengths")
    return value


def _nonnegative_int(row: dict[str, str], field: str) -> int:
    try:
        value = int(float(row[field]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"trace field {field} is not an integer") from error
    if value < 0:
        raise ValueError("trace requires nonnegative token lengths")
    return value


def _normalize_arrivals(rows: Iterable[TraceRequest]) -> list[TraceRequest]:
    values = list(rows)
    if not values:
        raise ValueError("trace window contains no usable requests")
    start = values[0].timestamp_seconds
    if any(row.timestamp_seconds < start for row in values):
        raise ValueError("trace timestamps must be nondecreasing")
    return [TraceRequest(
        source=row.source,
        source_row=row.source_row,
        timestamp_seconds=row.timestamp_seconds,
        arrival_seconds=row.timestamp_seconds - start,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        session_id=row.session_id,
        model=row.model,
        request_type=row.request_type,
    ) for row in values]


def load_burstgpt_trace(
    path: Path, *, start_row: int, request_count: int, include_failures: bool = False,
) -> list[TraceRequest]:
    if start_row < 0 or request_count <= 0:
        raise ValueError("trace window bounds are invalid")
    accepted = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        for source_row, row in enumerate(csv.DictReader(stream), 1):
            if source_row <= start_row:
                continue
            try:
                input_tokens = _nonnegative_int(row, "Request tokens")
                output_tokens = _nonnegative_int(row, "Response tokens")
                timestamp = float(row["Timestamp"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"BurstGPT schema mismatch at row {source_row}") from error
            if (input_tokens == 0 or output_tokens == 0) and not include_failures:
                continue
            accepted.append(TraceRequest(
                source="BurstGPT-v2.0",
                source_row=source_row,
                timestamp_seconds=timestamp,
                arrival_seconds=0.0,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                session_id=str(row.get("Session ID", "")),
                model=str(row.get("Model", "")),
                request_type=str(row.get("Log Type", "")),
            ))
            if len(accepted) == request_count:
                break
    return _normalize_arrivals(accepted)


def load_azure_trace(path: Path, *, start_row: int, request_count: int) -> list[TraceRequest]:
    if start_row < 0 or request_count <= 0:
        raise ValueError("trace window bounds are invalid")
    accepted = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        for source_row, row in enumerate(csv.DictReader(stream), 1):
            if source_row <= start_row:
                continue
            try:
                input_tokens = _positive_int(row, "ContextTokens")
                output_tokens = _positive_int(row, "GeneratedTokens")
                timestamp = datetime.fromisoformat(row["TIMESTAMP"]).timestamp()
            except (KeyError, TypeError, ValueError) as error:
                if isinstance(error, ValueError) and "positive token lengths" in str(error):
                    raise
                raise ValueError(f"Azure trace schema mismatch at row {source_row}") from error
            accepted.append(TraceRequest(
                source="Azure-LLM-Inference-2024",
                source_row=source_row,
                timestamp_seconds=timestamp,
                arrival_seconds=0.0,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ))
            if len(accepted) == request_count:
                break
    return _normalize_arrivals(accepted)


def build_trace_driven_replay(
    trace: Sequence[TraceRequest], prompts: Sequence[PublicPrompt], *, seed: int,
    time_scale: float, maximum_output_tokens: int,
) -> list[ReplayRequest]:
    if time_scale <= 0 or maximum_output_tokens <= 0 or not prompts:
        raise ValueError("replay parameters are invalid")
    generator = random.Random(seed)
    available = list(prompts)
    replay = []
    for request in trace:
        candidates = available or list(prompts)
        distances = [abs(row.local_tokens - request.input_tokens) for row in candidates]
        best = min(distances)
        nearest = [row for row, distance in zip(candidates, distances) if distance == best]
        nearest.sort(key=lambda row: (row.dataset, row.record_id))
        prompt = nearest[generator.randrange(len(nearest))]
        if prompt in available:
            available.remove(prompt)
        replay.append(ReplayRequest(
            trace_source=request.source,
            trace_row=request.source_row,
            prompt_dataset=prompt.dataset,
            prompt_id=prompt.record_id,
            prompt=prompt.prompt,
            arrival_seconds=request.arrival_seconds * time_scale,
            source_input_tokens=request.input_tokens,
            local_input_tokens=prompt.local_tokens,
            output_tokens=min(request.output_tokens, maximum_output_tokens),
            provenance="trace-driven-public-content-synthetic-replay",
        ))
    return replay
