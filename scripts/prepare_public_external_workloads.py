from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from llama_lab.public_workloads import (  # noqa: E402
    PublicPrompt,
    TraceRequest,
    load_azure_trace,
    load_burstgpt_trace,
    load_longbench_records,
)
from production_journey import cuda_environment, request_json, terminate_process, wait_ready  # noqa: E402


TASKS = ("multifieldqa_en", "2wikimqa", "triviaqa")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def post(base_url: str, endpoint: str, payload: dict) -> dict:
    status, body = request_json(f"{base_url}{endpoint}", payload)
    if status != 200 or "error" in body:
        raise RuntimeError(f"tokenizer endpoint failed: {status} {body}")
    return body


def exact_prefix(base_url: str, prompt: str, target: int) -> tuple[str, int]:
    tokens = post(base_url, "/tokenize", {"content": prompt})["tokens"]
    if len(tokens) < target:
        raise ValueError(f"official prompt has {len(tokens)} tokens, below requested {target}")
    candidates = []
    for count in range(max(1, target - 16), min(len(tokens), target + 16) + 1):
        text = post(base_url, "/detokenize", {"tokens": tokens[:count]})["content"]
        actual = len(post(base_url, "/tokenize", {"content": text})["tokens"])
        candidates.append((abs(actual - target), actual, text))
        if actual == target:
            return text, actual
    _, actual, text = min(candidates)
    raise AssertionError(f"cannot construct exact {target}-token public prompt; nearest={actual}")


def load_prompts(data_root: Path, config_root: Path) -> list[PublicPrompt]:
    templates = json.loads((config_root / "dataset2prompt.json").read_text(encoding="utf-8"))
    prompts = []
    for task in TASKS:
        prompts.extend(load_longbench_records(
            data_root / f"{task}.jsonl", expected_dataset=task, prompt_template=templates[task],
        ))
    return prompts


def usable_trace(rows: list[TraceRequest], count: int) -> list[TraceRequest]:
    filtered = [row for row in rows if 128 <= row.input_tokens <= 1024 and row.output_tokens > 0]
    if len(filtered) < count:
        raise ValueError(f"official trace has only {len(filtered)} usable requests")
    selected = filtered[:count]
    start = selected[0].timestamp_seconds
    return [replace(row, arrival_seconds=row.timestamp_seconds - start) for row in selected]


def prompt_row(
    base_url: str, trace: TraceRequest, source: PublicPrompt, *, output_cap: int,
) -> dict:
    target = trace.input_tokens
    text, actual = exact_prefix(base_url, source.prompt, target)
    return {
        "trace_source": trace.source,
        "trace_row": trace.source_row,
        "source_arrival_seconds": trace.arrival_seconds,
        "source_input_tokens": trace.input_tokens,
        "source_output_tokens": trace.output_tokens,
        "prompt_dataset": source.dataset,
        "prompt_id": source.record_id,
        "prompt": text,
        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "actual_local_input_tokens": actual,
        "requested_output_tokens": min(trace.output_tokens, output_cap),
        "provenance": "trace-driven-public-content-synthetic-replay",
    }


def quality_cases(base_url: str, prompts: list[PublicPrompt], count: int, maximum_tokens: int) -> list[dict]:
    if count % len(TASKS):
        raise ValueError("quality case count must allocate equally across LongBench-E tasks")
    per_task = count // len(TASKS)
    selected = []
    for task in TASKS:
        task_selected = 0
        candidates = sorted(
            (row for row in prompts if row.dataset == task),
            key=lambda row: (row.source_length, row.record_id),
        )
        for prompt in candidates:
            tokens = len(post(base_url, "/tokenize", {"content": prompt.prompt})["tokens"])
            if tokens > maximum_tokens:
                continue
            selected.append({
                "dataset": prompt.dataset,
                "record_id": prompt.record_id,
                "language": prompt.language,
                "source_length": prompt.source_length,
                "actual_local_input_tokens": tokens,
                "prompt": prompt.prompt,
                "prompt_sha256": hashlib.sha256(prompt.prompt.encode("utf-8")).hexdigest(),
                "answers": list(prompt.answers),
            })
            task_selected += 1
            if task_selected == per_task:
                break
        if task_selected != per_task:
            raise ValueError(
                f"only {task_selected}/{per_task} {task} LongBench cases fit the production Paged envelope"
            )
    if len(selected) < count:
        raise ValueError(f"only {len(selected)} LongBench cases fit the production Paged envelope")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--longbench-data", type=Path, required=True)
    parser.add_argument("--longbench-config", type=Path, required=True)
    parser.add_argument("--longbench-archive", type=Path, required=True)
    parser.add_argument("--burstgpt", type=Path, required=True)
    parser.add_argument("--azure", type=Path)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8350)
    parser.add_argument("--requests-per-trace", type=int, default=24)
    parser.add_argument("--output-cap", type=int, default=8)
    parser.add_argument("--quality-cases", type=int, default=6)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    log_path = ROOT / "results/raw/public-workload-tokenizer.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.longbench_data, args.longbench_config)
    traces = {
        "burstgpt": usable_trace(load_burstgpt_trace(
            args.burstgpt, start_row=0, request_count=20000, include_failures=False,
        ), args.requests_per_trace),
    }
    if args.azure is not None:
        traces["azure-code"] = usable_trace(load_azure_trace(
            args.azure, start_row=0, request_count=20000,
        ), args.requests_per_trace)

    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            [str(args.server.resolve()), "-m", str(args.model.resolve()), "--host", "127.0.0.1",
             "--port", str(args.port), "-ngl", "99", "--no-warmup"],
            cwd=ROOT, env=cuda_environment(), stdout=log_file, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            wait_ready(base_url, process, log_path, attempts=180)
            ordered_prompts = sorted(prompts, key=lambda row: (row.dataset, row.record_id))
            replay = {}
            for trace_name, trace_rows in traces.items():
                replay[trace_name] = [
                    prompt_row(
                        base_url, request, ordered_prompts[index % len(ordered_prompts)],
                        output_cap=args.output_cap,
                    )
                    for index, request in enumerate(trace_rows)
                ]
            quality = quality_cases(base_url, prompts, args.quality_cases, maximum_tokens=2048)
        finally:
            terminate_process(process)

    sources = {
        "longbench": {
            "name": "THUDM/LongBench", "archive_sha256": sha256(args.longbench_archive),
            "repository_revision": subprocess.check_output(
                ["git", "-C", str(args.longbench_config.parents[1]), "rev-parse", "HEAD"], text=True,
            ).strip(),
            "official_url": "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip",
        },
        "burstgpt": {
            "name": "BurstGPT-v2.0/BurstGPT_1.csv", "sha256": sha256(args.burstgpt),
            "official_url": "https://github.com/HPMLL/BurstGPT/releases/tag/v2.0",
        },
    }
    if args.azure is not None:
        sources["azure-code"] = {
            "name": "AzureLLMInferenceTrace_code_1week.csv", "sha256": sha256(args.azure),
            "official_url": "https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md",
        }
    payload = {
        "schema_version": 1,
        "corpus_version": "public-external-v1.0.0",
        "selection_rule": (
            "Contiguous official trace windows filtered only by the production 128..1024-token envelope; "
            "LongBench records are ordered by public dataset/id before round-robin content assignment; "
            "quality cases are complete, untruncated records within the verified 2048-token Paged envelope; "
            "no latency outcome is consulted."
        ),
        "longbench_e_boundary": (
            "LongBench-E was downloaded and reviewed, but its complete records exceed the current "
            "2048-token production Paged envelope; it is not truncated or scored as LongBench-E."
        ),
        "joint_distribution_claim": "forbidden",
        "sources": sources,
        "performance_replays": replay,
        "quality_cases": quality,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "replays": {name: len(rows) for name, rows in replay.items()},
        "quality_cases": len(quality),
    }))


if __name__ == "__main__":
    main()
