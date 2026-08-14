from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from production_journey import cuda_environment, request_json  # noqa: E402


SOURCES = (
    "docs/architecture.md",
    "docs/interview-notes.md",
    "docs/research/restricted-paged-decode-attention.md",
)
TARGETS = (128, 512, 1024)
STREAMS = 8


def normalize(text: str) -> str:
    text = re.sub(r"```[^\n]*\n", "", text).replace("```", "")
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def post(base_url: str, endpoint: str, payload: dict) -> dict:
    status, body = request_json(f"{base_url}{endpoint}", payload)
    if status != 200 or "error" in body:
        raise RuntimeError(f"tokenizer endpoint failed: {status} {body}")
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8340")
    parser.add_argument("--output", type=Path, default=ROOT / "config/batched_paged_workloads_v1.json")
    parser.add_argument("--corpus-version", default="1.0.0")
    parser.add_argument("--launch-server", type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    process = None
    log = None
    if args.launch_server:
        if not args.model:
            parser.error("--model is required with --launch-server")
        log_path = ROOT / "results/raw/batched-corpus-tokenizer.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("wb")
        process = subprocess.Popen(
            [str(args.launch_server.resolve()), "-m", str(args.model.resolve()), "--host", "127.0.0.1",
             "--port", args.base_url.rsplit(":", 1)[-1], "-ngl", "99", "--no-warmup"],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env=cuda_environment(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        for _ in range(120):
            try:
                status, _ = request_json(f"{args.base_url}/tokenize", {"content": "ready"})
                if status == 200:
                    break
            except Exception:
                pass
            if process.poll() is not None:
                raise RuntimeError("tokenizer server exited before becoming ready")
            time.sleep(0.25)
        else:
            raise RuntimeError("tokenizer server did not become ready")
    try:
        source_rows, joined = [], []
        for relative in SOURCES:
            raw = (ROOT / relative).read_bytes()
            source_rows.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest()})
            joined.append(normalize(raw.decode("utf-8")))
        content = " ".join(joined)
        tokens = post(args.base_url, "/tokenize", {"content": content})["tokens"]
        maximum_start = len(tokens) - max(TARGETS) - 32
        if maximum_start <= 0:
            raise ValueError("source corpus is too short")
        starts = [round(index * maximum_start / (STREAMS - 1)) for index in range(STREAMS)]
        workloads = []
        for target in TARGETS:
            prompts = []
            for stream, start in enumerate(starts):
                candidates = []
                for count in range(target - 16, target + 17):
                    prompt = post(args.base_url, "/detokenize", {"tokens": tokens[start:start + count]})["content"]
                    actual = len(post(args.base_url, "/tokenize", {"content": prompt})["tokens"])
                    candidates.append((abs(actual - target), actual, count, prompt))
                    if actual == target:
                        break
                difference, actual, count, prompt = min(candidates)
                if difference:
                    raise AssertionError(f"cannot construct exact {target}-token stream {stream}; nearest={actual}")
                prompts.append({
                    "stream": stream, "prompt": prompt, "actual_prompt_tokens": actual,
                    "joined_source_token_span": [start, start + count],
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                })
            workloads.append({"context_tokens": target, "prompts": prompts})
        payload = {
            "schema_version": 1,
            "corpus_version": args.corpus_version,
            "selection_rule": "Eight deterministic, evenly spaced source-bound token spans per context; exact tokenizer round-trip; no performance outcome consulted.",
            "sources": source_rows,
            "target_context_tokens": list(TARGETS),
            "streams_per_context": STREAMS,
            "workloads": workloads,
        }
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "contexts": list(TARGETS), "streams": STREAMS}))
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if log is not None:
            log.close()


if __name__ == "__main__":
    main()
