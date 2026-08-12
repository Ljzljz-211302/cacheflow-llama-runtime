from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from production_journey import request_json  # noqa: E402


SOURCES = (
    ("architecture", "system-design", "docs/architecture.md"),
    ("interview", "interview-knowledge", "docs/interview-notes.md"),
    ("paged-research", "research-method", "docs/research/restricted-paged-decode-attention.md"),
)
TARGETS = (64, 128, 256, 512, 1024, 2048)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_markdown(text: str) -> str:
    text = re.sub(r"```[^\n]*\n", "", text)
    text = text.replace("```", "")
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
    parser.add_argument("--base-url", default="http://127.0.0.1:8330")
    parser.add_argument("--output", type=Path, default=ROOT / "config/paged_objective_workloads_v3.json")
    parser.add_argument("--corpus-version", default="3.0.0")
    args = parser.parse_args()

    workloads = []
    for family, category, relative in SOURCES:
        source_path = ROOT / relative
        source_bytes = source_path.read_bytes()
        normalized = normalize_markdown(source_bytes.decode("utf-8"))
        tokens = post(args.base_url, "/tokenize", {"content": normalized})["tokens"]
        if len(tokens) < max(TARGETS):
            raise ValueError(f"source {relative} has only {len(tokens)} tokens")
        for target in TARGETS:
            candidates = []
            for source_token_count in range(max(1, target - 16), target + 17):
                selected = tokens[:source_token_count]
                prompt = post(args.base_url, "/detokenize", {"tokens": selected})["content"]
                roundtrip = post(args.base_url, "/tokenize", {"content": prompt})["tokens"]
                candidates.append((abs(len(roundtrip) - target), len(roundtrip), source_token_count, prompt))
                if len(roundtrip) == target:
                    break
            difference, actual, source_token_count, prompt = min(candidates)
            if difference != 0:
                raise AssertionError(f"cannot construct exact {target}-token prefix for {family}; nearest={actual}")
            workloads.append({
                "id": f"{family}-{target}",
                "category": category,
                "length_bucket_tokens": target,
                "source": relative,
                "source_sha256": sha256_bytes(source_bytes),
                "preprocessing": "UTF-8 Markdown normalization, detokenized tokenizer prefix, exact retokenization",
                "source_token_span": [0, source_token_count],
                "actual_prompt_tokens": actual,
                "prompt": prompt,
            })

    payload = {
        "schema_version": 2,
        "corpus_version": args.corpus_version,
        "selection_rule": (
            "Deterministic source-bound prefixes from three project documents at six fixed token "
            "lengths; no latency outcome is read and no prompt is deleted after measurement."
        ),
        "tokenizer_endpoint": "llama-server /tokenize and /detokenize",
        "target_context_tokens": list(TARGETS),
        "workloads": workloads,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "workloads": len(workloads)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
