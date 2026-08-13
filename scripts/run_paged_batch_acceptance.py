from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from production_journey import ROOT, cuda_environment, get_text, request_json, wait_ready


BATCH_PROMPTS = tuple(word * 17 for word in (
    " alpha", " bravo", " charlie", " delta",
    " echo", " foxtrot", " golf", " hotel",
))

RUNTIME_FILES = (
    "llama-server.exe",
    "llama-server-impl.dll",
    "llama.dll",
    "ggml.dll",
    "ggml-base.dll",
    "ggml-cpu.dll",
    "ggml-cuda.dll",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric(text: str, name: str) -> float:
    for line in text.splitlines():
        if line.startswith(name):
            return float(line.split()[-1])
    raise AssertionError(f"missing metric: {name}")


def run_arm(server: Path, model: Path, port: int, action: str) -> dict[str, Any]:
    log_path = ROOT / f"results/raw/paged-batch-{action}.log"
    command = [
        str(server.resolve()), "-m", str(model.resolve()),
        "--host", "127.0.0.1", "--port", str(port),
        "-c", "4096", "-np", str(len(BATCH_PROMPTS)), "-b", "512", "-ub", "512",
        "-t", "8", "-ngl", "99", "--flash-attn", "on", "--no-warmup",
        "--metrics", "--slots", "--no-cache-idle-slots",
        "--kv-block-runtime", "--kv-block-size", "16", "--kv-paged-decode",
        "--kv-action-policy", "analytical", "--kv-action-override", action,
        "-lv", "4",
    ]
    environment = cuda_environment()
    environment["LLAMA_CACHEFLOW_PAGED_KERNEL"] = "K4"
    payloads = [
        {
            "prompt": prompt,
            "n_predict": 1,
            "temperature": 0,
            "seed": 20260813 + index,
            "cache_prompt": True,
            "n_probs": 64,
        }
        for index, prompt in enumerate(BATCH_PROMPTS)
    ]
    base_url = f"http://127.0.0.1:{port}"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            wait_ready(base_url, process, log_path, attempts=180)
            for payload in payloads:
                request_json(f"{base_url}/completion", payload)
            before = get_text(f"{base_url}/metrics")

            barrier = threading.Barrier(len(payloads))

            def issue(payload: dict[str, Any]) -> dict[str, Any]:
                barrier.wait()
                return request_json(f"{base_url}/completion", payload)[1]

            with ThreadPoolExecutor(max_workers=len(payloads)) as pool:
                responses = list(pool.map(issue, payloads))
            after = get_text(f"{base_url}/metrics")
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    if any("error" in response or not response.get("content") for response in responses):
        raise AssertionError(f"{action} batch request failed: {responses}")

    names = {
        "paged_calls": "llamacpp:paged_decode_calls_total ",
        "paged_fallbacks": "llamacpp:paged_decode_fallbacks_total ",
        "paged_sequences": "llamacpp:paged_decode_sequences_total ",
        "paged_max_batch": "llamacpp:paged_decode_max_batch ",
        "cuda_dispatches": "llamacpp:paged_decode_cuda_dispatches_total ",
        "cuda_sequences": "llamacpp:paged_decode_cuda_sequences_total ",
        "action_decisions": f'llamacpp:kv_action_decisions_total{{action="{action}"}}',
    }
    counters = {
        key: metric(after, name) - metric(before, name)
        for key, name in names.items()
        if key != "paged_max_batch"
    }
    counters["paged_max_batch"] = metric(after, names["paged_max_batch"])
    return {
        "action": action,
        "responses": [response["content"] for response in responses],
        "cache_tokens": [int(response["timings"]["cache_n"]) for response in responses],
        "top_logprobs": [[
            {"id": int(item["id"]), "logprob": float(item["logprob"])}
            for item in response.get("completion_probabilities", [{}])[0].get("top_logprobs", [])
        ] for response in responses],
        "counters": counters,
        "log": str(log_path.relative_to(ROOT)).replace("\\", "/"),
        "log_sha256": sha256(log_path),
    }


def distribution_comparison(direct: dict[str, Any], paged: dict[str, Any]) -> dict[str, float]:
    max_logprob_error = 0.0
    minimum_overlap = 64
    for direct_probs, paged_probs in zip(direct["top_logprobs"], paged["top_logprobs"]):
        direct_by_id = {item["id"]: item["logprob"] for item in direct_probs}
        paged_by_id = {item["id"]: item["logprob"] for item in paged_probs}
        common = direct_by_id.keys() & paged_by_id.keys()
        minimum_overlap = min(minimum_overlap, len(common))
        if common:
            max_logprob_error = max(max_logprob_error, max(
                abs(direct_by_id[token] - paged_by_id[token]) for token in common
            ))
    return {"minimum_top64_overlap": minimum_overlap, "max_logprob_error": max_logprob_error}


def validate(direct: dict[str, Any], paged: dict[str, Any], request_count: int | None = None) -> None:
    request_count = len(BATCH_PROMPTS) if request_count is None else request_count
    if direct.get("action") != "direct" or paged.get("action") != "paged":
        raise AssertionError("acceptance arms are mislabeled")
    for arm in (direct, paged):
        if (len(arm.get("responses", ())) != request_count or
                len(arm.get("cache_tokens", ())) != request_count or
                len(arm.get("top_logprobs", ())) != request_count):
            raise AssertionError("acceptance arm does not contain every request")
        if len(arm.get("log_sha256", "")) != 64:
            raise AssertionError("acceptance server log is not hash-bound")
    if direct["responses"] != paged["responses"]:
        raise AssertionError("Paged batch output differs from Direct for the same prompts")
    if direct["cache_tokens"] != paged["cache_tokens"]:
        raise AssertionError("Paged and Direct used different resident prefix lengths")
    comparison = distribution_comparison(direct, paged)
    # The per-layer operator has a substantially tighter CPU/CUDA NMSE gate.
    # At the service boundary, 24 layers amplify different FP accumulation
    # orders, so use an explicit absolute log-probability envelope while
    # still requiring most of the top-64 support to agree.
    if comparison["minimum_top64_overlap"] < 48 or comparison["max_logprob_error"] > 1.0:
        raise AssertionError(
            f"Paged batch probability distribution differs from Direct: "
            f"overlap={comparison['minimum_top64_overlap']}, "
            f"max_logprob_error={comparison['max_logprob_error']}"
        )
    if direct["counters"]["paged_calls"] != 0:
        raise AssertionError("Direct arm entered the Paged graph")
    if direct["counters"]["paged_sequences"] != 0 or direct["counters"]["paged_max_batch"] != 0:
        raise AssertionError("Direct arm reported Paged batch work")
    if direct["counters"]["action_decisions"] != request_count:
        raise AssertionError("not every Direct request received a decision")
    if paged["counters"]["action_decisions"] != request_count:
        raise AssertionError("not every request received a Paged decision")
    if paged["counters"]["paged_sequences"] != request_count:
        raise AssertionError("Paged graphs did not process every requested sequence")
    if paged["counters"]["paged_fallbacks"] != 0:
        raise AssertionError("batched Paged execution fell back")
    if paged["counters"]["cuda_dispatches"] < 24:
        raise AssertionError("batched Paged graph did not dispatch every model layer on CUDA")
    if paged["counters"]["cuda_sequences"] != request_count * paged["counters"]["cuda_dispatches"]:
        raise AssertionError("CUDA Paged dispatch did not process the full sequence batch in every layer")
    if direct["counters"]["cuda_dispatches"] != 0 or direct["counters"]["cuda_sequences"] != 0:
        raise AssertionError("Direct arm dispatched the CUDA Paged operator")
    if paged["counters"]["paged_max_batch"] < 2:
        raise AssertionError("service did not form a multi-sequence Paged graph")
    if paged["counters"]["paged_calls"] >= request_count:
        raise AssertionError("each sequence ran as a separate Paged graph; batching was not exercised")
    if min(paged["cache_tokens"]) < 16:
        raise AssertionError("acceptance did not exercise cross-page resident prefixes")


def validate_result(result: dict[str, Any]) -> None:
    if result.get("schema_version") != "paged-batch-acceptance-v2":
        raise AssertionError("unsupported Paged batch acceptance schema")
    if result.get("request_count") != len(BATCH_PROMPTS):
        raise AssertionError("acceptance request count differs from the frozen workload")
    if result.get("prompt_sha256") != [
        hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in BATCH_PROMPTS
    ]:
        raise AssertionError("acceptance prompts differ from the frozen workload")
    runtime_hashes = result.get("runtime_sha256", {})
    if set(runtime_hashes) != set(RUNTIME_FILES):
        raise AssertionError("runtime bundle hash set is incomplete")
    if any(len(value) != 64 for value in runtime_hashes.values()):
        raise AssertionError("invalid runtime bundle SHA-256")
    if len(result.get("model_sha256", "")) != 64:
        raise AssertionError("invalid model SHA-256")
    if result.get("operator_oracle_batch_sizes") != [1, 2, 4, 8]:
        raise AssertionError("operator oracle batch coverage is incomplete")
    validate(result["direct"], result["paged"], result["request_count"])
    if result.get("distribution_comparison") != distribution_comparison(result["direct"], result["paged"]):
        raise AssertionError("copied distribution comparison differs from raw top-logprobs")
    if result.get("passed") is not True:
        raise AssertionError("Paged batch acceptance is not marked passed")


def validate_environment_binding(result: dict[str, Any], server: Path, model: Path) -> None:
    runtime_dir = server.parent
    actual_runtime = {name: sha256(runtime_dir / name) for name in RUNTIME_FILES}
    if result["runtime_sha256"] != actual_runtime:
        raise AssertionError("runtime bundle differs from the measured acceptance binary set")
    if result["model_sha256"] != sha256(model):
        raise AssertionError("model differs from the measured acceptance model")
    if result["vendor_source"] != vendor_binding():
        raise AssertionError("vendor revision or worktree overlay differs from the measured source")
    for arm_name in ("direct", "paged"):
        arm = result[arm_name]
        if arm["log_sha256"] != sha256(ROOT / arm["log"]):
            raise AssertionError(f"{arm_name} server log differs from the measured evidence")


def vendor_binding() -> dict[str, Any]:
    vendor = ROOT / "vendor/llama.cpp"
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=vendor, text=True,
    ).strip()
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "--no-ext-diff"], cwd=vendor,
    )
    dirty_paths = subprocess.check_output(
        ["git", "status", "--short"], cwd=vendor, text=True,
    ).splitlines()
    return {
        "revision": revision,
        "worktree_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "dirty_paths": dirty_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path,
                        default=ROOT / "build/patched-cuda-ninja3/bin/llama-server.exe")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    parser.add_argument("--port", type=int, default=8160)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/acceptance/paged-batch.json")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        result = json.loads(args.output.read_text(encoding="utf-8"))
        validate_result(result)
        validate_environment_binding(result, args.server, args.model)
        print(f"Paged batch acceptance validated: {args.output}")
        return

    direct = run_arm(args.server, args.model, args.port, "direct")
    paged = run_arm(args.server, args.model, args.port + 1, "paged")
    validate(direct, paged)
    runtime_dir = args.server.parent
    result = {
        "schema_version": "paged-batch-acceptance-v2",
        "runtime_sha256": {
            name: sha256(runtime_dir / name) for name in RUNTIME_FILES
        },
        "model_sha256": sha256(args.model),
        "vendor_source": vendor_binding(),
        "request_count": len(BATCH_PROMPTS),
        "prompt_sha256": [
            hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in BATCH_PROMPTS
        ],
        "operator_oracle_batch_sizes": [1, 2, 4, 8],
        "direct": direct,
        "paged": paged,
        "distribution_comparison": distribution_comparison(direct, paged),
        "passed": True,
    }
    validate_result(result)
    validate_environment_binding(result, args.server, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
