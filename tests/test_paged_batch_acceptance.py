from __future__ import annotations

import copy
import hashlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_paged_batch_acceptance import (  # noqa: E402
    BATCH_PROMPTS, RUNTIME_FILES, distribution_comparison, validate_result,
)


def arm(action: str) -> dict:
    request_count = len(BATCH_PROMPTS)
    paged = action == "paged"
    result = {
        "action": action,
        "responses": [str(index) for index in range(request_count)],
        "cache_tokens": [17] * request_count,
        "top_logprobs": [[
            {"id": token, "logprob": -0.01 * token} for token in range(64)
        ] for _ in range(request_count)],
        "counters": {
            "paged_calls": 2 if paged else 0,
            "paged_fallbacks": 0,
            "paged_sequences": request_count if paged else 0,
            "paged_max_batch": 4 if paged else 0,
            "action_decisions": request_count,
            "cuda_dispatches": 24 if paged else 0,
            "cuda_sequences": 24 * request_count if paged else 0,
        },
        "log": f"results/raw/paged-batch-{action}.log",
        "log_sha256": "e" * 64,
    }
    return result


def valid_result() -> dict:
    result = {
        "schema_version": "paged-batch-acceptance-v2",
        "runtime_sha256": {name: "a" * 64 for name in RUNTIME_FILES},
        "model_sha256": "b" * 64,
        "vendor_source": {
            "revision": "c" * 40,
            "worktree_diff_sha256": "d" * 64,
            "dirty_paths": [],
        },
        "request_count": len(BATCH_PROMPTS),
        "prompt_sha256": [hashlib.sha256(prompt.encode()).hexdigest() for prompt in BATCH_PROMPTS],
        "operator_oracle_batch_sizes": [1, 2, 4, 8],
        "direct": arm("direct"),
        "paged": arm("paged"),
        "passed": True,
    }
    result["distribution_comparison"] = distribution_comparison(result["direct"], result["paged"])
    return result


class PagedBatchAcceptanceTests(unittest.TestCase):
    def test_valid_multi_sequence_batch_passes(self) -> None:
        validate_result(valid_result())

    def test_semantic_tampering_is_rejected(self) -> None:
        mutations = (
            lambda result: result["paged"]["counters"].__setitem__("paged_max_batch", 1),
            lambda result: result["paged"]["counters"].__setitem__("paged_sequences", 7),
            lambda result: result["paged"]["counters"].__setitem__("paged_fallbacks", 1),
            lambda result: result["paged"]["counters"].__setitem__("paged_calls", 8),
            lambda result: result["paged"]["counters"].__setitem__("cuda_dispatches", 0),
            lambda result: result["paged"]["counters"].__setitem__("cuda_sequences", 191),
            lambda result: result["paged"]["responses"].__setitem__(0, "different"),
            lambda result: result["paged"]["top_logprobs"][0][0].__setitem__("logprob", 1.0),
            lambda result: result.__setitem__("prompt_sha256", ["0" * 64] * 8),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                result = copy.deepcopy(valid_result())
                mutate(result)
                with self.assertRaises(AssertionError):
                    validate_result(result)


if __name__ == "__main__":
    unittest.main()
