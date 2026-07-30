# Cache-aware iteration-level inference scheduling

## Problem

`llama-server` keeps each finished conversation in an idle slot so a later
request can reuse its prompt KV state. Upstream b9632 chooses the idle slot with
the largest longest-common-prefix (LCP) ratio when it exceeds
`--slot-prompt-similarity`; otherwise it falls back to LRU.

Maximizing reuse for the arriving request ignores the cache destroyed by that
choice. Reusing an 80-token prefix can evict a 480-token conversation even when
another slot can reuse 60 tokens while evicting only 10. The greedy choice saves
20 tokens now but can force hundreds of tokens of prefill on the next turn.

## Policy

For each idle slot:

```text
reuse   = LCP(slot.prompt, request.prompt)
evicted = max(slot.cached_tokens - reuse, 0)
score   = reuse - lambda * evicted
```

Candidates below the existing similarity threshold are excluded. The highest
positive score wins; equal scores use the older `t_last_used`. If no candidate
has positive net benefit, the original LRU fallback runs.

`lambda=0` is a compatibility mode: because every candidate is compared against
the same request length, maximizing reusable tokens is equivalent to maximizing
the upstream LCP ratio. A positive lambda represents the estimated future value
of one cached token relative to one unit of current prefill.

## Invariants

- Explicit `id_slot` requests bypass the policy exactly as upstream.
- Busy slots are never candidates.
- The existing similarity threshold remains authoritative.
- Negative penalties are rejected at CLI parsing.
- Non-positive net benefit cannot evict a cache through the cache-aware path.
- LRU remains the total fallback, so admission cannot fail solely because of the
  new score.

## Complexity

Upstream already computes an LCP for every idle slot. The policy adds constant
arithmetic and a tie-break per candidate, so asymptotic selection remains
`O(number_of_idle_slots × prompt_prefix_comparison)` with `O(number_of_slots)`
temporary metadata.

## Engine metrics

The patch adds Prometheus counters/gauges for:

- cache-aware and LRU selection counts;
- estimated reused and evicted tokens;
- actual prompt tokens reused by prompt evaluation;
- logical KV sequence tokens/capacity;
- model, context/KV, compute, and total bytes from
  `llama_get_memory_breakdown()`.
- token-level scheduler iterations and scheduled decode tokens;
- scheduled prefill tokens and prompt chunk counts.
- proactive KV pressure, victim, reclaimed-token, and unresolved-admission
  counters.

Logical KV token utilization and allocated bytes intentionally remain separate:
the buffer is generally allocated up front, while logical occupancy changes per
request.

## Controlled conflict workload

Two slots are seeded by explicit ID:

- slot A: 480 tokens, sharing an 80-token prefix with the conflict request;
- slot B: 70 tokens, sharing a 60-token prefix.

The conflict request is followed by another turn of conversation A. With
`lambda=0`, slot A is selected, 400 cached tokens are destroyed, and the followup
must re-prefill about 410 tokens. With `lambda=0.5`, slot B is selected, only 10
tokens are destroyed, and the followup processes about 10 tokens.

Five fresh-server trials are summarized by medians in `results/engine_ab.csv`;
per-trial data remains in `results/engine_ab_trials.csv`.

## Limitations

- The current lambda is static; production traffic should estimate future reuse
  probability from traces or learn it online.
- Token count is a proxy for prefill cost. Heterogeneous adapters, multimodal
  inputs, or tiered KV storage need a richer cost model.
- The current patched build is CPU-only because this machine has the CUDA
  runtime but no CUDA Toolkit. The same C++ path compiles independently of the
  backend, but GPU A/B remains future validation.
- The workload is adversarial and demonstrates the mechanism; it is not a claim
  about average production hit rate.

## Chunked prefill result

The refactor moves prefill allocation behind the same `InferenceScheduler`
interface. Decode/speculative tokens consume the iteration budget first. The
remaining tokens are distributed in round-robin chunks; `--prefill-chunk-size
0` preserves upstream greedy allocation exactly.

The CPU experiment in `results/prefill_ab.csv` is deliberately retained even
though it is negative. For this 0.5B Q4 model, fixed 128-token and 256-token
chunks increased active-decode wall time relative to greedy prefill. Small
chunks reduce GEMM efficiency and add more inference iterations; the cost was
larger than the latency-isolation benefit on this backend. This rules out a
backend-independent fixed chunk size and motivates the next implementation:
online, hardware-aware chunk calibration instead of presenting a negative
optimization as a speedup.
