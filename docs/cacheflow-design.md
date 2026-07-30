# CacheFlow: project specification

## Positioning

CacheFlow is a personally implemented LLM serving control plane. `llama.cpp` is
one replaceable execution backend; it is not counted as personal code. CacheFlow
owns request admission, lifecycle, continuous dispatch, slot placement, prefix
indexing, hierarchical KV checkpointing, backpressure, deadlines, cancellation,
metrics, and the OpenAI-compatible gateway.

## Problem

A raw model server optimizes individual requests but does not know application
conversation identity, revisit probability, tenant wait time, or disk cache
budget. Under many interleaved conversations, valuable KV state is overwritten,
long prompts are repeatedly prefetched, and a cache-greedy policy can starve old
requests. CacheFlow adds a serving-layer policy with explicit state and SLOs.

## Architecture

```text
OpenAI client
    │
    ▼
HTTP gateway ── admission/backpressure ── model-aware router
                                         ├─ quality constraint
                                         ├─ latency cost model
                                         ├─ KV/VRAM estimator
                                         └─ online EWMA calibration
                                                    │
                                                    ▼
                                         request state machine
                                         │
                                         ▼
                              continuous scheduler loop
                              ├─ aging/fairness
                              ├─ prefix benefit
                              ├─ eviction cost
                              └─ deadline/cancellation
                                         │
               ┌─────────────────────────┴────────────────────────┐
               ▼                                                  ▼
       L1 resident KV slots                              L2 checkpoint store
       + radix prefix index                              + byte-budgeted LRU
               │                                                  │
               └──────────────────────┬───────────────────────────┘
                                      ▼
                             Executor protocol
                                      │
                                      ▼
                        llama.cpp explicit-slot API
```

## Domain model

A request moves through:

```text
NEW → TOKENIZING → QUEUED → RESTORING → RUNNING → COMPLETED
                         ↘ CANCELLED / TIMED_OUT / FAILED
```

Only terminal states may resolve a client future. A slot is either idle or owns
one running request; after completion it retains that conversation's KV state.
The scheduler, not the executor, chooses the slot.

## Scheduling score

For queued request `r` and free slot `s`:

```text
score(r, s) = prefix_reuse
            - eviction_penalty × discarded_L1_tokens
            + wait_age_weight × waiting_ms
            + restore_bonus when an L2 checkpoint exists
```

Requests past `max_wait_ms` enter an urgent lane ordered by deadline/FIFO, which
prevents a hot conversation from starving cold requests. Dispatch is continuous:
whenever any slot finishes, the next request-slot pair is chosen without waiting
for a fixed batch barrier.

## Model-aware routing

Each backend model has a profile containing transformer architecture, weight
precision, quality guard score, measured prefill/decode curves, current VRAM,
and concurrency capacity. For a request, CacheFlow first rejects candidates that
violate a hard quality floor, context limit, or estimated VRAM budget. Remaining
models are scored by predicted service cost:

```text
predicted_ms = prefill_curve(context_bucket)
             + decode_curve(output_tokens, concurrency)

route_score  = quality_weight × normalized_quality
             - latency_weight × predicted_ms / latency_slo
             - memory_weight  × estimated_kv_bytes / free_vram
```

The KV estimate uses model layers, KV heads, head dimension, cache dtype, context,
and active sequences rather than treating every quantization as the same. After
completion, an EWMA updates the selected model's context/concurrency bucket, so
the router adapts to thermal throttling and workload drift. Every decision emits
eligible/rejected candidates and score components for interview/debug inspection.

This is a model-serving algorithm, not a claim to train or alter transformer
weights. Q4/Q8/F16 and future parameter scales are interchangeable profiles.

## Hierarchical KV cache

- L1 is the set of live llama.cpp slots and is indexed by a token radix trie.
- Before a valuable L1 conversation is overwritten, CacheFlow can save it through
  the executor's checkpoint API.
- L2 stores checkpoint metadata under a strict byte budget and evicts by LRU.
- A revisited conversation can restore its checkpoint into an idle slot before
  inference, avoiding full prefill.
- Checkpoint failures degrade to recomputation and never corrupt request state.

## Executor boundary

The executor protocol exposes only:

- `tokenize(messages)`;
- `complete(request, slot_id)`;
- `save_slot(slot_id, checkpoint_name)`;
- `restore_slot(slot_id, checkpoint_name)`;
- `erase_slot(slot_id)`.

Tests use a deterministic fake executor. The production adapter uses llama.cpp
native `/completion`, `/tokenize`, and `/slots/{id}` endpoints. Replacing llama.cpp
does not change scheduling or cache code.

## Observability

Prometheus metrics cover model-route decisions/prediction error, queue depth, admissions/rejections, state transitions,
running slots, wait/runtime histograms, prefix reuse, prefill processed tokens,
L1 evictions, L2 saves/restores/hits/misses/bytes, timeouts, cancellations, and
backend errors. A `/debug/state` endpoint exposes sanitized scheduler state.

## Acceptance criteria

1. Pure unit tests cover state transitions, trie updates, fairness, backpressure,
   cancellation, deadline expiry, cache policy, and L2 budget eviction.
2. A fake-executor integration test runs concurrent requests deterministically.
3. A real Qwen request passes through the CacheFlow OpenAI endpoint.
4. A multi-conversation trace demonstrates L2 restore and reduced re-prefill.
5. A model-routing trace compares static, latency-only, and constrained adaptive
   routing across context lengths, quality floors, and VRAM budgets.
6. Fault injection proves checkpoint/backend failures recover or fail cleanly.
7. One command runs tests, starts the backend/gateway, replays traces, and emits
   an interview report with limitations and negative results.

## Non-goals

- Reimplementing transformer kernels, tokenization, or CUDA graphs.
- Claiming CacheFlow is a replacement for vLLM in production.
- Counting model files, binaries, or vendored llama.cpp as personal work.
