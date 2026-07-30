# llama.cpp engine patch

`0001-cache-aware-slot-scheduler.patch` is applied to the pinned upstream commit
`acd79d603cb2e1c84c0886137b80f1ad649b6857` by `scripts/bootstrap.ps1`.

The patch contains the project-owned C++ contribution:

- eviction-cost-aware KV-cache slot selection;
- an extracted iteration-level `InferenceScheduler` module;
- decode-first token budgeting and configurable fair chunked prefill;
- engine-native KV, prompt-cache, scheduler, and memory metrics;
- a native C++ scheduler test target covering compatibility and fairness.

The complete upstream repository remains under `vendor/llama.cpp` and is ignored
by the top-level Git repository, so personal changes remain auditable as one
reviewable patch instead of being mixed with unchanged upstream source.
