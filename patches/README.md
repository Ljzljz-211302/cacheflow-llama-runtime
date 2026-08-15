# llama.cpp engine patch

`0001-cache-aware-slot-scheduler.patch` and
`0002-public-workload-routing.patch` are applied in order to the pinned upstream
commit `acd79d603cb2e1c84c0886137b80f1ad649b6857` by `scripts/bootstrap.ps1`.

The patch contains the project-owned C++ contribution:

- eviction-cost-aware KV-cache slot selection;
- an extracted iteration-level `InferenceScheduler` module;
- decode-first token budgeting and configurable fair chunked prefill;
- proactive unified-KV capacity admission and recency-aware victim planning;
- engine-native KV, prompt-cache, scheduler, and memory metrics;
- a native C++ scheduler test target covering compatibility and fairness.

The complete upstream repository remains under `vendor/llama.cpp` and is ignored
by the top-level Git repository, so personal changes remain auditable as one
reviewable patch series instead of being mixed with unchanged upstream source.
The second patch freezes the request-lifecycle and phase/action-homogeneous
routing used by the public-workload experiment; it also retains a pre-existing
Windows CUDA compilation alias that was present in the measured source tree.
