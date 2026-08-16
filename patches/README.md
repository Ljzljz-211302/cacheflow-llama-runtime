# llama.cpp engine source and audit patches

The complete project-owned C++/CUDA source is published directly in this
repository on branch `engine/llama.cpp`. The `main` branch pins that exact
snapshot through the `vendor/llama.cpp` submodule, and `scripts/bootstrap.ps1`
checks the pinned commit before downloading runtime artifacts.

`0001-cache-aware-slot-scheduler.patch` and
`0002-public-workload-routing.patch` are retained as audit exports relative to
upstream commit `acd79d603cb2e1c84c0886137b80f1ad649b6857`; bootstrap does not
reapply them because they are already present in the engine snapshot.

The patch contains the project-owned C++ contribution:

- eviction-cost-aware KV-cache slot selection;
- an extracted iteration-level `InferenceScheduler` module;
- decode-first token budgeting and configurable fair chunked prefill;
- proactive unified-KV capacity admission and recency-aware victim planning;
- engine-native KV, prompt-cache, scheduler, and memory metrics;
- a native C++ scheduler test target covering compatibility and fairness.

The full source is browsable without cloning another repository, while the
patch series keeps personal changes reviewable against the fixed upstream.
The second patch freezes the request-lifecycle and phase/action-homogeneous
routing used by the public-workload experiment; it also retains a pre-existing
Windows CUDA compilation alias that was present in the measured source tree.
