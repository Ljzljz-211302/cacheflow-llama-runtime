# Objective Paged-vs-Direct prompt-matrix report

- Frozen corpus: `2.0.0`; 6 workloads
- Design: 30 randomized matched-process blocks × Direct/Paged × every workload
- Raw workload-arm observations: 360
- Provenance boundary: Model and vendor-diff hashes were reconstructed and verified after measurement; they strengthen auditability but are not contemporaneous preregistration or a run-start binding.
- Primary median matched-block regression: -7.96%
- Process-block cluster bootstrap 95% interval: [-17.07%, -0.40%]
- Preregistered median-CI upper bound: +5.00%
- P95 regression / limit: +158.62% / +20.00%
- Worst workload median / limit: +44.66% / +5.00%
- Required page coverage passed: True → **FAIL**

| Workload | Category | Actual tokens | Pages | Median regression | P95 regression |
|---|---|---:|---:|---:|---:|
| controlled-cross-page | controlled | 17 | 2 | -5.92% | +122.32% |
| database-zh-cross-page | natural-zh | 19 | 2 | -14.49% | +105.29% |
| machine-learning-zh-cross-page | natural-zh | 19 | 2 | +44.66% | +164.80% |
| attention-en-cross-page | natural-en | 18 | 2 | -2.94% | +197.24% |
| cuda-mixed-cross-page | natural-mixed | 20 | 2 | -24.98% | +151.50% |
| code-cpp-cross-page | code | 20 | 2 | -3.17% | +102.71% |

Positive values mean Paged is slower. Blocks randomize independent process order; they are not shared-hot-state Trial Pairs. The P95 gate is the tail of block-workload regressions, not request-latency P95.
