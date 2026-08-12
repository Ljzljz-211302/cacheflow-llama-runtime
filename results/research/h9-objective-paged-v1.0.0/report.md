# Objective Paged-vs-Direct prompt-matrix report

- Frozen corpus: `1.0.0`; 6 workloads
- Design: 30 process pairs × Direct/Paged × every workload
- Raw workload-arm observations: 360
- Primary median paired regression: -8.59%
- Pair-cluster bootstrap 95% interval: [-18.16%, -2.12%]
- Preregistered upper bound: +5.00% → **PASS**

| Workload | Category | Actual tokens | Pages | Median regression | P95 regression |
|---|---|---:|---:|---:|---:|
| controlled-cross-page | controlled | 17 | 2 | +2.43% | +243.28% |
| database-zh | natural-zh | 15 | 1 | -12.95% | +131.99% |
| machine-learning-zh | natural-zh | 14 | 1 | -3.26% | +88.75% |
| attention-en | natural-en | 12 | 1 | -7.14% | +140.50% |
| cuda-mixed | natural-mixed | 11 | 1 | -0.04% | +201.72% |
| code-cpp | code | 15 | 1 | -26.63% | +95.51% |

Positive values mean Paged is slower. Results are stratified rather than inferred from one synthetic prompt.
