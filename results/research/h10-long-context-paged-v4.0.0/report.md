# Objective Paged-vs-Direct prompt-matrix report

- Frozen corpus: `3.0.0`; 18 workloads
- Design: 10 randomized matched-process blocks × Direct/Paged × every workload
- Raw workload-arm observations: 360
- Primary median matched-block regression: +50.35%
- Process-block cluster bootstrap 95% interval: [+49.19%, +51.19%]
- Preregistered median-CI upper bound: +5.00%
- P95 regression / limit: +63.74% / +20.00%
- Worst workload median / limit: +53.46% / +10.00%
- Required page coverage passed: True → **FAIL**

| Workload | Category | Actual tokens | Pages | Median regression | P95 regression |
|---|---|---:|---:|---:|---:|
| architecture-64 | system-design | 64 | 4 | +11.89% | +35.93% |
| architecture-128 | system-design | 128 | 8 | +24.71% | +49.63% |
| architecture-256 | system-design | 256 | 16 | +53.05% | +61.56% |
| architecture-512 | system-design | 512 | 32 | +49.00% | +72.02% |
| architecture-1024 | system-design | 1024 | 64 | +48.48% | +55.75% |
| architecture-2048 | system-design | 2048 | 128 | +40.82% | +55.57% |
| interview-64 | interview-knowledge | 64 | 4 | +15.32% | +42.45% |
| interview-128 | interview-knowledge | 128 | 8 | +20.79% | +49.92% |
| interview-256 | interview-knowledge | 256 | 16 | +51.45% | +61.44% |
| interview-512 | interview-knowledge | 512 | 32 | +53.11% | +60.13% |
| interview-1024 | interview-knowledge | 1024 | 64 | +47.37% | +63.72% |
| interview-2048 | interview-knowledge | 2048 | 128 | +52.50% | +60.57% |
| paged-research-64 | research-method | 64 | 4 | -1.26% | +28.58% |
| paged-research-128 | research-method | 128 | 8 | +24.00% | +47.33% |
| paged-research-256 | research-method | 256 | 16 | +40.55% | +54.45% |
| paged-research-512 | research-method | 512 | 32 | +49.31% | +70.08% |
| paged-research-1024 | research-method | 1024 | 64 | +53.46% | +62.11% |
| paged-research-2048 | research-method | 2048 | 128 | +51.72% | +63.34% |

Positive values mean Paged is slower. Blocks randomize independent process order; they are not shared-hot-state Trial Pairs. The P95 gate is the tail of block-workload regressions, not request-latency P95.
