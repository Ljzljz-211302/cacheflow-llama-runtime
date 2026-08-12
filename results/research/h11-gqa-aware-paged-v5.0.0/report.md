# Objective Paged-vs-Direct prompt-matrix report

- Frozen corpus: `4.0.0`; 18 workloads
- Design: 10 randomized matched-process blocks × Direct/Paged × every workload
- Raw workload-arm observations: 360
- Primary median matched-block regression: +2.33%
- Process-block cluster bootstrap 95% interval: [+0.22%, +5.51%]
- Preregistered median-CI upper bound: +5.00%
- P95 regression / limit: +25.74% / +20.00%
- Worst workload median / limit: +7.76% / +10.00%
- Required page coverage passed: True → **FAIL**

| Workload | Category | Actual tokens | Pages | Median regression | P95 regression |
|---|---|---:|---:|---:|---:|
| architecture-64 | system-design | 64 | 4 | -7.81% | +23.13% |
| architecture-128 | system-design | 128 | 8 | +1.31% | +10.33% |
| architecture-256 | system-design | 256 | 16 | -12.62% | -0.72% |
| architecture-512 | system-design | 512 | 32 | +3.77% | +9.24% |
| architecture-1024 | system-design | 1024 | 64 | +0.64% | +9.61% |
| architecture-2048 | system-design | 2048 | 128 | +0.07% | +11.63% |
| interview-64 | interview-knowledge | 64 | 4 | -5.77% | +4.18% |
| interview-128 | interview-knowledge | 128 | 8 | +4.12% | +36.31% |
| interview-256 | interview-knowledge | 256 | 16 | +1.65% | +22.19% |
| interview-512 | interview-knowledge | 512 | 32 | +7.76% | +28.53% |
| interview-1024 | interview-knowledge | 1024 | 64 | +4.86% | +19.13% |
| interview-2048 | interview-knowledge | 2048 | 128 | +7.15% | +32.79% |
| paged-research-64 | research-method | 64 | 4 | +0.51% | +25.51% |
| paged-research-128 | research-method | 128 | 8 | +3.25% | +25.20% |
| paged-research-256 | research-method | 256 | 16 | -2.05% | +3.70% |
| paged-research-512 | research-method | 512 | 32 | +3.40% | +26.44% |
| paged-research-1024 | research-method | 1024 | 64 | -0.34% | +3.71% |
| paged-research-2048 | research-method | 2048 | 128 | +2.20% | +9.62% |

Positive values mean Paged is slower. Blocks randomize independent process order; they are not shared-hot-state Trial Pairs. The P95 gate is the tail of block-workload regressions, not request-latency P95.
