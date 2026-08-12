# Objective Paged-vs-Direct prompt-matrix report

- Frozen corpus: `4.0.0`; 18 workloads
- Design: 12 randomized matched-process blocks × Direct/Paged × every workload
- Raw workload-arm observations: 432
- Primary median matched-block regression: +4.48%
- Process-block cluster bootstrap 95% interval: [+3.54%, +5.45%]
- Preregistered median-CI upper bound: +5.00%
- P95 regression / limit: +16.81% / +20.00%
- Worst workload median / limit: +6.50% / +10.00%
- Required page coverage passed: True → **FAIL**

| Workload | Category | Actual tokens | Pages | Median regression | P95 regression |
|---|---|---:|---:|---:|---:|
| architecture-64 | system-design | 64 | 4 | +2.52% | +10.50% |
| architecture-128 | system-design | 128 | 8 | +3.14% | +8.29% |
| architecture-256 | system-design | 256 | 16 | +2.03% | +7.75% |
| architecture-512 | system-design | 512 | 32 | +3.80% | +18.31% |
| architecture-1024 | system-design | 1024 | 64 | +3.95% | +11.95% |
| architecture-2048 | system-design | 2048 | 128 | +6.50% | +19.16% |
| interview-64 | interview-knowledge | 64 | 4 | +2.76% | +10.01% |
| interview-128 | interview-knowledge | 128 | 8 | +4.42% | +11.88% |
| interview-256 | interview-knowledge | 256 | 16 | +4.46% | +12.17% |
| interview-512 | interview-knowledge | 512 | 32 | +2.92% | +15.73% |
| interview-1024 | interview-knowledge | 1024 | 64 | +4.97% | +8.58% |
| interview-2048 | interview-knowledge | 2048 | 128 | +4.38% | +9.60% |
| paged-research-64 | research-method | 64 | 4 | +2.75% | +6.32% |
| paged-research-128 | research-method | 128 | 8 | +1.98% | +13.57% |
| paged-research-256 | research-method | 256 | 16 | +1.10% | +4.47% |
| paged-research-512 | research-method | 512 | 32 | +5.13% | +12.30% |
| paged-research-1024 | research-method | 1024 | 64 | +4.25% | +21.34% |
| paged-research-2048 | research-method | 2048 | 128 | +3.71% | +10.69% |

Positive values mean Paged is slower. Blocks randomize independent process order; they are not shared-hot-state Trial Pairs. The P95 gate is the tail of block-workload regressions, not request-latency P95.
