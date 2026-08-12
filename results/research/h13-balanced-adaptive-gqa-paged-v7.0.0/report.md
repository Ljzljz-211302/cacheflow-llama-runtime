# Objective Paged-vs-Direct prompt-matrix report

- Frozen corpus: `4.0.0`; 18 workloads
- Design: 12 randomized matched-process blocks × Direct/Paged × every workload
- Raw workload-arm observations: 432
- Primary median matched-block regression: +3.98%
- Process-block cluster bootstrap 95% interval: [+2.50%, +5.38%]
- Preregistered median-CI upper bound: +5.00%
- P95 regression / limit: +13.34% / +20.00%
- Worst workload median / limit: +8.76% / +10.00%
- Required page coverage passed: True → **FAIL**

| Workload | Category | Actual tokens | Pages | Median regression | P95 regression |
|---|---|---:|---:|---:|---:|
| architecture-64 | system-design | 64 | 4 | -1.92% | +10.10% |
| architecture-128 | system-design | 128 | 8 | +2.51% | +11.28% |
| architecture-256 | system-design | 256 | 16 | +0.53% | +9.12% |
| architecture-512 | system-design | 512 | 32 | +1.70% | +9.70% |
| architecture-1024 | system-design | 1024 | 64 | +3.18% | +14.51% |
| architecture-2048 | system-design | 2048 | 128 | +8.76% | +14.63% |
| interview-64 | interview-knowledge | 64 | 4 | +0.51% | +9.01% |
| interview-128 | interview-knowledge | 128 | 8 | +3.35% | +10.31% |
| interview-256 | interview-knowledge | 256 | 16 | +3.07% | +4.90% |
| interview-512 | interview-knowledge | 512 | 32 | +1.40% | +8.67% |
| interview-1024 | interview-knowledge | 1024 | 64 | +3.98% | +10.95% |
| interview-2048 | interview-knowledge | 2048 | 128 | +5.92% | +12.94% |
| paged-research-64 | research-method | 64 | 4 | +0.80% | +10.27% |
| paged-research-128 | research-method | 128 | 8 | +1.79% | +10.10% |
| paged-research-256 | research-method | 256 | 16 | +2.72% | +11.76% |
| paged-research-512 | research-method | 512 | 32 | +2.08% | +9.93% |
| paged-research-1024 | research-method | 1024 | 64 | +2.82% | +11.73% |
| paged-research-2048 | research-method | 2048 | 128 | +4.50% | +12.90% |

Positive values mean Paged is slower. Blocks randomize independent process order; they are not shared-hot-state Trial Pairs. The P95 gate is the tail of block-workload regressions, not request-latency P95.
