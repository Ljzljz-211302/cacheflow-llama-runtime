# Objective Paged-vs-Direct prompt-matrix report

- Frozen corpus: `4.0.0`; 18 workloads
- Design: 10 randomized matched-process blocks × Direct/Paged × every workload
- Raw workload-arm observations: 360
- Primary median matched-block regression: +5.93%
- Process-block cluster bootstrap 95% interval: [+4.86%, +7.27%]
- Preregistered median-CI upper bound: +5.00%
- P95 regression / limit: +17.61% / +20.00%
- Worst workload median / limit: +9.99% / +10.00%
- Required page coverage passed: True → **FAIL**

| Workload | Category | Actual tokens | Pages | Median regression | P95 regression |
|---|---|---:|---:|---:|---:|
| architecture-64 | system-design | 64 | 4 | +0.60% | +6.03% |
| architecture-128 | system-design | 128 | 8 | +4.12% | +11.37% |
| architecture-256 | system-design | 256 | 16 | +2.01% | +12.06% |
| architecture-512 | system-design | 512 | 32 | +3.76% | +16.60% |
| architecture-1024 | system-design | 1024 | 64 | +9.99% | +15.61% |
| architecture-2048 | system-design | 2048 | 128 | +6.18% | +12.55% |
| interview-64 | interview-knowledge | 64 | 4 | +2.24% | +11.62% |
| interview-128 | interview-knowledge | 128 | 8 | +3.56% | +21.05% |
| interview-256 | interview-knowledge | 256 | 16 | +1.49% | +14.65% |
| interview-512 | interview-knowledge | 512 | 32 | -0.16% | +7.44% |
| interview-1024 | interview-knowledge | 1024 | 64 | +9.06% | +20.87% |
| interview-2048 | interview-knowledge | 2048 | 128 | +6.02% | +8.08% |
| paged-research-64 | research-method | 64 | 4 | +0.98% | +13.23% |
| paged-research-128 | research-method | 128 | 8 | +4.41% | +16.42% |
| paged-research-256 | research-method | 256 | 16 | +4.87% | +27.63% |
| paged-research-512 | research-method | 512 | 32 | +4.90% | +14.74% |
| paged-research-1024 | research-method | 1024 | 64 | +6.00% | +18.67% |
| paged-research-2048 | research-method | 2048 | 128 | +6.93% | +17.34% |

Positive values mean Paged is slower. Blocks randomize independent process order; they are not shared-hot-state Trial Pairs. The P95 gate is the tail of block-workload regressions, not request-latency P95.
