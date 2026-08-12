# Objective Paged-vs-Direct prompt-matrix report

- Frozen corpus: `3.0.0`; 18 workloads
- Design: 10 randomized matched-process blocks × Direct/Paged × every workload
- Raw workload-arm observations: 360
- Primary median matched-block regression: +0.00%
- Process-block cluster bootstrap 95% interval: [+0.00%, +0.00%]
- Preregistered median-CI upper bound: +5.00%
- P95 regression / limit: +0.00% / +20.00%
- Worst workload median / limit: +0.00% / +10.00%
- Required page coverage passed: True → **PASS**

| Workload | Category | Actual tokens | Pages | Median regression | P95 regression |
|---|---|---:|---:|---:|---:|
| architecture-64 | system-design | 64 | 4 | +0.00% | +0.00% |
| architecture-128 | system-design | 128 | 8 | +0.00% | +0.00% |
| architecture-256 | system-design | 256 | 16 | +0.00% | +0.00% |
| architecture-512 | system-design | 512 | 32 | +0.00% | +0.00% |
| architecture-1024 | system-design | 1024 | 64 | +0.00% | +0.00% |
| architecture-2048 | system-design | 2048 | 128 | +0.00% | +0.00% |
| interview-64 | interview-knowledge | 64 | 4 | +0.00% | +0.00% |
| interview-128 | interview-knowledge | 128 | 8 | +0.00% | +0.00% |
| interview-256 | interview-knowledge | 256 | 16 | +0.00% | +0.00% |
| interview-512 | interview-knowledge | 512 | 32 | +0.00% | +0.00% |
| interview-1024 | interview-knowledge | 1024 | 64 | +0.00% | +0.00% |
| interview-2048 | interview-knowledge | 2048 | 128 | +0.00% | +0.00% |
| paged-research-64 | research-method | 64 | 4 | +0.00% | +0.00% |
| paged-research-128 | research-method | 128 | 8 | +0.00% | +0.00% |
| paged-research-256 | research-method | 256 | 16 | +0.00% | +0.00% |
| paged-research-512 | research-method | 512 | 32 | +0.00% | +0.00% |
| paged-research-1024 | research-method | 1024 | 64 | +0.00% | +0.00% |
| paged-research-2048 | research-method | 2048 | 128 | +0.00% | +0.00% |

Positive values mean Paged is slower. Blocks randomize independent process order; they are not shared-hot-state Trial Pairs. The P95 gate is the tail of block-workload regressions, not request-latency P95.
