# Batched Paged Decode objective performance result

- Promotion: **FAIL**
- Batch-8 throughput gain median: -3.22%
- Matched-process-block bootstrap 95% interval: [-4.90%, -0.12%]
- Batch-8 raw batched-wave P95 Direct/Paged: 131.459/152.495 ms
- Batch-8 raw batched-wave P95 latency regression: +16.00%
- Worst cell median batched-wave latency regression: +50.49%

- Exact output-token matches: 1052/1080
- Top-64 minimum overlap / maximum common logprob error: 58 / 0.685652

- Probability rows compared / incomplete: 1032 / 48

## Throughput by batch

| Batch | Median gain | 95% interval |
|---:|---:|---:|
| 1 | -6.66% | [-18.68%, +32.28%] |
| 2 | -6.04% | [-12.03%, +1.79%] |
| 4 | -2.84% | [-12.60%, +4.95%] |
| 8 | -3.22% | [-4.90%, -0.12%] |

GPU memory is descriptive only: Direct and Paged share the same allocator, so this experiment does not claim a capacity or fragmentation advantage.
