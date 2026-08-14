# Batched Paged Decode objective performance result

- Promotion: **PASS**
- Batch-8 throughput gain median: +1.54%
- Matched-process-block bootstrap 95% interval: [-2.06%, +3.86%]
- Batch-8 raw batched-wave P95 Direct/Paged: 142.532/141.449 ms
- Batch-8 raw batched-wave P95 latency regression: -0.76%
- Worst cell median batched-wave latency regression: +12.05%

- Execution mode: `contiguous_fastpath`
- Contiguous fast-path calls/sequences: 216/1728
- Custom Paged graph/CUDA dispatches: 0/0

- Exact output-token matches: 1728/1728
- Top-64 minimum overlap / maximum common logprob error: 64 / 0.000000

- Probability rows compared / incomplete: 1728 / 0

## Throughput by batch

| Batch | Median gain | 95% interval |
|---:|---:|---:|
| 8 | +1.54% | [-2.06%, +3.86%] |

GPU memory is descriptive only: Direct and Paged share the same allocator, so this experiment does not claim a capacity or fragmentation advantage.
