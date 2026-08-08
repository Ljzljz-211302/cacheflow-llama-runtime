# Issue #7 production Paged service experiment

- Protocol: `1.0.0` (`ba22fade51f3c901d113bbc48105cca8917765ff84d3a5d323e30c14b8a537cc`)
- Device: NVIDIA GeForce RTX 4050 Laptop GPU (compute capability 8.9)
- Paired trials: 10, alternating AB/BA, no outcome-based deletion
- Direct client latency: median 9.100 ms, P95 29.160 ms
- Paged client latency: median 30.118 ms, P95 37.861 ms
- Paired Paged - Direct: median +10.886 ms, P95 +26.905 ms
- Paired median bootstrap 95% interval: [+0.952, +22.929] ms
- P95 regression: +29.84% (preregistered promotion limit: +5.00%)
- Correctness: exact output match in every pair; 10 production Paged graph entries; 0 fallbacks
- Mechanism replay: 24 `cacheflow_paged_decode_fattn_k1<64>` launches (one per model layer); profiler timing is non-primary

The opt-in production promotion gate **did not pass**. This result applies only to the preregistered Qwen2.5-0.5B, batch-1, short-context envelope. It is not a long-context, memory-bound, or universal speedup claim.
