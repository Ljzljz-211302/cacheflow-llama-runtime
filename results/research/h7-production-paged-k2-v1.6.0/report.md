# Issue #7 production Paged service experiment

- Protocol: `1.6.0` (`fec6133ad994f29166c460f5250c6ac715b8866302e60e5a2685ea596947de4e`)
- Device: NVIDIA GeForce RTX 4050 Laptop GPU (compute capability 8.9)
- Paired trials: 20, alternating AB/BA, no outcome-based deletion
- Direct client latency: median 19.177 ms, P95 32.095 ms
- Paged client latency: median 19.661 ms, P95 34.579 ms
- Paired Paged - Direct: median +0.575 ms, P95 +24.327 ms
- Paired median bootstrap 95% interval: [-3.820, +10.659] ms
- P95 regression: +7.74% (preregistered promotion limit: +5.00%)
- Correctness: exact output match in every pair; 20 production Paged graph entries; 0 fallbacks
- Kernel variant: K2-T2-lane0-softmax
- Mechanism replay: 24 `cacheflow_paged_decode_fattn_k2_t2` launches (one per model layer); profiler timing is non-primary

The opt-in production promotion gate **did not pass**. This result applies only to the preregistered Qwen2.5-0.5B, batch-1, short-context envelope. It is not a long-context, memory-bound, or universal speedup claim.
