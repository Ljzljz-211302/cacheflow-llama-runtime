# Issue #7 production Paged service experiment

- Protocol: `1.1.0` (`f3df4c362ff7a79c7dba4fd5a419203132c574c2d874f0617599a49ce3707c9e`)
- Device: NVIDIA GeForce RTX 4050 Laptop GPU (compute capability 8.9)
- Paired trials: 10, alternating AB/BA, no outcome-based deletion
- Direct client latency: median 16.054 ms, P95 27.354 ms
- Paged client latency: median 21.404 ms, P95 29.210 ms
- Paired Paged - Direct: median +2.705 ms, P95 +15.259 ms
- Paired median bootstrap 95% interval: [-1.185, +12.019] ms
- P95 regression: +6.78% (preregistered promotion limit: +5.00%)
- Correctness: exact output match in every pair; 10 production Paged graph entries; 0 fallbacks
- Mechanism replay: 24 `cacheflow_paged_decode_fattn_k1<64>` launches (one per model layer); profiler timing is non-primary

The opt-in production promotion gate **did not pass**. This result applies only to the preregistered Qwen2.5-0.5B, batch-1, short-context envelope. It is not a long-context, memory-bound, or universal speedup claim.
