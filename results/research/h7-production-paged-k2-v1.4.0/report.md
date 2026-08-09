# Issue #7 production Paged service experiment

- Protocol: `1.4.0` (`82da87e46dc2f3caae4d103f90d3a777973445a49b13847cf308def8d33f6108`)
- Device: NVIDIA GeForce RTX 4050 Laptop GPU (compute capability 8.9)
- Paired trials: 20, alternating AB/BA, no outcome-based deletion
- Direct client latency: median 19.675 ms, P95 33.960 ms
- Paged client latency: median 20.809 ms, P95 34.414 ms
- Paired Paged - Direct: median +0.933 ms, P95 +12.154 ms
- Paired median bootstrap 95% interval: [-0.472, +3.225] ms
- P95 regression: +1.34% (preregistered promotion limit: +5.00%)
- Correctness: exact output match in every pair; 20 production Paged graph entries; 0 fallbacks
- Kernel variant: K2-T2-warp
- Mechanism replay: 24 `cacheflow_paged_decode_fattn_k2_t2` launches (one per model layer); profiler timing is non-primary

The opt-in production promotion gate **did not pass**. This result applies only to the preregistered Qwen2.5-0.5B, batch-1, short-context envelope. It is not a long-context, memory-bound, or universal speedup claim.
