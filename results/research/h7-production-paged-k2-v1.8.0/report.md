# Issue #7 production Paged service experiment

- Protocol: `1.8.0` (`27ec548f5370cb1f355bf1fb4bec3c46aee3b839205f823e3f74f77f0d51869d`)
- Device: NVIDIA GeForce RTX 4050 Laptop GPU (compute capability 8.9)
- Paired trials: 20, alternating AB/BA, no outcome-based deletion
- Direct client latency: median 9.735 ms, P95 21.191 ms
- Paged client latency: median 20.253 ms, P95 32.764 ms
- Paired Paged - Direct: median +10.084 ms, P95 +15.233 ms
- Paired median bootstrap 95% interval: [+0.080, +12.761] ms
- P95 regression: +54.61% (preregistered promotion limit: +5.00%)
- Correctness: exact output match in every pair; 20 production Paged graph entries; 0 fallbacks
- Kernel variant: K2-T2-parallel-softmax-cached-layout
- Mechanism replay: 24 `cacheflow_paged_decode_fattn_k2_t2` launches (one per model layer); profiler timing is non-primary

The opt-in production promotion gate **did not pass**. This result applies only to the preregistered Qwen2.5-0.5B, batch-1, short-context envelope. It is not a long-context, memory-bound, or universal speedup claim.
