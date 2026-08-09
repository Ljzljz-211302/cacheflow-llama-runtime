# Issue #7 production Paged service experiment

- Protocol: `1.7.0` (`63640305925161236fd2578af50e6912624e653825b637b3fb1984a7bfa2dd57`)
- Device: NVIDIA GeForce RTX 4050 Laptop GPU (compute capability 8.9)
- Paired trials: 20, alternating AB/BA, no outcome-based deletion
- Direct client latency: median 20.305 ms, P95 33.390 ms
- Paged client latency: median 20.327 ms, P95 35.094 ms
- Paired Paged - Direct: median +0.167 ms, P95 +15.308 ms
- Paired median bootstrap 95% interval: [-9.502, +2.700] ms
- P95 regression: +5.10% (preregistered promotion limit: +5.00%)
- Correctness: exact output match in every pair; 20 production Paged graph entries; 0 fallbacks
- Kernel variant: K2-T2-parallel-short-softmax
- Mechanism replay: 24 `cacheflow_paged_decode_fattn_k2_t2` launches (one per model layer); profiler timing is non-primary

The opt-in production promotion gate **did not pass**. This result applies only to the preregistered Qwen2.5-0.5B, batch-1, short-context envelope. It is not a long-context, memory-bound, or universal speedup claim.
