# Issue #7 production Paged service experiment

- Protocol: `1.5.0` (`f45989060a09bf59cd4e52c9286dd40d606d0f8208ed7d46647a7e88a1069268`)
- Device: NVIDIA GeForce RTX 4050 Laptop GPU (compute capability 8.9)
- Paired trials: 20, alternating AB/BA, no outcome-based deletion
- Direct client latency: median 19.510 ms, P95 33.792 ms
- Paged client latency: median 21.131 ms, P95 34.147 ms
- Paired Paged - Direct: median +0.976 ms, P95 +22.534 ms
- Paired median bootstrap 95% interval: [-3.563, +6.165] ms
- P95 regression: +1.05% (preregistered promotion limit: +5.00%)
- Correctness: exact output match in every pair; 20 production Paged graph entries; 0 fallbacks
- Kernel variant: K2-T2-one-warp-head
- Mechanism replay: 24 `cacheflow_paged_decode_fattn_k2_t2` launches (one per model layer); profiler timing is non-primary

The opt-in production promotion gate **did not pass**. This result applies only to the preregistered Qwen2.5-0.5B, batch-1, short-context envelope. It is not a long-context, memory-bound, or universal speedup claim.
