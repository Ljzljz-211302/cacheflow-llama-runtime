# Issue #7 production Paged service experiment

- Protocol: `1.2.0` (`d18d22699fd963a63e58e0707713507666d5f7b11f60a08a808f1812b1c4d23f`)
- Device: NVIDIA GeForce RTX 4050 Laptop GPU (compute capability 8.9)
- Paired trials: 20, alternating AB/BA, no outcome-based deletion
- Direct client latency: median 19.407 ms, P95 33.043 ms
- Paged client latency: median 21.833 ms, P95 33.913 ms
- Paired Paged - Direct: median +4.269 ms, P95 +13.687 ms
- Paired median bootstrap 95% interval: [+0.786, +12.164] ms
- P95 regression: +2.63% (preregistered promotion limit: +5.00%)
- Correctness: exact output match in every pair; 20 production Paged graph entries; 0 fallbacks
- Kernel variant: K2-T2
- Mechanism replay: 24 `cacheflow_paged_decode_fattn_k2_t2` launches (one per model layer); profiler timing is non-primary

The opt-in production promotion gate **did not pass**. This result applies only to the preregistered Qwen2.5-0.5B, batch-1, short-context envelope. It is not a long-context, memory-bound, or universal speedup claim.
