# Issue #7 production Paged service experiment

- Protocol: `1.3.0` (`44db11883cbbfcb0e4311306a367e097ba2f2a62390fd64aa3e381c33a041882`)
- Device: NVIDIA GeForce RTX 4050 Laptop GPU (compute capability 8.9)
- Paired trials: 20, alternating AB/BA, no outcome-based deletion
- Direct client latency: median 9.742 ms, P95 23.570 ms
- Paged client latency: median 21.029 ms, P95 32.837 ms
- Paired Paged - Direct: median +9.741 ms, P95 +16.794 ms
- Paired median bootstrap 95% interval: [+0.963, +12.324] ms
- P95 regression: +39.31% (preregistered promotion limit: +5.00%)
- Correctness: exact output match in every pair; 20 production Paged graph entries; 0 fallbacks
- Kernel variant: K2-T2-native-Q
- Mechanism replay: 24 `cacheflow_paged_decode_fattn_k2_t2` launches (one per model layer); profiler timing is non-primary

The opt-in production promotion gate **did not pass**. This result applies only to the preregistered Qwen2.5-0.5B, batch-1, short-context envelope. It is not a long-context, memory-bound, or universal speedup claim.
