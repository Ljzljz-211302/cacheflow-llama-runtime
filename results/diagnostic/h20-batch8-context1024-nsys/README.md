# H20 batch-8/context-1024 NSYS root-cause captures

These captures are diagnostic evidence, not a preregistered performance result. They were used to locate the H19 regression before the H20 protocol was frozen. The formal acceptance result is `results/research/h20-paged-hybrid-batch8-v6.1.0`.

The measurement worktree contained one pre-existing vendor overlay. Its exact 374-byte diff is archived at `patches/h20-vendor-overlay.patch`; SHA-256 `c2a48bcc5623301f1f5f7c80c5e53b75da9ef16b6a7d0c17142971c279d865e7` matches the frozen protocol and execution-start binding. It is a compile-compatibility alias, not part of the H20 routing intervention.

All runs use Qwen2.5-0.5B on the same RTX 4050 Laptop GPU and native batch 8. SQLite rows are the exported NSYS trace; `.nsys-rep` is retained as the raw profiler capture. Kernel timing is profiler-perturbed and supports mechanism selection only.

| Capture | Target kernel | Launches in selected decode range | Grid X | Mean duration |
|---|---|---:|---:|---:|
| `direct` | upstream `flash_attn_ext_f16<64,64,1,8>` | 96 | 40 | 26.078 μs |
| `paged` | original custom K4 | 96 | 256 | 40.543 μs |
| `paged` | K4 merge | 96 | 112 | 1.621 μs |
| `paged-adaptive3` | K4 with host/kernel partition 128 | 24 | 128 | 41.626 μs |
| `paged-k5` | experimental WMMA K5, partition 128 | 24 | 128 | 46.693 μs |
| `paged-k5-p256` | experimental WMMA K5, partition 256 | 24 | 64 | 48.687 μs |

The comparison rejects two shallow fixes. Halving K4 grid/scratch did not reduce arithmetic time, and the first WMMA K5 was slower than K4. The production fix therefore separates Paged KV/layout semantics from arithmetic dispatch: contiguous physical pages use upstream attention, while fragmented pages retain the oracle-validated custom K4.

File hashes are recorded in `manifest.json`. The partial 24-launch captures are not compared as end-to-end latency trials; they only compare target-kernel geometry and mean duration under NSYS.

`health-server.log`, `paged-adaptive.*`, and `paged-adaptive2.*` are retained aborted setup attempts. They contain no reported mechanism number and are listed only so the diagnostic directory has a complete audit trail rather than silently deleting failed profiler work.
