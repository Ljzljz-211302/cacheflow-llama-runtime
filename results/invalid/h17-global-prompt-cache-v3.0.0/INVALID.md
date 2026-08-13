# H17 v3 invalidation record

The device-side counter fix worked: every sequence-layer execution was counted across CUDA Graph replay, batch 8 formed exactly one graph in every primary cell, and fallback remained zero. Four secondary batch-4 cells nevertheless split each nominal wave into two graphs. The service log and configuration identify a second, process-wide RAM prompt cache in addition to the controlled per-slot KV cache; its task reassignment made graph composition a confounder.

The v4 replacement disables only the process-wide RAM prompt cache with `--cache-ram 0`, retains per-slot `cache_prompt` reuse, and keeps the workload, matrix, sample size, primary batch, statistics, and performance thresholds unchanged.
