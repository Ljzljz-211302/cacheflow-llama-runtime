# H18 v4 invalidation record

Disabling the process-wide RAM prompt cache reproduced the same four secondary batch-4 graph splits as v3, while device counters still proved all 24 layer executions for every sequence and primary batch 8 remained one graph per wave. This falsifies the cache-confound hypothesis: the split is deterministic production scheduler behavior for those inputs.

V4 is excluded because its blanket one-graph requirement conflated requested application batch with realized scheduler graph size. V5 retains the strict one-graph requirement for the preregistered primary batch 8, while secondary batch sizes report realized sequences per graph and still require complete device-counted execution and zero fallback. Performance thresholds are unchanged.
