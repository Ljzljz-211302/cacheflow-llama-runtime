# H15 v1 invalidation record

This run is retained but excluded from performance inference. The v1 client used a thread barrier to release independent HTTP requests. Backend counters proved that all requested sequences reached CUDA with zero fallback, but the scheduler split a nominal wave into a variable number of graphs. Consequently the configured client concurrency was not a controlled kernel batch size, and the preregistered one-graph-per-wave invariant failed.

The replacement v2 protocol uses the server's native multiple-prompt `/completion` request so all completion tasks exist before scheduling. No H15 timing observation was used to set or change the v2 performance thresholds.
