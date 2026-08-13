# H16 v2 invalidation record

This run is retained but excluded from performance inference. Native multiple-prompt requests did create controlled Paged graphs, process every sequence, and produce zero fallback. However, the original CUDA dispatch metric incremented on the host submission path. CUDA Graph replay bypasses that host path, so the metric counted graph capture rather than every actual kernel execution and could not satisfy the v2 execution-evidence contract.

The root fix moves the counter into the Paged CUDA kernels. Device-side atomic increments execute during both ordinary launch and CUDA Graph replay. The replacement v3 protocol binds the rebuilt runtime and retains all original statistical thresholds.
