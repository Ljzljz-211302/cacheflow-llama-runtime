# H2 service-level Nsight causal chain

Three no-profiler paired trials own the request-level effect. A separate replay with identical seeds/configuration links each trial/mode/server PID and deterministic request ID to scheduler counters, KV actions, Nsight Systems CUDA events, and TTFT.

| Trial | TTFT P95 delta always-upstream (ms) | Engine execute delta (us) |
|---:|---:|---:|
| 1 | +354.462 | +258596 |
| 2 | +31.212 | +88904 |
| 3 | +346.820 | +439769 |

Paired median TTFT delta: +346.820 ms; observed range [+31.212, +354.462] ms. Paired median Engine execute delta: +258596 us. With only three pairs, the range is reported as uncertainty rather than a high-confidence population interval.

The preregistered causal gate passed: policy decisions changed, prefill shape changed, custom KV CUDA work changed, and TTFT/Engine outcomes were material. The no-profiler result recorded decision delta +26, prefill chunk delta +39, KV launch delta +0, and TTFT P95 delta +346.820 ms.

The NSYS replay contains 6 linked trial/mode processes and 72 request observations. For every process, PID-filtered NSYS custom-kernel counts exactly equal the runtime Prometheus counter. `causal-links.json` is the machine-readable join.

In this run TTFT worsened together with aggregate Engine execute time; it closes the scheduler/action/CUDA/request linkage but does not reproduce the earlier execute-time sign reversal. The evidence does not establish a universal policy effect, and it does not provide NCU occupancy/DRAM counters.
