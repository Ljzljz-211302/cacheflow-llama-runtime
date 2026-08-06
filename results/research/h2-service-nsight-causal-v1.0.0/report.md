# H2 service-level Nsight causal chain

Three no-profiler paired trials own the request-level effect. A separate replay with identical seeds/configuration links each trial/mode/server PID and deterministic request ID to scheduler counters, KV actions, Nsight Systems CUDA events, and TTFT.

| Trial | TTFT P95 delta always-upstream (ms) | Engine execute delta (us) |
|---:|---:|---:|
| 1 | +46.572 | +181209 |
| 2 | +126.382 | +139825 |
| 3 | +108.810 | +358590 |

Paired median TTFT delta: +108.810 ms; observed range [+46.572, +126.382] ms. Paired median Engine execute delta: +181209 us. With only three pairs, the range is reported as uncertainty rather than a high-confidence population interval.

The preregistered causal gate passed: policy decisions changed, prefill shape changed, custom KV CUDA work changed, and TTFT/Engine outcomes were material. The no-profiler result recorded decision delta +22, prefill chunk delta +32, KV launch delta +2, and TTFT P95 delta +108.810 ms.

The NSYS replay contains 6 linked trial/mode processes and 72 request observations. For every process, PID-filtered NSYS custom-kernel counts exactly equal the runtime Prometheus counter. `causal-links.json` is the machine-readable join.

In this run TTFT worsened together with aggregate Engine execute time; it closes the scheduler/action/CUDA/request linkage but does not reproduce the earlier execute-time sign reversal. The evidence does not establish a universal policy effect, and it does not provide NCU occupancy/DRAM counters.
