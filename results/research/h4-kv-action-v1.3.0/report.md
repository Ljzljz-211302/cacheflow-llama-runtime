# H4 unified KV action policy report

The matched-workload replay compares only actions with complete real-service implementations. Remap and Paged remain capability-masked; Paged recorded zero production decisions.

`observed_cost_ms` is the server policy's internal counter delta from scheduler snapshot through that slot's first completed useful target decode. HTTP round-trip time is retained separately and never enters regret or harm.

| Model | Median regret (ms) | P95 regret (ms) | Harmful rate |
|---|---:|---:|---:|
| H0 | 0.000 | 1.953 | 0.00% |
| A1 | 0.000 | 1.953 | 0.00% |
| T1 | 0.000 | 1.516 | 0.00% |
| L1 | 0.000 | 1.953 | 0.00% |

L1 made no held-out switch because its conservative bound did not beat H0. It therefore matched H0. T1 differed from H0; the complete metrics above define the retained result. The selected production behavior is H0 execution with L1 shadow recommendations.

Decision overhead: p99 1.000 us; observed max 1997.800 us; scheduler/action ratio p99 0.0644%; hot-loop allocations 0; direct CUDA synchronization symbols 0.

The action servers can expose different stateful feature values; their maximum normalized feature deltas are retained in report.json and checked against protocol gates. The shared model input is the real H0 anchor, so this is not described as an exact cloned-state causal counterfactual.

Each model's JSON summary includes a 10,000-resample paired trace-cluster bootstrap 95% CI for mean regret delta versus H0.

The observed maximum is a Windows wall-clock measurement and includes thread preemption. It is reported without trimming.
