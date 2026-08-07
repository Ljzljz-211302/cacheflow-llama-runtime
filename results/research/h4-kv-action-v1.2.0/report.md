# H4 unified KV action policy report

The confirmatory replay compares only actions with complete real-service implementations. Remap and Paged remain capability-masked; Paged recorded zero production decisions.

`observed_cost_ms` is the server policy's internal counter delta from scheduler snapshot through that slot's first completed useful target decode. HTTP round-trip time is retained separately and never enters regret or harm.

| Model | Median regret (ms) | P95 regret (ms) | Harmful rate |
|---|---:|---:|---:|
| H0 | 0.000 | 1.488 | 0.00% |
| A1 | 0.000 | 1.488 | 0.00% |
| T1 | 0.000 | 1.488 | 7.50% |
| L1 | 0.000 | 1.488 | 0.00% |

L1 made no held-out switch because its conservative bound did not beat H0. It therefore matched H0. T1 produced a higher harmful-decision rate than H0. The selected production behavior is H0 execution with L1 shadow recommendations.

Decision overhead: p99 0.600 us; observed max 242.700 us; scheduler/action ratio p99 0.0460%; hot-loop allocations 0; CUDA synchronizations 0.

Each model's JSON summary includes a 10,000-resample paired trace-cluster bootstrap 95% CI for mean regret delta versus H0.

The observed maximum is a Windows wall-clock measurement and includes thread preemption. It is reported without trimming.
