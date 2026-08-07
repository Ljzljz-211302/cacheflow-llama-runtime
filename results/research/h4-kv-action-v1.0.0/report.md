# H4 unified KV action policy report

The confirmatory replay compares only actions with complete real-service implementations. Remap and Paged remain capability-masked; Paged recorded zero production decisions.

| Model | Median regret (ms) | P95 regret (ms) | Harmful rate |
|---|---:|---:|---:|
| H0 | 0.000 | 7.205 | 0.00% |
| A1 | 0.000 | 7.205 | 0.00% |
| T1 | 0.000 | 3.078 | 6.25% |
| L1 | 0.000 | 7.205 | 0.00% |

L1 made no held-out switch because its conservative bound did not beat H0. It therefore matched H0. T1 produced a higher harmful-decision rate than H0. The selected production behavior is H0 execution with L1 shadow recommendations.

Decision overhead: p99 0.800 us; observed max 481.000 us; scheduler/action ratio p99 0.0345%; hot-loop allocations 0.

The observed maximum is a Windows wall-clock measurement and includes thread preemption. It is reported without trimming.
