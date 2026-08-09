# H4 unified KV action policy report

The matched-workload replay compares only actions with complete real-service implementations. Remap and Paged remain capability-masked; Paged recorded zero production decisions.

`observed_cost_ms` is the server policy's internal counter delta from scheduler snapshot through that slot's first completed useful target decode. HTTP round-trip time is retained separately and never enters regret or harm.

| Model | Median regret (ms) | P95 regret (ms) | Harmful rate |
|---|---:|---:|---:|
| H0 | 0.000 | 3.169 | 0.00% |
| A1 | 0.000 | 3.169 | 0.00% |
| T1 | 0.000 | 1.128 | 1.25% |
| L1 | 0.000 | 3.169 | 0.00% |
| D1 | 0.000 | 3.169 | 0.00% |
| D2 | 0.000 | 3.169 | 0.00% |
| D3 | 0.000 | 0.000 | 1.25% |

L1 made no held-out switch because its independent absolute-cost bounds did not beat H0. It therefore matched H0. T1 produced a higher harmful-decision rate than H0.

D1 made 0 held-out switches but failed the retained paired-replay gates: minimum_switches_vs_h0, mean_regret_delta_ci95_upper_negative. H0 therefore remains selected.

D1 predicts the paired complete-action delta `candidate - H0`, conditions models by runtime regime, adds four pre-registered mechanism interactions, and adds a one-sided held-out calibration offset before the switch margin.

D2 made 0 held-out switches but failed the retained paired-replay gates: minimum_switches_vs_h0, mean_regret_delta_ci95_upper_negative. H0 therefore remains selected.

D2 preserves D1's signed-delta and fail-closed logic but fits and calibrates separate models on the pre-registered `context_tokens <= 512` and `> 512` bands. Its stricter 0.75 ms margin was locked before this confirmatory collection.

D3 made 24 held-out switches and passed its pre-registered 5% harmful-rate budget: total gain 41.480 ms, total harm 0.972 ms. This authorizes only a monitored canary.

D3 treats isolated slow switches as an explicit risk budget instead of requiring the tautological H0 harmful rate of zero. It still requires a negative paired CI, non-worse P95, at most 5% harmful decisions, and total measured gain above total harm.

| D1 ablation | Median regret (ms) | P95 regret (ms) | Cumulative regret (ms) | Switches |
|---|---:|---:|---:|---:|
| D1-I0-no-interactions | 0.000 | 3.169 | 42.590 | 0 |
| D1-R0-pooled-regimes | 0.000 | 3.169 | 42.590 | 0 |
| D1-C0-no-calibration | 0.000 | 3.169 | 42.590 | 0 |

Decision overhead: p99 0.900 us; observed max 431.400 us; scheduler/action ratio p99 0.0874%; hot-loop allocations 0; direct CUDA synchronization symbols 0.

The action servers can expose different stateful feature values; their maximum normalized feature deltas are retained in report.json and checked against protocol gates. The shared model input is the real H0 anchor, so this is not described as an exact cloned-state causal counterfactual.

Each model's JSON summary includes a 10,000-resample paired trace-cluster bootstrap 95% CI for mean regret delta versus H0.

The observed maximum is a Windows wall-clock measurement and includes thread preemption. It is reported without trimming.
