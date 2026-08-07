# H3 restricted paged-decode report

## Scope

This is a decode-only CUDA prototype, not a production llama.cpp dispatch integration. The primary effects are 20-pair no-profiler CUDA-event measurements. NSYS replays only bind kernel identity and launch count; profiler durations are not primary timings.

## No-profiler paired results

| Regime | GPU improvement median [95% CI] | End-to-end improvement median [95% CI] | Class |
|---|---:|---:|---|
| q05-boundary-16-identity | -11.82% [-13.33%, -7.20%] | -5.86% [-10.04%, +6.16%] | material-loss |
| q05-boundary-17-fragmented | -13.22% [-191.74%, -8.50%] | -9.60% [-143.27%, -4.75%] | material-loss |
| q05-medium-b1-fragmented | -12.68% [-12.80%, -12.13%] | -12.36% [-12.76%, -12.26%] | material-loss |
| q05-long-b1-fragmented | -13.05% [-13.06%, -12.97%] | -12.85% [-12.98%, -12.36%] | material-loss |
| q05-long-b4-fragmented | -12.17% [-12.29%, -12.07%] | -12.19% [-12.41%, -11.94%] | material-loss |
| q7-boundary-17-fragmented | -9.41% [-11.76%, -5.56%] | -5.72% [-9.78%, -0.61%] | material-loss |
| q7-medium-b1-fragmented | -11.49% [-11.77%, -11.17%] | -11.13% [-11.27%, -10.78%] | material-loss |
| q7-long-b1-fragmented | -11.46% [-11.60%, -11.42%] | -11.41% [-11.52%, -11.29%] | material-loss |
| q7-long-b4-fragmented | -9.87% [-10.24%, -8.94%] | -9.74% [-10.14%, -8.22%] | material-loss |

## Next kernel decision

The preregistered rule selected **K2-GQA-reuse**: a long-context ratio-7 regime regresses by at least 3 percent. This is a mechanism hypothesis, not a demonstrated speedup; the selected variant must still be implemented and profiled.

## Evidence boundaries

- NSYS coverage is complete for 4 method-specific captures.
- NCU hardware counters are incomplete (ERR_NVGPUCTRPERM, driver incompatibility), so memory-bound classification, occupancy explanation, and DRAM byte attribution are prohibited.
- The D64 shape matches the local Qwen2.5-0.5B geometry. D128 matches the Qwen2.5-7B kernel geometry only and is not end-to-end 7B serving evidence.
- Negative, neutral, and uncertain regimes are retained without outcome-based deletion.
