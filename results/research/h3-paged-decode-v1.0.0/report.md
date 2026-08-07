# H3 restricted paged-decode report

## Scope

This is a decode-only CUDA prototype, not a production llama.cpp dispatch integration. The primary effects are 20-pair no-profiler CUDA-event measurements. NSYS replays only bind kernel identity and launch count; profiler durations are not primary timings.

Every regime validates both CUDA arms against an independent CPU FP32 paged-attention oracle before timing; maximum absolute error was 0.000000036.

## No-profiler paired results

| Regime | GPU improvement median [95% CI] | End-to-end improvement median [95% CI] | Class |
|---|---:|---:|---|
| q05-boundary-16-identity | -6.46% [-11.43%, +29.56%] | -4.93% [-8.41%, +33.49%] | uncertain |
| q05-boundary-17-fragmented | -9.83% [-13.53%, -6.46%] | -5.92% [-15.06%, +5.13%] | material-loss |
| q05-medium-b1-fragmented | -12.71% [-12.80%, -12.12%] | -12.00% [-12.24%, -10.56%] | material-loss |
| q05-long-b1-fragmented | -13.07% [-13.32%, -12.91%] | -12.78% [-13.44%, -12.56%] | material-loss |
| q05-long-b4-fragmented | -11.69% [-12.06%, -9.85%] | -11.33% [-11.62%, -8.61%] | material-loss |
| q7-boundary-17-fragmented | -7.05% [-11.69%, -5.72%] | -6.59% [-9.29%, -1.93%] | material-loss |
| q7-medium-b1-fragmented | -11.66% [-11.78%, -11.23%] | -11.13% [-11.58%, -10.90%] | material-loss |
| q7-long-b1-fragmented | -11.54% [-11.68%, -11.32%] | -11.20% [-11.42%, -11.09%] | material-loss |
| q7-long-b4-fragmented | -9.85% [-10.16%, -9.30%] | -9.36% [-9.94%, -8.22%] | material-loss |

## Next kernel decision

The preregistered rule selected **K2-GQA-reuse**: a long-context ratio-7 regime regresses by at least 3 percent. This is a mechanism hypothesis, not a demonstrated speedup; the selected variant must still be implemented and profiled.

## Evidence boundaries

- NSYS coverage is complete for 4 method-specific captures.
- NCU hardware counters are incomplete (ERR_NVGPUCTRPERM, driver incompatibility), so memory-bound classification, occupancy explanation, and DRAM byte attribution are prohibited.
- The D64 shape matches the local Qwen2.5-0.5B geometry. D128 matches the Qwen2.5-7B kernel geometry only and is not end-to-end 7B serving evidence.
- Negative, neutral, and uncertain regimes are retained without outcome-based deletion.
