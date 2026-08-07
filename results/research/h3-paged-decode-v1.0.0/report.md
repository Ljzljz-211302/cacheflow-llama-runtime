# H3 restricted paged-decode report

## Scope

This is a decode-only CUDA prototype, not a production llama.cpp dispatch integration. The primary effects are 20-pair no-profiler CUDA-event measurements. NSYS replays only bind kernel identity and launch count; profiler durations are not primary timings.

Every regime validates both CUDA arms against an independent CPU FP32 paged-attention oracle before timing; maximum absolute error was 0.000000036.

Device: NVIDIA GeForce RTX 4050 Laptop GPU (sm_89), 6141 MiB total and 5921 MiB free before run. Per-regime input/output/workspace and peak device-allocation accounting is retained in `report.json`; every resource gate passed.

## No-profiler paired results

| Regime | GPU improvement median [95% CI] | End-to-end improvement median [95% CI] | Class |
|---|---:|---:|---|
| q05-boundary-16-identity | +0.00% [-10.09%, +6.00%] | -2.85% [-4.73%, -1.48%] | uncertain |
| q05-boundary-17-fragmented | +22.87% [+0.00%, +47.08%] | +6.97% [-2.55%, +8.78%] | uncertain |
| q05-medium-b1-fragmented | -12.73% [-13.02%, -12.46%] | -12.32% [-12.51%, -12.01%] | material-loss |
| q05-long-b1-fragmented | -13.05% [-13.12%, -12.91%] | -12.95% [-13.42%, -12.42%] | material-loss |
| q05-long-b4-fragmented | -12.21% [-12.37%, -12.05%] | -12.06% [-12.52%, -11.84%] | material-loss |
| q7-boundary-17-fragmented | -11.49% [-16.91%, -6.57%] | -8.14% [-15.22%, -3.92%] | material-loss |
| q7-medium-b1-fragmented | -11.28% [-12.00%, -11.22%] | -11.19% [-11.69%, -10.47%] | material-loss |
| q7-long-b1-fragmented | -11.59% [-11.78%, -11.37%] | -11.26% [-11.58%, -10.74%] | material-loss |
| q7-long-b4-fragmented | -10.41% [-11.53%, -9.61%] | -10.24% [-12.38%, -9.20%] | material-loss |

## Next kernel decision

The preregistered rule selected **K2-GQA-reuse**: a long-context ratio-7 regime regresses by at least 3 percent. This is a mechanism hypothesis, not a demonstrated speedup; the selected variant must still be implemented and profiled.

## Evidence boundaries

- NSYS coverage is complete for 4 method-specific captures.
- NCU hardware counters are incomplete (ERR_NVGPUCTRPERM, driver incompatibility), so memory-bound classification, occupancy explanation, and DRAM byte attribution are prohibited.
- The D64 shape matches the local Qwen2.5-0.5B geometry. D128 matches the Qwen2.5-7B kernel geometry only and is not end-to-end 7B serving evidence.
- Negative, neutral, and uncertain regimes are retained without outcome-based deletion.
