# H2 KV movement profiling report

## Result boundary

No-profiler paired CUDA-event and end-to-end timings are the primary effects. Nsight Systems traces are separate mechanism observations and are never used as latency samples. Nsight Compute hardware-counter claims are allowed only when all configured metrics were captured.

## Regimes and uncertainty

| Regime | Layout | Blocks | GPU improvement, median [95% CI] | End-to-end improvement, median [95% CI] | Effect | NSYS scalar/vector kernel ms | Launches scalar/vector |
|---|---|---:|---:|---:|---|---:|---:|
| aligned-small | aligned | 1 | +57.10% [+38.51%, +57.14%] | +36.58% [+22.52%, +47.38%] | material-win | 0.166241/0.050432 | 10/10 |
| aligned-transition | aligned | 16 | +5.24% [+4.55%, +5.57%] | +4.73% [+3.22%, +5.75%] | neutral | 4.713588/4.355699 | 10/10 |
| aligned-large | aligned | 32 | +3.93% [+3.90%, +4.70%] | +3.94% [+3.66%, +4.49%] | neutral | 9.286687/8.912796 | 10/10 |
| misaligned-small | misaligned | 1 | -137.94% [-140.55%, -137.84%] | -113.02% [-114.82%, -110.77%] | material-loss | 0.167328/0.336576 | 10/10 |

The 95% intervals are deterministic paired percentile-bootstrap intervals of the median, using the fixed sample count and seed in the protocol. No outcome-based outlier deletion or optional stopping is used.

## Causal interpretation

- Equal scalar/vector launch counts in each NSYS capture rule out a reduction in launch count as the cause of vectorization gains. Kernel duration changes remain visible in the same profiler range.
- The misaligned regime is retained as a layout counterexample: the vector kernel executes its scalar fallback lanes with a grid sized for vector work. Its paired end-to-end effect is reported even when it loses.
- Effective payload GB/s in `report.json` is logical bytes divided by CUDA-event time. It is not hardware DRAM throughput and cannot establish a roofline ceiling.

## Nsight Compute boundary

NCU hardware counters are incomplete (observed: ERR_NVGPUCTRPERM, installed profiler/driver compatibility check). Therefore this report prohibits memory-bound, roofline, achieved-occupancy, and hardware-DRAM-byte claims. The failed commands and logs are retained rather than replaced with estimates.

## Raw evidence map

For every regime, `raw/<regime>-no-profiler.csv` contains 20 paired trials. `profiles/<regime>/nsys.nsys-rep` is the native report and `nsys.sqlite` is the parser input. `manifest.json` records commands, tool versions, revisions, dirty flags, binary/protocol hashes, and every trace hash.
