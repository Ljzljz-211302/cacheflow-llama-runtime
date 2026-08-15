# Official public workload Direct/Paged result

- Promotion: **FAIL**
- Trace sources: azure-code, burstgpt
- Median throughput gain: -0.04%
- Matched-process-block bootstrap 95% interval: [-1.54%, +0.87%]
- Request P95 Direct/Paged: 2482.391/2511.827 ms
- P95 latency regression: +1.19%
- P95 regression block-bootstrap 95% interval: [-1.51%, +3.13%]
- Maximum arrival slip: 5.605 ms
- Exact trace outputs: 565/576
- Top-probability minimum overlap: 19
- Maximum common-token logprob error: 0.225273
- LongBench paired outputs: 6/6
- Maximum Direct/Paged LongBench score delta: 0.000000

BurstGPT/Azure arrival traces and LongBench text are separately sourced and then matched for replay. This is trace-driven public-content synthetic replay, not a claim about their joint production distribution.
