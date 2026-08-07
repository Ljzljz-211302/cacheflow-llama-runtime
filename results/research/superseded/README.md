# Superseded research artifacts

`h4-kv-action-v1.0.0` is retained for audit only. The final review found that
its `observed_cost_ms` used client HTTP round-trip time instead of the
preregistered internal boundary from scheduler snapshot through the same
slot's first useful decode. It also evaluated a batch-fit seven-feature Ridge
rather than the production nine-feature online Ridge and did not enforce the
registered pair/bootstrap/CUDA-sync gates.

`h4-kv-action-v1.1.0` fixed those defects, but the next review found four
remaining contract problems: its formal rows replaced the runtime feature
vector with a synthetic vector, its bootstrap resampled paired regimes rather
than whole traces, collection order was not preregistered and balanced, and the
zero-CUDA-sync claim came from a benchmark literal rather than an auditable
source constraint.

Do not cite v1.0.0 or v1.1.0 as formal evidence. Both remain byte-for-byte
preserved beneath this directory so the negative audit trail is not erased.
