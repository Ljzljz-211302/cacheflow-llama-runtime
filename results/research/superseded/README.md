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

`h4-kv-action-v1.2.0` added runtime H0 anchors, balanced collection order,
trace-cluster bootstrap, and source-bound lexical synchronization audit. It is
still superseded: action costs came from servers whose stateful feature vectors
differed from the shared H0 anchor, one physical Recompute observation was
duplicated across both regimes, replay ignored physical observation order, raw
Prometheus snapshots were not retained, and the lexical audit was described too
strongly as runtime zero-sync evidence. Do not cite v1.2.0 as formal evidence.
