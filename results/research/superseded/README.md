# Superseded research artifacts

`h4-kv-action-v1.0.0` is retained for audit only. The final review found that
its `observed_cost_ms` used client HTTP round-trip time instead of the
preregistered internal boundary from scheduler snapshot through the same
slot's first useful decode. It also evaluated a batch-fit seven-feature Ridge
rather than the production nine-feature online Ridge and did not enforce the
registered pair/bootstrap/CUDA-sync gates.

Do not cite v1.0.0 as formal evidence. The corrected protocol and artifact are
version 1.1.0; v1.0.0 remains byte-for-byte preserved beneath this directory so
the negative audit trail is not erased.
