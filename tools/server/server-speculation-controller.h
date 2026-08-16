#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

struct server_speculation_config {
    bool adaptive = false;
    size_t min_draft_tokens = 1;
    size_t max_draft_tokens = 8;
    double acceptance_alpha = 0.2;
    double disable_below_acceptance = 0.20;
    double grow_above_acceptance = 0.75;
    double kv_pressure_threshold = 0.85;
    size_t cooldown_iterations = 8;
    size_t warmup_observations = 4;
    size_t min_evidence_tokens = 24;
};

struct server_speculation_observation {
    int sequence_id = -1;
    size_t drafted_tokens = 0;
    size_t accepted_tokens = 0;
    double kv_pressure = 0.0;
};

struct server_speculation_state {
    size_t draft_tokens = 0;
    double acceptance_ewma = 1.0;
    uint64_t drafted_total = 0;
    uint64_t accepted_total = 0;
    size_t observations = 0;
    size_t cooldown_remaining = 0;
    bool disabled = false;
};

// Per-sequence feedback controller for speculative draft length. It owns no
// model state; the engine supplies accepted/drafted counts and KV pressure.
class server_speculation_controller {
public:
    explicit server_speculation_controller(server_speculation_config config = {});

    void configure(server_speculation_config config);
    size_t recommend(int sequence_id, size_t configured_max, double kv_pressure);
    void observe(const server_speculation_observation & observation);
    server_speculation_state state(int sequence_id) const;
    uint64_t disabled_low_acceptance_total() const { return disabled_low_acceptance_total_; }
    void reset(int sequence_id);

private:
    server_speculation_config config_;
    std::unordered_map<int, server_speculation_state> states_;
    uint64_t disabled_low_acceptance_total_ = 0;

    server_speculation_state & get_or_create(int sequence_id);
};
