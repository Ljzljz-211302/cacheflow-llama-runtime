#pragma once

#include "server-model-cost-model.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

struct server_scheduler_config {
    float  min_prefix_similarity = 0.0f;
    float  eviction_penalty      = 0.0f;
    size_t prefill_chunk_size    = 0; // zero preserves the upstream greedy policy
    bool   adaptive_prefill      = false;
    size_t prefill_chunk_min     = 16;
    size_t prefill_chunk_max     = 512;
    double target_iteration_ms   = 25.0;
    bool   adaptive_greedy_fallback = false;
};

struct server_slot_candidate {
    int     id                   = -1;
    size_t  common_prefix_tokens = 0;
    size_t  cached_tokens        = 0;
    int64_t last_used            = -1;
};

struct server_slot_plan {
    int     id             = -1;
    size_t  reused_tokens  = 0;
    size_t  evicted_tokens = 0;
    double  score          = -std::numeric_limits<double>::infinity();
    int64_t last_used      = std::numeric_limits<int64_t>::max();

    bool found() const { return id >= 0; }
};

struct server_prefill_candidate {
    int    id               = -1;
    size_t remaining_tokens = 0;
};

struct server_prefill_allocation {
    int    id     = -1;
    size_t tokens = 0;
};

struct server_prefill_plan {
    std::vector<server_prefill_allocation> allocations;
    size_t token_budget = 0;
    size_t tokens_scheduled = 0;
    size_t effective_chunk_size = 0;

    size_t quota(int id) const;
    bool equivalent_allocations(const server_prefill_plan & other) const;
};

struct server_scheduler_observation {
    size_t decode_tokens = 0;
    size_t prefill_tokens = 0;
    double elapsed_ms = 0.0;
    size_t prefill_sequences = 0;
};

struct server_scheduler_state {
    bool adaptive_prefill = false;
    size_t effective_chunk_size = 0;
    double iteration_latency_ewma_ms = 0.0;
    uint64_t observations = 0;
    server_model_cost_state cost;
};

// Owns request-to-slot placement and iteration-level prefill allocation.
// The server supplies snapshots; the module returns plans without touching
// llama_context, which keeps policy independently testable.
class server_inference_scheduler {
public:
    explicit server_inference_scheduler(server_scheduler_config config = {});

    void configure(server_scheduler_config config);

    server_slot_plan select_slot(
            const std::vector<server_slot_candidate> & candidates,
            size_t input_tokens) const;

    server_prefill_plan plan_prefill(
            const std::vector<server_prefill_candidate> & candidates,
            size_t token_budget);

    // Pure counterfactual used by benefit gating. It exactly reproduces the
    // upstream greedy allocator without mutating the CacheFlow fairness cursor.
    static server_prefill_plan plan_upstream_prefill(
            const std::vector<server_prefill_candidate> & candidates,
            size_t token_budget);

    void observe_iteration(const server_scheduler_observation & observation);
    server_scheduler_state state() const;

private:
    server_scheduler_config config_;
    int last_prefill_slot_ = -1;
    size_t adaptive_chunk_size_ = 0;
    double iteration_latency_ewma_ms_ = 0.0;
    uint64_t observations_ = 0;
    server_model_cost_model cost_model_;
};
