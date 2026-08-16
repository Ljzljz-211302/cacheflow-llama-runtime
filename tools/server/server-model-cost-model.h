#pragma once

#include <cstddef>
#include <cstdint>

struct server_model_cost_state {
    double decode_ms_per_token = 0.0;
    double prefill_ms_per_token = 0.0;
    double fixed_ms = 0.0;
    uint64_t observations = 0;
    double confidence = 0.0;
};

// Online exponentially weighted latency model:
// elapsed = decode_tokens * D + prefill_tokens * P + fixed.
class server_model_cost_model {
public:
    explicit server_model_cost_model(double decay = 0.95);
    void reset();
    void observe(size_t decode_tokens, size_t prefill_tokens, double elapsed_ms);
    server_model_cost_state state() const;
    size_t recommend_prefill_chunk(
            double target_iteration_ms,
            size_t decode_tokens,
            size_t active_prefill_sequences,
            size_t minimum,
            size_t maximum) const;

private:
    double decay_;
    double normal_[3][3]{};
    double rhs_[3]{};
    uint64_t observations_ = 0;

    server_model_cost_state solve() const;
};
