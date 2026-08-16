#ifdef NDEBUG
#undef NDEBUG
#endif

#include "server-model-cost-model.h"

#include <cassert>
#include <cmath>
#include <initializer_list>

int main() {
    server_model_cost_model model(0.99);
    for (size_t round = 0; round < 8; ++round) {
        for (size_t decode : { 0u, 1u, 4u }) {
            for (size_t prefill : { 0u, 32u, 128u, 512u }) {
                if (decode == 0 && prefill == 0) continue;
                model.observe(decode, prefill, 1.5 + decode * 3.0 + prefill * 0.02);
            }
        }
    }
    const auto state = model.state();
    assert(std::abs(state.decode_ms_per_token - 3.0) < 0.05);
    assert(std::abs(state.prefill_ms_per_token - 0.02) < 0.002);
    assert(std::abs(state.fixed_ms - 1.5) < 0.05);
    const size_t recommended = model.recommend_prefill_chunk(12.0, 1, 2, 16, 512);
    assert(recommended >= 180 && recommended <= 190);
    return 0;
}
