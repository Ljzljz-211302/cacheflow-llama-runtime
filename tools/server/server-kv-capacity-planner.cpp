#include "server-kv-capacity-planner.h"

#include <algorithm>
#include <cmath>
#include <limits>

server_kv_capacity_planner::server_kv_capacity_planner(double reuse_half_life_ms) :
        reuse_half_life_us_(std::max(reuse_half_life_ms, 1.0) * 1000.0) {}

server_kv_capacity_plan server_kv_capacity_planner::plan(
        size_t capacity_tokens,
        size_t projected_tokens,
        int64_t now_us,
        std::vector<server_kv_cache_candidate> candidates) const {
    server_kv_capacity_plan result;
    result.capacity_tokens = capacity_tokens;
    result.projected_tokens = projected_tokens;
    result.deficit_tokens = projected_tokens > capacity_tokens
            ? projected_tokens - capacity_tokens
            : 0;
    if (result.deficit_tokens == 0) {
        return result;
    }

    while (result.reclaimed_tokens < result.deficit_tokens && !candidates.empty()) {
        const size_t remaining = result.deficit_tokens - result.reclaimed_tokens;
        auto best = candidates.begin();
        double best_probability = std::numeric_limits<double>::infinity();
        size_t best_overshoot = std::numeric_limits<size_t>::max();

        for (auto it = candidates.begin(); it != candidates.end(); ++it) {
            const int64_t age_us = std::max<int64_t>(0, now_us - it->last_used_us);
            const double probability = std::exp2(-(double) age_us / reuse_half_life_us_);
            const size_t overshoot = it->cached_tokens > remaining
                    ? it->cached_tokens - remaining
                    : 0;
            if (probability < best_probability ||
                    (probability == best_probability && overshoot < best_overshoot)) {
                best = it;
                best_probability = probability;
                best_overshoot = overshoot;
            }
        }

        result.victim_ids.push_back(best->id);
        result.reclaimed_tokens += best->cached_tokens;
        result.expected_recompute_tokens += best_probability * best->cached_tokens;
        candidates.erase(best);
    }
    return result;
}
