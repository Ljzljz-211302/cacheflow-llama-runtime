#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct server_kv_cache_candidate {
    int id = -1;
    size_t cached_tokens = 0;
    int64_t last_used_us = -1;
};

struct server_kv_capacity_plan {
    size_t capacity_tokens = 0;
    size_t projected_tokens = 0;
    size_t deficit_tokens = 0;
    size_t reclaimed_tokens = 0;
    double expected_recompute_tokens = 0.0;
    std::vector<int> victim_ids;

    bool fits() const {
        return projected_tokens <= capacity_tokens + reclaimed_tokens;
    }
};

// Plans proactive eviction from unified KV memory. Recent caches carry a
// higher expected reuse value; equally valuable candidates minimize excess
// reclamation so the engine does not purge more KV than the batch needs.
class server_kv_capacity_planner {
public:
    explicit server_kv_capacity_planner(double reuse_half_life_ms = 30000.0);

    server_kv_capacity_plan plan(
            size_t capacity_tokens,
            size_t projected_tokens,
            int64_t now_us,
            std::vector<server_kv_cache_candidate> candidates) const;

private:
    double reuse_half_life_us_;
};
