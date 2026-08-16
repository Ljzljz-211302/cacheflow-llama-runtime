#include "server-kv-action-policy.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

static std::atomic<bool> track_allocations{false};
static std::atomic<uint64_t> tracked_allocations{0};

void * operator new(std::size_t size) {
    if (track_allocations.load(std::memory_order_relaxed)) {
        tracked_allocations.fetch_add(1, std::memory_order_relaxed);
    }
    if (void * value = std::malloc(size)) return value;
    throw std::bad_alloc();
}

void operator delete(void * value) noexcept { std::free(value); }
void operator delete(void * value, std::size_t) noexcept { std::free(value); }

static server_kv_action_request request(bool ood = false) {
    server_kv_action_request value;
    value.phase = server_kv_action_phase::serve;
    value.direct_legal = true;
    value.cached_tokens = ood ? (1ULL << 30) : 1024;
    value.expected_decode_tokens = 128;
    value.kv_bytes = ood ? (1ULL << 50) : 32ULL * 1024 * 1024;
    value.page_count = ood ? (1ULL << 24) : 64;
    value.contiguous_pages = 16;
    value.reuse_distance = ood ? UINT64_MAX : 3;
    value.kv_pressure = 0.85;
    value.device_bandwidth_bytes_per_ms = 100.0 * 1024 * 1024;
    value.host_bandwidth_bytes_per_ms = 12.0 * 1024 * 1024;
    value.launch_ms = 0.01;
    value.prefill_ms_per_token = 0.03;
    value.decode_ms_per_token = 0.02;
    value.paged_decode_multiplier = 1.13;
    return value;
}

static server_kv_action_capabilities capabilities(size_t count) {
    server_kv_action_capabilities value;
    value.recompute = true;
    if (count >= 2) value.device_swap = true;
    if (count >= 4) {
        value.direct = true;
        value.remap = true;
    }
    if (count >= 6) {
        value.paged = true;
        value.host_swap = true;
    }
    return value;
}

static uint64_t percentile(std::vector<uint64_t> values, double quantile) {
    const size_t index = std::min(values.size() - 1,
            (size_t) (quantile * (values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
}

static void run_case(size_t candidate_count, bool hot, bool ood) {
    server_kv_action_policy_config config;
    config.model = server_kv_action_model::learned;
    config.minimum_observations = 4;
    config.confidence_beta = 1.0;
    config.shadow = true;
    server_kv_action_policy policy(config);
    auto input = request(ood);
    auto available = capabilities(candidate_count);

    if (hot) {
        for (size_t i = 0; i < 8; ++i) {
            const auto decision = policy.choose(input, available);
            for (size_t action = 0; action < server_kv_action_count; ++action) {
                if (!decision.estimates[action].eligible) continue;
                policy.observe(decision, {
                    static_cast<server_kv_action>(action),
                    1.0 + action * 0.5,
                    false,
                });
            }
        }
    }

    constexpr size_t iterations = 1000000;
    std::vector<uint64_t> durations(iterations);
    tracked_allocations = 0;
    track_allocations = true;
    for (size_t i = 0; i < iterations; ++i) {
        const auto started = std::chrono::steady_clock::now();
        const auto decision = policy.choose(input, available);
        const auto finished = std::chrono::steady_clock::now();
        durations[i] = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                finished - started).count();
        if (decision.id == 0) std::abort();
    }
    track_allocations = false;
    const auto allocations = tracked_allocations.load();
    const auto maximum = *std::max_element(durations.begin(), durations.end());
    std::printf(
            "{\"candidates\":%zu,\"state\":\"%s\",\"ood\":%s,"
            "\"iterations\":%zu,\"p50_ns\":%llu,\"p95_ns\":%llu,"
            "\"p99_ns\":%llu,\"max_ns\":%llu,\"allocations\":%llu}\n",
            candidate_count, hot ? "hot" : "cold", ood ? "true" : "false", iterations,
            (unsigned long long) percentile(durations, 0.50),
            (unsigned long long) percentile(durations, 0.95),
            (unsigned long long) percentile(durations, 0.99),
            (unsigned long long) maximum,
            (unsigned long long) allocations);
}

int main() {
    run_case(1, false, false);
    run_case(2, false, false);
    run_case(4, true, false);
    run_case(6, true, false);
    run_case(6, true, true);
    return 0;
}
