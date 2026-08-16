#ifdef NDEBUG
#undef NDEBUG
#endif

#include "server-kv-action-policy.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <string>

static server_kv_action_request baseline_request() {
    server_kv_action_request request;
    request.phase = server_kv_action_phase::serve;
    request.cached_tokens = 1024;
    request.expected_decode_tokens = 128;
    request.kv_bytes = 32ULL * 1024 * 1024;
    request.page_count = 64;
    request.contiguous_pages = 16;
    request.reuse_distance = 2;
    request.kv_pressure = 0.75;
    request.device_bandwidth_bytes_per_ms = 100ULL * 1024 * 1024;
    request.host_bandwidth_bytes_per_ms = 12ULL * 1024 * 1024;
    request.launch_ms = 0.01;
    request.prefill_ms_per_token = 0.03;
    request.decode_ms_per_token = 0.02;
    return request;
}

int main() {
    server_kv_action_policy_config analytical_config;
    analytical_config.model = server_kv_action_model::analytical;
    server_kv_action_policy policy(analytical_config);
    auto request = baseline_request();

    // A prototype kernel is not a production capability.  It must remain
    // ineligible even when its analytical estimate would otherwise win.
    server_kv_action_capabilities capabilities;
    capabilities.direct = true;
    capabilities.remap = true;
    capabilities.paged = false;
    capabilities.device_swap = true;
    capabilities.host_swap = true;
    capabilities.recompute = true;
    request.direct_legal = false;
    request.paged_decode_multiplier = 0.01;
    const auto guarded = policy.choose(request, capabilities);
    assert(guarded.action != server_kv_action::paged);
    assert(guarded.estimate(server_kv_action::paged).reason ==
            server_kv_action_rejection::capability_unavailable);

    // Recompute is the deterministic safety fallback when no optimized
    // action is both available and legal.
    capabilities = {};
    capabilities.recompute = true;
    request.direct_legal = false;
    const auto fallback = policy.choose(request, capabilities);
    assert(fallback.action == server_kv_action::recompute);
    assert(fallback.safe_fallback_used);

    // Non-finite observations never enter a cost comparison.
    capabilities.direct = true;
    request.direct_legal = true;
    request.kv_pressure = std::numeric_limits<double>::quiet_NaN();
    const auto invalid = policy.choose(request, capabilities);
    assert(invalid.action == server_kv_action::recompute);
    assert(invalid.reason == server_kv_action_reason::invalid_features);

    // Worked analytical examples use literal expectations independent of the
    // policy implementation: 32 MiB / 12 MiB/ms ~= 2.67 ms is cheaper than
    // recomputing 1024 * 0.03 = 30.72 ms.
    request = baseline_request();
    request.direct_legal = false;
    capabilities = {};
    capabilities.device_swap = true;
    capabilities.recompute = true;
    const auto long_cache = policy.choose(request, capabilities);
    assert(long_cache.action == server_kv_action::device_swap);
    assert(std::abs(long_cache.estimate(server_kv_action::device_swap).cost_ms -
            (32.0 / 12.0 + 0.02 + 128.0 * 0.02)) < 1e-9);

    // Eight cached tokens cost 0.24 ms to recompute, so retaining the same
    // 32 MiB payload would be a measurable wrong enable.
    request.cached_tokens = 8;
    const auto short_cache = policy.choose(request, capabilities);
    assert(short_cache.action == server_kv_action::recompute);
    assert(!short_cache.safe_fallback_used);
    assert(short_cache.baseline_action == server_kv_action::device_swap);

    request = baseline_request();
    request.direct_legal = true;
    capabilities.direct = true;
    const auto resident = policy.choose(request, capabilities);
    assert(resident.action == server_kv_action::direct);

    // A1 must use both pressure and reuse-distance rather than merely carry
    // those fields through the API. High pressure penalizes resident actions;
    // distant reuse discounts preempt-time retention cost.
    request.kv_pressure = 1.0;
    const auto pressured = policy.choose(request, capabilities);
    assert(pressured.estimate(server_kv_action::direct).cost_ms >
            resident.estimate(server_kv_action::direct).cost_ms);
    request.phase = server_kv_action_phase::preempt;
    request.reuse_probability = 1.0;
    request.reuse_distance = 0;
    const auto near_reuse = policy.choose(request, capabilities);
    request.reuse_distance = 1000000;
    const auto distant_reuse = policy.choose(request, capabilities);
    assert(distant_reuse.estimate(server_kv_action::device_swap).cost_ms <
            near_reuse.estimate(server_kv_action::device_swap).cost_ms);

    server_kv_action_policy fixed({server_kv_action_model::fixed_rule});
    request.direct_legal = false;
    capabilities.host_swap = true;
    const auto heuristic = fixed.choose(request, capabilities);
    assert(heuristic.action == server_kv_action::device_swap);
    assert(heuristic.reason == server_kv_action_reason::fixed_rule);

    request = baseline_request();
    request.phase = server_kv_action_phase::preempt;
    capabilities.remap = true;
    capabilities.paged = true;
    const auto preempt = policy.choose(request, capabilities);
    assert(preempt.estimate(server_kv_action::direct).reason ==
            server_kv_action_rejection::illegal_for_phase);
    assert(preempt.estimate(server_kv_action::remap).reason ==
            server_kv_action_rejection::illegal_for_phase);
    assert(preempt.estimate(server_kv_action::paged).reason ==
            server_kv_action_rejection::illegal_for_phase);

    const auto snapshot = policy.snapshot();
    assert(snapshot.decisions == 10);
    assert(snapshot.action_decisions[static_cast<size_t>(server_kv_action::recompute)] == 3);
    assert(snapshot.safe_fallbacks == 2);
    assert(snapshot.invalid_feature_fallbacks == 1);
    assert(snapshot.reason_decisions[static_cast<size_t>(
            server_kv_action_reason::invalid_features)] == 1);
    assert(snapshot.action_reason_decisions[static_cast<size_t>(server_kv_action::recompute)]
            [static_cast<size_t>(server_kv_action_reason::invalid_features)] == 1);
    assert(std::string(server_kv_action_name(server_kv_action::paged)) == "paged");
    assert(std::string(server_kv_action_reason_name(server_kv_action_reason::cold_start)) ==
            "cold_start");
    assert(snapshot.total_decision_nanoseconds >= snapshot.maximum_decision_nanoseconds);

    const auto failed = policy.choose(baseline_request(), capabilities);
    policy.observe(failed, {failed.action, 3.0, true});
    assert(policy.snapshot().action_failures[static_cast<size_t>(failed.action)] == 1);

    server_kv_action_policy_config learned_config;
    learned_config.model = server_kv_action_model::learned;
    learned_config.minimum_observations = 3;
    learned_config.ridge_lambda = 0.01;
    learned_config.confidence_beta = 0.0;
    learned_config.switch_margin_ms = 0.1;
    learned_config.shadow = false;
    server_kv_action_policy learned(learned_config);
    request = baseline_request();
    request.direct_legal = false;
    capabilities = {};
    capabilities.device_swap = true;
    capabilities.recompute = true;
    auto cold = learned.choose(request, capabilities);
    assert(cold.action == server_kv_action::device_swap);
    assert(cold.reason == server_kv_action_reason::cold_start);
    for (size_t i = 0; i < 8; ++i) {
        const auto training = learned.choose(request, capabilities);
        learned.observe(training, {server_kv_action::device_swap, 8.0, false});
    learned.observe(training, {server_kv_action::recompute, 2.0, false});
    }
    const auto active_learned = learned.choose(request, capabilities);
    assert(active_learned.baseline_action == server_kv_action::device_swap);
    assert(active_learned.recommended_action == server_kv_action::recompute);
    assert(active_learned.action == server_kv_action::recompute);
    assert(active_learned.reason == server_kv_action_reason::positive_lower_bound);

    learned_config.shadow = true;
    server_kv_action_policy shadow(learned_config);
    for (size_t i = 0; i < 8; ++i) {
        const auto training = shadow.choose(request, capabilities);
        shadow.observe(training, {server_kv_action::device_swap, 8.0, false});
        shadow.observe(training, {server_kv_action::recompute, 2.0, false});
    }
    const auto shadowed = shadow.choose(request, capabilities);
    assert(shadowed.recommended_action == server_kv_action::recompute);
    assert(shadowed.action == server_kv_action::device_swap);
    assert(shadowed.reason == server_kv_action_reason::shadow_baseline);
    assert(shadow.snapshot().shadow_decisions >= 1);
    assert(shadow.snapshot().observation_total_milliseconds[
            static_cast<size_t>(server_kv_action::recompute)] == 16.0);
    return 0;
}
