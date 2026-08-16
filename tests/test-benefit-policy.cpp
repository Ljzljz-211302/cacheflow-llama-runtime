#ifdef NDEBUG
#undef NDEBUG
#endif

#include "server-benefit-policy.h"

#include <cassert>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>

static server_benefit_features workload(bool cuda = false) {
    server_benefit_features value;
    value.backend = cuda ? server_benefit_backend::cuda : server_benefit_backend::cpu;
    value.batch_width = 4;
    value.decode_tokens = 2;
    value.upstream_prefill_tokens = 256;
    value.cacheflow_prefill_tokens = 128;
    value.cacheflow_chunks = 2;
    value.active_sequences = 4;
    value.remaining_prefill_tokens = 512;
    value.maximum_remaining_tokens = 180;
    value.kv_pressure = 0.45;
    return value;
}

int main() {
    server_benefit_config config;
    config.mode = server_benefit_mode::learned;
    config.minimum_observations = 4;
    config.exploration_interval = 3;
    config.confidence_beta = 0.05;
    config.safety_margin_ms = 0.1;
    config.drift_consecutive_limit = 2;
    config.cooldown_decisions = 3;

    // Cold start is safe: establish the upstream baseline before probing.
    server_benefit_policy policy(config);
    const auto features = workload();
    for (size_t i = 0; i < config.minimum_observations; ++i) {
        const auto decision = policy.choose(features);
        assert(decision.action == server_benefit_action::upstream);
        assert(decision.reason == server_benefit_reason::cold_start);
        policy.observe(decision, { 20.0, false, false });
    }

    // Exploration is deterministic and bounded; learned selection requires
    // evidence that CacheFlow's pessimistic cost beats upstream's optimistic cost.
    auto probe = policy.choose(features);
    assert(probe.action == server_benefit_action::cacheflow);
    assert(probe.reason == server_benefit_reason::safe_exploration);
    policy.observe(probe, { 10.0, false, false });
    size_t guard = 0;
    while (policy.snapshot(server_benefit_backend::cpu).cacheflow_observations <
            config.minimum_observations && guard++ < 32) {
        const auto decision = policy.choose(features);
        policy.observe(decision, {
            decision.action == server_benefit_action::cacheflow ? 10.0 : 20.0,
            false,
            false,
        });
    }
    assert(guard < 32);
    auto learned = policy.choose(features);
    assert(learned.action == server_benefit_action::cacheflow);
    assert(learned.reason == server_benefit_reason::positive_lower_bound);
    assert(learned.predicted_benefit_ms > learned.uncertainty_ms);
    const auto converged = policy.snapshot(server_benefit_backend::cpu);
    assert(converged.last_predicted_benefit_ms == learned.predicted_benefit_ms);
    assert(converged.last_uncertainty_ms == learned.uncertainty_ms);
    uint64_t timed_decisions = 0;
    for (const auto count : converged.choose_duration_us_buckets) {
        timed_decisions += count;
    }
    assert(timed_decisions == converged.upstream_decisions + converged.cacheflow_decisions);
    assert(converged.choose_duration_us_sum > 0.0);

    // Inconclusive evidence triggers only a finite number of sparse probes.
    auto ambiguous_config = config;
    ambiguous_config.minimum_observations = 1;
    ambiguous_config.exploration_interval = 1;
    ambiguous_config.confidence_beta = 100.0;
    server_benefit_policy ambiguous(ambiguous_config);
    for (size_t i = 0; i < 20; ++i) {
        const auto decision = ambiguous.choose(features);
        ambiguous.observe(decision, { 20.0, false, false });
    }
    const auto ambiguous_state = ambiguous.snapshot(server_benefit_backend::cpu);
    assert(ambiguous_state.exploration_decisions == 3);
    assert(ambiguous.choose(features).action == server_benefit_action::upstream);

    // CPU evidence must never leak into the CUDA model.
    auto cuda_decision = policy.choose(workload(true));
    assert(cuda_decision.action == server_benefit_action::upstream);
    assert(cuda_decision.reason == server_benefit_reason::cold_start);

    // Unsafe pressure suppresses probing and learned enablement.
    auto pressured = features;
    pressured.kv_pressure = 0.99;
    auto pressure_decision = policy.choose(pressured);
    assert(pressure_decision.action == server_benefit_action::upstream);
    assert(pressure_decision.reason == server_benefit_reason::safety_fallback);

    auto lone_prefill = features;
    lone_prefill.decode_tokens = 0;
    lone_prefill.active_sequences = 1;
    lone_prefill.cacheflow_chunks = 1;
    auto throughput_only = policy.choose(lone_prefill);
    assert(throughput_only.action == server_benefit_action::upstream);
    assert(throughput_only.reason == server_benefit_reason::safety_fallback);

    // Consecutive severe residuals trip a backend-local cooldown.
    policy.observe(learned, { 100.0, true, true });
    policy.observe(learned, { 100.0, true, true });
    auto cooldown = policy.choose(features);
    assert(cooldown.action == server_benefit_action::upstream);
    assert(cooldown.reason == server_benefit_reason::drift_cooldown);
    assert(policy.snapshot(server_benefit_backend::cpu).drift_events == 1);

    // Fixed policies support controlled A/B experiments without bypassing the seam.
    config.mode = server_benefit_mode::always_cacheflow;
    server_benefit_policy always(config);
    assert(always.choose(features).action == server_benefit_action::cacheflow);
    config.mode = server_benefit_mode::upstream;
    server_benefit_policy upstream(config);
    assert(upstream.choose(features).action == server_benefit_action::upstream);

    // Replaying the same ordered snapshots and feedback through the production
    // seam is bit-for-bit deterministic at the decision level.
    config.mode = server_benefit_mode::learned;
    server_benefit_policy replay_a(config);
    server_benefit_policy replay_b(config);
    for (size_t i = 0; i < 30; ++i) {
        auto replay_features = workload(i % 5 == 0);
        replay_features.decode_tokens += i % 3;
        replay_features.kv_pressure += (i % 4) * 0.03;
        const auto left = replay_a.choose(replay_features);
        const auto right = replay_b.choose(replay_features);
        assert(left.id == right.id);
        assert(left.action == right.action);
        assert(left.reason == right.reason);
        assert(std::abs(left.predicted_benefit_ms - right.predicted_benefit_ms) < 1e-12);
        const double elapsed = left.action == server_benefit_action::cacheflow ? 11.0 : 19.0;
        replay_a.observe(left, { elapsed, false, false });
        replay_b.observe(right, { elapsed, false, false });
    }

    // Learned evidence survives a process restart. Checkpoints are bound to
    // an explicit compatibility key so another model/backend cannot consume
    // stale latency evidence.
    const auto checkpoint_path = std::filesystem::temp_directory_path() /
            ("cacheflow-benefit-policy-" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");
    std::filesystem::remove(checkpoint_path);
    config.checkpoint_path = checkpoint_path.string();
    config.checkpoint_compatibility_key = "model=qwen-test;backend=cpu";
    config.checkpoint_interval = 1;
    server_benefit_decision before_restart;
    {
        server_benefit_policy persistent(config);
        for (size_t i = 0; i < 24; ++i) {
            const auto decision = persistent.choose(features);
            persistent.observe(decision, {
                decision.action == server_benefit_action::cacheflow ? 9.0 : 21.0,
                false,
                false,
            });
        }
        before_restart = persistent.choose(features);
        assert(persistent.flush_checkpoint());
    }
    {
        server_benefit_policy restored(config);
        const auto status = restored.checkpoint_snapshot();
        assert(status.restored == 1);
        const auto after_restart = restored.choose(features);
        assert(after_restart.action == before_restart.action);
        assert(after_restart.reason == before_restart.reason);
        assert(std::abs(after_restart.predicted_benefit_ms - before_restart.predicted_benefit_ms) < 1e-12);
        assert(std::abs(after_restart.uncertainty_ms - before_restart.uncertainty_ms) < 1e-12);
    }
    {
        auto incompatible_config = config;
        incompatible_config.checkpoint_compatibility_key = "model=different;backend=cpu";
        server_benefit_policy incompatible(incompatible_config);
        assert(incompatible.checkpoint_snapshot().incompatible == 1);
        assert(incompatible.snapshot(server_benefit_backend::cpu).upstream_observations == 0);
    }

    // Corruption never produces decisions from partially parsed coefficients.
    {
        std::ofstream corrupt(checkpoint_path, std::ios::binary | std::ios::trunc);
        corrupt << "{\"schema_version\":1,\"state\":\"truncated";
    }
    {
        server_benefit_policy corrupted(config);
        assert(corrupted.checkpoint_snapshot().restore_failures == 1);
        assert(corrupted.snapshot(server_benefit_backend::cpu).upstream_observations == 0);
        assert(corrupted.choose(features).reason == server_benefit_reason::cold_start);
    }
    std::filesystem::remove(checkpoint_path);
    std::filesystem::remove(checkpoint_path.string() + ".tmp");

    return 0;
}
