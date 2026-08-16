#pragma once

#include "server-benefit-checkpoint.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

enum class server_benefit_backend { cpu, cuda };
enum class server_benefit_action { upstream, cacheflow };
enum class server_benefit_mode { upstream, always_cacheflow, rule, learned };

enum class server_benefit_reason {
    fixed_upstream,
    fixed_cacheflow,
    rule_match,
    rule_reject,
    cold_start,
    safe_exploration,
    positive_lower_bound,
    insufficient_evidence,
    safety_fallback,
    drift_cooldown,
};

constexpr size_t server_benefit_feature_count = 10;
constexpr size_t server_benefit_choose_duration_bucket_count = 9;

struct server_benefit_config {
    server_benefit_mode mode = server_benefit_mode::upstream;
    size_t minimum_observations = 6;
    size_t exploration_interval = 8;
    double ridge_lambda = 1.0;
    double confidence_beta = 1.0;
    double safety_margin_ms = 0.25;
    double maximum_kv_pressure = 0.90;
    double target_iteration_ms = 40.0;
    double slo_penalty = 2.0;
    double failure_penalty_ms = 1000.0;
    double drift_ratio = 2.5;
    size_t drift_consecutive_limit = 3;
    size_t cooldown_decisions = 16;
    std::string checkpoint_path;
    std::string checkpoint_compatibility_key;
    size_t checkpoint_interval = 128;
};

struct server_benefit_features {
    server_benefit_backend backend = server_benefit_backend::cpu;
    size_t batch_width = 0;
    size_t decode_tokens = 0;
    size_t upstream_prefill_tokens = 0;
    size_t cacheflow_prefill_tokens = 0;
    size_t cacheflow_chunks = 0;
    size_t active_sequences = 0;
    size_t remaining_prefill_tokens = 0;
    size_t maximum_remaining_tokens = 0;
    double kv_pressure = 0.0;
};

struct server_benefit_feedback {
    double elapsed_ms = 0.0;
    bool slo_violation = false;
    bool execution_failed = false;
};

struct server_benefit_decision {
    uint64_t id = 0;
    server_benefit_backend backend = server_benefit_backend::cpu;
    server_benefit_action action = server_benefit_action::upstream;
    server_benefit_reason reason = server_benefit_reason::fixed_upstream;
    double predicted_benefit_ms = 0.0;
    double uncertainty_ms = 0.0;

    // Opaque observation context retained so feedback is attributed to the
    // exact decision that produced it, not to a later mutable snapshot.
    std::array<double, server_benefit_feature_count> model_features{};
};

struct server_benefit_snapshot {
    uint64_t upstream_observations = 0;
    uint64_t cacheflow_observations = 0;
    uint64_t upstream_decisions = 0;
    uint64_t cacheflow_decisions = 0;
    uint64_t exploration_decisions = 0;
    uint64_t cold_start_decisions = 0;
    uint64_t positive_lower_bound_decisions = 0;
    uint64_t insufficient_evidence_decisions = 0;
    uint64_t safety_fallbacks = 0;
    uint64_t drift_events = 0;
    size_t cooldown_remaining = 0;
    double upstream_cost_ms = 0.0;
    double cacheflow_cost_ms = 0.0;
    double last_predicted_benefit_ms = 0.0;
    double last_uncertainty_ms = 0.0;
    // Exact (non-cumulative) bins with upper bounds
    // 1/2/5/10/20/50/100/250/+Inf microseconds.
    std::array<uint64_t, server_benefit_choose_duration_bucket_count> choose_duration_us_buckets{};
    double choose_duration_us_sum = 0.0;
};

struct server_benefit_persistence_snapshot {
    uint64_t restored = 0;
    uint64_t incompatible = 0;
    uint64_t restore_failures = 0;
    uint64_t checkpoints_enqueued = 0;
    uint64_t checkpoints_coalesced = 0;
    uint64_t saves_completed = 0;
    uint64_t save_failures = 0;
    uint64_t pending = 0;
};

// Conservative contextual policy for the production scheduling seam. It owns
// backend-local online models and exposes only choose/observe/snapshot.
class server_benefit_policy {
public:
    explicit server_benefit_policy(server_benefit_config config = {});
    ~server_benefit_policy();

    void configure(server_benefit_config config);
    server_benefit_decision choose(const server_benefit_features & features);
    void observe(const server_benefit_decision & decision, const server_benefit_feedback & feedback);
    server_benefit_snapshot snapshot(server_benefit_backend backend) const;
    bool flush_checkpoint();
    server_benefit_persistence_snapshot checkpoint_snapshot() const;

private:
    static constexpr size_t feature_count = server_benefit_feature_count;

    struct ridge_model {
        double normal[feature_count][feature_count]{};
        double rhs[feature_count]{};
        uint64_t observations = 0;
        double residual_variance_ewma = 0.0;

        void reset(double lambda);
        void observe(const std::array<double, feature_count> & x, double cost);
        double predict(const std::array<double, feature_count> & x) const;
        double radius(const std::array<double, feature_count> & x, double beta) const;
    };

    struct backend_state {
        ridge_model upstream;
        ridge_model cacheflow;
        uint64_t upstream_decisions = 0;
        uint64_t cacheflow_decisions = 0;
        uint64_t exploration_decisions = 0;
        uint64_t cold_start_decisions = 0;
        uint64_t positive_lower_bound_decisions = 0;
        uint64_t insufficient_evidence_decisions = 0;
        uint64_t safety_fallbacks = 0;
        uint64_t drift_events = 0;
        size_t upstream_bad_residuals = 0;
        size_t cacheflow_bad_residuals = 0;
        size_t cooldown_remaining = 0;
        uint64_t decisions_since_probe = 0;
        double last_predicted_benefit_ms = 0.0;
        double last_uncertainty_ms = 0.0;
        std::array<uint64_t, server_benefit_choose_duration_bucket_count> choose_duration_us_buckets{};
        double choose_duration_us_sum = 0.0;
    };

    server_benefit_config config_;
    backend_state cpu_;
    backend_state cuda_;
    uint64_t next_id_ = 1;
    std::unique_ptr<server_benefit_checkpoint> checkpoint_;
    size_t observations_since_checkpoint_ = 0;
    uint64_t checkpoints_restored_ = 0;
    uint64_t checkpoints_incompatible_ = 0;
    uint64_t checkpoint_restore_failures_ = 0;
    uint64_t checkpoint_enqueue_failures_ = 0;

    backend_state & state(server_benefit_backend backend);
    const backend_state & state(server_benefit_backend backend) const;
    std::array<double, feature_count> normalize(const server_benefit_features & features) const;
    std::string encode_checkpoint() const;
    enum class checkpoint_restore_result {
        SERVER_BENEFIT_CHECKPOINT_RESTORE_RESULT_RESTORED,
        SERVER_BENEFIT_CHECKPOINT_RESTORE_RESULT_INCOMPATIBLE,
        SERVER_BENEFIT_CHECKPOINT_RESTORE_RESULT_INVALID,
    };
    checkpoint_restore_result restore_checkpoint(const std::string & payload);
};
