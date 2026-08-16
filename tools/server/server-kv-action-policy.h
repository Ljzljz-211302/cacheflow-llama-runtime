#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

enum class server_kv_action : uint8_t {
    direct,
    remap,
    paged,
    device_swap,
    host_swap,
    recompute,
};

enum class server_kv_action_phase : uint8_t {
    serve,
    preempt,
};

enum class server_kv_action_model : uint8_t {
    fixed_rule,
    analytical,
    learned,
};

enum class server_kv_action_rejection : uint8_t {
    none,
    capability_unavailable,
    illegal_for_phase,
    invalid_features,
};

enum class server_kv_action_reason : uint8_t {
    minimum_estimated_cost,
    fixed_rule,
    no_optimized_action,
    invalid_features,
    cold_start,
    uncertainty_fallback,
    positive_lower_bound,
    shadow_baseline,
};

constexpr size_t server_kv_action_count = 6;
constexpr size_t server_kv_action_feature_count = 9;
constexpr size_t server_kv_action_reason_count = 8;

const char * server_kv_action_name(server_kv_action action);
const char * server_kv_action_reason_name(server_kv_action_reason reason);

struct server_kv_action_capabilities {
    bool direct = false;
    bool remap = false;
    bool paged = false;
    bool device_swap = false;
    bool host_swap = false;
    bool recompute = true;
};

struct server_kv_action_request {
    server_kv_action_phase phase = server_kv_action_phase::serve;
    bool direct_legal = false;
    size_t cached_tokens = 0;
    size_t expected_decode_tokens = 0;
    uint64_t kv_bytes = 0;
    size_t page_count = 0;
    size_t contiguous_pages = 0;
    uint64_t reuse_distance = 0;
    double reuse_probability = 1.0;
    double kv_pressure = 0.0;
    double device_bandwidth_bytes_per_ms = 1.0;
    double host_bandwidth_bytes_per_ms = 1.0;
    double launch_ms = 0.0;
    double prefill_ms_per_token = 0.0;
    double decode_ms_per_token = 0.0;
    double paged_decode_multiplier = 1.0;
};

struct server_kv_action_estimate {
    bool eligible = false;
    double cost_ms = std::numeric_limits<double>::infinity();
    double uncertainty_ms = 0.0;
    server_kv_action_rejection reason = server_kv_action_rejection::none;
};

struct server_kv_action_decision {
    uint64_t id = 0;
    server_kv_action action = server_kv_action::recompute;
    server_kv_action baseline_action = server_kv_action::recompute;
    server_kv_action recommended_action = server_kv_action::recompute;
    server_kv_action_reason reason = server_kv_action_reason::no_optimized_action;
    bool safe_fallback_used = false;
    std::array<server_kv_action_estimate, server_kv_action_count> estimates{};
    std::array<double, server_kv_action_feature_count> model_features{};

    const server_kv_action_estimate & estimate(server_kv_action candidate) const;
};

struct server_kv_action_policy_config {
    server_kv_action_model model = server_kv_action_model::learned;
    double host_swap_fixed_ms = 0.20;
    double remap_fragment_page_ms = 0.0005;
    double paged_fragment_page_ms = 0.0010;
    double pressure_reserve = 0.90;
    double switch_margin_ms = 0.25;
    size_t minimum_observations = 8;
    double ridge_lambda = 1.0;
    double confidence_beta = 1.0;
    double failure_penalty_ms = 1000.0;
    bool shadow = true;
};

struct server_kv_action_feedback {
    server_kv_action action = server_kv_action::recompute;
    double total_ms = 0.0;
    bool execution_failed = false;
};

struct server_kv_action_policy_snapshot {
    uint64_t decisions = 0;
    std::array<uint64_t, server_kv_action_count> action_decisions{};
    std::array<uint64_t, server_kv_action_reason_count> reason_decisions{};
    std::array<std::array<uint64_t, server_kv_action_reason_count>, server_kv_action_count>
            action_reason_decisions{};
    uint64_t safe_fallbacks = 0;
    uint64_t invalid_feature_fallbacks = 0;
    uint64_t cold_start_fallbacks = 0;
    uint64_t uncertainty_fallbacks = 0;
    uint64_t shadow_decisions = 0;
    std::array<uint64_t, server_kv_action_count> observations{};
    std::array<uint64_t, server_kv_action_count> action_failures{};
    std::array<double, server_kv_action_count> observation_total_milliseconds{};
    std::array<double, server_kv_action_feature_count> last_model_features{};
    uint64_t total_decision_nanoseconds = 0;
    uint64_t maximum_decision_nanoseconds = 0;
};

// Deep, pure decision module. Callers provide one immutable runtime snapshot;
// the module owns legality, cost estimation, action ranking and fail-closed
// fallback. It never performs KV mutation.
class server_kv_action_policy {
public:
    explicit server_kv_action_policy(server_kv_action_policy_config config = {});
    void configure(server_kv_action_policy_config config);
    server_kv_action_decision choose(
            const server_kv_action_request & request,
            const server_kv_action_capabilities & capabilities);
    void observe(const server_kv_action_decision & decision,
            const server_kv_action_feedback & feedback);
    server_kv_action_policy_snapshot snapshot() const { return snapshot_; }

private:
    server_kv_action_policy_config config_;
    server_kv_action_policy_snapshot snapshot_;
    uint64_t next_id_ = 1;

    struct ridge_model {
        double inverse[server_kv_action_feature_count][server_kv_action_feature_count]{};
        double rhs[server_kv_action_feature_count]{};
        uint64_t observations = 0;
        double residual_variance_ewma = 0.0;

        void reset(double lambda);
        void observe(const std::array<double, server_kv_action_feature_count> & x, double cost);
        double predict(const std::array<double, server_kv_action_feature_count> & x) const;
        double radius(const std::array<double, server_kv_action_feature_count> & x, double beta) const;
    };
    std::array<ridge_model, server_kv_action_count> learned_{};

    static std::array<double, server_kv_action_feature_count> normalize(
            const server_kv_action_request & request);
};
