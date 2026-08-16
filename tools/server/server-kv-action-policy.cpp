#include "server-kv-action-policy.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace {

size_t index_of(server_kv_action action) {
    return static_cast<size_t>(action);
}

bool capability(const server_kv_action_capabilities & value, server_kv_action action) {
    switch (action) {
        case server_kv_action::direct:      return value.direct;
        case server_kv_action::remap:       return value.remap;
        case server_kv_action::paged:       return value.paged;
        case server_kv_action::device_swap: return value.device_swap;
        case server_kv_action::host_swap:   return value.host_swap;
        case server_kv_action::recompute:   return value.recompute;
    }
    return false;
}

bool finite_request(const server_kv_action_request & value) {
    const double fields[] = {
        value.reuse_probability,
        value.kv_pressure,
        value.device_bandwidth_bytes_per_ms,
        value.host_bandwidth_bytes_per_ms,
        value.launch_ms,
        value.prefill_ms_per_token,
        value.decode_ms_per_token,
        value.paged_decode_multiplier,
    };
    return std::all_of(std::begin(fields), std::end(fields), [](double item) {
        return std::isfinite(item) && item >= 0.0;
    }) && value.reuse_probability <= 1.0 && value.kv_pressure <= 1.0 &&
            value.device_bandwidth_bytes_per_ms > 0.0 &&
            value.host_bandwidth_bytes_per_ms > 0.0;
}

bool legal(const server_kv_action_request & request, server_kv_action action) {
    if (request.phase == server_kv_action_phase::preempt) {
        return action == server_kv_action::device_swap ||
                action == server_kv_action::host_swap ||
                action == server_kv_action::recompute;
    }
    if (action == server_kv_action::direct) return request.direct_legal;
    return true;
}

double estimate_cost(
        const server_kv_action_policy_config & config,
        const server_kv_action_request & request,
        server_kv_action action) {
    const double bytes = static_cast<double>(request.kv_bytes);
    const double device_transfer = bytes / request.device_bandwidth_bytes_per_ms;
    const double host_transfer = bytes / request.host_bandwidth_bytes_per_ms;
    const double decode = request.expected_decode_tokens * request.decode_ms_per_token;
    const double recompute = request.cached_tokens * request.prefill_ms_per_token;
    const size_t fragmented_pages = request.page_count > request.contiguous_pages
            ? request.page_count - request.contiguous_pages : 0;
    const double reuse_decay = 1.0 /
            (1.0 + std::log1p((double) request.reuse_distance) / 20.0);
    const double future = request.phase == server_kv_action_phase::preempt
            ? request.reuse_probability * reuse_decay : 1.0;
    const double pressure_excess = std::max(0.0,
            request.kv_pressure - config.pressure_reserve);
    const double pressure_cost = pressure_excess * (device_transfer + recompute);

    switch (action) {
        case server_kv_action::direct:
            return decode + pressure_cost;
        case server_kv_action::remap:
            return decode + device_transfer + 2.0 * request.launch_ms +
                    fragmented_pages * config.remap_fragment_page_ms + 2.0 * pressure_cost;
        case server_kv_action::paged:
            return decode * request.paged_decode_multiplier + request.launch_ms +
                    fragmented_pages * config.paged_fragment_page_ms + 0.25 * pressure_cost;
        case server_kv_action::device_swap:
            // CUDA-managed swap crosses the device/host seam once per phase.
            return (request.phase == server_kv_action_phase::serve ? host_transfer :
                    host_transfer * (1.0 + future)) + 2.0 * request.launch_ms +
                    (request.phase == server_kv_action_phase::serve ? decode + pressure_cost : 0.0);
        case server_kv_action::host_swap:
            // Transactional host/file storage adds serialization/checksum cost.
            return (request.phase == server_kv_action_phase::serve ? host_transfer :
                    host_transfer * (1.0 + future)) + config.host_swap_fixed_ms +
                    (request.phase == server_kv_action_phase::serve ? decode + pressure_cost : 0.0);
        case server_kv_action::recompute:
            return future * recompute + decode + pressure_cost;
    }
    return std::numeric_limits<double>::infinity();
}

} // namespace

const char * server_kv_action_name(server_kv_action action) {
    static constexpr const char * names[] = {
        "direct", "remap", "paged", "device_swap", "host_swap", "recompute",
    };
    static_assert(std::size(names) == server_kv_action_count);
    const size_t index = index_of(action);
    return index < std::size(names) ? names[index] : "unknown";
}

const char * server_kv_action_reason_name(server_kv_action_reason reason) {
    static constexpr const char * names[] = {
        "minimum_estimated_cost", "fixed_rule", "no_optimized_action", "invalid_features",
        "cold_start", "uncertainty_fallback", "positive_lower_bound", "shadow_baseline",
    };
    static_assert(std::size(names) == server_kv_action_reason_count);
    const size_t index = static_cast<size_t>(reason);
    return index < std::size(names) ? names[index] : "unknown";
}

const server_kv_action_estimate & server_kv_action_decision::estimate(
        server_kv_action candidate) const {
    return estimates.at(index_of(candidate));
}

server_kv_action_policy::server_kv_action_policy(server_kv_action_policy_config config) {
    configure(config);
}

void server_kv_action_policy::configure(server_kv_action_policy_config config) {
    config.host_swap_fixed_ms = std::max(0.0, config.host_swap_fixed_ms);
    config.remap_fragment_page_ms = std::max(0.0, config.remap_fragment_page_ms);
    config.paged_fragment_page_ms = std::max(0.0, config.paged_fragment_page_ms);
    config.pressure_reserve = std::clamp(config.pressure_reserve, 0.0, 1.0);
    config.switch_margin_ms = std::max(0.0, config.switch_margin_ms);
    config.minimum_observations = std::max<size_t>(1, config.minimum_observations);
    config.ridge_lambda = std::max(1e-9, config.ridge_lambda);
    config.confidence_beta = std::max(0.0, config.confidence_beta);
    config.failure_penalty_ms = std::max(0.0, config.failure_penalty_ms);
    config_ = config;
    snapshot_ = {};
    next_id_ = 1;
    for (auto & model : learned_) model.reset(config_.ridge_lambda);
}

void server_kv_action_policy::ridge_model::reset(double lambda) {
    observations = 0;
    residual_variance_ewma = 0.0;
    for (size_t row = 0; row < server_kv_action_feature_count; ++row) {
        rhs[row] = 0.0;
        for (size_t column = 0; column < server_kv_action_feature_count; ++column) {
            inverse[row][column] = row == column ? 1.0 / lambda : 0.0;
        }
    }
}

double server_kv_action_policy::ridge_model::predict(
        const std::array<double, server_kv_action_feature_count> & x) const {
    double result = 0.0;
    for (size_t row = 0; row < server_kv_action_feature_count; ++row) {
        double theta = 0.0;
        for (size_t column = 0; column < server_kv_action_feature_count; ++column) {
            theta += inverse[row][column] * rhs[column];
        }
        result += theta * x[row];
    }
    return std::max(0.0, result);
}

double server_kv_action_policy::ridge_model::radius(
        const std::array<double, server_kv_action_feature_count> & x, double beta) const {
    double quadratic = 0.0;
    for (size_t row = 0; row < server_kv_action_feature_count; ++row) {
        for (size_t column = 0; column < server_kv_action_feature_count; ++column) {
            quadratic += x[row] * inverse[row][column] * x[column];
        }
    }
    const double sigma = std::sqrt(std::max(1e-9, residual_variance_ewma));
    return beta * sigma * std::sqrt(std::max(0.0, quadratic));
}

void server_kv_action_policy::ridge_model::observe(
        const std::array<double, server_kv_action_feature_count> & x, double cost) {
    const double previous = predict(x);
    std::array<double, server_kv_action_feature_count> inverse_x{};
    double denominator = 1.0;
    for (size_t row = 0; row < server_kv_action_feature_count; ++row) {
        for (size_t column = 0; column < server_kv_action_feature_count; ++column) {
            inverse_x[row] += inverse[row][column] * x[column];
        }
        denominator += x[row] * inverse_x[row];
    }
    for (size_t row = 0; row < server_kv_action_feature_count; ++row) {
        for (size_t column = 0; column < server_kv_action_feature_count; ++column) {
            inverse[row][column] -= inverse_x[row] * inverse_x[column] / denominator;
        }
        rhs[row] += x[row] * cost;
    }
    const double residual = cost - previous;
    residual_variance_ewma = observations == 0 ? residual * residual :
            0.1 * residual * residual + 0.9 * residual_variance_ewma;
    observations++;
}

std::array<double, server_kv_action_feature_count> server_kv_action_policy::normalize(
        const server_kv_action_request & request) {
    const double fragmented = request.page_count > 0
            ? (double) (request.page_count - std::min(request.page_count, request.contiguous_pages)) /
                    request.page_count : 0.0;
    const double host_transfer_ms = (double) request.kv_bytes /
            request.host_bandwidth_bytes_per_ms;
    const auto bound = [](double value) { return std::clamp(value, 0.0, 4.0); };
    return {
        1.0,
        bound(request.cached_tokens / 4096.0),
        bound(request.expected_decode_tokens / 512.0),
        bound((double) request.kv_bytes / (1024.0 * 1024 * 1024)),
        fragmented,
        bound(std::log1p((double) request.reuse_distance) / 20.0),
        request.kv_pressure,
        bound(request.prefill_ms_per_token * 10.0),
        bound(host_transfer_ms / 100.0),
    };
}

server_kv_action_decision server_kv_action_policy::choose(
        const server_kv_action_request & request,
        const server_kv_action_capabilities & capabilities) {
    const auto started = std::chrono::steady_clock::now();
    server_kv_action_decision decision;
    decision.id = next_id_++;
    const auto finish = [this, started](const server_kv_action_decision & value) {
        const uint64_t elapsed = static_cast<uint64_t>(std::chrono::duration_cast<
                std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
        snapshot_.decisions++;
        snapshot_.action_decisions[index_of(value.action)]++;
        snapshot_.reason_decisions[static_cast<size_t>(value.reason)]++;
        snapshot_.action_reason_decisions[index_of(value.action)][static_cast<size_t>(value.reason)]++;
        snapshot_.safe_fallbacks += value.safe_fallback_used ? 1 : 0;
        snapshot_.invalid_feature_fallbacks +=
                value.reason == server_kv_action_reason::invalid_features ? 1 : 0;
        snapshot_.cold_start_fallbacks +=
                value.reason == server_kv_action_reason::cold_start ? 1 : 0;
        snapshot_.uncertainty_fallbacks +=
                value.reason == server_kv_action_reason::uncertainty_fallback ? 1 : 0;
        snapshot_.shadow_decisions +=
                value.reason == server_kv_action_reason::shadow_baseline ? 1 : 0;
        snapshot_.last_model_features = value.model_features;
        snapshot_.total_decision_nanoseconds += elapsed;
        snapshot_.maximum_decision_nanoseconds = std::max(
                snapshot_.maximum_decision_nanoseconds, elapsed);
        return value;
    };
    if (!finite_request(request)) {
        decision.reason = server_kv_action_reason::invalid_features;
        decision.safe_fallback_used = true;
        for (auto & estimate : decision.estimates) {
            estimate.reason = server_kv_action_rejection::invalid_features;
        }
        return finish(decision);
    }
    decision.model_features = normalize(request);

    for (size_t i = 0; i < server_kv_action_count; ++i) {
        const auto action = static_cast<server_kv_action>(i);
        auto & estimate = decision.estimates[i];
        if (!capability(capabilities, action)) {
            estimate.reason = server_kv_action_rejection::capability_unavailable;
            continue;
        }
        if (!legal(request, action)) {
            estimate.reason = server_kv_action_rejection::illegal_for_phase;
            continue;
        }
        estimate.eligible = true;
        estimate.cost_ms = estimate_cost(config_, request, action);
    }

    constexpr server_kv_action fixed_order[] = {
        server_kv_action::direct,
        server_kv_action::device_swap,
        server_kv_action::host_swap,
        server_kv_action::remap,
        server_kv_action::paged,
        server_kv_action::recompute,
    };
    for (const auto action : fixed_order) {
        if (!decision.estimate(action).eligible) continue;
        decision.baseline_action = action;
        break;
    }
    decision.action = decision.baseline_action;
    decision.recommended_action = decision.baseline_action;
    if (config_.model == server_kv_action_model::fixed_rule) {
        decision.reason = server_kv_action_reason::fixed_rule;
    } else if (config_.model == server_kv_action_model::analytical) {
        double best = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < server_kv_action_count; ++i) {
            const auto & estimate = decision.estimates[i];
            if (estimate.eligible && estimate.cost_ms < best) {
                best = estimate.cost_ms;
                decision.recommended_action = static_cast<server_kv_action>(i);
            }
        }
        const double baseline_cost = decision.estimate(decision.baseline_action).cost_ms;
        const double candidate_cost = decision.estimate(decision.recommended_action).cost_ms;
        if (candidate_cost + config_.switch_margin_ms < baseline_cost) {
            decision.action = decision.recommended_action;
            decision.reason = server_kv_action_reason::minimum_estimated_cost;
        } else {
            decision.reason = server_kv_action_reason::fixed_rule;
            decision.safe_fallback_used = decision.recommended_action != decision.baseline_action;
        }
    } else {
        const auto baseline_index = index_of(decision.baseline_action);
        if (learned_[baseline_index].observations < config_.minimum_observations) {
            decision.reason = server_kv_action_reason::cold_start;
            decision.safe_fallback_used = true;
        } else {
            for (size_t i = 0; i < server_kv_action_count; ++i) {
                if (!decision.estimates[i].eligible ||
                        learned_[i].observations < config_.minimum_observations) continue;
                decision.estimates[i].cost_ms = learned_[i].predict(decision.model_features);
                decision.estimates[i].uncertainty_ms = learned_[i].radius(
                        decision.model_features, config_.confidence_beta);
            }
            const auto & baseline = decision.estimates[baseline_index];
            const double baseline_lower = std::max(0.0,
                    baseline.cost_ms - baseline.uncertainty_ms);
            double best_upper = std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < server_kv_action_count; ++i) {
                if (i == baseline_index || !decision.estimates[i].eligible ||
                        learned_[i].observations < config_.minimum_observations) continue;
                const double upper = decision.estimates[i].cost_ms +
                        decision.estimates[i].uncertainty_ms;
                if (upper < best_upper) {
                    best_upper = upper;
                    decision.recommended_action = static_cast<server_kv_action>(i);
                }
            }
            if (decision.recommended_action != decision.baseline_action &&
                    best_upper + config_.switch_margin_ms < baseline_lower) {
                if (config_.shadow) {
                    decision.reason = server_kv_action_reason::shadow_baseline;
                    decision.safe_fallback_used = true;
                } else {
                    decision.action = decision.recommended_action;
                    decision.reason = server_kv_action_reason::positive_lower_bound;
                }
            } else {
                decision.reason = server_kv_action_reason::uncertainty_fallback;
                decision.safe_fallback_used = true;
            }
        }
    }
    if (!decision.estimate(decision.action).eligible && capabilities.recompute) {
        decision.action = server_kv_action::recompute;
        decision.baseline_action = server_kv_action::recompute;
        decision.recommended_action = server_kv_action::recompute;
        decision.safe_fallback_used = true;
        decision.reason = server_kv_action_reason::no_optimized_action;
    }
    bool optimized_eligible = false;
    for (size_t i = 0; i < index_of(server_kv_action::recompute); ++i) {
        optimized_eligible = optimized_eligible || decision.estimates[i].eligible;
    }
    if (!optimized_eligible && decision.action == server_kv_action::recompute) {
        decision.safe_fallback_used = true;
        decision.reason = server_kv_action_reason::no_optimized_action;
    }
    return finish(decision);
}

void server_kv_action_policy::observe(
        const server_kv_action_decision & decision,
        const server_kv_action_feedback & feedback) {
    const size_t action = index_of(feedback.action);
    if (decision.id == 0 || action >= server_kv_action_count ||
            !decision.estimates[action].eligible) return;
    double cost = feedback.execution_failed ? config_.failure_penalty_ms : feedback.total_ms;
    if (!std::isfinite(cost) || cost < 0.0) return;
    learned_[action].observe(decision.model_features, cost);
    snapshot_.observations[action]++;
    snapshot_.action_failures[action] += feedback.execution_failed ? 1 : 0;
    snapshot_.observation_total_milliseconds[action] += cost;
}
