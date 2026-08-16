#include "server-benefit-policy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

namespace {
using checkpoint_json = nlohmann::ordered_json;

constexpr std::array<double, server_benefit_choose_duration_bucket_count - 1>
        choose_duration_upper_bounds_us = { 1, 2, 5, 10, 20, 50, 100, 250 };

struct choose_duration_timer {
    std::array<uint64_t, server_benefit_choose_duration_bucket_count> & buckets;
    double & sum_us;
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();

    ~choose_duration_timer() {
        const double elapsed_us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - started).count();
        sum_us += elapsed_us;
        size_t bucket = 0;
        while (bucket < choose_duration_upper_bounds_us.size() &&
                elapsed_us > choose_duration_upper_bounds_us[bucket]) {
            ++bucket;
        }
        buckets[bucket]++;
    }
};

uint32_t checkpoint_crc32(const std::string & value) {
    uint32_t crc = 0xffffffffU;
    for (const unsigned char byte : value) {
        crc ^= byte;
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1U) ^ (0xedb88320U & (0U - (crc & 1U)));
        }
    }
    return ~crc;
}

checkpoint_json checkpoint_configuration(const server_benefit_config & config) {
    return {
        {"minimum_observations", config.minimum_observations},
        {"exploration_interval", config.exploration_interval},
        {"ridge_lambda", config.ridge_lambda},
        {"confidence_beta", config.confidence_beta},
        {"safety_margin_ms", config.safety_margin_ms},
        {"maximum_kv_pressure", config.maximum_kv_pressure},
        {"target_iteration_ms", config.target_iteration_ms},
        {"slo_penalty", config.slo_penalty},
        {"failure_penalty_ms", config.failure_penalty_ms},
        {"drift_ratio", config.drift_ratio},
        {"drift_consecutive_limit", config.drift_consecutive_limit},
        {"cooldown_decisions", config.cooldown_decisions},
    };
}

std::string checkpoint_checksum_input(const std::string & compatibility,
        const checkpoint_json & configuration, const checkpoint_json & state) {
    return compatibility + "\n" + configuration.dump() + "\n" + state.dump();
}

template <size_t N>
bool invert(const double (&input)[N][N], double (&output)[N][N]) {
    double matrix[N][2 * N]{};
    for (size_t row = 0; row < N; ++row) {
        for (size_t column = 0; column < N; ++column) matrix[row][column] = input[row][column];
        matrix[row][N + row] = 1.0;
    }
    for (size_t pivot = 0; pivot < N; ++pivot) {
        size_t best = pivot;
        for (size_t row = pivot + 1; row < N; ++row) {
            if (std::abs(matrix[row][pivot]) > std::abs(matrix[best][pivot])) best = row;
        }
        if (std::abs(matrix[best][pivot]) < 1e-12) return false;
        if (best != pivot) {
            for (size_t column = 0; column < 2 * N; ++column) {
                std::swap(matrix[pivot][column], matrix[best][column]);
            }
        }
        const double divisor = matrix[pivot][pivot];
        for (size_t column = 0; column < 2 * N; ++column) matrix[pivot][column] /= divisor;
        for (size_t row = 0; row < N; ++row) {
            if (row == pivot) continue;
            const double factor = matrix[row][pivot];
            for (size_t column = 0; column < 2 * N; ++column) {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
        }
    }
    for (size_t row = 0; row < N; ++row) {
        for (size_t column = 0; column < N; ++column) output[row][column] = matrix[row][N + column];
    }
    return true;
}
}

void server_benefit_policy::ridge_model::reset(double lambda) {
    observations = 0;
    residual_variance_ewma = 0.0;
    for (size_t row = 0; row < feature_count; ++row) {
        rhs[row] = 0.0;
        for (size_t column = 0; column < feature_count; ++column) normal[row][column] = 0.0;
        normal[row][row] = lambda;
    }
}

void server_benefit_policy::ridge_model::observe(
        const std::array<double, feature_count> & x, double cost) {
    const double residual = cost - predict(x);
    constexpr double alpha = 0.2;
    residual_variance_ewma = observations == 0 ? residual * residual :
            alpha * residual * residual + (1.0 - alpha) * residual_variance_ewma;
    for (size_t row = 0; row < feature_count; ++row) {
        rhs[row] += x[row] * cost;
        for (size_t column = 0; column < feature_count; ++column) {
            normal[row][column] += x[row] * x[column];
        }
    }
    observations++;
}

double server_benefit_policy::ridge_model::predict(
        const std::array<double, feature_count> & x) const {
    double inverse[feature_count][feature_count]{};
    if (!invert(normal, inverse)) return 0.0;
    double prediction = 0.0;
    for (size_t row = 0; row < feature_count; ++row) {
        double theta = 0.0;
        for (size_t column = 0; column < feature_count; ++column) theta += inverse[row][column] * rhs[column];
        prediction += x[row] * theta;
    }
    return std::max(0.0, prediction);
}

double server_benefit_policy::ridge_model::radius(
        const std::array<double, feature_count> & x, double beta) const {
    double inverse[feature_count][feature_count]{};
    if (!invert(normal, inverse)) return 1e9;
    double variance = 0.0;
    for (size_t row = 0; row < feature_count; ++row) {
        for (size_t column = 0; column < feature_count; ++column) {
            variance += x[row] * inverse[row][column] * x[column];
        }
    }
    const double noise_ms = std::max(1.0, std::sqrt(std::max(0.0, residual_variance_ewma)));
    return beta * noise_ms * std::sqrt(std::max(0.0, variance));
}

server_benefit_policy::server_benefit_policy(server_benefit_config config) {
    configure(config);
}

server_benefit_policy::~server_benefit_policy() {
    try {
        flush_checkpoint();
    } catch (...) {
        // Destructors cannot surface persistence failures. The synchronous
        // shutdown path can call flush_checkpoint(), and metrics retain errors.
    }
}

void server_benefit_policy::configure(server_benefit_config config) {
    if (checkpoint_) {
        flush_checkpoint();
        checkpoint_.reset();
    }
    config_ = config;
    config_.minimum_observations = std::max<size_t>(1, config_.minimum_observations);
    config_.exploration_interval = std::max<size_t>(1, config_.exploration_interval);
    config_.ridge_lambda = std::max(1e-6, config_.ridge_lambda);
    config_.drift_consecutive_limit = std::max<size_t>(1, config_.drift_consecutive_limit);
    config_.checkpoint_interval = std::max<size_t>(1, config_.checkpoint_interval);
    cpu_ = {};
    cuda_ = {};
    cpu_.upstream.reset(config_.ridge_lambda);
    cpu_.cacheflow.reset(config_.ridge_lambda);
    cuda_.upstream.reset(config_.ridge_lambda);
    cuda_.cacheflow.reset(config_.ridge_lambda);
    next_id_ = 1;
    observations_since_checkpoint_ = 0;
    checkpoints_restored_ = 0;
    checkpoints_incompatible_ = 0;
    checkpoint_restore_failures_ = 0;
    checkpoint_enqueue_failures_ = 0;
    if (!config_.checkpoint_path.empty() && config_.mode == server_benefit_mode::learned) {
        if (config_.checkpoint_compatibility_key.empty()) {
            throw std::invalid_argument("benefit checkpoint requires a non-empty compatibility key");
        }
        checkpoint_ = std::make_unique<server_benefit_checkpoint>(
                server_benefit_create_file_checkpoint_store(config_.checkpoint_path));
        const auto loaded = checkpoint_->load();
        if (loaded.status == server_benefit_checkpoint_load_status::SERVER_BENEFIT_CHECKPOINT_LOAD_STATUS_RESTORED) {
            const auto result = restore_checkpoint(loaded.payload);
            if (result == checkpoint_restore_result::SERVER_BENEFIT_CHECKPOINT_RESTORE_RESULT_RESTORED) checkpoints_restored_++;
            else if (result == checkpoint_restore_result::SERVER_BENEFIT_CHECKPOINT_RESTORE_RESULT_INCOMPATIBLE) checkpoints_incompatible_++;
            else checkpoint_restore_failures_++;
        } else if (loaded.status == server_benefit_checkpoint_load_status::SERVER_BENEFIT_CHECKPOINT_LOAD_STATUS_IO_ERROR) {
            checkpoint_restore_failures_++;
        }
    }
}

server_benefit_policy::backend_state & server_benefit_policy::state(server_benefit_backend backend) {
    return backend == server_benefit_backend::cuda ? cuda_ : cpu_;
}

const server_benefit_policy::backend_state & server_benefit_policy::state(server_benefit_backend backend) const {
    return backend == server_benefit_backend::cuda ? cuda_ : cpu_;
}

std::array<double, server_benefit_policy::feature_count> server_benefit_policy::normalize(
        const server_benefit_features & value) const {
    const double upstream = std::max<size_t>(1, value.upstream_prefill_tokens);
    return {
        1.0,
        std::min(1.0, value.batch_width / 32.0),
        std::min(1.0, value.decode_tokens / 32.0),
        std::min(1.0, value.upstream_prefill_tokens / 2048.0),
        std::clamp((upstream - std::min<double>(upstream, value.cacheflow_prefill_tokens)) / upstream, 0.0, 1.0),
        std::min(1.0, value.cacheflow_chunks / 32.0),
        std::min(1.0, value.active_sequences / 32.0),
        std::min(1.0, value.remaining_prefill_tokens / 8192.0),
        value.remaining_prefill_tokens == 0 ? 0.0 : std::min(1.0,
                (double) value.maximum_remaining_tokens / value.remaining_prefill_tokens),
        std::clamp(value.kv_pressure, 0.0, 1.0),
    };
}

server_benefit_decision server_benefit_policy::choose(const server_benefit_features & features) {
    auto & backend = state(features.backend);
    choose_duration_timer duration_timer{
        backend.choose_duration_us_buckets,
        backend.choose_duration_us_sum,
    };
    server_benefit_decision decision;
    decision.id = next_id_++;
    decision.backend = features.backend;
    decision.model_features = normalize(features);

    auto select = [&](server_benefit_action action, server_benefit_reason reason) {
        decision.action = action;
        decision.reason = reason;
        if (action == server_benefit_action::cacheflow) backend.cacheflow_decisions++;
        else backend.upstream_decisions++;
        if (reason == server_benefit_reason::cold_start) backend.cold_start_decisions++;
        if (reason == server_benefit_reason::positive_lower_bound) backend.positive_lower_bound_decisions++;
        if (reason == server_benefit_reason::insufficient_evidence) backend.insufficient_evidence_decisions++;
    };

    if (config_.mode == server_benefit_mode::upstream) {
        select(server_benefit_action::upstream, server_benefit_reason::fixed_upstream);
        return decision;
    }
    if (config_.mode == server_benefit_mode::always_cacheflow) {
        select(server_benefit_action::cacheflow, server_benefit_reason::fixed_cacheflow);
        return decision;
    }
    if (features.kv_pressure > config_.maximum_kv_pressure) {
        backend.safety_fallbacks++;
        select(server_benefit_action::upstream, server_benefit_reason::safety_fallback);
        return decision;
    }
    if (config_.mode == server_benefit_mode::rule) {
        const bool match = features.cacheflow_prefill_tokens < features.upstream_prefill_tokens &&
                features.active_sequences > 1 && features.cacheflow_chunks > 1;
        select(match ? server_benefit_action::cacheflow : server_benefit_action::upstream,
                match ? server_benefit_reason::rule_match : server_benefit_reason::rule_reject);
        return decision;
    }
    // Chunking a lone prefill only sacrifices throughput. With multiple
    // prefills it can improve fairness/TTFT even before decode begins, so the
    // statistical model decides inside that structurally useful region.
    const bool structurally_safe = features.active_sequences > 1 &&
            features.cacheflow_chunks > 1;
    if (!structurally_safe) {
        backend.safety_fallbacks++;
        select(server_benefit_action::upstream, server_benefit_reason::safety_fallback);
        return decision;
    }
    if (backend.cooldown_remaining > 0) {
        backend.cooldown_remaining--;
        select(server_benefit_action::upstream, server_benefit_reason::drift_cooldown);
        return decision;
    }
    if (backend.upstream.observations < config_.minimum_observations) {
        select(server_benefit_action::upstream, server_benefit_reason::cold_start);
        return decision;
    }
    if (backend.cacheflow.observations < config_.minimum_observations) {
        backend.decisions_since_probe++;
        if (backend.cacheflow.observations == 0 ||
                backend.decisions_since_probe >= config_.exploration_interval) {
            backend.decisions_since_probe = 0;
            backend.exploration_decisions++;
            select(server_benefit_action::cacheflow, server_benefit_reason::safe_exploration);
        } else {
            select(server_benefit_action::upstream, server_benefit_reason::cold_start);
        }
        return decision;
    }

    const double upstream_cost = backend.upstream.predict(decision.model_features);
    const double cacheflow_cost = backend.cacheflow.predict(decision.model_features);
    const double upstream_radius = backend.upstream.radius(decision.model_features, config_.confidence_beta);
    const double cacheflow_radius = backend.cacheflow.radius(decision.model_features, config_.confidence_beta);
    decision.predicted_benefit_ms = upstream_cost - cacheflow_cost;
    decision.uncertainty_ms = upstream_radius + cacheflow_radius + config_.safety_margin_ms;
    backend.last_predicted_benefit_ms = decision.predicted_benefit_ms;
    backend.last_uncertainty_ms = decision.uncertainty_ms;
    const bool proven = cacheflow_cost + cacheflow_radius + config_.safety_margin_ms <
            std::max(0.0, upstream_cost - upstream_radius);
    if (proven) {
        backend.decisions_since_probe = 0;
        select(server_benefit_action::cacheflow, server_benefit_reason::positive_lower_bound);
        return decision;
    }

    // An inconclusive model must not become permanently stuck after reaching
    // the minimum sample count. Continue sparse exploration up to a strict
    // backend-local budget, then fail closed if the lower bound is still not
    // positive.
    backend.decisions_since_probe++;
    const uint64_t exploration_budget = config_.minimum_observations * 3;
    if (backend.cacheflow.observations < exploration_budget &&
            backend.decisions_since_probe >= config_.exploration_interval) {
        backend.decisions_since_probe = 0;
        backend.exploration_decisions++;
        select(server_benefit_action::cacheflow, server_benefit_reason::safe_exploration);
    } else {
        select(server_benefit_action::upstream, server_benefit_reason::insufficient_evidence);
    }
    return decision;
}

void server_benefit_policy::observe(
        const server_benefit_decision & decision, const server_benefit_feedback & feedback) {
    if (feedback.elapsed_ms <= 0.0) return;
    auto & backend = state(decision.backend);
    auto & model = decision.action == server_benefit_action::cacheflow ? backend.cacheflow : backend.upstream;
    auto & bad_residuals = decision.action == server_benefit_action::cacheflow
            ? backend.cacheflow_bad_residuals : backend.upstream_bad_residuals;
    const double predicted = model.predict(decision.model_features);
    double cost = feedback.elapsed_ms;
    if (feedback.slo_violation || feedback.elapsed_ms > config_.target_iteration_ms) {
        cost += config_.slo_penalty * std::max(0.0, feedback.elapsed_ms - config_.target_iteration_ms);
    }
    if (feedback.execution_failed) cost += config_.failure_penalty_ms;

    const double residual = std::abs(cost - predicted);
    const bool mature = model.observations >= config_.minimum_observations * 3;
    // An SLO miss changes the optimized cost but is not itself model drift:
    // a consistently slow backend can be predicted accurately. Drift requires
    // an execution failure or a mature model's unexpectedly large residual.
    const bool severe = feedback.execution_failed ||
            (mature && residual > config_.drift_ratio * std::max(1.0, predicted));
    if (severe) bad_residuals++;
    else bad_residuals = 0;

    model.observe(decision.model_features, cost);
    if (bad_residuals >= config_.drift_consecutive_limit) {
        backend.drift_events++;
        bad_residuals = 0;
        backend.cooldown_remaining = config_.cooldown_decisions;
        model.reset(config_.ridge_lambda);
    }
    if (checkpoint_ && ++observations_since_checkpoint_ >= config_.checkpoint_interval) {
        try {
            checkpoint_->enqueue(encode_checkpoint());
            observations_since_checkpoint_ = 0;
        } catch (...) {
            checkpoint_enqueue_failures_++;
        }
    }
}

server_benefit_snapshot server_benefit_policy::snapshot(server_benefit_backend backend_id) const {
    const auto & backend = state(backend_id);
    std::array<double, feature_count> reference{};
    reference[0] = 1.0;
    return {
        backend.upstream.observations,
        backend.cacheflow.observations,
        backend.upstream_decisions,
        backend.cacheflow_decisions,
        backend.exploration_decisions,
        backend.cold_start_decisions,
        backend.positive_lower_bound_decisions,
        backend.insufficient_evidence_decisions,
        backend.safety_fallbacks,
        backend.drift_events,
        backend.cooldown_remaining,
        backend.upstream.predict(reference),
        backend.cacheflow.predict(reference),
        backend.last_predicted_benefit_ms,
        backend.last_uncertainty_ms,
        backend.choose_duration_us_buckets,
        backend.choose_duration_us_sum,
    };
}

std::string server_benefit_policy::encode_checkpoint() const {
    auto encode_model = [](const ridge_model & model) {
        checkpoint_json normal = checkpoint_json::array();
        for (size_t row = 0; row < feature_count; ++row) {
            checkpoint_json values = checkpoint_json::array();
            for (size_t column = 0; column < feature_count; ++column) {
                values.push_back(model.normal[row][column]);
            }
            normal.push_back(std::move(values));
        }
        checkpoint_json rhs = checkpoint_json::array();
        for (const double value : model.rhs) rhs.push_back(value);
        return checkpoint_json{
            {"normal", std::move(normal)},
            {"rhs", std::move(rhs)},
            {"observations", model.observations},
            {"residual_variance_ewma", model.residual_variance_ewma},
        };
    };
    auto encode_backend = [&](const backend_state & backend) {
        return checkpoint_json{
            {"upstream", encode_model(backend.upstream)},
            {"cacheflow", encode_model(backend.cacheflow)},
            {"upstream_decisions", backend.upstream_decisions},
            {"cacheflow_decisions", backend.cacheflow_decisions},
            {"exploration_decisions", backend.exploration_decisions},
            {"cold_start_decisions", backend.cold_start_decisions},
            {"positive_lower_bound_decisions", backend.positive_lower_bound_decisions},
            {"insufficient_evidence_decisions", backend.insufficient_evidence_decisions},
            {"safety_fallbacks", backend.safety_fallbacks},
            {"drift_events", backend.drift_events},
            {"upstream_bad_residuals", backend.upstream_bad_residuals},
            {"cacheflow_bad_residuals", backend.cacheflow_bad_residuals},
            {"cooldown_remaining", backend.cooldown_remaining},
            {"decisions_since_probe", backend.decisions_since_probe},
            {"last_predicted_benefit_ms", backend.last_predicted_benefit_ms},
            {"last_uncertainty_ms", backend.last_uncertainty_ms},
        };
    };

    checkpoint_json state = {
        {"next_id", next_id_},
        {"cpu", encode_backend(cpu_)},
        {"cuda", encode_backend(cuda_)},
    };
    const checkpoint_json configuration = checkpoint_configuration(config_);
    const uint32_t checksum = checkpoint_crc32(checkpoint_checksum_input(
            config_.checkpoint_compatibility_key, configuration, state));
    checkpoint_json envelope = {
        {"schema_version", 1},
        {"feature_count", feature_count},
        {"compatibility_key", config_.checkpoint_compatibility_key},
        {"configuration", configuration},
        {"state", std::move(state)},
        {"crc32", checksum},
    };
    return envelope.dump();
}

server_benefit_policy::checkpoint_restore_result server_benefit_policy::restore_checkpoint(
        const std::string & payload) {
    try {
        const checkpoint_json envelope = checkpoint_json::parse(payload);
        if (envelope.at("schema_version").get<int>() != 1 ||
                envelope.at("feature_count").get<size_t>() != feature_count) {
            return checkpoint_restore_result::SERVER_BENEFIT_CHECKPOINT_RESTORE_RESULT_INCOMPATIBLE;
        }
        const std::string compatibility = envelope.at("compatibility_key").get<std::string>();
        const checkpoint_json configuration = envelope.at("configuration");
        if (compatibility != config_.checkpoint_compatibility_key ||
                configuration != checkpoint_configuration(config_)) {
            return checkpoint_restore_result::SERVER_BENEFIT_CHECKPOINT_RESTORE_RESULT_INCOMPATIBLE;
        }
        const checkpoint_json & state_json = envelope.at("state");
        const uint32_t expected_crc = checkpoint_crc32(
                checkpoint_checksum_input(compatibility, configuration, state_json));
        if (envelope.at("crc32").get<uint32_t>() != expected_crc) {
            return checkpoint_restore_result::SERVER_BENEFIT_CHECKPOINT_RESTORE_RESULT_INVALID;
        }

        auto finite = [](double value) {
            return std::isfinite(value) && std::abs(value) <= 1e18;
        };
        auto load_size = [](const checkpoint_json & value) {
            const uint64_t parsed = value.get<uint64_t>();
            if (parsed > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw std::out_of_range("checkpoint size does not fit size_t");
            }
            return static_cast<size_t>(parsed);
        };
        auto load_model = [&](const checkpoint_json & value, ridge_model & model) {
            const auto & normal = value.at("normal");
            const auto & rhs = value.at("rhs");
            if (!normal.is_array() || normal.size() != feature_count ||
                    !rhs.is_array() || rhs.size() != feature_count) {
                throw std::invalid_argument("invalid ridge checkpoint dimensions");
            }
            for (size_t row = 0; row < feature_count; ++row) {
                if (!normal[row].is_array() || normal[row].size() != feature_count) {
                    throw std::invalid_argument("invalid ridge checkpoint matrix row");
                }
                model.rhs[row] = rhs[row].get<double>();
                if (!finite(model.rhs[row])) throw std::invalid_argument("non-finite ridge rhs");
                for (size_t column = 0; column < feature_count; ++column) {
                    model.normal[row][column] = normal[row][column].get<double>();
                    if (!finite(model.normal[row][column])) {
                        throw std::invalid_argument("non-finite ridge matrix");
                    }
                }
                if (model.normal[row][row] < config_.ridge_lambda) {
                    throw std::invalid_argument("ridge matrix lost regularization diagonal");
                }
            }
            for (size_t row = 0; row < feature_count; ++row) {
                for (size_t column = row + 1; column < feature_count; ++column) {
                    const double scale = std::max({1.0, std::abs(model.normal[row][column]),
                            std::abs(model.normal[column][row])});
                    if (std::abs(model.normal[row][column] - model.normal[column][row]) > 1e-10 * scale) {
                        throw std::invalid_argument("ridge matrix is not symmetric");
                    }
                }
            }
            model.observations = value.at("observations").get<uint64_t>();
            if (model.observations > 1000000000000ULL) {
                throw std::invalid_argument("ridge observation count is implausible");
            }
            model.residual_variance_ewma = value.at("residual_variance_ewma").get<double>();
            if (!finite(model.residual_variance_ewma) || model.residual_variance_ewma < 0.0) {
                throw std::invalid_argument("invalid residual variance");
            }
        };
        auto load_backend = [&](const checkpoint_json & value, backend_state & backend) {
            load_model(value.at("upstream"), backend.upstream);
            load_model(value.at("cacheflow"), backend.cacheflow);
            backend.upstream_decisions = value.at("upstream_decisions").get<uint64_t>();
            backend.cacheflow_decisions = value.at("cacheflow_decisions").get<uint64_t>();
            backend.exploration_decisions = value.at("exploration_decisions").get<uint64_t>();
            backend.cold_start_decisions = value.at("cold_start_decisions").get<uint64_t>();
            backend.positive_lower_bound_decisions = value.at("positive_lower_bound_decisions").get<uint64_t>();
            backend.insufficient_evidence_decisions = value.at("insufficient_evidence_decisions").get<uint64_t>();
            backend.safety_fallbacks = value.at("safety_fallbacks").get<uint64_t>();
            backend.drift_events = value.at("drift_events").get<uint64_t>();
            backend.upstream_bad_residuals = load_size(value.at("upstream_bad_residuals"));
            backend.cacheflow_bad_residuals = load_size(value.at("cacheflow_bad_residuals"));
            backend.cooldown_remaining = load_size(value.at("cooldown_remaining"));
            backend.decisions_since_probe = value.at("decisions_since_probe").get<uint64_t>();
            backend.last_predicted_benefit_ms = value.at("last_predicted_benefit_ms").get<double>();
            backend.last_uncertainty_ms = value.at("last_uncertainty_ms").get<double>();
            if (!finite(backend.last_predicted_benefit_ms) ||
                    !finite(backend.last_uncertainty_ms) || backend.last_uncertainty_ms < 0.0) {
                throw std::invalid_argument("invalid benefit prediction telemetry");
            }
        };

        backend_state restored_cpu{};
        backend_state restored_cuda{};
        load_backend(state_json.at("cpu"), restored_cpu);
        load_backend(state_json.at("cuda"), restored_cuda);
        const uint64_t restored_next_id = state_json.at("next_id").get<uint64_t>();
        if (restored_next_id == 0) throw std::invalid_argument("invalid next decision id");
        cpu_ = std::move(restored_cpu);
        cuda_ = std::move(restored_cuda);
        next_id_ = restored_next_id;
        return checkpoint_restore_result::SERVER_BENEFIT_CHECKPOINT_RESTORE_RESULT_RESTORED;
    } catch (...) {
        return checkpoint_restore_result::SERVER_BENEFIT_CHECKPOINT_RESTORE_RESULT_INVALID;
    }
}

bool server_benefit_policy::flush_checkpoint() {
    if (!checkpoint_) return true;
    if (observations_since_checkpoint_ > 0) {
        try {
            checkpoint_->enqueue(encode_checkpoint());
            observations_since_checkpoint_ = 0;
        } catch (...) {
            checkpoint_enqueue_failures_++;
            return false;
        }
    }
    return checkpoint_->flush();
}

server_benefit_persistence_snapshot server_benefit_policy::checkpoint_snapshot() const {
    server_benefit_persistence_snapshot result;
    result.restored = checkpoints_restored_;
    result.incompatible = checkpoints_incompatible_;
    result.restore_failures = checkpoint_restore_failures_;
    result.save_failures = checkpoint_enqueue_failures_;
    if (checkpoint_) {
        const auto io = checkpoint_->snapshot();
        result.checkpoints_enqueued = io.enqueued;
        result.checkpoints_coalesced = io.coalesced;
        result.saves_completed = io.saves_completed;
        result.save_failures += io.save_failures;
        result.pending = io.pending;
    }
    return result;
}
