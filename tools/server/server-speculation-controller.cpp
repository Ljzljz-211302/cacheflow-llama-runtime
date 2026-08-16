#include "server-speculation-controller.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

server_speculation_controller::server_speculation_controller(server_speculation_config config) {
    configure(config);
}

void server_speculation_controller::configure(server_speculation_config config) {
    config.min_draft_tokens = std::max<size_t>(1, config.min_draft_tokens);
    config.max_draft_tokens = std::max(config.min_draft_tokens, config.max_draft_tokens);
    config.acceptance_alpha = std::clamp(config.acceptance_alpha, 0.01, 1.0);
    config.disable_below_acceptance = std::clamp(config.disable_below_acceptance, 0.0, 1.0);
    config.grow_above_acceptance = std::clamp(config.grow_above_acceptance, 0.0, 1.0);
    config.kv_pressure_threshold = std::clamp(config.kv_pressure_threshold, 0.0, 1.0);
    config_ = config;
    states_.clear();
    disabled_low_acceptance_total_ = 0;
}

server_speculation_state & server_speculation_controller::get_or_create(int sequence_id) {
    auto [it, inserted] = states_.emplace(sequence_id, server_speculation_state{});
    if (inserted) it->second.draft_tokens = config_.max_draft_tokens;
    return it->second;
}

size_t server_speculation_controller::recommend(
        int sequence_id,
        size_t configured_max,
        double kv_pressure) {
    if (configured_max == 0) return 0;
    if (!config_.adaptive) return configured_max;
    auto & value = get_or_create(sequence_id);
    if (value.cooldown_remaining > 0) {
        value.cooldown_remaining--;
        value.disabled = true;
        return 0;
    }
    value.disabled = false;
    size_t result = std::min({value.draft_tokens, configured_max, config_.max_draft_tokens});
    if (kv_pressure >= config_.kv_pressure_threshold) {
        result = std::max(config_.min_draft_tokens, result / 2);
    }
    return result;
}

void server_speculation_controller::observe(
        const server_speculation_observation & observation) {
    if (observation.drafted_tokens == 0) return;
    auto & value = get_or_create(observation.sequence_id);
    const size_t accepted = std::min(observation.accepted_tokens, observation.drafted_tokens);
    const double ratio = (double) accepted / observation.drafted_tokens;
    value.acceptance_ewma = value.observations == 0
            ? ratio
            : config_.acceptance_alpha * ratio +
                    (1.0 - config_.acceptance_alpha) * value.acceptance_ewma;
    value.drafted_total += observation.drafted_tokens;
    value.accepted_total += accepted;
    value.observations++;
    if (!config_.adaptive) return;

    const bool has_evidence = value.observations >= config_.warmup_observations &&
            value.drafted_total >= config_.min_evidence_tokens;
    if (has_evidence &&
            value.acceptance_ewma < config_.disable_below_acceptance) {
        if (!value.disabled && value.cooldown_remaining == 0) {
            disabled_low_acceptance_total_++;
        }
        value.cooldown_remaining = config_.cooldown_iterations;
        value.disabled = true;
        value.draft_tokens = config_.min_draft_tokens;
        return;
    }
    if (observation.kv_pressure >= config_.kv_pressure_threshold ||
            (has_evidence && value.acceptance_ewma < 0.40)) {
        value.draft_tokens = std::max(config_.min_draft_tokens, value.draft_tokens / 2);
    } else if (value.acceptance_ewma >= config_.grow_above_acceptance) {
        value.draft_tokens = std::min(config_.max_draft_tokens, value.draft_tokens + 1);
    }
}

server_speculation_state server_speculation_controller::state(int sequence_id) const {
    const auto it = states_.find(sequence_id);
    if (it != states_.end()) return it->second;
    server_speculation_state result;
    result.draft_tokens = config_.max_draft_tokens;
    return result;
}

void server_speculation_controller::reset(int sequence_id) {
    states_.erase(sequence_id);
}
