#include "server-inference-scheduler.h"

#include <algorithm>
#include <cmath>

size_t server_prefill_plan::quota(int id) const {
    const auto it = std::find_if(allocations.begin(), allocations.end(),
            [id](const server_prefill_allocation & allocation) {
                return allocation.id == id;
            });
    return it == allocations.end() ? 0 : it->tokens;
}

bool server_prefill_plan::equivalent_allocations(const server_prefill_plan & other) const {
    if (tokens_scheduled != other.tokens_scheduled || allocations.size() != other.allocations.size()) {
        return false;
    }
    return std::all_of(allocations.begin(), allocations.end(), [&](const auto & allocation) {
        return other.quota(allocation.id) == allocation.tokens;
    });
}

server_prefill_plan server_inference_scheduler::plan_upstream_prefill(
        const std::vector<server_prefill_candidate> & candidates,
        size_t token_budget) {
    server_prefill_plan plan;
    plan.token_budget = token_budget;
    if (token_budget == 0) return plan;
    auto ordered = candidates;
    std::sort(ordered.begin(), ordered.end(), [](const auto & left, const auto & right) {
        return left.id < right.id;
    });
    size_t remaining = token_budget;
    for (const auto & candidate : ordered) {
        const size_t allocation = std::min(candidate.remaining_tokens, remaining);
        if (allocation == 0) continue;
        plan.allocations.push_back({candidate.id, allocation});
        plan.tokens_scheduled += allocation;
        remaining -= allocation;
        if (remaining == 0) break;
    }
    return plan;
}

server_inference_scheduler::server_inference_scheduler(server_scheduler_config config) {
    configure(config);
}

void server_inference_scheduler::configure(server_scheduler_config config) {
    config.min_prefix_similarity = std::max(config.min_prefix_similarity, 0.0f);
    config.eviction_penalty = std::max(config.eviction_penalty, 0.0f);
    config.prefill_chunk_min = std::max<size_t>(1, config.prefill_chunk_min);
    config.prefill_chunk_max = std::max(config.prefill_chunk_min, config.prefill_chunk_max);
    config.target_iteration_ms = std::max(0.1, config.target_iteration_ms);
    config_ = config;
    adaptive_chunk_size_ = std::clamp(
            config.prefill_chunk_size == 0 ? config.prefill_chunk_max : config.prefill_chunk_size,
            config.prefill_chunk_min,
            config.prefill_chunk_max);
    iteration_latency_ewma_ms_ = 0.0;
    observations_ = 0;
    cost_model_.reset();
}

server_slot_plan server_inference_scheduler::select_slot(
        const std::vector<server_slot_candidate> & candidates,
        size_t input_tokens) const {
    server_slot_plan best;
    if (input_tokens == 0 || config_.min_prefix_similarity <= 0.0f) {
        return best;
    }

    for (const auto & candidate : candidates) {
        if (candidate.cached_tokens == 0 || candidate.common_prefix_tokens == 0) {
            continue;
        }

        const double similarity = (double) candidate.common_prefix_tokens / input_tokens;
        if (similarity <= config_.min_prefix_similarity) {
            continue;
        }

        const size_t evicted = candidate.cached_tokens > candidate.common_prefix_tokens
                ? candidate.cached_tokens - candidate.common_prefix_tokens
                : 0;
        const double score = candidate.common_prefix_tokens - config_.eviction_penalty * evicted;
        const bool better = score > best.score ||
                (score == best.score && candidate.last_used < best.last_used);
        if (score > 0.0 && better) {
            best = {
                candidate.id,
                candidate.common_prefix_tokens,
                evicted,
                score,
                candidate.last_used,
            };
        }
    }
    return best;
}

server_prefill_plan server_inference_scheduler::plan_prefill(
        const std::vector<server_prefill_candidate> & candidates,
        size_t token_budget) {
    server_prefill_plan plan;
    plan.token_budget = token_budget;
    const size_t chunk_size = config_.adaptive_prefill
            ? (config_.adaptive_greedy_fallback ? 0 : adaptive_chunk_size_)
            : config_.prefill_chunk_size;
    plan.effective_chunk_size = chunk_size;
    if (token_budget == 0 || candidates.empty()) {
        return plan;
    }

    // Stable slot ordering plus a rotating start gives deterministic round
    // robin fairness across decode iterations.
    std::vector<server_prefill_candidate> ordered;
    ordered.reserve(candidates.size());
    for (const auto & candidate : candidates) {
        if (candidate.remaining_tokens > 0) {
            ordered.push_back(candidate);
        }
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto & left, const auto & right) {
        return left.id < right.id;
    });
    if (ordered.empty()) {
        return plan;
    }

    size_t start = 0;
    if (chunk_size != 0) {
        const auto next = std::find_if(ordered.begin(), ordered.end(), [this](const auto & candidate) {
            return candidate.id > last_prefill_slot_;
        });
        if (next != ordered.end()) {
            start = (size_t) std::distance(ordered.begin(), next);
        }
    }

    size_t remaining_budget = token_budget;
    for (size_t offset = 0; offset < ordered.size() && remaining_budget > 0; ++offset) {
        const auto & candidate = ordered[(start + offset) % ordered.size()];
        const size_t per_slot_limit = chunk_size == 0
                ? remaining_budget
                : chunk_size;
        const size_t allocation = std::min({
                candidate.remaining_tokens,
                per_slot_limit,
                remaining_budget,
        });
        if (allocation == 0) {
            continue;
        }
        plan.allocations.push_back({candidate.id, allocation});
        plan.tokens_scheduled += allocation;
        remaining_budget -= allocation;
        if (chunk_size != 0) {
            last_prefill_slot_ = candidate.id;
        }
    }
    return plan;
}

void server_inference_scheduler::observe_iteration(
        const server_scheduler_observation & observation) {
    if (observation.elapsed_ms <= 0.0) return;
    constexpr double alpha = 0.2;
    iteration_latency_ewma_ms_ = observations_ == 0
            ? observation.elapsed_ms
            : alpha * observation.elapsed_ms + (1.0 - alpha) * iteration_latency_ewma_ms_;
    observations_++;
    cost_model_.observe(
            observation.decode_tokens, observation.prefill_tokens, observation.elapsed_ms);
    if (!config_.adaptive_prefill || observation.prefill_tokens == 0 ||
            config_.adaptive_greedy_fallback) return;

    if (observation.decode_tokens == 0) {
        // No latency-sensitive decode work is waiting: consume prompt at the
        // configured throughput ceiling.
        adaptive_chunk_size_ = config_.prefill_chunk_max;
        return;
    }

    const auto cost_state = cost_model_.state();
    if (cost_state.observations >= 3 && cost_state.prefill_ms_per_token > 1e-6) {
        size_t recommended = cost_model_.recommend_prefill_chunk(
                config_.target_iteration_ms,
                observation.decode_tokens,
                observation.prefill_sequences,
                config_.prefill_chunk_min,
                config_.prefill_chunk_max);
        // Regression is least reliable during the first mixed iterations.
        // Never shrink while the measured iteration still has clear latency
        // headroom; use that headroom to gather a more informative sample.
        if (observation.elapsed_ms < config_.target_iteration_ms * 0.70) {
            recommended = std::max(recommended, std::min(
                    config_.prefill_chunk_max,
                    adaptive_chunk_size_ + config_.prefill_chunk_min));
        }
        adaptive_chunk_size_ = std::clamp(
                (adaptive_chunk_size_ + recommended) / 2,
                config_.prefill_chunk_min,
                config_.prefill_chunk_max);
    } else if (observation.elapsed_ms > config_.target_iteration_ms) {
        adaptive_chunk_size_ = std::max(
                config_.prefill_chunk_min,
                adaptive_chunk_size_ / 2);
    } else if (observation.elapsed_ms < config_.target_iteration_ms * 0.70) {
        // Additive growth avoids oscillating directly back to a large chunk.
        adaptive_chunk_size_ = std::min(
                config_.prefill_chunk_max,
                adaptive_chunk_size_ + config_.prefill_chunk_min);
    }
}

server_scheduler_state server_inference_scheduler::state() const {
    return {
        config_.adaptive_prefill,
        config_.adaptive_prefill
                ? (config_.adaptive_greedy_fallback ? 0 : adaptive_chunk_size_)
                : config_.prefill_chunk_size,
        iteration_latency_ewma_ms_,
        observations_,
        cost_model_.state(),
    };
}
