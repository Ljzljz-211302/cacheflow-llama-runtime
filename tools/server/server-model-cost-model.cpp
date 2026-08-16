#include "server-model-cost-model.h"

#include <algorithm>
#include <cmath>

server_model_cost_model::server_model_cost_model(double decay) :
        decay_(std::clamp(decay, 0.5, 0.999)) {
    reset();
}

void server_model_cost_model::reset() {
    observations_ = 0;
    for (auto & row : normal_) for (double & value : row) value = 0.0;
    for (double & value : rhs_) value = 0.0;
}

void server_model_cost_model::observe(
        size_t decode_tokens, size_t prefill_tokens, double elapsed_ms) {
    if (elapsed_ms <= 0.0 || (decode_tokens == 0 && prefill_tokens == 0)) return;
    const double features[3] = {
        (double) decode_tokens,
        (double) prefill_tokens,
        1.0,
    };
    for (size_t row = 0; row < 3; ++row) {
        rhs_[row] = decay_ * rhs_[row] + features[row] * elapsed_ms;
        for (size_t column = 0; column < 3; ++column) {
            normal_[row][column] = decay_ * normal_[row][column] +
                    features[row] * features[column];
        }
    }
    observations_++;
}

server_model_cost_state server_model_cost_model::solve() const {
    double matrix[3][4]{};
    for (size_t row = 0; row < 3; ++row) {
        for (size_t column = 0; column < 3; ++column) {
            matrix[row][column] = normal_[row][column];
        }
        matrix[row][row] += 1e-6;
        matrix[row][3] = rhs_[row];
    }
    for (size_t pivot = 0; pivot < 3; ++pivot) {
        size_t best = pivot;
        for (size_t row = pivot + 1; row < 3; ++row) {
            if (std::abs(matrix[row][pivot]) > std::abs(matrix[best][pivot])) best = row;
        }
        if (best != pivot) {
            for (size_t column = pivot; column < 4; ++column) {
                std::swap(matrix[pivot][column], matrix[best][column]);
            }
        }
        const double divisor = matrix[pivot][pivot];
        if (std::abs(divisor) < 1e-9) continue;
        for (size_t column = pivot; column < 4; ++column) matrix[pivot][column] /= divisor;
        for (size_t row = 0; row < 3; ++row) {
            if (row == pivot) continue;
            const double factor = matrix[row][pivot];
            for (size_t column = pivot; column < 4; ++column) {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
        }
    }
    return {
        std::max(0.0, matrix[0][3]),
        std::max(0.0, matrix[1][3]),
        std::max(0.0, matrix[2][3]),
        observations_,
        std::min(1.0, observations_ / 12.0),
    };
}

server_model_cost_state server_model_cost_model::state() const {
    return solve();
}

size_t server_model_cost_model::recommend_prefill_chunk(
        double target_iteration_ms,
        size_t decode_tokens,
        size_t active_prefill_sequences,
        size_t minimum,
        size_t maximum) const {
    const auto estimate = solve();
    if (estimate.observations < 3 || estimate.prefill_ms_per_token <= 1e-6) return maximum;
    const double decode_cost = estimate.fixed_ms + decode_tokens * estimate.decode_ms_per_token;
    const double available = std::max(0.0, target_iteration_ms - decode_cost);
    const double total_tokens = available / estimate.prefill_ms_per_token;
    const size_t sequences = std::max<size_t>(1, active_prefill_sequences);
    const size_t per_sequence = (size_t) std::floor(total_tokens / sequences);
    return std::clamp(per_sequence, minimum, maximum);
}
