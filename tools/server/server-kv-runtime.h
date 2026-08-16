#pragma once

#include "server-kv-block-manager.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

enum class server_kv_residency : uint8_t {
    resident,
    swapped,
};

struct server_kv_prefix_share_plan {
    server_kv_sequence_id donor = -1;
    server_kv_sequence_id destination = -1;
    size_t matched_tokens = 0;
    size_t matched_blocks = 0;

    bool found() const { return donor >= 0 && matched_tokens > 0; }
};

struct server_kv_runtime_sequence {
    server_kv_sequence_id id = -1;
    server_kv_residency residency = server_kv_residency::resident;
    std::vector<server_kv_token> committed_tokens;
    size_t reserved_tokens = 0;
    uint64_t last_access_us = 0;
};

struct server_kv_runtime_snapshot {
    server_kv_block_snapshot blocks;
    std::vector<server_kv_runtime_sequence> sequences;
};

// Transactional coordinator for logical block ownership, resident-prefix
// discovery and preempt/restore state. Physical llama/CUDA mutations are
// executed by the engine only after a returned plan succeeds.
class server_kv_runtime {
public:
    server_kv_runtime(size_t block_size, size_t capacity_blocks);

    bool synchronize(
            server_kv_sequence_id sequence_id,
            const std::vector<server_kv_token> & committed_tokens,
            size_t reserve_tokens,
            uint64_t now_us,
            std::string * error = nullptr);

    server_kv_prefix_share_plan plan_prefix_share(
            server_kv_sequence_id destination,
            const std::vector<server_kv_token> & prompt,
            const std::vector<server_kv_sequence_id> & eligible_donors = {}) const;

    bool preempt(server_kv_sequence_id sequence_id, uint64_t now_us);
    bool restore(server_kv_sequence_id sequence_id, uint64_t now_us, std::string * error = nullptr);
    bool release(server_kv_sequence_id sequence_id);
    bool is_swapped(server_kv_sequence_id sequence_id) const;

    server_kv_runtime_snapshot snapshot() const;
    std::string validate() const;

private:
    size_t block_size_;
    server_kv_block_manager blocks_;
    std::unordered_map<server_kv_sequence_id, server_kv_runtime_sequence> sequences_;

    static size_t common_prefix(
            const std::vector<server_kv_token> & left,
            const std::vector<server_kv_token> & right);
};
