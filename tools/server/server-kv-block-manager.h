#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

using server_kv_block_id = uint64_t;
using server_kv_sequence_id = int64_t;
using server_kv_token = int32_t;

struct server_kv_attach_result {
    bool admitted = false;
    size_t shared_blocks = 0;
    size_t allocated_blocks = 0;
    size_t reserved_blocks = 0;
    size_t matched_tokens = 0;
    std::string reason;
};

struct server_kv_write_result {
    bool writable = false;
    bool copied = false;
    server_kv_block_id block_id = 0;
    std::string reason;
};

struct server_kv_block_view {
    server_kv_block_id id = 0;
    server_kv_block_id parent = 0;
    size_t token_count = 0;
    size_t ref_count = 0;
    bool prefix_indexed = false;
    uint64_t last_access_us = 0;
};

struct server_kv_sequence_view {
    server_kv_sequence_id id = -1;
    std::vector<server_kv_block_id> blocks;
    size_t prompt_tokens = 0;
    size_t reserved_blocks = 0;
};

struct server_kv_block_snapshot {
    size_t block_size = 0;
    size_t capacity_blocks = 0;
    size_t allocated_blocks = 0;
    size_t reserved_blocks = 0;
    size_t shared_blocks = 0;
    uint64_t copy_on_write_total = 0;
    std::vector<server_kv_block_view> blocks;
    std::vector<server_kv_sequence_view> sequences;
};

// Owns logical KV blocks, sequence Block Tables and the block-prefix index.
// Full immutable blocks may be shared. Partial tail blocks stay private. A
// shared final block must be cloned through make_tail_writable() before a
// backend mutates it.
class server_kv_block_manager {
public:
    server_kv_block_manager(size_t block_size, size_t capacity_blocks);

    server_kv_attach_result attach(
            server_kv_sequence_id sequence_id,
            const std::vector<server_kv_token> & prompt,
            size_t reserve_tokens,
            uint64_t now_us);

    bool append(
            server_kv_sequence_id sequence_id,
            const std::vector<server_kv_token> & tokens,
            uint64_t now_us,
            std::string * error = nullptr);

    server_kv_write_result make_tail_writable(
            server_kv_sequence_id sequence_id,
            uint64_t now_us);

    size_t longest_prefix_blocks(const std::vector<server_kv_token> & prompt) const;
    bool release(server_kv_sequence_id sequence_id);
    server_kv_block_snapshot snapshot() const;

    // Returns an empty string when every ownership/capacity invariant holds.
    std::string validate() const;

private:
    struct block {
        server_kv_block_id id = 0;
        server_kv_block_id parent = 0;
        std::vector<server_kv_token> tokens;
        size_t ref_count = 0;
        uint64_t hash = 0;
        uint64_t last_access_us = 0;
        bool prefix_indexed = false;
    };

    struct sequence {
        server_kv_sequence_id id = -1;
        std::vector<server_kv_block_id> blocks;
        size_t prompt_tokens = 0;
        size_t reserved_blocks = 0;
    };

    size_t block_size_;
    size_t capacity_blocks_;
    server_kv_block_id next_block_id_ = 1;
    size_t reserved_blocks_ = 0;
    uint64_t copy_on_write_total_ = 0;
    std::unordered_map<server_kv_block_id, block> blocks_;
    std::unordered_map<server_kv_sequence_id, sequence> sequences_;
    std::unordered_map<uint64_t, std::vector<server_kv_block_id>> prefix_index_;

    size_t free_unreserved_blocks() const;
    uint64_t chain_hash(server_kv_block_id parent, const std::vector<server_kv_token> & tokens) const;
    server_kv_block_id find_prefix_block(
            server_kv_block_id parent,
            const std::vector<server_kv_token> & tokens) const;
    server_kv_block_id allocate_block(
            server_kv_block_id parent,
            std::vector<server_kv_token> tokens,
            bool prefix_indexed,
            uint64_t now_us);
    void index_block(server_kv_block_id id);
    void unindex_block(server_kv_block_id id);
    void release_block(server_kv_block_id id);
};
