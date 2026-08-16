#include "server-kv-block-manager.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

server_kv_block_manager::server_kv_block_manager(size_t block_size, size_t capacity_blocks) :
        block_size_(block_size), capacity_blocks_(capacity_blocks) {
    if (block_size == 0 || capacity_blocks == 0) {
        throw std::invalid_argument("KV block size and capacity must be positive");
    }
}

size_t server_kv_block_manager::free_unreserved_blocks() const {
    const size_t committed = blocks_.size() + reserved_blocks_;
    return committed < capacity_blocks_ ? capacity_blocks_ - committed : 0;
}

uint64_t server_kv_block_manager::chain_hash(
        server_kv_block_id parent,
        const std::vector<server_kv_token> & tokens) const {
    uint64_t hash = 1469598103934665603ULL;
    const auto parent_it = blocks_.find(parent);
    const uint64_t parent_hash = parent_it == blocks_.end() ? 0 : parent_it->second.hash;
    hash ^= parent_hash;
    hash *= 1099511628211ULL;
    for (server_kv_token token : tokens) {
        const uint32_t value = (uint32_t) token;
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= (value >> shift) & 0xffU;
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

server_kv_block_id server_kv_block_manager::find_prefix_block(
        server_kv_block_id parent,
        const std::vector<server_kv_token> & tokens) const {
    const uint64_t hash = chain_hash(parent, tokens);
    const auto index_it = prefix_index_.find(hash);
    if (index_it == prefix_index_.end()) {
        return 0;
    }
    for (server_kv_block_id id : index_it->second) {
        const auto block_it = blocks_.find(id);
        if (block_it != blocks_.end() && block_it->second.prefix_indexed &&
                block_it->second.parent == parent && block_it->second.tokens == tokens) {
            return id;
        }
    }
    return 0;
}

server_kv_block_id server_kv_block_manager::allocate_block(
        server_kv_block_id parent,
        std::vector<server_kv_token> tokens,
        bool prefix_indexed,
        uint64_t now_us) {
    const server_kv_block_id id = next_block_id_++;
    block value;
    value.id = id;
    value.parent = parent;
    value.hash = chain_hash(parent, tokens);
    value.tokens = std::move(tokens);
    value.ref_count = 1;
    value.last_access_us = now_us;
    // A prefix-indexed block must be reachable from the root through an
    // entirely indexed chain. COW creates a deliberately private block; its
    // descendants must remain private even after they become full.
    const auto parent_it = blocks_.find(parent);
    prefix_indexed = prefix_indexed &&
            (parent == 0 || (parent_it != blocks_.end() && parent_it->second.prefix_indexed));
    value.prefix_indexed = prefix_indexed;
    blocks_.emplace(id, std::move(value));
    if (prefix_indexed) {
        index_block(id);
    }
    return id;
}

void server_kv_block_manager::index_block(server_kv_block_id id) {
    auto & value = blocks_.at(id);
    value.prefix_indexed = true;
    prefix_index_[value.hash].push_back(id);
}

void server_kv_block_manager::unindex_block(server_kv_block_id id) {
    auto block_it = blocks_.find(id);
    if (block_it == blocks_.end() || !block_it->second.prefix_indexed) {
        return;
    }
    auto index_it = prefix_index_.find(block_it->second.hash);
    if (index_it != prefix_index_.end()) {
        auto & ids = index_it->second;
        ids.erase(std::remove(ids.begin(), ids.end(), id), ids.end());
        if (ids.empty()) {
            prefix_index_.erase(index_it);
        }
    }
    block_it->second.prefix_indexed = false;
}

void server_kv_block_manager::release_block(server_kv_block_id id) {
    auto it = blocks_.find(id);
    if (it == blocks_.end() || it->second.ref_count == 0) {
        throw std::logic_error("releasing unknown or unreferenced KV block");
    }
    if (--it->second.ref_count == 0) {
        unindex_block(id);
        blocks_.erase(id);
    }
}

server_kv_attach_result server_kv_block_manager::attach(
        server_kv_sequence_id sequence_id,
        const std::vector<server_kv_token> & prompt,
        size_t reserve_tokens,
        uint64_t now_us) {
    server_kv_attach_result result;
    if (sequences_.count(sequence_id) != 0) {
        result.reason = "sequence already attached";
        return result;
    }

    const size_t prompt_blocks = (prompt.size() + block_size_ - 1) / block_size_;
    std::vector<server_kv_block_id> shared;
    shared.reserve(prompt_blocks);
    server_kv_block_id parent = 0;
    size_t offset = 0;
    while (offset + block_size_ <= prompt.size()) {
        std::vector<server_kv_token> chunk(
                prompt.begin() + offset, prompt.begin() + offset + block_size_);
        const server_kv_block_id found = find_prefix_block(parent, chunk);
        if (found == 0) {
            break;
        }
        shared.push_back(found);
        parent = found;
        offset += block_size_;
    }

    // An exact partial tail is reusable too. It is indexed as an immutable
    // prefix and cloned on the first append by either owner.
    if (offset < prompt.size()) {
        std::vector<server_kv_token> chunk(prompt.begin() + offset, prompt.end());
        const server_kv_block_id found = find_prefix_block(parent, chunk);
        if (found != 0) {
            shared.push_back(found);
            parent = found;
            offset = prompt.size();
        }
    }

    const size_t new_blocks = prompt_blocks - shared.size();
    const size_t reserve_blocks = (reserve_tokens + block_size_ - 1) / block_size_;
    if (new_blocks + reserve_blocks > free_unreserved_blocks()) {
        result.reason = "insufficient KV block capacity";
        return result;
    }

    sequence seq;
    seq.id = sequence_id;
    seq.prompt_tokens = prompt.size();
    seq.reserved_blocks = reserve_blocks;
    seq.blocks = shared;
    for (server_kv_block_id id : shared) {
        auto & value = blocks_.at(id);
        value.ref_count++;
        value.last_access_us = now_us;
    }

    while (offset < prompt.size()) {
        const size_t count = std::min(block_size_, prompt.size() - offset);
        std::vector<server_kv_token> chunk(prompt.begin() + offset, prompt.begin() + offset + count);
        const server_kv_block_id id = allocate_block(parent, std::move(chunk), true, now_us);
        seq.blocks.push_back(id);
        parent = id;
        offset += count;
    }

    reserved_blocks_ += reserve_blocks;
    sequences_.emplace(sequence_id, std::move(seq));
    result.admitted = true;
    result.shared_blocks = shared.size();
    result.allocated_blocks = new_blocks;
    result.reserved_blocks = reserve_blocks;
    result.matched_tokens = 0;
    for (server_kv_block_id id : shared) result.matched_tokens += blocks_.at(id).tokens.size();
    return result;
}

bool server_kv_block_manager::append(
        server_kv_sequence_id sequence_id,
        const std::vector<server_kv_token> & tokens,
        uint64_t now_us,
        std::string * error) {
    auto seq_it = sequences_.find(sequence_id);
    if (seq_it == sequences_.end()) {
        if (error) *error = "unknown sequence";
        return false;
    }
    if (tokens.empty()) {
        return true;
    }
    auto & seq = seq_it->second;
    size_t tail_free = 0;
    if (!seq.blocks.empty()) {
        const auto & tail = blocks_.at(seq.blocks.back());
        if (tail.tokens.size() < block_size_) {
            tail_free = block_size_ - tail.tokens.size();
        }
    }
    const size_t after_tail = tokens.size() > tail_free ? tokens.size() - tail_free : 0;
    const size_t needed_blocks = (after_tail + block_size_ - 1) / block_size_;
    const bool cow_needed = tail_free > 0 && blocks_.at(seq.blocks.back()).ref_count > 1;
    const size_t total_allocations = needed_blocks + (cow_needed ? 1 : 0);
    const size_t reserved_for_all = std::min(total_allocations, seq.reserved_blocks);
    const size_t extra = total_allocations - reserved_for_all;
    if (extra > free_unreserved_blocks()) {
        if (error) *error = "insufficient KV block capacity";
        return false;
    }

    size_t offset = 0;
    if (tail_free > 0) {
        const auto writable = make_tail_writable(sequence_id, now_us);
        if (!writable.writable) {
            if (error) *error = writable.reason;
            return false;
        }
        auto & tail = blocks_.at(seq.blocks.back());
        const bool keep_canonical = tail.prefix_indexed;
        unindex_block(tail.id);
        const size_t count = std::min(tail_free, tokens.size());
        tail.tokens.insert(tail.tokens.end(), tokens.begin(), tokens.begin() + count);
        tail.last_access_us = now_us;
        offset += count;
        if (keep_canonical &&
                (tail.parent == 0 || blocks_.at(tail.parent).prefix_indexed)) {
            tail.hash = chain_hash(tail.parent, tail.tokens);
            index_block(tail.id);
        }
    }

    const size_t from_reserve = std::min(needed_blocks, seq.reserved_blocks);
    server_kv_block_id parent = seq.blocks.empty() ? 0 : seq.blocks.back();
    while (offset < tokens.size()) {
        const size_t count = std::min(block_size_, tokens.size() - offset);
        std::vector<server_kv_token> chunk(tokens.begin() + offset, tokens.begin() + offset + count);
        const server_kv_block_id id = allocate_block(parent, std::move(chunk), count == block_size_, now_us);
        seq.blocks.push_back(id);
        parent = id;
        offset += count;
    }

    seq.reserved_blocks -= from_reserve;
    reserved_blocks_ -= from_reserve;
    seq.prompt_tokens += tokens.size();
    return true;
}

server_kv_write_result server_kv_block_manager::make_tail_writable(
        server_kv_sequence_id sequence_id,
        uint64_t now_us) {
    server_kv_write_result result;
    auto seq_it = sequences_.find(sequence_id);
    if (seq_it == sequences_.end() || seq_it->second.blocks.empty()) {
        result.reason = "sequence has no KV tail";
        return result;
    }
    auto & seq = seq_it->second;
    const server_kv_block_id old_id = seq.blocks.back();
    const auto old_it = blocks_.find(old_id);
    if (old_it->second.ref_count == 1) {
        old_it->second.last_access_us = now_us;
        result.writable = true;
        result.block_id = old_id;
        return result;
    }

    const bool consume_reserve = seq.reserved_blocks > 0;
    if (!consume_reserve && free_unreserved_blocks() == 0) {
        result.reason = "insufficient KV block capacity for copy-on-write";
        return result;
    }
    // Copy metadata before insertion: unordered_map growth may rehash and
    // invalidate references/iterators into blocks_.
    const server_kv_block_id parent = old_it->second.parent;
    const std::vector<server_kv_token> copied_tokens = old_it->second.tokens;
    const server_kv_block_id new_id = allocate_block(
            parent, copied_tokens, false, now_us);
    blocks_.at(old_id).ref_count--;
    seq.blocks.back() = new_id;
    if (consume_reserve) {
        seq.reserved_blocks--;
        reserved_blocks_--;
    }
    result.writable = true;
    result.copied = true;
    result.block_id = new_id;
    copy_on_write_total_++;
    return result;
}

size_t server_kv_block_manager::longest_prefix_blocks(
        const std::vector<server_kv_token> & prompt) const {
    size_t matched = 0;
    size_t offset = 0;
    server_kv_block_id parent = 0;
    while (offset + block_size_ <= prompt.size()) {
        std::vector<server_kv_token> chunk(
                prompt.begin() + offset, prompt.begin() + offset + block_size_);
        const server_kv_block_id found = find_prefix_block(parent, chunk);
        if (found == 0) {
            break;
        }
        matched++;
        parent = found;
        offset += block_size_;
    }
    return matched;
}

bool server_kv_block_manager::release(server_kv_sequence_id sequence_id) {
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        return false;
    }
    for (server_kv_block_id id : it->second.blocks) {
        release_block(id);
    }
    reserved_blocks_ -= it->second.reserved_blocks;
    sequences_.erase(it);
    return true;
}

server_kv_block_snapshot server_kv_block_manager::snapshot() const {
    server_kv_block_snapshot result;
    result.block_size = block_size_;
    result.capacity_blocks = capacity_blocks_;
    result.allocated_blocks = blocks_.size();
    result.reserved_blocks = reserved_blocks_;
    result.copy_on_write_total = copy_on_write_total_;
    for (const auto & item : blocks_) {
        const auto & value = item.second;
        if (value.ref_count > 1) {
            result.shared_blocks++;
        }
        result.blocks.push_back({
            value.id, value.parent, value.tokens.size(), value.ref_count,
            value.prefix_indexed, value.last_access_us,
        });
    }
    for (const auto & item : sequences_) {
        const auto & value = item.second;
        result.sequences.push_back({
            value.id, value.blocks, value.prompt_tokens, value.reserved_blocks,
        });
    }
    std::sort(result.blocks.begin(), result.blocks.end(),
            [](const auto & left, const auto & right) { return left.id < right.id; });
    std::sort(result.sequences.begin(), result.sequences.end(),
            [](const auto & left, const auto & right) { return left.id < right.id; });
    return result;
}

std::string server_kv_block_manager::validate() const {
    if (blocks_.size() + reserved_blocks_ > capacity_blocks_) {
        return "allocated plus reserved blocks exceed capacity";
    }
    size_t sequence_reservations = 0;
    std::unordered_map<server_kv_block_id, size_t> references;
    for (const auto & item : sequences_) {
        const auto & seq = item.second;
        sequence_reservations += seq.reserved_blocks;
        for (server_kv_block_id id : seq.blocks) {
            if (blocks_.count(id) == 0) {
                return "sequence references missing block";
            }
            references[id]++;
        }
    }
    if (sequence_reservations != reserved_blocks_) {
        return "reservation accounting mismatch";
    }
    for (const auto & item : blocks_) {
        const auto & value = item.second;
        if (value.tokens.empty() || value.tokens.size() > block_size_) {
            return "invalid block token count";
        }
        if (references[value.id] != value.ref_count) {
            return "block reference count mismatch";
        }
        if (value.prefix_indexed && value.parent != 0 &&
                !blocks_.at(value.parent).prefix_indexed) {
            return "prefix index contains a block with a private parent";
        }
        if (value.parent != 0 && blocks_.count(value.parent) == 0) {
            return "block references missing parent";
        }
    }
    return {};
}
