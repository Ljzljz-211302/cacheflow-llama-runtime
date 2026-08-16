#include "server-kv-runtime.h"

#include <algorithm>
#include <sstream>
#include <unordered_set>

server_kv_runtime::server_kv_runtime(size_t block_size, size_t capacity_blocks) :
        block_size_(block_size), blocks_(block_size, capacity_blocks) {}

size_t server_kv_runtime::common_prefix(
        const std::vector<server_kv_token> & left,
        const std::vector<server_kv_token> & right) {
    const size_t count = std::min(left.size(), right.size());
    size_t index = 0;
    while (index < count && left[index] == right[index]) index++;
    return index;
}

bool server_kv_runtime::synchronize(
        server_kv_sequence_id sequence_id,
        const std::vector<server_kv_token> & committed_tokens,
        size_t reserve_tokens,
        uint64_t now_us,
        std::string * error) {
    const auto current = sequences_.find(sequence_id);
    if (current != sequences_.end() &&
            current->second.residency == server_kv_residency::resident &&
            current->second.committed_tokens == committed_tokens &&
            current->second.reserved_tokens == reserve_tokens) {
        current->second.last_access_us = now_us;
        return true;
    }

    if (committed_tokens.empty()) {
        release(sequence_id);
        return true;
    }

    // Mutate a complete candidate copy and commit only after admission. This
    // preserves the old Block Table when a larger synchronization cannot fit.
    server_kv_block_manager candidate = blocks_;
    const bool extends_current = current != sequences_.end() &&
            current->second.residency == server_kv_residency::resident &&
            current->second.reserved_tokens == reserve_tokens &&
            reserve_tokens == 0 &&
            committed_tokens.size() > current->second.committed_tokens.size() &&
            std::equal(current->second.committed_tokens.begin(),
                    current->second.committed_tokens.end(), committed_tokens.begin());
    if (extends_current) {
        const std::vector<server_kv_token> appended(
                committed_tokens.begin() + current->second.committed_tokens.size(),
                committed_tokens.end());
        if (!candidate.append(sequence_id, appended, now_us, error)) return false;
    } else {
        candidate.release(sequence_id);
        const auto attached = candidate.attach(
                sequence_id, committed_tokens, reserve_tokens, now_us);
        if (!attached.admitted) {
            if (error) *error = attached.reason;
            return false;
        }
    }
    const std::string invalid = candidate.validate();
    if (!invalid.empty()) {
        if (error) *error = invalid;
        return false;
    }
    blocks_ = std::move(candidate);
    sequences_[sequence_id] = {
        sequence_id,
        server_kv_residency::resident,
        committed_tokens,
        reserve_tokens,
        now_us,
    };
    return true;
}

server_kv_prefix_share_plan server_kv_runtime::plan_prefix_share(
        server_kv_sequence_id destination,
        const std::vector<server_kv_token> & prompt,
        const std::vector<server_kv_sequence_id> & eligible_donors) const {
    std::unordered_set<server_kv_sequence_id> eligible(
            eligible_donors.begin(), eligible_donors.end());
    server_kv_prefix_share_plan best;
    best.destination = destination;
    for (const auto & item : sequences_) {
        const auto & candidate = item.second;
        if (candidate.id == destination || candidate.residency != server_kv_residency::resident) continue;
        if (!eligible.empty() && eligible.count(candidate.id) == 0) continue;
        const size_t matched = common_prefix(candidate.committed_tokens, prompt);
        if (matched > best.matched_tokens ||
                (matched == best.matched_tokens && matched > 0 &&
                 candidate.last_access_us < sequences_.at(best.donor).last_access_us)) {
            best.donor = candidate.id;
            best.matched_tokens = matched;
            best.matched_blocks = (matched + block_size_ - 1) / block_size_;
        }
    }
    return best;
}

bool server_kv_runtime::preempt(server_kv_sequence_id sequence_id, uint64_t now_us) {
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end() || it->second.residency != server_kv_residency::resident) return false;
    if (!blocks_.release(sequence_id)) return false;
    it->second.residency = server_kv_residency::swapped;
    it->second.last_access_us = now_us;
    return true;
}

bool server_kv_runtime::restore(
        server_kv_sequence_id sequence_id,
        uint64_t now_us,
        std::string * error) {
    const auto it = sequences_.find(sequence_id);
    if (it == sequences_.end() || it->second.residency != server_kv_residency::swapped) {
        if (error) *error = "sequence is not swapped";
        return false;
    }
    const auto tokens = it->second.committed_tokens;
    const size_t reserve = it->second.reserved_tokens;
    const auto attached = blocks_.attach(sequence_id, tokens, reserve, now_us);
    if (!attached.admitted) {
        if (error) *error = attached.reason;
        return false;
    }
    it->second.residency = server_kv_residency::resident;
    it->second.last_access_us = now_us;
    return true;
}

bool server_kv_runtime::release(server_kv_sequence_id sequence_id) {
    const auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) return false;
    if (it->second.residency == server_kv_residency::resident) {
        blocks_.release(sequence_id);
    }
    sequences_.erase(it);
    return true;
}

bool server_kv_runtime::is_swapped(server_kv_sequence_id sequence_id) const {
    const auto it = sequences_.find(sequence_id);
    return it != sequences_.end() && it->second.residency == server_kv_residency::swapped;
}

server_kv_runtime_snapshot server_kv_runtime::snapshot() const {
    server_kv_runtime_snapshot result;
    result.blocks = blocks_.snapshot();
    for (const auto & item : sequences_) result.sequences.push_back(item.second);
    std::sort(result.sequences.begin(), result.sequences.end(),
            [](const auto & left, const auto & right) { return left.id < right.id; });
    return result;
}

std::string server_kv_runtime::validate() const {
    const std::string block_error = blocks_.validate();
    if (!block_error.empty()) return block_error;
    const auto block_snapshot = blocks_.snapshot();
    std::unordered_set<server_kv_sequence_id> resident_block_tables;
    for (const auto & sequence : block_snapshot.sequences) {
        resident_block_tables.insert(sequence.id);
    }
    for (const auto & item : sequences_) {
        const auto & sequence = item.second;
        const bool has_table = resident_block_tables.count(sequence.id) != 0;
        if ((sequence.residency == server_kv_residency::resident) != has_table) {
            return "runtime residency and Block Table disagree";
        }
        if (sequence.committed_tokens.empty()) {
            return "runtime retains an empty sequence";
        }
    }
    if (resident_block_tables.size() > sequences_.size()) {
        return "Block Table has no runtime sequence";
    }
    return {};
}
