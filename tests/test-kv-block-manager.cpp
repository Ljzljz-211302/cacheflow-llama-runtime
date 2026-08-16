#ifdef NDEBUG
#undef NDEBUG
#endif

#include "server-kv-block-manager.h"

#include <cassert>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

static std::vector<server_kv_token> tokens(int begin, int count) {
    std::vector<server_kv_token> result;
    for (int i = 0; i < count; ++i) result.push_back(begin + i);
    return result;
}

static void test_prefix_blocks_and_partial_tail_are_shared() {
    server_kv_block_manager manager(4, 16);
    const auto first = manager.attach(1, tokens(0, 10), 0, 10);
    const auto second = manager.attach(2, tokens(0, 10), 0, 20);
    assert(first.admitted && second.admitted);
    assert(first.allocated_blocks == 3);
    assert(second.shared_blocks == 3);
    assert(second.allocated_blocks == 0);
    assert(second.matched_tokens == 10);
    const auto snapshot = manager.snapshot();
    assert(snapshot.allocated_blocks == 3);
    assert(snapshot.shared_blocks == 3);
    assert(manager.validate().empty());
}

static void test_copy_on_write_clones_shared_tail() {
    server_kv_block_manager manager(4, 8);
    assert(manager.attach(1, tokens(0, 8), 4, 10).admitted);
    assert(manager.attach(2, tokens(0, 8), 4, 20).admitted);
    const auto before = manager.snapshot();
    assert(before.allocated_blocks == 2);
    const auto writable = manager.make_tail_writable(2, 30);
    assert(writable.writable && writable.copied);
    const auto after = manager.snapshot();
    assert(after.allocated_blocks == 3);
    assert(after.copy_on_write_total == 1);
    assert(after.sequences[0].blocks.back() != after.sequences[1].blocks.back());
    assert(manager.validate().empty());
}

static void test_append_automatically_cows_shared_partial_tail() {
    server_kv_block_manager manager(4, 8);
    assert(manager.attach(1, tokens(0, 6), 0, 10).admitted);
    assert(manager.attach(2, tokens(0, 6), 0, 20).admitted);
    std::string error;
    assert(manager.append(2, tokens(6, 1), 30, &error));
    const auto after = manager.snapshot();
    assert(after.copy_on_write_total == 1);
    assert(after.sequences[0].blocks.back() != after.sequences[1].blocks.back());
    assert(manager.validate().empty());
}

static void test_reservation_enforces_admission_and_is_consumed_by_append() {
    server_kv_block_manager manager(4, 4);
    const auto admitted = manager.attach(1, tokens(0, 4), 8, 1);
    assert(admitted.admitted);
    assert(admitted.reserved_blocks == 2);
    assert(!manager.attach(2, tokens(10, 8), 0, 2).admitted);
    std::string error;
    assert(manager.append(1, tokens(100, 8), 3, &error));
    const auto snapshot = manager.snapshot();
    assert(snapshot.allocated_blocks == 3);
    assert(snapshot.reserved_blocks == 0);
    assert(manager.validate().empty());
}

static void test_release_preserves_shared_prefix_until_last_owner() {
    server_kv_block_manager manager(4, 8);
    assert(manager.attach(1, tokens(0, 8), 0, 1).admitted);
    assert(manager.attach(2, tokens(0, 8), 0, 2).admitted);
    assert(manager.release(1));
    assert(manager.snapshot().allocated_blocks == 2);
    assert(manager.longest_prefix_blocks(tokens(0, 8)) == 2);
    assert(manager.release(2));
    assert(manager.snapshot().allocated_blocks == 0);
    assert(manager.longest_prefix_blocks(tokens(0, 8)) == 0);
    assert(manager.validate().empty());
}

static void test_failed_attach_is_atomic() {
    server_kv_block_manager manager(4, 2);
    assert(manager.attach(1, tokens(0, 8), 0, 1).admitted);
    const auto before = manager.snapshot();
    const auto failed = manager.attach(2, tokens(20, 4), 0, 2);
    assert(!failed.admitted);
    const auto after = manager.snapshot();
    assert(before.allocated_blocks == after.allocated_blocks);
    assert(before.sequences.size() == after.sequences.size());
    assert(manager.validate().empty());
}

static void test_cow_descendants_never_enter_root_prefix_index() {
    server_kv_block_manager manager(4, 16);
    assert(manager.attach(1, tokens(0, 8), 8, 1).admitted);
    assert(manager.attach(2, tokens(0, 8), 8, 2).admitted);
    assert(manager.make_tail_writable(2, 3).copied);
    std::string error;
    assert(manager.append(2, tokens(8, 8), 4, &error));

    const auto snapshot = manager.snapshot();
    for (const auto & sequence : snapshot.sequences) {
        if (sequence.id != 2) continue;
        assert(sequence.blocks.size() == 4);
        assert(!snapshot.blocks[sequence.blocks[1] - 1].prefix_indexed);
        assert(!snapshot.blocks[sequence.blocks[2] - 1].prefix_indexed);
        assert(!snapshot.blocks[sequence.blocks[3] - 1].prefix_indexed);
    }
    // Only the original two-block canonical chain remains reusable.
    assert(manager.longest_prefix_blocks(tokens(0, 16)) == 2);
    assert(manager.validate().empty());
}

static void test_randomized_operations_preserve_capacity_and_references() {
    server_kv_block_manager manager(4, 32);
    std::mt19937 rng(20260730);
    std::vector<server_kv_sequence_id> active;
    server_kv_sequence_id next_id = 1;

    for (uint64_t step = 1; step <= 2000; ++step) {
        const int operation = (int) (rng() % 4);
        if (operation == 0 || active.empty()) {
            const int length = 1 + (int) (rng() % 20);
            auto prompt = tokens((rng() % 3 == 0) ? 0 : (int) (rng() % 1000), length);
            const auto result = manager.attach(next_id, prompt, rng() % 9, step);
            if (result.admitted) active.push_back(next_id);
            next_id++;
        } else {
            const size_t index = rng() % active.size();
            const auto id = active[index];
            if (operation == 1) {
                std::string error;
                manager.append(id, tokens((int) (rng() % 1000), 1 + rng() % 7), step, &error);
            } else if (operation == 2) {
                manager.make_tail_writable(id, step);
            } else {
                assert(manager.release(id));
                active.erase(active.begin() + index);
            }
        }
        assert(manager.validate().empty());
        const auto snapshot = manager.snapshot();
        assert(snapshot.allocated_blocks + snapshot.reserved_blocks <= snapshot.capacity_blocks);
    }

    for (auto id : active) assert(manager.release(id));
    assert(manager.snapshot().allocated_blocks == 0);
    assert(manager.validate().empty());
}

int main() {
    test_prefix_blocks_and_partial_tail_are_shared();
    test_copy_on_write_clones_shared_tail();
    test_append_automatically_cows_shared_partial_tail();
    test_reservation_enforces_admission_and_is_consumed_by_append();
    test_release_preserves_shared_prefix_until_last_owner();
    test_failed_attach_is_atomic();
    test_cow_descendants_never_enter_root_prefix_index();
    test_randomized_operations_preserve_capacity_and_references();
    return 0;
}
