#ifdef NDEBUG
#undef NDEBUG
#endif

#include "server-kv-runtime.h"

#include <algorithm>
#include <cassert>
#include <random>
#include <string>
#include <vector>

static std::vector<server_kv_token> tokens(int begin, int count) {
    std::vector<server_kv_token> result;
    for (int i = 0; i < count; ++i) result.push_back(begin + i);
    return result;
}

static void test_prefix_plan_includes_shared_partial_tail() {
    server_kv_runtime runtime(4, 16);
    assert(runtime.synchronize(1, tokens(0, 10), 0, 10));
    auto plan = runtime.plan_prefix_share(2, tokens(0, 9));
    assert(plan.donor == 1 && plan.matched_tokens == 9 && plan.matched_blocks == 3);
    assert(runtime.preempt(1, 20));
    assert(runtime.is_swapped(1));
    assert(!runtime.plan_prefix_share(2, tokens(0, 9)).found());
    assert(runtime.restore(1, 30));
    assert(!runtime.is_swapped(1));
    assert(runtime.plan_prefix_share(2, tokens(0, 9)).matched_tokens == 9);
    assert(runtime.validate().empty());
}

static void test_failed_growth_preserves_old_table_atomically() {
    server_kv_runtime runtime(4, 3);
    assert(runtime.synchronize(1, tokens(0, 4), 4, 1));
    const auto before = runtime.snapshot();
    std::string error;
    assert(!runtime.synchronize(1, tokens(0, 12), 4, 2, &error));
    const auto after = runtime.snapshot();
    assert(!error.empty());
    assert(after.sequences[0].committed_tokens == before.sequences[0].committed_tokens);
    assert(after.blocks.allocated_blocks == before.blocks.allocated_blocks);
    assert(after.blocks.reserved_blocks == before.blocks.reserved_blocks);
    assert(runtime.validate().empty());
}

static void test_incremental_growth_cows_shared_physical_tail() {
    server_kv_runtime runtime(4, 16);
    assert(runtime.synchronize(1, tokens(0, 6), 0, 1));
    assert(runtime.synchronize(2, tokens(0, 6), 0, 2));
    assert(runtime.snapshot().blocks.shared_blocks == 2);
    assert(runtime.synchronize(2, tokens(0, 7), 0, 3));
    const auto after = runtime.snapshot();
    assert(after.blocks.copy_on_write_total == 1);
    assert(after.blocks.sequences[0].blocks.back() != after.blocks.sequences[1].blocks.back());
    assert(runtime.validate().empty());
}

static void test_eligible_donor_filter_and_release() {
    server_kv_runtime runtime(4, 16);
    assert(runtime.synchronize(1, tokens(0, 8), 0, 1));
    assert(runtime.synchronize(2, tokens(0, 12), 0, 2));
    assert(runtime.plan_prefix_share(3, tokens(0, 12), {1}).donor == 1);
    assert(runtime.plan_prefix_share(3, tokens(0, 12), {2}).donor == 2);
    assert(runtime.release(2));
    assert(runtime.plan_prefix_share(3, tokens(0, 12)).matched_tokens == 8);
}

static void test_randomized_residency_transitions_preserve_invariants() {
    server_kv_runtime runtime(4, 40);
    std::mt19937 rng(20260730);
    for (uint64_t step = 1; step <= 1000; ++step) {
        const int id = 1 + rng() % 12;
        const auto snapshot = runtime.snapshot();
        auto found = std::find_if(snapshot.sequences.begin(), snapshot.sequences.end(),
                [id](const auto & value) { return value.id == id; });
        if (found == snapshot.sequences.end()) {
            runtime.synchronize(id, tokens((rng() % 3) * 100, 1 + rng() % 16), rng() % 5, step);
        } else if (found->residency == server_kv_residency::resident && rng() % 3 == 0) {
            runtime.preempt(id, step);
        } else if (found->residency == server_kv_residency::swapped && rng() % 2 == 0) {
            std::string error;
            runtime.restore(id, step, &error);
        } else if (rng() % 4 == 0) {
            runtime.release(id);
        } else if (found->residency == server_kv_residency::resident) {
            runtime.synchronize(id, tokens((rng() % 3) * 100, 1 + rng() % 20), rng() % 5, step);
        }
        assert(runtime.validate().empty());
    }
}

int main() {
    test_prefix_plan_includes_shared_partial_tail();
    test_failed_growth_preserves_old_table_atomically();
    test_incremental_growth_cows_shared_physical_tail();
    test_eligible_donor_filter_and_release();
    test_randomized_residency_transitions_preserve_invariants();
    return 0;
}
