#ifdef NDEBUG
#undef NDEBUG
#endif

#include "server-inference-scheduler.h"

#include <cassert>
#include <vector>

static void test_cost_aware_slot_selection() {
    server_inference_scheduler lcp({0.1f, 0.0f, 0});
    server_inference_scheduler cost_aware({0.1f, 0.5f, 0});
    const std::vector<server_slot_candidate> candidates = {
        {0, 80, 1000, 20},
        {1, 60,   70, 10},
    };
    assert(lcp.select_slot(candidates, 100).id == 0);
    const auto selected = cost_aware.select_slot(candidates, 100);
    assert(selected.id == 1);
    assert(selected.reused_tokens == 60);
    assert(selected.evicted_tokens == 10);
}

static void test_threshold_negative_benefit_and_lru_tie_break() {
    server_inference_scheduler scheduler({0.1f, 1.0f, 0});
    assert(!scheduler.select_slot({{0, 10, 10, 1}}, 100).found());
    assert(!scheduler.select_slot({{0, 20, 1000, 1}}, 100).found());

    scheduler.configure({0.1f, 0.5f, 0});
    assert(scheduler.select_slot({{0, 50, 60, 20}, {1, 50, 60, 10}}, 100).id == 1);
}

static void test_chunked_prefill_shares_an_iteration_budget() {
    server_inference_scheduler scheduler({0.1f, 0.5f, 32});
    const auto plan = scheduler.plan_prefill({{0, 100}, {1, 100}, {2, 5}}, 64);
    assert(plan.tokens_scheduled == 64);
    assert(plan.quota(0) == 32);
    assert(plan.quota(1) == 32);
    assert(plan.quota(2) == 0);

    // The rotating cursor prevents slot 2 from starving in the next iteration.
    const auto next = scheduler.plan_prefill({{0, 68}, {1, 68}, {2, 5}}, 32);
    assert(next.quota(2) == 5);
    assert(next.quota(0) == 27);
}

static void test_zero_chunk_preserves_greedy_upstream_behavior() {
    server_inference_scheduler scheduler({0.1f, 0.0f, 0});
    const auto plan = scheduler.plan_prefill({{0, 10}, {1, 100}}, 64);
    assert(plan.quota(0) == 10);
    assert(plan.quota(1) == 54);

    // Unlimited mode remains deterministic instead of rotating the first slot.
    const auto next = scheduler.plan_prefill({{0, 100}, {1, 100}}, 64);
    assert(next.quota(0) == 64);
    assert(next.quota(1) == 0);
}

static void test_shadow_upstream_plan_is_pure_and_matches_greedy() {
    server_inference_scheduler scheduler({0.1f, 0.5f, 32});
    const std::vector<server_prefill_candidate> candidates = {{2, 10}, {0, 100}, {1, 20}};
    const auto shadow = server_inference_scheduler::plan_upstream_prefill(candidates, 64);
    assert(shadow.effective_chunk_size == 0);
    assert(shadow.quota(0) == 64);
    assert(shadow.quota(1) == 0);
    assert(shadow.quota(2) == 0);

    // Shadow planning must not advance the CacheFlow round-robin cursor.
    const auto first = scheduler.plan_prefill(candidates, 32);
    assert(first.quota(0) == 32);

    server_prefill_plan reordered;
    reordered.tokens_scheduled = 30;
    reordered.allocations = {{2, 10}, {1, 20}};
    server_prefill_plan same_quotas;
    same_quotas.tokens_scheduled = 30;
    same_quotas.allocations = {{1, 20}, {2, 10}};
    assert(reordered.equivalent_allocations(same_quotas));
    same_quotas.allocations[0].tokens = 19;
    assert(!reordered.equivalent_allocations(same_quotas));
}

static void test_adaptive_chunk_responds_to_real_iteration_feedback() {
    server_scheduler_config config;
    config.adaptive_prefill = true;
    config.prefill_chunk_size = 128;
    config.prefill_chunk_min = 16;
    config.prefill_chunk_max = 256;
    config.target_iteration_ms = 20.0;
    server_inference_scheduler scheduler(config);

    assert(scheduler.plan_prefill({{0, 1000}}, 512).quota(0) == 128);
    scheduler.observe_iteration({1, 128, 50.0});
    assert(scheduler.state().effective_chunk_size == 64);
    scheduler.observe_iteration({1, 64, 40.0});
    assert(scheduler.state().effective_chunk_size == 32);
    scheduler.observe_iteration({1, 32, 10.0});
    assert(scheduler.state().effective_chunk_size >= 16);
    assert(scheduler.state().effective_chunk_size <= 256);
    assert(scheduler.state().cost.observations == 3);
    scheduler.observe_iteration({0, 48, 100.0});
    assert(scheduler.state().effective_chunk_size == 256);
    assert(scheduler.state().observations == 4);
}

static void test_adaptive_cpu_safety_fallback_selects_greedy_action() {
    server_scheduler_config config;
    config.adaptive_prefill = true;
    config.prefill_chunk_size = 128;
    config.adaptive_greedy_fallback = true;
    server_inference_scheduler scheduler(config);

    const auto first = scheduler.plan_prefill({{0, 1000}, {1, 1000}}, 512);
    assert(first.effective_chunk_size == 0);
    assert(first.quota(0) == 512);
    scheduler.observe_iteration({1, 512, 100.0, 1});
    assert(scheduler.state().effective_chunk_size == 0);
    assert(scheduler.state().cost.observations == 1);
}

int main() {
    test_cost_aware_slot_selection();
    test_threshold_negative_benefit_and_lru_tie_break();
    test_chunked_prefill_shares_an_iteration_budget();
    test_zero_chunk_preserves_greedy_upstream_behavior();
    test_shadow_upstream_plan_is_pure_and_matches_greedy();
    test_adaptive_chunk_responds_to_real_iteration_feedback();
    test_adaptive_cpu_safety_fallback_selects_greedy_action();
    return 0;
}
