#ifdef NDEBUG
#undef NDEBUG
#endif

#include "server-kv-capacity-planner.h"

#include <cassert>

static void test_no_pressure_keeps_all_caches() {
    server_kv_capacity_planner planner;
    const auto plan = planner.plan(1000, 900, 1000000, {{0, 200, 1}});
    assert(plan.fits());
    assert(plan.victim_ids.empty());
}

static void test_old_cache_is_evicted_before_recent_cache() {
    server_kv_capacity_planner planner(1000);
    const auto plan = planner.plan(1000, 1100, 10000000, {
        {0, 150, 9900000},
        {1, 150, 1000000},
    });
    assert(plan.fits());
    assert(plan.victim_ids.size() == 1);
    assert(plan.victim_ids[0] == 1);
}

static void test_reclaims_multiple_slots_and_reports_unresolved_pressure() {
    server_kv_capacity_planner planner;
    const auto enough = planner.plan(1000, 1300, 1000000, {
        {0, 100, 0}, {1, 250, 0},
    });
    assert(enough.fits());
    assert(enough.reclaimed_tokens >= 300);

    const auto insufficient = planner.plan(1000, 1500, 1000000, {{0, 100, 0}});
    assert(!insufficient.fits());
}

int main() {
    test_no_pressure_keeps_all_caches();
    test_old_cache_is_evicted_before_recent_cache();
    test_reclaims_multiple_slots_and_reports_unresolved_pressure();
    return 0;
}
