#ifdef NDEBUG
#undef NDEBUG
#endif

#include "server-speculation-controller.h"

#include <cassert>

static server_speculation_config adaptive_config() {
    server_speculation_config config;
    config.adaptive = true;
    config.min_draft_tokens = 1;
    config.max_draft_tokens = 8;
    config.acceptance_alpha = 0.5;
    config.cooldown_iterations = 3;
    config.warmup_observations = 2;
    config.min_evidence_tokens = 8;
    return config;
}

static void test_high_acceptance_grows_and_pressure_shrinks() {
    server_speculation_controller controller(adaptive_config());
    assert(controller.recommend(1, 6, 0.2) == 6);
    controller.observe({1, 6, 6, 0.2});
    assert(controller.state(1).draft_tokens == 8);
    assert(controller.recommend(1, 8, 0.9) == 4);
    controller.observe({1, 4, 4, 0.9});
    assert(controller.state(1).draft_tokens == 4);
}

static void test_low_acceptance_disables_then_recovers() {
    server_speculation_controller controller(adaptive_config());
    controller.observe({2, 8, 0, 0.2});
    controller.observe({2, 4, 0, 0.2});
    assert(controller.state(2).disabled);
    assert(controller.disabled_low_acceptance_total() == 1);
    assert(controller.recommend(2, 8, 0.2) == 0);
    assert(controller.recommend(2, 8, 0.2) == 0);
    assert(controller.recommend(2, 8, 0.2) == 0);
    assert(controller.recommend(2, 8, 0.2) == 1);
}

static void test_sequences_are_independent_and_fixed_mode_is_compatible() {
    server_speculation_controller adaptive(adaptive_config());
    adaptive.observe({1, 8, 0, 0.0});
    assert(adaptive.state(1).draft_tokens == 8);
    adaptive.observe({1, 8, 0, 0.0});
    assert(adaptive.state(1).draft_tokens == 1);
    assert(adaptive.state(1).disabled);
    assert(adaptive.state(2).draft_tokens == 8);

    server_speculation_controller fixed;
    assert(fixed.recommend(1, 5, 1.0) == 5);
}

int main() {
    test_high_acceptance_grows_and_pressure_shrinks();
    test_low_acceptance_disables_then_recovers();
    test_sequences_are_independent_and_fixed_mode_is_compatible();
    return 0;
}
