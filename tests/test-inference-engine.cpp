#ifdef NDEBUG
#undef NDEBUG
#endif

#include "server-inference-engine.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>

int main() {
    server_inference_engine engine;
    const auto trace = std::filesystem::temp_directory_path() / "cacheflow-engine-trace-test.json";
    std::filesystem::remove(trace);
    engine.configure_trace(trace);
    engine.scheduler().configure({0.1f, 0.25f, 64, true, 16, 256, 25.0});
    server_benefit_config benefit_config;
    benefit_config.mode = server_benefit_mode::always_cacheflow;
    engine.benefit_policy().configure(benefit_config);
    assert(engine.benefit_policy().choose({}).action == server_benefit_action::cacheflow);
    server_kv_action_capabilities action_capabilities;
    action_capabilities.recompute = true;
    const auto action = engine.kv_action_policy().choose({}, action_capabilities);
    assert(action.action == server_kv_action::recompute);
    engine.set_kv_runtime(std::make_unique<server_kv_runtime>(16, 64));
    engine.set_swap_store(server_kv_create_host_swap_store(1 << 20));
    engine.set_runtime(std::make_unique<server_deterministic_runtime>(
            server_deterministic_runtime_config{{0}, 0}));

    auto & step = engine.begin_prepared_iteration();
    assert(engine.publish_plan(step, {1, 7, 8}));
    llama_batch batch{};
    engine.execute(step, [&]() {
        const auto decoded = engine.runtime().decode(nullptr, batch);
        assert(decoded.code == 0 && decoded.committed);
        assert(step.record_executed(8));
    });
    assert(engine.commit(step));
    engine.flush_trace();
    std::ifstream trace_input(trace);
    const std::string trace_text((std::istreambuf_iterator<char>(trace_input)), {});
    assert(trace_text.find("\"name\":\"plan\"") != std::string::npos);
    assert(trace_text.find("\"name\":\"execute\"") != std::string::npos);
    trace_input.close();
    std::filesystem::remove(trace);

    const auto snapshot = engine.snapshot();
    assert(snapshot.iteration_phase == server_iteration_phase::committed);
    assert(snapshot.iteration_plan.decode_tokens == 1);
    assert(snapshot.iteration_plan.prefill_tokens == 7);
    assert(snapshot.block_runtime_enabled && snapshot.swap_store_enabled);
    assert(snapshot.cpu_benefit.cacheflow_decisions == 1);
    assert(snapshot.kv_action_policy.decisions == 1);

    bool prepared = false;
    bool executed = false;
    const bool step_committed = engine.step(
            [&]() {
                prepared = true;
                return true;
            },
            [&](server_inference_iteration & iteration) {
                assert(prepared);
                assert(engine.publish_plan(iteration, {0, 1, 1}));
                engine.execute(iteration, [&]() {
                    executed = true;
                    assert(iteration.record_executed(1));
                });
                assert(engine.commit_with(iteration, []() {}));
            });
    assert(step_committed && executed);
    assert(engine.snapshot().iteration_phase == server_iteration_phase::committed);

    const bool step_aborted = engine.step(
            []() { return true; },
            [&](server_inference_iteration & iteration) {
                assert(engine.publish_plan(iteration, {1, 0, 1}));
                iteration.abort("injected step failure");
            });
    assert(!step_aborted);
    assert(engine.snapshot().iteration_phase == server_iteration_phase::aborted);

    server_sequence_state sequence;
    assert(engine.transition(sequence, SLOT_STATE_STARTED));
    assert(engine.transition(sequence, SLOT_STATE_PROCESSING_PROMPT));
    assert(engine.transition(sequence, SLOT_STATE_DONE_PROMPT));
    assert(engine.transition(sequence, SLOT_STATE_GENERATING));
    assert(!engine.transition(sequence, SLOT_STATE_STARTED));
    assert(engine.transition(sequence, SLOT_STATE_IDLE));

    // Starting another iteration discards no committed state, but creates a
    // fresh transaction boundary owned by the engine.
    auto & next = engine.begin_prepared_iteration();
    assert(next.phase() == server_iteration_phase::prepared);
    assert(engine.snapshot().iteration_phase == server_iteration_phase::prepared);
    return 0;
}
