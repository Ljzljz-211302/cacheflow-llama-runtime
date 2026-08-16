#pragma once

#include "server-benefit-policy.h"
#include "server-inference-iteration.h"
#include "server-inference-scheduler.h"
#include "server-kv-capacity-planner.h"
#include "server-kv-action-policy.h"
#include "server-kv-runtime.h"
#include "server-kv-swap-store.h"
#include "server-runtime-adapter.h"
#include "server-speculation-controller.h"

#include <memory>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

enum slot_state : uint8_t {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER,
    SLOT_STATE_STARTED,
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

class server_sequence_state {
public:
    slot_state phase() const { return phase_; }

private:
    friend class server_inference_engine;
    slot_state phase_ = SLOT_STATE_IDLE;
};

struct server_engine_snapshot {
    server_iteration_phase iteration_phase = server_iteration_phase::created;
    server_iteration_plan_summary iteration_plan;
    server_scheduler_state scheduler;
    bool block_runtime_enabled = false;
    bool swap_store_enabled = false;
    server_benefit_snapshot cpu_benefit;
    server_benefit_snapshot cuda_benefit;
    server_benefit_persistence_snapshot benefit_persistence;
    server_kv_action_policy_snapshot kv_action_policy;
};

// Owns the inference control plane. server_context remains the protocol and
// llama object adapter; scheduling, iteration transactions, KV lifecycle,
// speculation policy, and the runtime seam live here.
class server_inference_engine {
public:
    server_inference_engine();
    ~server_inference_engine();

    server_inference_iteration & begin_prepared_iteration();
    bool publish_plan(server_inference_iteration & iteration, server_iteration_plan_summary plan);

    // Owns the production iteration order. The callbacks adapt llama-server's
    // protocol/model objects, but cannot reorder the transaction phases.
    template <typename Prepare, typename PlanExecute>
    bool step(Prepare && prepare, PlanExecute && plan_execute) {
        if (!measure("prepare", std::forward<Prepare>(prepare))) return false;
        auto & iteration = begin_prepared_iteration();
        try {
            std::forward<PlanExecute>(plan_execute)(iteration);
        } catch (...) {
            iteration.abort("iteration callback raised an exception");
            flush_trace();
            throw;
        }
        flush_trace();
        return !iteration.is_aborted();
    }

    template <typename Execute>
    void execute(server_inference_iteration & iteration, Execute && execute_batch) {
        if (&iteration != &iteration_ || !iteration.begin_execute()) {
            throw std::logic_error("invalid inference-engine execute transition");
        }
        measure("execute", std::forward<Execute>(execute_batch));
    }

    bool commit(server_inference_iteration & iteration);

    template <typename BeforeCommit>
    bool commit_with(server_inference_iteration & iteration, BeforeCommit && before_commit) {
        return measure("commit", [&]() {
            std::forward<BeforeCommit>(before_commit)();
            if (iteration.is_aborted()) return false;
            return commit(iteration);
        });
    }
    bool transition(server_sequence_state & sequence, slot_state next);
    void configure_trace(std::filesystem::path output);
    void flush_trace() const;

    template <typename Operation>
    auto measure(const char * name, Operation && operation) -> decltype(operation()) {
        const uint64_t started = clock_us();
        if constexpr (std::is_void_v<decltype(operation())>) {
            try {
                std::forward<Operation>(operation)();
                record_trace(name, started, clock_us());
            } catch (...) {
                record_trace(name, started, clock_us());
                throw;
            }
        } else {
            try {
                auto result = std::forward<Operation>(operation)();
                record_trace(name, started, clock_us());
                return result;
            } catch (...) {
                record_trace(name, started, clock_us());
                throw;
            }
        }
    }
    server_engine_snapshot snapshot() const;

    server_inference_scheduler & scheduler() { return scheduler_; }
    server_benefit_policy & benefit_policy() { return benefit_policy_; }
    server_kv_action_policy & kv_action_policy() { return kv_action_policy_; }
    server_kv_capacity_planner & capacity_planner() { return capacity_planner_; }
    server_speculation_controller & speculation() { return speculation_; }
    server_runtime_adapter & runtime() { return *runtime_; }

    server_kv_runtime * kv_runtime() { return kv_runtime_.get(); }
    const server_kv_runtime * kv_runtime() const { return kv_runtime_.get(); }
    void set_kv_runtime(std::unique_ptr<server_kv_runtime> value) { kv_runtime_ = std::move(value); }

    server_kv_swap_store * swap_store() { return swap_store_.get(); }
    const server_kv_swap_store * swap_store() const { return swap_store_.get(); }
    void set_swap_store(std::unique_ptr<server_kv_swap_store> value) { swap_store_ = std::move(value); }

    // Deterministic tests and production use exactly this ownership seam.
    void set_runtime(std::unique_ptr<server_runtime_adapter> value);

private:
    server_inference_scheduler scheduler_;
    server_benefit_policy benefit_policy_;
    server_kv_action_policy kv_action_policy_;
    server_kv_capacity_planner capacity_planner_;
    server_speculation_controller speculation_;
    std::unique_ptr<server_kv_runtime> kv_runtime_;
    std::unique_ptr<server_kv_swap_store> swap_store_;
    std::unique_ptr<server_runtime_adapter> runtime_;
    server_inference_iteration iteration_;
    struct trace_event {
        std::string name;
        uint64_t started_us = 0;
        uint64_t duration_us = 0;
    };
    std::filesystem::path trace_output_;
    std::vector<trace_event> trace_events_;
    uint64_t plan_started_us_ = 0;

    static uint64_t clock_us();
    void record_trace(const char * name, uint64_t started, uint64_t finished);
};
