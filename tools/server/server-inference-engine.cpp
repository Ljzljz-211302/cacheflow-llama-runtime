#include "server-inference-engine.h"

#include <stdexcept>
#include <chrono>
#include <fstream>

server_inference_engine::server_inference_engine() :
        runtime_(server_create_llama_runtime()) {
}

server_inference_engine::~server_inference_engine() {
    try { benefit_policy_.flush_checkpoint(); } catch (...) {}
    try { flush_trace(); } catch (...) {}
}

uint64_t server_inference_engine::clock_us() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

void server_inference_engine::record_trace(
        const char * name, uint64_t started, uint64_t finished) {
    if (trace_output_.empty()) return;
    trace_events_.push_back({name, started, finished - started});
}

void server_inference_engine::configure_trace(std::filesystem::path output) {
    trace_output_ = std::move(output);
    trace_events_.clear();
}

void server_inference_engine::flush_trace() const {
    if (trace_output_.empty()) return;
    if (trace_output_.has_parent_path()) {
        std::filesystem::create_directories(trace_output_.parent_path());
    }
    std::ofstream output(trace_output_, std::ios::trunc);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output << "{\"traceEvents\":[";
    for (size_t i = 0; i < trace_events_.size(); ++i) {
        const auto & event = trace_events_[i];
        if (i) output << ',';
        output << "{\"name\":\"" << event.name
               << "\",\"cat\":\"inference-engine\",\"ph\":\"X\",\"pid\":1,\"tid\":1,\"ts\":"
               << event.started_us << ",\"dur\":" << event.duration_us << '}';
    }
    output << "]}";
}

server_inference_iteration & server_inference_engine::begin_prepared_iteration() {
    iteration_ = server_inference_iteration{};
    if (!iteration_.prepare()) throw std::logic_error("failed to prepare inference iteration");
    plan_started_us_ = clock_us();
    return iteration_;
}

bool server_inference_engine::publish_plan(
        server_inference_iteration & iteration,
        server_iteration_plan_summary plan) {
    if (&iteration != &iteration_ || !iteration.plan(plan)) return false;
    const uint64_t finished = clock_us();
    record_trace("plan", plan_started_us_, finished);
    return true;
}

bool server_inference_engine::commit(server_inference_iteration & iteration) {
    return &iteration == &iteration_ && iteration.begin_commit() && iteration.commit();
}

bool server_inference_engine::transition(server_sequence_state & sequence, slot_state next) {
    const slot_state current = sequence.phase_;
    const bool valid =
            (current == SLOT_STATE_IDLE && (next == SLOT_STATE_STARTED || next == SLOT_STATE_WAIT_OTHER)) ||
            (current == SLOT_STATE_WAIT_OTHER && next == SLOT_STATE_DONE_PROMPT) ||
            (current == SLOT_STATE_STARTED && next == SLOT_STATE_PROCESSING_PROMPT) ||
            (current == SLOT_STATE_PROCESSING_PROMPT && next == SLOT_STATE_DONE_PROMPT) ||
            (current == SLOT_STATE_DONE_PROMPT && next == SLOT_STATE_GENERATING) ||
            (current != SLOT_STATE_IDLE && next == SLOT_STATE_IDLE);
    if (!valid) return false;
    sequence.phase_ = next;
    return true;
}

server_engine_snapshot server_inference_engine::snapshot() const {
    return {
        iteration_.phase(),
        iteration_.plan_summary(),
        scheduler_.state(),
        kv_runtime_ != nullptr,
        swap_store_ != nullptr,
        benefit_policy_.snapshot(server_benefit_backend::cpu),
        benefit_policy_.snapshot(server_benefit_backend::cuda),
        benefit_policy_.checkpoint_snapshot(),
        kv_action_policy_.snapshot(),
    };
}

void server_inference_engine::set_runtime(std::unique_ptr<server_runtime_adapter> value) {
    if (!value) throw std::invalid_argument("inference engine runtime cannot be null");
    runtime_ = std::move(value);
}
