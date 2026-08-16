#include "server-inference-iteration.h"

bool server_inference_iteration::prepare() {
    if (phase_ != server_iteration_phase::created) return false;
    phase_ = server_iteration_phase::prepared;
    return true;
}

bool server_inference_iteration::plan(server_iteration_plan_summary value) {
    if (phase_ != server_iteration_phase::prepared) return false;
    plan_ = value;
    phase_ = server_iteration_phase::planned;
    return true;
}

bool server_inference_iteration::begin_execute() {
    if (phase_ != server_iteration_phase::planned) return false;
    phase_ = server_iteration_phase::executing;
    return true;
}

bool server_inference_iteration::record_executed(size_t tokens) {
    if (phase_ != server_iteration_phase::executing ||
            tokens > plan_.physical_batch_tokens - executed_tokens_) return false;
    executed_tokens_ += tokens;
    return true;
}

bool server_inference_iteration::begin_commit() {
    if (phase_ != server_iteration_phase::executing ||
            executed_tokens_ != plan_.physical_batch_tokens) return false;
    phase_ = server_iteration_phase::committing;
    return true;
}

bool server_inference_iteration::commit() {
    if (phase_ != server_iteration_phase::committing) return false;
    phase_ = server_iteration_phase::committed;
    return true;
}

void server_inference_iteration::abort(std::string reason) {
    if (phase_ == server_iteration_phase::committed) return;
    phase_ = server_iteration_phase::aborted;
    error_ = std::move(reason);
}

std::string server_inference_iteration::validate() const {
    if (executed_tokens_ > plan_.physical_batch_tokens) return "execution exceeded immutable physical batch";
    if ((phase_ == server_iteration_phase::committing || phase_ == server_iteration_phase::committed) &&
            executed_tokens_ != plan_.physical_batch_tokens) {
        return "iteration committed a partial physical batch";
    }
    if (phase_ == server_iteration_phase::aborted && error_.empty()) return "aborted iteration has no reason";
    return {};
}
