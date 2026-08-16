#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

enum class server_iteration_phase : uint8_t {
    created,
    prepared,
    planned,
    executing,
    committing,
    committed,
    aborted,
};

struct server_iteration_plan_summary {
    size_t decode_tokens = 0;
    size_t prefill_tokens = 0;
    size_t physical_batch_tokens = 0;
};

// Enforces the prepare -> immutable plan -> execute -> commit transaction
// boundary independently from llama-server's request and HTTP state.
class server_inference_iteration {
public:
    bool prepare();
    bool plan(server_iteration_plan_summary value);
    bool begin_execute();
    bool record_executed(size_t tokens);
    bool begin_commit();
    bool commit();
    void abort(std::string reason);

    server_iteration_phase phase() const { return phase_; }
    const server_iteration_plan_summary & plan_summary() const { return plan_; }
    size_t executed_tokens() const { return executed_tokens_; }
    const std::string & error() const { return error_; }
    bool is_aborted() const { return phase_ == server_iteration_phase::aborted; }
    std::string validate() const;

private:
    server_iteration_phase phase_ = server_iteration_phase::created;
    server_iteration_plan_summary plan_;
    size_t executed_tokens_ = 0;
    std::string error_;
};
