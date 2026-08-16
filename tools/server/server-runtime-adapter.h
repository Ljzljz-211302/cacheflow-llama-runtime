#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct server_runtime_decode_result {
    int code = 0;
    bool committed = false;
    uint64_t elapsed_us = 0;
};

// Narrow production/test seam around state-changing llama runtime calls.
class server_runtime_adapter {
public:
    virtual ~server_runtime_adapter() = default;
    virtual server_runtime_decode_result decode(llama_context * context, llama_batch batch) = 0;
};

struct server_deterministic_runtime_config {
    std::vector<int> decode_results;
    uint64_t fixed_delay_us = 0;
};

class server_deterministic_runtime final : public server_runtime_adapter {
public:
    explicit server_deterministic_runtime(server_deterministic_runtime_config config);
    server_runtime_decode_result decode(llama_context * context, llama_batch batch) override;
    size_t calls() const { return calls_; }

private:
    server_deterministic_runtime_config config_;
    size_t calls_ = 0;
};

std::unique_ptr<server_runtime_adapter> server_create_llama_runtime();
