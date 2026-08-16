#include "server-runtime-adapter.h"

#include "ggml.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <utility>

namespace {

class server_llama_runtime final : public server_runtime_adapter {
public:
    server_llama_runtime() {
        if (const char * value = std::getenv("CACHEFLOW_TEST_DECODE_FAILURES")) {
            failures_remaining_ = std::max(0, std::atoi(value));
        }
        if (const char * value = std::getenv("CACHEFLOW_TEST_DECODE_FAILURE_CODE")) {
            failure_code_ = std::atoi(value);
        }
        if (const char * value = std::getenv("CACHEFLOW_TEST_POST_SUCCESS_FAILURES")) {
            post_success_failures_remaining_ = std::max(0, std::atoi(value));
        }
        if (const char * value = std::getenv("CACHEFLOW_TEST_POST_SUCCESS_FAILURE_AFTER")) {
            post_success_failure_after_ = std::max(0, std::atoi(value));
        }
    }

    server_runtime_decode_result decode(llama_context * context, llama_batch batch) override {
        const int64_t started = ggml_time_us();
        int code = 0;
        if (failures_remaining_ > 0) {
            --failures_remaining_;
            code = failure_code_;
        } else {
            code = llama_decode(context, batch);
            if (code == 0 && post_success_failures_remaining_ > 0) {
                if (successful_decodes_++ >= post_success_failure_after_) {
                    --post_success_failures_remaining_;
                    LOG_WRN("CacheFlow test injected a late CUDA failure after successful llama_decode\n");
                    code = failure_code_;
                }
            }
        }
        return { code, code == 0, (uint64_t) (ggml_time_us() - started) };
    }

private:
    int failures_remaining_ = 0;
    int failure_code_ = 1;
    int post_success_failures_remaining_ = 0;
    int post_success_failure_after_ = 0;
    int successful_decodes_ = 0;
};

} // namespace

server_deterministic_runtime::server_deterministic_runtime(
        server_deterministic_runtime_config config) : config_(std::move(config)) {}

server_runtime_decode_result server_deterministic_runtime::decode(
        llama_context *, llama_batch) {
    const auto started = std::chrono::steady_clock::now();
    if (config_.fixed_delay_us > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(config_.fixed_delay_us));
    }
    const int code = calls_ < config_.decode_results.size()
            ? config_.decode_results[calls_]
            : 0;
    ++calls_;
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count();
    return { code, code == 0, (uint64_t) elapsed };
}

std::unique_ptr<server_runtime_adapter> server_create_llama_runtime() {
    return std::make_unique<server_llama_runtime>();
}
