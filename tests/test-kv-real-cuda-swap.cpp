#include "ggml-backend.h"
#include "llama.h"

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", message);
        std::exit(1);
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }
    ggml_backend_load_all();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    require(model != nullptr, "model load");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string prompt =
            "CacheFlow validates physical KV swap with a real transformer prompt. "
            "The restored sequence must preserve every key, value, and cell position.";
    const int token_count = -llama_tokenize(
            vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
    require(token_count > 0, "prompt token count");
    std::vector<llama_token> tokens(token_count);
    require(llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(),
            tokens.size(), true, true) == token_count, "prompt tokenization");

    auto context_params = llama_context_default_params();
    context_params.n_ctx = 512;
    context_params.n_batch = 256;
    context_params.n_seq_max = 2;
    context_params.kv_unified = false;
    llama_context * context = llama_init_from_model(model, context_params);
    require(context != nullptr, "context initialization");
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    require(llama_decode(context, batch) == 0, "prompt decode");
    llama_synchronize(context);
    llama_memory_t memory = llama_get_memory(context);
    require(llama_memory_seq_pos_max(memory, 0) >= 0, "decoded sequence residency");
    require(llama_memory_seq_can_swap_cuda(memory, 0), "CUDA swap capability");

    const size_t state_size = llama_state_seq_get_size_ext(
            context, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
    std::vector<uint8_t> before(state_size);
    std::vector<uint8_t> after(state_size);
    require(state_size > 20, "non-empty serialized KV state");
    require(llama_state_seq_get_data_ext(context, before.data(), before.size(), 0,
            LLAMA_STATE_SEQ_FLAGS_NONE) == before.size(), "state snapshot before swap");
    const llama_pos max_position = llama_memory_seq_pos_max(memory, 0);
    const auto out_start = std::chrono::steady_clock::now();
    require(llama_memory_seq_swap_out_cuda(memory, 0), "CUDA swap-out enqueue");
    require(llama_memory_seq_rm(memory, 0, -1, -1), "sequence remove after swap-out");
    require(llama_memory_seq_pos_max(memory, 0) == -1, "sequence eviction");
    const auto restore_start = std::chrono::steady_clock::now();
    require(llama_memory_seq_swap_in_cuda(memory, 0), "CUDA swap-in");
    const auto restore_stop = std::chrono::steady_clock::now();
    require(llama_memory_seq_pos_max(memory, 0) == max_position, "restored cell positions");
    require(llama_state_seq_get_data_ext(context, after.data(), after.size(), 0,
            LLAMA_STATE_SEQ_FLAGS_NONE) == after.size(), "state snapshot after restore");
    require(before == after, "byte-identical restored KV state");
    llama_cacheflow_cuda_stats stats{};
    require(llama_memory_cacheflow_cuda_stats(memory, &stats), "CUDA stats capability");
    require(stats.copy_bytes >= 2 * 319488, "physical swap bytes counted");
    require(stats.swap_bytes == stats.copy_bytes, "swap-only byte accounting");
    require(stats.swap_out_microseconds > 0 && stats.swap_in_microseconds > 0,
            "CUDA event timings counted");
    require(stats.events_waited >= 2 && stats.pinned_bytes_current == 0 &&
            stats.pinned_bytes_peak >= 319488 && stats.backend_errors == 0,
            "CUDA event, pinned-memory, and error metrics");
    const double enqueue_ms = std::chrono::duration<double, std::milli>(
            restore_start - out_start).count();
    const double restore_ms = std::chrono::duration<double, std::milli>(
            restore_stop - restore_start).count();
    std::printf("real_cuda_swap,state_bytes=%zu,swap_out_enqueue_ms=%.3f,restore_ms=%.3f,"
            "cuda_swap_out_us=%llu,cuda_swap_in_us=%llu,pinned_peak=%llu\n",
            state_size, enqueue_ms, restore_ms,
            (unsigned long long) stats.swap_out_microseconds,
            (unsigned long long) stats.swap_in_microseconds,
            (unsigned long long) stats.pinned_bytes_peak);
    llama_free(context);
    llama_model_free(model);
    return 0;
}
