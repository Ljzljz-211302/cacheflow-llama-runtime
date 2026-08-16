#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-inference-scheduler.h"
#include "server-inference-iteration.h"
#include "server-inference-engine.h"
#include "server-kv-capacity-planner.h"
#include "server-kv-runtime.h"
#include "server-kv-swap-store.h"
#include "server-speculation-controller.h"
#include "server-runtime-adapter.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include "ggml-cpp.h"
#include "ggml-backend.h"

// TODO: tmp until the mtmd draft processing is refactored [TAG_MTMD_DRAFT_PROCESSING]
#include "../../src/llama-ext.h"

#include <algorithm>
#include <cstddef>
#include <cinttypes>
#include <exception>
#include <memory>
#include <filesystem>
#include <sstream>
#include <utility>
#include <fstream>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

constexpr int HTTP_POLLING_SECONDS = 1;

static std::pair<uint64_t, uint64_t> server_cuda_paged_fattn_stats() {
    using get_stats_fn = void (*)(uint64_t *, uint64_t *);
    for (size_t index = 0; index < ggml_backend_reg_count(); ++index) {
        auto * function = reinterpret_cast<get_stats_fn>(ggml_backend_reg_get_proc_address(
                ggml_backend_reg_get(index), "ggml_backend_cuda_get_paged_fattn_stats"));
        if (function != nullptr) {
            uint64_t dispatches = 0;
            uint64_t sequences = 0;
            function(&dispatches, &sequences);
            return { dispatches, sequences };
        }
    }
    return { 0, 0 };
}

static uint32_t server_n_outputs_max(const common_params & params) {
    const uint32_t n_batch  = params.n_batch;

    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return n_batch;
    }

    const uint32_t n_outputs_per_seq = 1 + common_speculative_n_max(&params.speculative);

    const uint64_t n_outputs = (uint64_t) params.n_parallel * n_outputs_per_seq;

    return std::max<uint32_t>(1, std::min<uint64_t>(n_batch, n_outputs));
}

enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
};

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;
    std::array<llama_context *, 2> mtgt = {nullptr, nullptr}; // [0] for main context, [1] for optional draft context

    // speculative decoding
    common_speculative * spec;

    llama_tokens spec_draft;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_keep      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;

    int32_t n_prompt_tokens_cache     = 0;
    int32_t n_prompt_tokens_processed = 0;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    server_sequence_state lifecycle;

    server_prompt prompt;
    llama_tokens cuda_swapped_tokens;
    llama_tokens store_swapped_tokens;
    server_kv_swap_handle store_swap_handle = 0;
    size_t kv_action_bytes = 0;
    server_kv_action_decision pending_kv_action;
    server_kv_action pending_kv_action_effective = server_kv_action::recompute;
    int64_t pending_kv_action_started_us = 0;
    bool has_pending_kv_action = false;
    // Decode routing lasts for the complete generation. The policy observation
    // itself is one-shot, so it cannot also serve as persistent route state.
    bool paged_decode_requested = false;

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        GGML_ASSERT(prompt.data.size() == 0);

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_WRN(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return false;
        }

        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }

        return true;
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear(bool allow_processing) {
        if (!allow_processing) {
            GGML_ASSERT(!is_processing());
        }

        SLT_INF(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        common_context_seq_rm(ctx_tgt, id, -1, -1);
        if (ctx_dft) {
            common_context_seq_rm(ctx_dft, id, -1, -1);
        }

        prompt.tokens.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // stats
    size_t n_sent_text = 0; // number of sent text character

    int64_t t_print_last = 0;
    int64_t t_start_process_prompt;
    int64_t t_start_generation;

    double t_prompt_processing = 0.0; // ms
    double t_token_generation = 0.0;  // ms

    std::function<void(int /* id_slot */)> callback_on_release;

    // Speculative decoding stats
    int32_t n_draft_total = 0;      // Total draft tokens generated
    int32_t n_draft_accepted = 0;   // Draft tokens actually accepted

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        n_prompt_tokens_cache = 0;
        has_pending_kv_action = false;
        paged_decode_requested = false;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        // clear speculative decoding stats
        n_draft_total = 0;
        n_draft_accepted = 0;

        task_prev = std::move(task);
        task.reset();

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();
        mtgt[0] = ctx_tgt;
        mtgt[1] = nullptr;
        if (ctx_dft && llama_get_ctx_other(ctx_dft) != ctx_tgt) {
            // TODO: in the future, figure out how to infuse target embeddings to the images
            //       for now, we re-decode the same chunk in both ctx_tgt and ctx_dft
            //       maybe we simply need to call `common_speculative_process()` ?
            //       [TAG_MTMD_DRAFT_PROCESSING]
            mtgt[1] = ctx_dft;
        }
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd() || (spec && common_speculative_need_embd(spec));
    }

    bool need_embd_nextn() const {
        GGML_ASSERT(task);
        return spec && common_speculative_need_embd_nextn(spec);
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    // (MTP supports splitting — uses task->need_embd() not need_embd())
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx_tgt) && llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type && are_lora_equal(lora, other_slot.lora);
    }

    bool has_budget(const common_params & global_params) {
        GGML_ASSERT(task);

        if (task->params.n_predict == -1 && global_params.n_predict == -1) {
            return true; // limitless
        }

        n_remaining = -1;

        if (task->params.n_predict != -1) {
            n_remaining = task->params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool is_processing() const {
        return lifecycle.phase() != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    void update_batch(llama_batch & batch) {
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.n_tokens;

            common_batch_add(batch, sampled, prompt.tokens.pos_next(), { this->id }, true);

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.n_tokens);
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.n_tokens + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            common_batch_add(batch, sampled, pos0++, { this->id }, true);
            for (auto token : spec_draft) {
                common_batch_add(batch, token, pos0++, { this->id }, true);
            }
        }

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used        =  ggml_time_us();
            t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear(false);
            }

            reset();

            callback_on_release(id);
        }
    }

    result_timings get_timings() const {
        result_timings timings;
        timings.cache_n = n_prompt_tokens_cache;

        timings.prompt_n            = n_prompt_tokens_processed;
        timings.prompt_ms           = t_prompt_processing;
        timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
        timings.prompt_per_second   = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        timings.predicted_n            = n_decoded;
        timings.predicted_ms           = t_token_generation;
        timings.predicted_per_token_ms = t_token_generation / n_decoded;
        timings.predicted_per_second   = 1e3 / t_token_generation * n_decoded;

        // Add speculative metrics
        if (n_draft_total > 0) {
            timings.draft_n          = n_draft_total;
            timings.draft_n_accepted = n_draft_accepted;
        }

        return timings;
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (n_decoded < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        t_print_last = t_now;

        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this, "n_decoded = %6d, tg = %6.2f t/s\n", n_decoded, n_gen_second);
    }

    void print_timings_pp() const {
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;
        const double f_progress = (float) prompt.n_tokens() / task->n_tokens();

        if (t_prompt_processing < 3000.0) {
            return;
        }

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                n_prompt_tokens_processed, f_progress, t_prompt_processing / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt        =       t_prompt_processing / n_prompt_tokens_processed;
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        const double t_gen        =       t_token_generation / n_decoded;
        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_processing, n_prompt_tokens_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_token_generation, n_decoded, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_processing + t_token_generation, n_prompt_tokens_processed + n_decoded);

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        if (n_draft_total > 0) {
            const float draft_ratio = (float) n_draft_accepted / n_draft_total;
            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated)\n",
                    draft_ratio, n_draft_accepted, n_draft_total);
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = n_prompt_tokens_processed;
            res["n_prompt_tokens_cache"]     = n_prompt_tokens_cache;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = {
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining},
                    {"n_decoded",      n_decoded},
                }
            };

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(lifecycle.phase() == SLOT_STATE_DONE_PROMPT);

        common_context_seq_rm(ctx_tgt, other.id,     -1, -1);
        common_context_seq_cp(ctx_tgt, id, other.id, -1, -1);

        if (ctx_dft) {
            common_context_seq_rm(ctx_dft, other.id,     -1, -1);
            common_context_seq_cp(ctx_dft, id, other.id, -1, -1);
        }

        other.n_decoded   = n_decoded;
        other.n_remaining = n_remaining;
        other.i_batch     = i_batch;

        other.t_start_process_prompt    = t_start_process_prompt;
        other.t_prompt_processing       = t_prompt_processing;
        other.n_prompt_tokens_cache     = n_prompt_tokens_cache;
        other.n_prompt_tokens_processed = n_prompt_tokens_processed;

        other.prompt = prompt.clone();
        other.init_sampler();
    }

    // returns 0 on success
    // caller need to update prompt.tokens after a successful call to keep track of the processing progress
    int process_mtmd_chunk(size_t idx, size_t & n_tokens_out) {
        GGML_ASSERT(mctx);
        const auto & input_tokens = task->tokens;
        auto & chunk = input_tokens.find_chunk(idx);
        int32_t res = 0;

        auto try_decode = [&]() -> int32_t {
            if (mbatch) {
                float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
                if (embd) {
                    for (auto * lctx : mtgt) {
                        if (lctx == nullptr) {
                            continue;
                        }
                        llama_pos new_n_past; // unused for now
                        res = mtmd_helper_decode_image_chunk(
                            mctx,
                            lctx,
                            chunk.get(),
                            embd,
                            prompt.tokens.pos_next(),
                            id,
                            llama_n_batch(lctx),
                            &new_n_past
                        );
                        if (res != 0) {
                            SLT_ERR(*this, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                            return -1;
                        }
                    }
                    n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                    return 0; // success
                }
            }
            return 1; // (non-error) need to create & encode batch
        };

        // if the batch is already exist, try searching & encode
        res = try_decode();
        if (res == 0) {
            return 0;
        } else if (res < 0) {
            // fatal error
            return res;
        }

        // otherwise, the batch is either uninitialized or is used up
        // we need to create & encode a new batch
        mbatch.reset(mtmd_batch_init(mctx));
        res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
        GGML_ASSERT(res == 0); // we should never have an empty batch

        // try batching as much as possible
        int n_added = 1;
        size_t idx_cur = idx;
        while (res == 0) {
            auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
            if (next_chunk == nullptr) {
                break;
            }
            res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
            n_added += (res == 0 ? 1 : 0);
            idx_cur = next_idx;
            SLT_DBG(*this, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
            // if res != 0, batch is full or chunk is not compatible -> this loop breaks
        }

        // TODO @ngxson : move this log line to debug when it become more stable
        SLT_INF(*this, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

        res = mtmd_batch_encode(mbatch.get());
        if (res != 0) {
            SLT_ERR(*this, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
            return -1;
        }

        return try_decode();
    }
};



//
// server_metrics
//

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t n_prompt_tokens_cached_total    = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t n_prompt_tokens_cached    = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    uint64_t n_slot_cache_selections_total = 0;
    uint64_t n_slot_lru_selections_total   = 0;
    uint64_t n_slot_reused_tokens_total    = 0;
    uint64_t n_slot_evicted_tokens_total   = 0;
    uint64_t n_scheduler_iterations_total     = 0;
    uint64_t n_decode_tokens_scheduled_total  = 0;
    uint64_t n_prefill_tokens_scheduled_total = 0;
    uint64_t n_prefill_chunks_scheduled_total = 0;
    uint64_t batch_tokens = 0;
    uint64_t batch_sequences = 0;
    double prefill_starvation_ms = 0.0;
    uint64_t n_kv_admission_pressure_total = 0;
    uint64_t n_kv_proactive_evictions_total = 0;
    uint64_t n_kv_proactive_reclaimed_tokens_total = 0;
    uint64_t n_kv_admission_failures_total = 0;
    uint64_t n_requests_preempted_total = 0;
    uint64_t n_requests_swapped_total = 0;
    uint64_t n_requests_restored_total = 0;
    uint64_t n_requests_recomputed_total = 0;
    uint64_t n_draft_tokens_total = 0;
    uint64_t n_draft_tokens_accepted_total = 0;
    double t_speculation_draft_total_ms = 0.0;
    server_latency_histogram time_to_first_token;
    server_latency_histogram time_per_output_token;
    server_latency_histogram request_latency;
    server_latency_histogram request_queue;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const server_slot & slot) {
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed       += slot.n_prompt_tokens_processed;
        n_prompt_tokens_cached_total    += slot.n_prompt_tokens_cache;
        n_prompt_tokens_cached          += slot.n_prompt_tokens_cache;
        t_prompt_processing             += slot.t_prompt_processing;
        t_prompt_processing_total       += slot.t_prompt_processing;

        n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
    }

    void on_prediction(const server_slot & slot) {
        n_tokens_predicted_total   += slot.n_decoded;
        n_tokens_predicted         += slot.n_decoded;
        t_tokens_generation        += slot.t_token_generation;
        t_tokens_generation_total  += slot.t_token_generation;
    }

    void on_decoded(const std::vector<server_slot> & slots) {
        n_decode_total++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
            n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }
    }

    void on_slot_selected(bool cache_aware, size_t reused_tokens, size_t evicted_tokens) {
        if (cache_aware) {
            n_slot_cache_selections_total++;
            n_slot_reused_tokens_total  += reused_tokens;
            n_slot_evicted_tokens_total += evicted_tokens;
        } else {
            n_slot_lru_selections_total++;
        }
    }

    void on_scheduler_plan(
            size_t decode_tokens,
            size_t decode_sequences,
            const server_prefill_plan & plan,
            double max_prefill_wait_ms) {
        n_scheduler_iterations_total++;
        n_decode_tokens_scheduled_total += decode_tokens;
        n_prefill_tokens_scheduled_total += plan.tokens_scheduled;
        n_prefill_chunks_scheduled_total += plan.allocations.size();
        batch_tokens = decode_tokens + plan.tokens_scheduled;
        batch_sequences = decode_sequences + plan.allocations.size();
        prefill_starvation_ms = max_prefill_wait_ms;
    }

    void on_kv_capacity_plan(const server_kv_capacity_plan & plan) {
        if (plan.deficit_tokens == 0) {
            return;
        }
        n_kv_admission_pressure_total++;
        n_kv_proactive_evictions_total += plan.victim_ids.size();
        n_kv_proactive_reclaimed_tokens_total += plan.reclaimed_tokens;
        n_kv_admission_failures_total += plan.fits() ? 0 : 1;
    }

    void on_speculation(size_t drafted, size_t accepted) {
        n_draft_tokens_total += drafted;
        n_draft_tokens_accepted_total += accepted;
    }

    void on_request_finished(const server_slot & slot) {
        if (!slot.task || slot.task->t_queued_us <= 0 || slot.t_start_process_prompt <= 0) return;
        const int64_t now = ggml_time_us();
        request_queue.observe(std::max<int64_t>(0,
                slot.t_start_process_prompt - slot.task->t_queued_us) / 1.e6);
        request_latency.observe(std::max<int64_t>(0,
                now - slot.task->t_queued_us) / 1.e6);
        if (slot.n_decoded > 0 && slot.t_start_generation > 0) {
            time_to_first_token.observe(std::max<int64_t>(0,
                    slot.t_start_generation - slot.task->t_queued_us) / 1.e6);
            time_per_output_token.observe((slot.t_token_generation / 1.e3) /
                    std::max(1, slot.n_decoded - 1));
        }
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        n_prompt_tokens_cached    = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};


//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    llama_batch batch {};

    llama_model_ptr model_dft;
    llama_context_ptr ctx_dft;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;
    int slots_debug = 0;
    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    json json_ui_settings = json::object();    // Primary: new name
    json json_webui_settings = json::object();    // Deprecated: use json_ui_settings instead (kept for compat)

    server_inference_engine inference_engine;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    void destroy() {
        spec.reset();
        ctx_dft.reset();
        model_dft.reset();

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;

        llama_batch_free(batch);
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        bool is_resume = sleeping;

        SRV_INF("loading model '%s'\n", params.model.path.c_str());

        params_base = params;
        params_base.n_outputs_max = server_n_outputs_max(params_base);
        if (!params_base.engine_trace_path.empty()) {
            inference_engine.configure_trace(params_base.engine_trace_path);
            SRV_INF("inference engine trace enabled: %s\n", params_base.engine_trace_path.c_str());
        }

        std::string & mmproj_path = params_base.mmproj.path;
        bool has_mmproj = !mmproj_path.empty();
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
            mparams.media_marker     = get_media_marker();
        }

        // optionally get the memory usage of mmproj
        if (has_mmproj && params_base.fit_params) {
            auto mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_INF("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB\n", total / (1024.0 * 1024.0));
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // optionally reserve VRAM for the draft / MTP context before fitting the target model
        if (params_base.fit_params) {
            const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                            params_base.speculative.types.end(),
                                            COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
            const bool has_draft = params_base.speculative.has_dft();

            if (has_draft || spec_mtp) {
                common_params params_dft = params_base;
                bool measure_model_bytes = true;

                if (has_draft) {
                    const auto & params_spec = params_base.speculative.draft;
                    params_dft.devices               = params_spec.devices;
                    params_dft.model                 = params_spec.mparams;
                    params_dft.n_gpu_layers          = params_spec.n_gpu_layers;
                    params_dft.cache_type_k          = params_spec.cache_type_k;
                    params_dft.cache_type_v          = params_spec.cache_type_v;
                    params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;
                } else {
                    // MTP draft context lives on the target model, only context+compute are new
                    measure_model_bytes = false;
                }

                params_dft.n_outputs_max = params_base.n_parallel;

                auto mparams_dft = common_model_params_to_llama(params_dft);
                auto cparams_dft = common_context_params_to_llama(params_dft);
                if (spec_mtp) {
                    cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                    cparams_dft.type_k   = params_base.speculative.draft.cache_type_k;
                    cparams_dft.type_v   = params_base.speculative.draft.cache_type_v;
                }
                cparams_dft.n_rs_seq = 0;

                std::vector<ggml_backend_dev_t> devs;
                uint32_t hp_ngl = 0;
                uint32_t hp_nct = 0;
                uint32_t hp_nex = 0;
                try {
                    auto dmd = common_get_device_memory_data(
                        params_dft.model.path.c_str(), &mparams_dft, &cparams_dft,
                        devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);

                    GGML_ASSERT(!params_base.fit_params_target.empty());
                    size_t total = 0;

                    std::vector<ggml_backend_dev_t> tgt_devices = params.devices;

                    if (tgt_devices.empty()) {
                        for(size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                           tgt_devices.push_back(ggml_backend_dev_get(i));
                        }
                    }

                    for (size_t j = 0; j < devs.size(); ++j) {
                        const size_t bytes = (measure_model_bytes ? dmd[j].model : 0) + dmd[j].context + dmd[j].compute;
                        total += bytes;
                        for (size_t i = 0; i < tgt_devices.size(); i++) {
                            if (tgt_devices[i] == devs[j]) {
                                SRV_DBG("[spec] adding %.2f MiB to fit_params_target for device %s\n",
                                        bytes / (1024.0 * 1024.0), ggml_backend_dev_name(devs[j]));
                                params_base.fit_params_target[i] += bytes;
                                break;
                            }
                        }
                    }
                    SRV_INF("[spec] estimated memory usage of %s is %.2f MiB\n",
                            has_draft ? "draft model" : "MTP context",
                            total / (1024.0 * 1024.0));
                } catch (const std::exception & e) {
                    SRV_WRN("[spec] failed to measure %s memory: %s\n",
                            has_draft ? "draft model" : "MTP context", e.what());
                }
            }
        }

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (params_base.speculative.has_dft()) {
            // TODO speculative: move to common/speculative.cpp?
            const auto & params_spec = params_base.speculative.draft;

            SRV_INF("loading draft model '%s'\n", params_spec.mparams.path.c_str());

            auto params_dft = params_base;

            params_dft.devices      = params_spec.devices;
            params_dft.model        = params_spec.mparams;
            params_dft.n_gpu_layers = params_spec.n_gpu_layers;
            params_dft.cache_type_k = params_spec.cache_type_k;
            params_dft.cache_type_v = params_spec.cache_type_v;

            if (params_spec.cpuparams.n_threads > 0) {
                params_dft.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
                params_dft.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
            }

            params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;

            auto mparams_dft = common_model_params_to_llama(params_dft);

            model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
            if (model_dft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                return false;
            }

            auto cparams = common_context_params_to_llama(params_dft);

            const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                            params_base.speculative.types.end(),
                                            COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();

            if (spec_mtp) {
                cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            }

            // note: for small models maybe we can set this to the maximum possible draft from all speculative types
            //       the extra memory for small models is likely negligible?
            cparams.n_rs_seq  = 0;
            cparams.ctx_other = ctx_tgt;

            ctx_dft.reset(llama_init_from_model(model_dft.get(), cparams));

            params_base.speculative.draft.ctx_tgt = ctx_tgt;
            params_base.speculative.draft.ctx_dft = ctx_dft.get();
        } else if (std::find(params_base.speculative.types.begin(), params_base.speculative.types.end(),
                             COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end()) {
            SRV_INF("creating MTP draft context against the target model '%s'\n",
                    params_base.model.path.c_str());

            auto cparams_mtp = common_context_params_to_llama(params_base);
            cparams_mtp.ctx_type      = LLAMA_CONTEXT_TYPE_MTP;
            cparams_mtp.type_k        = params_base.speculative.draft.cache_type_k;
            cparams_mtp.type_v        = params_base.speculative.draft.cache_type_v;
            cparams_mtp.n_rs_seq      = 0;
            cparams_mtp.n_outputs_max = params_base.n_parallel;
            cparams_mtp.ctx_other     = ctx_tgt;

            ctx_dft.reset(llama_init_from_model(model_tgt, cparams_mtp));
            if (ctx_dft == nullptr) {
                SRV_ERR("%s", "failed to create MTP context\n");
                return false;
            }

            params_base.speculative.draft.ctx_tgt = ctx_tgt;
            params_base.speculative.draft.ctx_dft = ctx_dft.get();
        }

        if (has_mmproj) {
            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        inference_engine.scheduler().configure({
            params_base.slot_prompt_similarity,
            params_base.slot_cache_eviction_penalty,
            (size_t) params_base.prefill_chunk_size,
            params_base.prefill_chunk_adaptive,
            (size_t) params_base.prefill_chunk_min,
            (size_t) params_base.prefill_chunk_max,
            params_base.prefill_target_iteration_ms,
            params_base.n_gpu_layers <= 0 && params_base.benefit_policy == "upstream",
        });
        server_benefit_mode benefit_mode = server_benefit_mode::upstream;
        if (params_base.benefit_policy == "always") benefit_mode = server_benefit_mode::always_cacheflow;
        else if (params_base.benefit_policy == "rule") benefit_mode = server_benefit_mode::rule;
        else if (params_base.benefit_policy == "learned") benefit_mode = server_benefit_mode::learned;
        server_benefit_config benefit_config;
        benefit_config.mode = benefit_mode;
        benefit_config.minimum_observations = (size_t) params_base.benefit_min_observations;
        benefit_config.exploration_interval = (size_t) params_base.benefit_exploration_interval;
        benefit_config.confidence_beta = params_base.benefit_confidence_beta;
        benefit_config.safety_margin_ms = params_base.benefit_safety_margin_ms;
        benefit_config.drift_ratio = params_base.benefit_drift_ratio;
        benefit_config.drift_consecutive_limit = (size_t) params_base.benefit_drift_consecutive;
        benefit_config.cooldown_decisions = (size_t) params_base.benefit_cooldown_decisions;
        benefit_config.target_iteration_ms = params_base.prefill_target_iteration_ms;
        benefit_config.checkpoint_path = params_base.benefit_checkpoint_path;
        benefit_config.checkpoint_interval = (size_t) params_base.benefit_checkpoint_interval;
        if (!benefit_config.checkpoint_path.empty()) {
            if (!params_base.benefit_checkpoint_key.empty()) {
                benefit_config.checkpoint_compatibility_key = params_base.benefit_checkpoint_key;
            } else {
                char description[256]{};
                llama_model_desc(model_tgt, description, sizeof(description));
                std::error_code identity_error;
                const auto model_file_size = std::filesystem::file_size(params_base.model.path, identity_error);
                const auto model_mtime = std::filesystem::last_write_time(params_base.model.path, identity_error);
                std::ostringstream identity;
                identity << "model_path=" << std::filesystem::absolute(params_base.model.path).lexically_normal().string()
                         << ";description=" << description
                         << ";model_bytes=" << llama_model_size(model_tgt)
                         << ";parameters=" << llama_model_n_params(model_tgt)
                         << ";gpu_layers=" << params_base.n_gpu_layers;
                if (!identity_error) {
                    identity << ";file_bytes=" << model_file_size
                             << ";mtime=" << model_mtime.time_since_epoch().count();
                }
                benefit_config.checkpoint_compatibility_key = identity.str();
            }
        }
        inference_engine.benefit_policy().configure(benefit_config);
        server_kv_action_policy_config kv_action_config;
        if (params_base.kv_action_policy == "fixed") {
            kv_action_config.model = server_kv_action_model::fixed_rule;
            kv_action_config.shadow = false;
        } else if (params_base.kv_action_policy == "analytical") {
            kv_action_config.model = server_kv_action_model::analytical;
            kv_action_config.shadow = false;
        } else {
            kv_action_config.model = server_kv_action_model::learned;
            kv_action_config.shadow = params_base.kv_action_policy == "learned-shadow";
        }
        kv_action_config.minimum_observations =
                (size_t) params_base.kv_action_min_observations;
        kv_action_config.confidence_beta = params_base.kv_action_confidence_beta;
        kv_action_config.switch_margin_ms = params_base.kv_action_switch_margin_ms;
        inference_engine.kv_action_policy().configure(kv_action_config);
        SRV_INF("CacheFlow policies: scheduler=%s, prefill=%s, benefit=%s, kv_action=%s, speculation=%s, kv_blocks=%s\n",
                params_base.scheduler_policy.c_str(), params_base.prefill_policy.c_str(),
                params_base.benefit_policy.c_str(),
                params_base.kv_action_policy.c_str(),
                params_base.spec_policy.c_str(), params_base.kv_block_runtime ? "enabled" : "disabled");
        inference_engine.speculation().configure({
            params_base.speculative_adaptive,
            1,
            (size_t) std::max(1, params_base.speculative.draft.n_max),
            0.2,
            params_base.speculative_disable_acceptance,
            0.75,
            params_base.speculative_kv_pressure,
            (size_t) params_base.speculative_cooldown,
            4,
            24,
        });
        if (params_base.kv_block_runtime) {
            const size_t block_size = (size_t) params_base.kv_block_size;
            const size_t capacity_blocks = ((size_t) n_ctx + block_size - 1) / block_size;
            inference_engine.set_kv_runtime(std::make_unique<server_kv_runtime>(block_size, capacity_blocks));
            const bool physical_block_capability = llama_memory_cacheflow_set_block_size(
                    llama_get_memory(ctx_tgt), (uint32_t) block_size);
            if (physical_block_capability) {
                SRV_INF("block KV runtime enabled with physical memory capability: block_size = %zu, capacity_blocks = %zu\n",
                        block_size, capacity_blocks);
            } else {
                SRV_WRN("%s", "block KV runtime is logical-only: active memory has no physical block capability; "
                        "physical KV operations remain on the upstream memory path\n");
            }
        } else {
            inference_engine.set_kv_runtime(nullptr);
        }
        inference_engine.set_swap_store(nullptr);
        if (params_base.kv_swap_budget_mib > 0) {
            const size_t budget_bytes = (size_t) params_base.kv_swap_budget_mib * 1024 * 1024;
            if (params_base.kv_swap_path.empty() || params_base.kv_swap_path == "memory") {
                inference_engine.set_swap_store(server_kv_create_host_swap_store(budget_bytes));
                SRV_INF("host KV swap enabled: budget = %d MiB\n", params_base.kv_swap_budget_mib);
            } else {
                inference_engine.set_swap_store(server_kv_create_file_swap_store(params_base.kv_swap_path, budget_bytes));
                SRV_INF("file KV swap enabled: path = %s, budget = %d MiB\n",
                        params_base.kv_swap_path.c_str(), params_base.kv_swap_budget_mib);
            }
            if (params_base.kv_swap_fault == "next-save") {
                inference_engine.swap_store()->inject(server_kv_swap_fault::next_save);
                SRV_WRN("%s\n", "injecting the next transactional KV-store save failure");
            } else if (params_base.kv_swap_fault == "next-restore") {
                inference_engine.swap_store()->inject(server_kv_swap_fault::next_restore);
                SRV_WRN("%s\n", "injecting the next transactional KV-store restore failure");
            }
        }

        // setup slots
        SRV_INF("initializing slots, n_slots = %d\n", params_base.n_parallel);

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        int n_ctx_slot = llama_n_ctx_seq(ctx_tgt);
        if (n_ctx_slot > n_ctx_train) {
            SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n", n_ctx_slot, n_ctx_train);
            n_ctx_slot = n_ctx_train;
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_WRN("%s", "speculative decoding will use checkpoints\n");
        }

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());
        }

        if (spec) {
            SRV_INF("%s", "speculative decoding context initialized\n");
        } else {
            ctx_dft.reset();
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft.get();
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot;

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            SLT_INF(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                auto * released = get_slot_by_id(id_slot);
                GGML_ASSERT(released != nullptr);
                GGML_ASSERT(inference_engine.transition(released->lifecycle, SLOT_STATE_IDLE));
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.reset();
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            batch = llama_batch_init(std::max(n_batch, params_base.n_parallel), 0, 1);
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_INF("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_INF("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_INF("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_INF("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_INF("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_INF("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_INF("%s", "context checkpoints disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.name.empty()) {
            model_name = params_base.model.name;
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task) {
            process_single_task(std::move(task));
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_INF("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_INF("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        // populate UI settings (from either new ui_config_json or deprecated webui_config_json)
        {
            const std::string & cfg = !params_base.ui_config_json.empty()
                ? params_base.ui_config_json
                : params_base.webui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                    json_webui_settings = json_settings; // deprecated: keep in sync
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                LOG_INF("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // thinking is enabled if:
            // 1. It's not explicitly disabled via --reasoning off
            // 2. The chat template supports it
            const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
            const bool enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
            SRV_INF("%s: chat template, thinking = %d\n", __func__, enable_thinking);

            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* allow_video           */ mctx ? mtmd_helper_support_video(mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    struct server_kv_action_inputs {
        server_kv_action_request request;
        server_kv_action_capabilities capabilities;
    };

    void restrict_kv_capabilities_to_override(server_kv_action_capabilities & capabilities) const {
        using override = common_params::kv_action_override;
        if (params_base.kv_action_override_mode == override::none) return;
        server_kv_action_capabilities forced;
        forced.recompute = true;
        switch (params_base.kv_action_override_mode) {
            case override::none: break;
            case override::direct: forced.direct = capabilities.direct; break;
            case override::remap: forced.remap = capabilities.remap; break;
            case override::paged: forced.paged = capabilities.paged; break;
            case override::recompute: break;
        }
        capabilities = forced;
    }

    server_kv_action_inputs build_kv_action_inputs(
            server_slot & target, const server_task & task) {
        const size_t resident_prefix = target.prompt.tokens.get_common_prefix(task.tokens);
        size_t donor_prefix = 0;
        const server_slot * donor = nullptr;
        for (const auto & candidate : slots) {
            if (candidate.id == target.id || candidate.is_processing()) continue;
            const size_t candidate_prefix = candidate.prompt.tokens.get_common_prefix(task.tokens);
            if (candidate_prefix > donor_prefix) {
                donor_prefix = candidate_prefix;
                donor = &candidate;
            }
        }
        const size_t cached_tokens = std::max({
            resident_prefix,
            donor_prefix,
            target.cuda_swapped_tokens.size(),
            target.store_swapped_tokens.size(),
        });
        server_kv_action_inputs result;
        auto & request = result.request;
        request.phase = server_kv_action_phase::serve;
        request.direct_legal = resident_prefix > 0 &&
                target.cuda_swapped_tokens.empty() && target.store_swapped_tokens.empty();
        request.cached_tokens = cached_tokens;
        request.expected_decode_tokens = task.params.n_predict > 0 ?
                static_cast<size_t>(task.params.n_predict) : 128;
        request.kv_bytes = donor && donor_prefix > resident_prefix ? donor->kv_action_bytes : target.kv_action_bytes;
        if (request.kv_bytes == 0 && cached_tokens > 0) {
            request.kv_bytes = llama_state_seq_get_size_ext(
                    ctx_tgt, donor && donor_prefix > resident_prefix ? donor->id : target.id,
                    LLAMA_STATE_SEQ_FLAGS_NONE);
        }
        const size_t block_size = (size_t) std::max(1, params_base.kv_block_size);
        request.page_count = (cached_tokens + block_size - 1) / block_size;
        request.contiguous_pages = request.direct_legal ? request.page_count : 0;
        request.reuse_distance = target.t_last_used < 0 ? 0 :
                (uint64_t) std::max<int64_t>(0, ggml_time_us() - target.t_last_used);
        request.device_bandwidth_bytes_per_ms = 100.0 * 1024 * 1024;
        request.host_bandwidth_bytes_per_ms = 12.0 * 1024 * 1024;
        request.launch_ms = 0.01;
        const auto measured = inference_engine.scheduler().state().cost;
        request.prefill_ms_per_token = measured.prefill_ms_per_token > 0.0 ?
                measured.prefill_ms_per_token : 0.03;
        request.decode_ms_per_token = measured.decode_ms_per_token > 0.0 ?
                measured.decode_ms_per_token : 0.02;

        if (const auto * runtime = inference_engine.kv_runtime()) {
            const auto blocks = runtime->snapshot().blocks;
            if (blocks.capacity_blocks > 0) {
                request.kv_pressure = (double) (blocks.allocated_blocks + blocks.reserved_blocks) /
                        blocks.capacity_blocks;
            }
        }
        const auto prefix_matches = [&task](const llama_tokens & cached) {
            if (cached.empty() || task.tokens.size() < cached.size()) return false;
            for (size_t i = 0; i < cached.size(); ++i) {
                if (cached[i] != task.tokens[i]) return false;
            }
            return true;
        };
        auto & capabilities = result.capabilities;
        capabilities.direct = request.direct_legal;
        capabilities.remap = params_base.kv_block_runtime && donor_prefix > resident_prefix;
        capabilities.paged = params_base.kv_paged_decode && resident_prefix > 0 &&
                resident_prefix <= 2048 && params_base.kv_block_size == 16;
        capabilities.device_swap = prefix_matches(target.cuda_swapped_tokens);
        capabilities.host_swap = target.store_swap_handle != 0 &&
                prefix_matches(target.store_swapped_tokens);
        capabilities.recompute = true;
        return result;
    }

    void choose_explicit_kv_action(server_slot & target, const server_task & task) {
        if (params_base.kv_action_override_mode == common_params::kv_action_override::none) return;
        auto inputs = build_kv_action_inputs(target, task);
        restrict_kv_capabilities_to_override(inputs.capabilities);
        const int64_t started = ggml_time_us();
        const auto decision = inference_engine.kv_action_policy().choose(
                inputs.request, inputs.capabilities);
        target.pending_kv_action = decision;
        target.pending_kv_action_effective = decision.action;
        target.pending_kv_action_started_us = started;
        target.has_pending_kv_action = true;
        target.paged_decode_requested = decision.action == server_kv_action::paged ||
                params_base.kv_action_override_mode == common_params::kv_action_override::paged;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // Select by net prefill benefit: reusable prefix minus the configured
        // cost of destroying cached tokens that do not belong to the new prompt.
        if (ret == nullptr && params_base.slot_prompt_similarity != 0.0f) {
            std::vector<server_slot_candidate> candidates;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                const llama_tokens & swapped_tokens = slot.store_swapped_tokens.empty()
                        ? slot.cuda_swapped_tokens : slot.store_swapped_tokens;
                size_t swapped_prefix = 0;
                while (swapped_prefix < swapped_tokens.size() &&
                        swapped_prefix < task.tokens.size() &&
                        swapped_tokens[swapped_prefix] == task.tokens[swapped_prefix]) {
                    ++swapped_prefix;
                }
                const size_t resident_prefix = slot.prompt.tokens.get_common_prefix(task.tokens);
                candidates.push_back({
                    slot.id,
                    std::max(resident_prefix, swapped_prefix),
                    std::max(slot.prompt.tokens.size(), swapped_tokens.size()),
                    slot.t_last_used,
                });
            }

            const auto selected = inference_engine.scheduler().select_slot(candidates, task.tokens.size());
            if (selected.found()) {
                ret = get_slot_by_id(selected.id);
                GGML_ASSERT(ret != nullptr);
                const float similarity = (float) selected.reused_tokens / task.tokens.size();
                const size_t retained_tokens = std::max(
                        ret->prompt.tokens.size(), std::max(
                            ret->cuda_swapped_tokens.size(), ret->store_swapped_tokens.size()));
                const float f_keep = retained_tokens > 0
                        ? (float) selected.reused_tokens / retained_tokens
                        : 0.0f;

                SLT_INF(*ret,
                        "selected slot by cache-aware score, score = %.3f, similarity = %.3f (> %.3f thold), "
                        "reused = %zu, evicted = %zu, penalty = %.3f, f_keep = %.3f\n",
                        selected.score, similarity, params_base.slot_prompt_similarity, selected.reused_tokens,
                        selected.evicted_tokens, params_base.slot_cache_eviction_penalty, f_keep);
                metrics.on_slot_selected(true, selected.reused_tokens, selected.evicted_tokens);

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);
                metrics.on_slot_selected(false, 0, ret->prompt.tokens.size());

                update_cache = true;
            }
        }

        if (ret) {
            auto inputs = build_kv_action_inputs(*ret, task);
            auto & action_request = inputs.request;
            auto & action_capabilities = inputs.capabilities;
            restrict_kv_capabilities_to_override(action_capabilities);
            const int64_t action_started_us = ggml_time_us();
            auto action_decision = inference_engine.kv_action_policy().choose(
                    action_request, action_capabilities);
            const bool store_restored = action_decision.action == server_kv_action::host_swap &&
                    try_restore_store_swap(*ret, task.tokens);
            const bool cuda_restored = action_decision.action == server_kv_action::device_swap &&
                    try_restore_cuda_swap(*ret, task.tokens);
            const bool restore_failed =
                    (action_decision.action == server_kv_action::host_swap && !store_restored) ||
                    (action_decision.action == server_kv_action::device_swap && !cuda_restored);
            if (restore_failed) {
                inference_engine.kv_action_policy().observe(action_decision, {
                    action_decision.action,
                    (ggml_time_us() - action_started_us) / 1000.0,
                    true,
                });
                server_kv_action_capabilities recompute_only;
                recompute_only.recompute = true;
                action_decision = inference_engine.kv_action_policy().choose(
                        action_request, recompute_only);
            }
            ret->pending_kv_action = action_decision;
            ret->pending_kv_action_effective = action_decision.action;
            ret->pending_kv_action_started_us = action_started_us;
            ret->has_pending_kv_action = true;
            ret->paged_decode_requested = action_decision.action == server_kv_action::paged ||
                    params_base.kv_action_override_mode == common_params::kv_action_override::paged;
            if (store_restored || cuda_restored) update_cache = false;
            update_cache = update_cache && prompt_cache;

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            if (update_cache) {
                SRV_INF("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                ret->prompt_save(*prompt_cache);

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear(false);
                }

                prompt_cache->update();

                SRV_INF("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }

            proactively_reclaim_kv(*ret, task);
        }

        return ret;
    }

    double current_kv_pressure() const {
        auto * memory = llama_get_memory(ctx_tgt);
        if (!memory || n_ctx <= 0) return 0.0;
        size_t used_tokens = 0;
        for (const auto & slot : slots) {
            const llama_pos pos_min = llama_memory_seq_pos_min(memory, slot.id);
            const llama_pos pos_max = llama_memory_seq_pos_max(memory, slot.id);
            if (pos_min >= 0 && pos_max >= pos_min) {
                used_tokens += (size_t) (pos_max - pos_min + 1);
            }
        }
        return std::min(1.0, (double) used_tokens / n_ctx);
    }

    void proactively_reclaim_kv(server_slot & selected, const server_task & task) {
        if (!params_base.kv_unified || task.type != SERVER_TASK_TYPE_COMPLETION) {
            return;
        }
        auto * memory = llama_get_memory(ctx_tgt);
        if (!memory) {
            return;
        }

        size_t used_tokens = 0;
        size_t selected_tokens = 0;
        std::vector<server_kv_cache_candidate> candidates;
        const int64_t now = ggml_time_us();
        for (const auto & slot : slots) {
            const llama_pos pos_min = llama_memory_seq_pos_min(memory, slot.id);
            const llama_pos pos_max = llama_memory_seq_pos_max(memory, slot.id);
            const size_t occupied = pos_min >= 0 && pos_max >= pos_min
                    ? (size_t) (pos_max - pos_min + 1)
                    : 0;
            used_tokens += occupied;
            if (slot.id == selected.id) {
                selected_tokens = occupied;
            } else if (!slot.is_processing() && occupied > 0) {
                candidates.push_back({slot.id, occupied, slot.t_last_used});
            }
        }

        // The selected sequence is truncated to its common prefix before new
        // prompt tokens enter llama_decode. Its projected footprint therefore
        // becomes the new prompt size rather than old + new.
        const size_t projected = used_tokens - selected_tokens + task.tokens.size();
        const auto plan = inference_engine.capacity_planner().plan((size_t) n_ctx, projected, now, candidates);
        metrics.on_kv_capacity_plan(plan);
        for (int victim_id : plan.victim_ids) {
            auto * victim = get_slot_by_id(victim_id);
            GGML_ASSERT(victim != nullptr && !victim->is_processing());
            SRV_WRN("proactively purging KV slot %d with %zu tokens; pressure = %zu, expected recompute = %.2f\n",
                    victim_id, victim->prompt.tokens.size(), plan.deficit_tokens,
                    plan.expected_recompute_tokens);
            bool saved = false;
            if (prompt_cache) {
                saved = victim->prompt_save(*prompt_cache);
                if (saved) prompt_cache->update();
            }
            mark_slot_preempted(*victim);
            victim->prompt_clear(false);
        }
        if (!plan.fits()) {
            SRV_WRN("KV admission remains over capacity by %zu tokens after reclaiming %zu; decode fallback remains active\n",
                    plan.deficit_tokens - plan.reclaimed_tokens, plan.reclaimed_tokens);
        }
    }

    // Last-resort path for physical fragmentation that cannot be predicted by
    // logical token spans. Use the same value-aware planner instead of purging
    // the first idle slot in vector order.
    bool try_clear_idle_slots() {
        if (!params_base.kv_unified) {
            return false;
        }

        std::vector<server_kv_cache_candidate> candidates;
        for (const auto & slot : slots) {
            if (!slot.is_processing() && slot.prompt.n_tokens() > 0) {
                candidates.push_back({slot.id, slot.prompt.tokens.size(), slot.t_last_used});
            }
        }
        // A one-token synthetic deficit asks for exactly one lowest-value
        // victim; llama_decode will retry and request another victim only if
        // fragmentation remains.
        const auto plan = inference_engine.capacity_planner().plan(0, 1, ggml_time_us(), candidates);
        if (plan.victim_ids.empty()) {
            return false;
        }
        metrics.on_kv_capacity_plan(plan);
        auto * victim = get_slot_by_id(plan.victim_ids.front());
        GGML_ASSERT(victim != nullptr && !victim->is_processing());
        SRV_WRN("purging planned KV victim slot %d with %zu tokens after decode allocation failure\n",
                victim->id, victim->prompt.tokens.size());
        bool saved = false;
        if (prompt_cache) {
            saved = victim->prompt_save(*prompt_cache);
            if (saved) prompt_cache->update();
        }
        mark_slot_preempted(*victim);
        victim->prompt_clear(false);
        return true;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    void try_adopt_shared_prefix(server_slot & destination, const server_task & task) {
        auto * kv_runtime = inference_engine.kv_runtime();
        if (!kv_runtime || !task.params.cache_prompt ||
                destination.prompt.tokens.has_mtmd || task.tokens.has_mtmd) {
            return;
        }
        std::vector<server_kv_sequence_id> eligible;
        for (const auto & donor : slots) {
            if (donor.id == destination.id || donor.prompt.tokens.empty() ||
                    donor.prompt.tokens.has_mtmd || !are_lora_equal(donor.lora, destination.lora)) {
                continue;
            }
            eligible.push_back(donor.id);
        }
        const auto plan = kv_runtime->plan_prefix_share(
                destination.id, task.tokens.get_tokens(), eligible);
        const size_t existing = destination.prompt.tokens.get_common_prefix(task.tokens);
        if (!plan.found() || plan.matched_tokens <= existing) return;

        auto * donor = get_slot_by_id((int) plan.donor);
        if (!donor || donor->prompt.tokens.size() < plan.matched_tokens) return;
        common_context_seq_rm(ctx_tgt, destination.id, -1, -1);
        llama_memory_seq_cp(llama_get_memory(ctx_tgt),
                donor->id, destination.id, 0, (llama_pos) plan.matched_tokens);
        if (ctx_dft) {
            common_context_seq_rm(ctx_dft.get(), destination.id, -1, -1);
            llama_memory_seq_cp(llama_get_memory(ctx_dft.get()),
                    donor->id, destination.id, 0, (llama_pos) plan.matched_tokens);
        }
        llama_tokens shared;
        shared.reserve(plan.matched_tokens);
        for (size_t i = 0; i < plan.matched_tokens; ++i) {
            shared.push_back(donor->prompt.tokens[i]);
        }
        destination.prompt.tokens.clear();
        destination.prompt.tokens.insert(shared);
        SRV_INF("shared %zu prefix KV blocks (%zu tokens) from slot %d to slot %d\n",
                plan.matched_blocks, plan.matched_tokens, donor->id, destination.id);
    }

    void synchronize_kv_runtime() {
        auto * kv_runtime = inference_engine.kv_runtime();
        if (!kv_runtime) return;
        const uint64_t now = (uint64_t) ggml_time_us();
        for (const auto & slot : slots) {
            if (slot.prompt.tokens.empty()) {
                if (!kv_runtime->is_swapped(slot.id)) {
                    kv_runtime->release(slot.id);
                }
                continue;
            }
            std::string error;
            if (!kv_runtime->synchronize(
                    slot.id, slot.prompt.tokens.get_tokens(), 0, now, &error)) {
                SRV_WRN("logical KV synchronization rejected for slot %d: %s\n",
                        slot.id, error.c_str());
            }
        }
        const std::string invalid = kv_runtime->validate();
        GGML_ASSERT(invalid.empty());
    }

    bool try_swap_out_store(server_slot & slot) {
        auto * kv_swap_store = inference_engine.swap_store();
        if (!kv_swap_store || ctx_dft || slot.prompt.tokens.empty()) return false;

        server_kv_swap_payload payload;
        payload.sequence_id = slot.id;
        const size_t state_size = llama_state_seq_get_size_ext(
                ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (state_size == 0) return false;
        payload.opaque_state.resize(state_size);
        const size_t written = llama_state_seq_get_data_ext(
                ctx_tgt, payload.opaque_state.data(), payload.opaque_state.size(),
                slot.id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (written != state_size) {
            SRV_WRN("failed to serialize KV state for slot %d: expected %zu bytes, got %zu\n",
                    slot.id, state_size, written);
            return false;
        }

        // Save is transactional: resident KV remains untouched until the store
        // has committed a complete checksummed record.
        const auto saved = kv_swap_store->save(payload);
        if (!saved.ok) {
            SRV_WRN("KV store swap-out failed for slot %d; keeping resident KV: %s\n",
                    slot.id, saved.error.c_str());
            return false;
        }
        const auto previous = slot.store_swap_handle;
        slot.store_swap_handle = saved.handle;
        slot.store_swapped_tokens = slot.prompt.tokens.get_tokens();
        slot.kv_action_bytes = state_size;
        common_context_seq_rm(ctx_tgt, slot.id, -1, -1);
        if (previous != 0) kv_swap_store->erase(previous);
        SRV_INF("stored KV state for slot %d with %zu tokens and %zu bytes\n",
                slot.id, slot.store_swapped_tokens.size(), state_size);
        return true;
    }

    bool try_restore_store_swap(server_slot & slot, const server_tokens & requested) {
        auto * kv_swap_store = inference_engine.swap_store();
        auto * kv_runtime = inference_engine.kv_runtime();
        if (!kv_swap_store || slot.store_swap_handle == 0 ||
                slot.store_swapped_tokens.empty() ||
                requested.size() < slot.store_swapped_tokens.size()) {
            return false;
        }
        for (size_t i = 0; i < slot.store_swapped_tokens.size(); ++i) {
            if (slot.store_swapped_tokens[i] != requested[i]) return false;
        }

        server_kv_swap_payload payload;
        std::string error;
        if (!kv_swap_store->restore(slot.store_swap_handle, payload, &error) ||
                payload.sequence_id != slot.id || payload.opaque_state.empty()) {
            SRV_WRN("KV store restore failed for slot %d; request will recompute: %s\n",
                    slot.id, error.c_str());
            kv_swap_store->erase(slot.store_swap_handle);
            slot.store_swap_handle = 0;
            slot.store_swapped_tokens.clear();
            slot.kv_action_bytes = 0;
            metrics.n_requests_recomputed_total++;
            return false;
        }
        const size_t consumed = llama_state_seq_set_data_ext(
                ctx_tgt, payload.opaque_state.data(), payload.opaque_state.size(),
                slot.id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (consumed != payload.opaque_state.size()) {
            SRV_WRN("KV store state rejected for slot %d; request will recompute\n", slot.id);
            common_context_seq_rm(ctx_tgt, slot.id, -1, -1);
            kv_swap_store->erase(slot.store_swap_handle);
            slot.store_swap_handle = 0;
            slot.store_swapped_tokens.clear();
            slot.kv_action_bytes = 0;
            metrics.n_requests_recomputed_total++;
            return false;
        }

        slot.prompt.tokens.clear();
        slot.prompt.tokens.insert(slot.store_swapped_tokens);
        if (!slot.prompt.tokens.empty()) {
            slot.prompt.tokens.keep_first(slot.prompt.tokens.size() - 1);
            common_context_seq_rm(ctx_tgt, slot.id, slot.prompt.tokens.pos_next(), -1);
        }
        kv_swap_store->erase(slot.store_swap_handle);
        slot.store_swap_handle = 0;
        slot.store_swapped_tokens.clear();
        slot.kv_action_bytes = 0;
        metrics.n_requests_restored_total++;
        if (kv_runtime && kv_runtime->is_swapped(slot.id)) {
            std::string logical_error;
            if (!kv_runtime->restore(slot.id, (uint64_t) ggml_time_us(), &logical_error)) {
                SRV_WRN("logical KV restore failed for slot %d: %s\n", slot.id, logical_error.c_str());
            }
        }
        SRV_INF("restored KV state from transactional store for slot %d with %zu prompt tokens\n",
                slot.id, slot.prompt.tokens.size());
        return true;
    }

    bool mark_slot_preempted(server_slot & slot) {
        auto * kv_runtime = inference_engine.kv_runtime();
        if (slot.prompt.tokens.empty()) return false;
        bool physically_swapped = try_swap_out_store(slot);
        auto * memory = llama_get_memory(ctx_tgt);
        if (!physically_swapped && !ctx_dft && llama_memory_seq_can_swap_cuda(memory, slot.id)) {
            slot.kv_action_bytes = llama_state_seq_get_size_ext(
                    ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE);
            llama_synchronize(ctx_tgt);
            physically_swapped = llama_memory_seq_swap_out_cuda(memory, slot.id);
            if (physically_swapped) slot.cuda_swapped_tokens = slot.prompt.tokens.get_tokens();
            else slot.kv_action_bytes = 0;
        }
        metrics.n_requests_preempted_total++;
        metrics.n_requests_swapped_total += physically_swapped ? 1 : 0;
        if (!kv_runtime) return physically_swapped;
        if (!kv_runtime->preempt(slot.id, (uint64_t) ggml_time_us())) {
            // The logical snapshot may lag a just-completed request by one
            // iteration. Synchronize it before retrying the transition.
            std::string error;
            if (kv_runtime->synchronize(
                    slot.id, slot.prompt.tokens.get_tokens(), 0,
                    (uint64_t) ggml_time_us(), &error)) {
                kv_runtime->preempt(slot.id, (uint64_t) ggml_time_us());
            }
        }
        return physically_swapped;
    }

    bool try_restore_cuda_swap(server_slot & slot, const server_tokens & requested) {
        auto * kv_runtime = inference_engine.kv_runtime();
        if (slot.cuda_swapped_tokens.empty() || requested.size() < slot.cuda_swapped_tokens.size()) {
            return false;
        }
        for (size_t i = 0; i < slot.cuda_swapped_tokens.size(); ++i) {
            if (slot.cuda_swapped_tokens[i] != requested[i]) return false;
        }
        if (!llama_memory_seq_swap_in_cuda(llama_get_memory(ctx_tgt), slot.id)) return false;
        slot.prompt.tokens.clear();
        slot.prompt.tokens.insert(slot.cuda_swapped_tokens);
        // The scheduler constructs its prefill candidates before the STARTED
        // state performs the usual one-token logits rollback. Do that rollback
        // at restore time so an exact prompt match still schedules one token.
        if (!slot.prompt.tokens.empty()) {
            slot.prompt.tokens.keep_first(slot.prompt.tokens.size() - 1);
            common_context_seq_rm(ctx_tgt, slot.id, slot.prompt.tokens.pos_next(), -1);
        }
        slot.cuda_swapped_tokens.clear();
        slot.kv_action_bytes = 0;
        metrics.n_requests_restored_total++;
        if (kv_runtime && kv_runtime->is_swapped(slot.id)) {
            std::string error;
            if (!kv_runtime->restore(slot.id, (uint64_t) ggml_time_us(), &error)) {
                SRV_WRN("logical KV restore failed for slot %d: %s\n", slot.id, error.c_str());
            }
        }
        SRV_INF("restored real CUDA KV stream for slot %d with %zu prompt tokens\n",
                slot.id, slot.prompt.tokens.size());
        return true;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        auto * kv_swap_store = inference_engine.swap_store();
        if (slot.store_swap_handle != 0) {
            if (kv_swap_store) kv_swap_store->erase(slot.store_swap_handle);
            slot.store_swap_handle = 0;
            slot.store_swapped_tokens.clear();
            slot.kv_action_bytes = 0;
            metrics.n_requests_recomputed_total++;
        }
        if (!slot.cuda_swapped_tokens.empty()) {
            llama_memory_seq_swap_erase_cuda(llama_get_memory(ctx_tgt), slot.id);
            slot.cuda_swapped_tokens.clear();
            slot.kv_action_bytes = 0;
            metrics.n_requests_recomputed_total++;
        }
        inference_engine.speculation().reset(slot.id);
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.tokens.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        try_adopt_shared_prefix(slot, task);

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool backend_sampling = true;

            backend_sampling &= task.params.sampling.backend_sampling;

            // TODO: speculative decoding requires multiple samples per batch - not supported yet
            backend_sampling &= !(slot.can_speculate());

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            backend_sampling &= !need_pre_sample_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());
        } else {
            slot.smpl.reset();
        }

        slot.task = std::make_unique<const server_task>(std::move(task));

        const slot_state initial_phase = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;
        GGML_ASSERT(inference_engine.transition(slot.lifecycle, initial_phase));

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_decoded = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), slot.n_decoded, slot.n_ctx);
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params_base)) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n", slot.n_decoded, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_decoded = %d, n_indent = %d\n", slot.n_decoded, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && (ggml_time_us() - slot.t_start_generation > 1000.0f*slot.task->params.t_max_predict_ms)) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n", slot.n_decoded, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n", slot.n_decoded, slot.n_remaining, result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            // TODO: optimize this with min-p optimization
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    // if multimodal is enabled, send an error and return false
    bool check_no_mtmd(const int id_task) {
        if (mctx) {
            send_error(id_task, "This feature is not supported by multimodal", ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }
        return true;
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.n_prompt_tokens_cache;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = (ggml_time_us() - slot.t_start_process_prompt) / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->timings = slot.get_timings();
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->timings         = slot.get_timings();
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        metrics.on_request_finished(slot);
        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_INF("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        auto & cur = slot.prompt.checkpoints.emplace_back();

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        cur.update_tgt(ctx_tgt,       slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        cur.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

        SLT_INF(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);
    }

    void process_single_task(server_task && task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_slot = task.id_slot;
                    const int id_task = task.id;

                    server_slot * slot = id_slot != -1 ? get_slot_by_id(id_slot) : get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (id_slot != -1) {
                            choose_explicit_kv_action(*slot, task);
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            slot->has_pending_kv_action = false;
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else {
                        if (id_slot != -1) {
                            choose_explicit_kv_action(*slot, task);
                        }
                        if (!launch_slot_with_task(*slot, std::move(task))) {
                            slot->has_pending_kv_action = false;
                            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_INF(slot, "%s", "saving idle slot to prompt cache\n");

                                const bool saved = slot.prompt_save(*prompt_cache);
                                if (saved) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                const bool physically_swapped = mark_slot_preempted(slot);
                                if (params_base.kv_unified || physically_swapped) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    slot.prompt_clear(false);
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    json slots_data = json::array();

                    int n_idle_slots       = 0;
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        json slot_data = slot.to_json(slots_debug == 0);

                        if (slot.is_processing()) {
                            n_processing_slots++;
                        } else {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot_data);
                    }
                    SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->slots_data          = std::move(slots_data);
                    res->n_idle_slots        = n_idle_slots;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->t_start             = metrics.t_start;

                    res->n_prompt_tokens_processed_total = metrics.n_prompt_tokens_processed_total;
                    res->n_prompt_tokens_cached_total    = metrics.n_prompt_tokens_cached_total;
                    res->t_prompt_processing_total       = metrics.t_prompt_processing_total;
                    res->n_tokens_predicted_total        = metrics.n_tokens_predicted_total;
                    res->t_tokens_generation_total       = metrics.t_tokens_generation_total;

                    res->n_tokens_max = metrics.n_tokens_max;

                    res->n_prompt_tokens_processed = metrics.n_prompt_tokens_processed;
                    res->n_prompt_tokens_cached    = metrics.n_prompt_tokens_cached;
                    res->t_prompt_processing       = metrics.t_prompt_processing;
                    res->n_tokens_predicted        = metrics.n_tokens_predicted;
                    res->t_tokens_generation       = metrics.t_tokens_generation;

                    res->n_decode_total          = metrics.n_decode_total;
                    res->n_busy_slots_total      = metrics.n_busy_slots_total;
                    res->n_slot_cache_selections_total = metrics.n_slot_cache_selections_total;
                    res->n_slot_lru_selections_total   = metrics.n_slot_lru_selections_total;
                    res->n_slot_reused_tokens_total    = metrics.n_slot_reused_tokens_total;
                    res->n_slot_evicted_tokens_total   = metrics.n_slot_evicted_tokens_total;
                    res->n_scheduler_iterations_total     = metrics.n_scheduler_iterations_total;
                    res->n_decode_tokens_scheduled_total  = metrics.n_decode_tokens_scheduled_total;
                    res->n_prefill_tokens_scheduled_total = metrics.n_prefill_tokens_scheduled_total;
                    res->n_prefill_chunks_scheduled_total = metrics.n_prefill_chunks_scheduled_total;
                    res->batch_tokens = metrics.batch_tokens;
                    res->batch_sequences = metrics.batch_sequences;
                    res->prefill_starvation_ms = metrics.prefill_starvation_ms;
                    res->n_kv_admission_pressure_total = metrics.n_kv_admission_pressure_total;
                    res->n_kv_proactive_evictions_total = metrics.n_kv_proactive_evictions_total;
                    res->n_kv_proactive_reclaimed_tokens_total = metrics.n_kv_proactive_reclaimed_tokens_total;
                    res->n_kv_admission_failures_total = metrics.n_kv_admission_failures_total;
                    res->n_requests_preempted_total = metrics.n_requests_preempted_total;
                    res->n_requests_swapped_total = metrics.n_requests_swapped_total;
                    res->n_requests_restored_total = metrics.n_requests_restored_total;
                    res->n_requests_recomputed_total = metrics.n_requests_recomputed_total;
                    res->n_draft_tokens_total = metrics.n_draft_tokens_total;
                    res->n_draft_tokens_accepted_total = metrics.n_draft_tokens_accepted_total;
                    res->n_speculation_disabled_low_acceptance_total =
                            inference_engine.speculation().disabled_low_acceptance_total();
                    const double target_ms_per_token = metrics.n_tokens_predicted_total
                            ? (double) metrics.t_tokens_generation_total /
                                    metrics.n_tokens_predicted_total
                            : 0.0;
                    res->speculation_net_saved_ms =
                            metrics.n_draft_tokens_accepted_total * target_ms_per_token -
                            metrics.t_speculation_draft_total_ms;
                    const auto engine_snapshot = inference_engine.snapshot();
                    res->benefit_cpu = engine_snapshot.cpu_benefit;
                    res->benefit_cuda = engine_snapshot.cuda_benefit;
                    res->benefit_persistence = engine_snapshot.benefit_persistence;
                    res->kv_action_policy = engine_snapshot.kv_action_policy;
                    res->kv_execution_status = llama_get_kv_execution_status(ctx_tgt);
                    res->time_to_first_token = metrics.time_to_first_token;
                    res->time_per_output_token = metrics.time_per_output_token;
                    res->request_latency = metrics.request_latency;
                    res->request_queue = metrics.request_queue;

                    const auto scheduler_state = inference_engine.scheduler().state();
                    res->adaptive_prefill_chunk = scheduler_state.effective_chunk_size;
                    res->adaptive_prefill_iteration_ms = scheduler_state.iteration_latency_ewma_ms;
                    for (const auto & slot : slots) {
                        res->adaptive_draft_length += inference_engine.speculation().state(slot.id).draft_tokens;
                    }
                    if (!slots.empty()) res->adaptive_draft_length /= slots.size();

                    if (const auto * kv_runtime = inference_engine.kv_runtime()) {
                        const auto kv_snapshot = kv_runtime->snapshot().blocks;
                        res->kv_blocks_used = kv_snapshot.allocated_blocks;
                        res->kv_blocks_free = kv_snapshot.capacity_blocks -
                                kv_snapshot.allocated_blocks - kv_snapshot.reserved_blocks;
                        res->kv_shared_blocks = kv_snapshot.shared_blocks;
                        res->kv_copy_on_write_total = kv_snapshot.copy_on_write_total;
                    }
                    if (const auto * kv_swap_store = inference_engine.swap_store()) {
                        const auto store_stats = kv_swap_store->stats();
                        res->kv_store_save_failures = store_stats.save_failures;
                        res->kv_store_restore_failures = store_stats.restore_failures;
                        res->kv_store_bytes_saved_total = store_stats.bytes_saved_total;
                        res->kv_store_bytes_restored_total = store_stats.bytes_restored_total;
                        res->kv_store_bytes_current = store_stats.bytes_current;
                        res->kv_store_bytes_peak = store_stats.bytes_peak;
                        res->kv_store_save_microseconds = store_stats.save_microseconds;
                        res->kv_store_restore_microseconds = store_stats.restore_microseconds;
                    }

                    auto memory = llama_get_memory(ctx_tgt);
                    if (memory) {
                        llama_memory_cacheflow_cuda_stats(memory, &res->cuda_kv);
                        for (const auto & slot : slots) {
                            const llama_pos pos_min = llama_memory_seq_pos_min(memory, slot.id);
                            const llama_pos pos_max = llama_memory_seq_pos_max(memory, slot.id);
                            if (pos_min >= 0 && pos_max >= pos_min) {
                                res->kv_cache_tokens += (uint64_t) (pos_max - pos_min + 1);
                            }
                        }
                    }
                    res->kv_cache_capacity_tokens = (uint64_t) n_ctx;

                    for (const auto & item : llama_get_memory_breakdown(ctx_tgt)) {
                        const auto & breakdown = item.second;
                        res->memory_model_bytes   += breakdown.model;
                        res->memory_context_bytes += breakdown.context;
                        res->memory_compute_bytes += breakdown.compute;
                    }

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }

                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const size_t token_count = slot->prompt.tokens.size();
                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    const llama_tokens & tokens = slot->prompt.tokens.get_tokens();
                    const size_t nwrite = llama_state_seq_save_file(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), token_count);

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = token_count;
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    if (!check_no_mtmd(task.id)) break;
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    llama_tokens tokens;
                    tokens.resize(slot->n_ctx);
                    size_t token_count = 0;
                    size_t nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), tokens.size(), &token_count);
                    if (nread == 0) {
                        slot->prompt.tokens.clear(); // KV may already been invalidated?
                        send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    tokens.resize(token_count);
                    slot->prompt.tokens.clear();
                    slot->prompt.tokens.insert(tokens);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = token_count;
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear(false);

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_INF("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }
    }

    bool prepare_iteration() {
        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_INF("%s", "all slots are idle\n");

                return false;
            }
        }

        {
            SRV_DBG("%s", "posting NEXT_RESPONSE\n");

            server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
            task.id = queue_tasks.get_new_id();
            queue_tasks.post(std::move(task));
        }

        // apply context-shift if needed
        // TODO: simplify and improve
        for (server_slot & slot : slots) {
            if (slot.lifecycle.phase() == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                const int n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                common_context_seq_rm (ctx_tgt, slot.id, n_keep            , n_keep + n_discard);
                common_context_seq_add(ctx_tgt, slot.id, n_keep + n_discard, slot.prompt.n_tokens(), -n_discard);

                if (ctx_dft) {
                    common_context_seq_rm (ctx_dft.get(), slot.id, n_keep            , n_keep + n_discard);
                    common_context_seq_add(ctx_dft.get(), slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);
                }

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.tokens.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        }

        return true;
    }

    void plan_execute_iteration(server_inference_iteration & iteration) {
        // start populating the batch for this iteration
        common_batch_clear(batch);

        // track if given slot can be batched with slots already in the batch
        server_slot * slot_batched = nullptr;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // determine which slots are generating and drafting
        for (auto & slot : slots) {
            if (slot.lifecycle.phase() != SLOT_STATE_GENERATING) {
                continue;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                continue;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_configured = slot.get_n_draft_max();
                const int n_draft_max = (int) inference_engine.speculation().recommend(
                        slot.id, (size_t) std::max(0, n_draft_configured), current_kv_pressure());

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .n_past   = */ slot.prompt.n_tokens(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        }

        // generate the actual drafts (if any)
        {
            const int64_t draft_started_us = ggml_time_us();
            common_speculative_draft(spec.get());
            if (!drafting.empty()) {
                metrics.t_speculation_draft_total_ms +=
                        std::max<int64_t>(0, ggml_time_us() - draft_started_us) / 1.e3;
            }
        }

        // make checkpoints if needed
        for (auto * slot_ptr : drafting) {
            auto & slot = *slot_ptr;

            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.n_draft_total += draft.size();

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }

                common_context_seq_rm(ctx_dft.get(), slot.id, ckpt.pos_max + 1, -1);
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft.get()));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
        }

        // update the batch with the sampled/drafted tokens
        for (auto * slot_ptr : generating) {
            auto & slot = *slot_ptr;

            slot.update_batch(batch);
        }

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        // Decode tokens were inserted above. Allocate only the remaining
        // iteration budget to prompt chunks, so long prefills cannot monopolize
        // latency-sensitive decode work.
        std::vector<server_prefill_candidate> prefill_candidates;
        for (auto & slot : slots) {
            if (!slot.is_processing() || !slot.can_split()) {
                continue;
            }
            if (slot.lifecycle.phase() != SLOT_STATE_STARTED &&
                    slot.lifecycle.phase() != SLOT_STATE_PROCESSING_PROMPT) {
                continue;
            }
            if (slot_batched && !slot_batched->can_batch_with(slot)) {
                continue;
            }

            size_t cached = slot.prompt.tokens.size();
            if (slot.lifecycle.phase() == SLOT_STATE_STARTED) {
                cached = slot.task->params.cache_prompt
                        ? slot.prompt.tokens.get_common_prefix(slot.task->tokens)
                        : 0;
            }
            const size_t total = (size_t) slot.task->n_tokens();
            // A fully cached prompt still has to re-evaluate its final token to
            // produce logits ([TAG_PROMPT_LOGITS] below). Giving it zero quota
            // deadlocks the iteration before that rollback can run.
            prefill_candidates.push_back({slot.id, std::max<size_t>(1,
                    total > cached ? total - cached : 0)});
        }

        const size_t decode_tokens_planned = (size_t) batch.n_tokens;
        const int64_t iteration_started_us = ggml_time_us();
        const size_t prefill_budget = n_batch > batch.n_tokens
                ? (size_t) (n_batch - batch.n_tokens) : 0;
        const auto cacheflow_prefill_plan = inference_engine.scheduler().plan_prefill(
                prefill_candidates, prefill_budget);
        const auto upstream_prefill_plan = server_inference_scheduler::plan_upstream_prefill(
                prefill_candidates, prefill_budget);
        server_benefit_decision benefit_decision;
        const bool benefit_decision_active = !prefill_candidates.empty() && prefill_budget > 0 &&
                !cacheflow_prefill_plan.equivalent_allocations(upstream_prefill_plan);
        server_prefill_plan prefill_plan = cacheflow_prefill_plan;
        if (benefit_decision_active) {
            size_t remaining_prefill_tokens = 0;
            size_t maximum_remaining_tokens = 0;
            for (const auto & candidate : prefill_candidates) {
                remaining_prefill_tokens += candidate.remaining_tokens;
                maximum_remaining_tokens = std::max(maximum_remaining_tokens, candidate.remaining_tokens);
            }
            double kv_pressure = 0.0;
            if (const auto * kv_runtime = inference_engine.kv_runtime()) {
                const auto kv = kv_runtime->snapshot().blocks;
                if (kv.capacity_blocks > 0) {
                    kv_pressure = (double) (kv.allocated_blocks + kv.reserved_blocks) /
                            kv.capacity_blocks;
                }
            }
            benefit_decision = inference_engine.benefit_policy().choose({
                params_base.n_gpu_layers > 0 ? server_benefit_backend::cuda : server_benefit_backend::cpu,
                generating.size() + prefill_candidates.size(),
                decode_tokens_planned,
                upstream_prefill_plan.tokens_scheduled,
                cacheflow_prefill_plan.tokens_scheduled,
                cacheflow_prefill_plan.allocations.size(),
                prefill_candidates.size(),
                remaining_prefill_tokens,
                maximum_remaining_tokens,
                kv_pressure,
            });
            if (benefit_decision.action == server_benefit_action::upstream) {
                prefill_plan = upstream_prefill_plan;
            }
        }
        double max_prefill_wait_ms = 0.0;
        for (const auto & candidate : prefill_candidates) {
            if (prefill_plan.quota(candidate.id) != 0) continue;
            const auto & waiting = slots.at((size_t) candidate.id);
            if (waiting.task && waiting.task->t_queued_us > 0) {
                max_prefill_wait_ms = std::max(max_prefill_wait_ms,
                        (iteration_started_us - waiting.task->t_queued_us) / 1.e3);
            }
        }
        metrics.on_scheduler_plan(
                (size_t) batch.n_tokens, generating.size(), prefill_plan, max_prefill_wait_ms);

        float  alora_scale       = -1.0f;
        size_t alora_disabled_id = 0;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.n_tokens == 0) {
            for (auto & slot : slots) {
                if (!slot.is_processing()) {
                    continue;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    continue;
                }

                // check if this is a child slot
                if (slot.lifecycle.phase() == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    continue;
                }

                // this slot still has a prompt to be processed
                if (slot.lifecycle.phase() == SLOT_STATE_PROCESSING_PROMPT ||
                        slot.lifecycle.phase() == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    const size_t prefill_quota = slot.can_split()
                            ? prefill_plan.quota(slot.id)
                            : (size_t) std::max(0, n_batch - batch.n_tokens);
                    if (prefill_quota == 0) {
                        continue;
                    }

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.n_tokens;

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.lifecycle.phase() == SLOT_STATE_STARTED) {
                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation = 0;

                        GGML_ASSERT(inference_engine.transition(
                                slot.lifecycle, SLOT_STATE_PROCESSING_PROMPT));

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            continue;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                continue;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            common_context_seq_rm (ctx_tgt, slot.id, head_p, head_c);
                                            common_context_seq_add(ctx_tgt, slot.id, head_c, head_c + n_match, kv_shift);

                                            if (ctx_dft) {
                                                common_context_seq_rm (ctx_dft.get(), slot.id, head_p, head_c);
                                                common_context_seq_add(ctx_dft.get(), slot.id, head_c, head_c + n_match, kv_shift);
                                            }

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - 4, 0);
                                    const int np1 = std::min<int>(n_past + 6, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&, func_name = __func__](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            LOG_INF("slot %12.*s: id %2d | task %d | Checking checkpoint with [%d, %d] against %d...\n", 12,
                                                func_name, (slot).id, ((slot).task ? (slot).task->id : -1), cur.pos_min, cur.pos_max, pos_min_thold);
                                            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                            if (cur.pos_max > pos_next) {
                                                return false;
                                            }
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        it->load_tgt(ctx_tgt,       slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        it->load_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                        pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                        n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                        SLT_WRN(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) it->size() / 1024 / 1024);
                                    }

                                    if (do_reset) {
                                        SLT_WRN(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next) {
                                        SLT_WRN(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.n_prompt_tokens_cache = n_past;
                        slot.n_prompt_tokens_processed = 0;

                        slot.prompt.tokens.keep_first(n_past);

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.n_tokens + slot.task->n_tokens() > n_batch) {
                            continue;
                        }
                    }

                    const int64_t t_current = ggml_time_us();
                    slot.t_prompt_processing = (t_current - slot.t_start_process_prompt) / 1e3;
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    common_context_seq_rm(ctx_tgt, slot.id, p0, -1);
                    if (ctx_dft) {
                        common_context_seq_rm(ctx_dft.get(), slot.id, p0, -1);
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    bool has_mtmd = false;

                    // check if we should process the image
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        // process the image
                        size_t n_tokens_out = 0;
                        int32_t res = slot.process_mtmd_chunk(cur_token_idx, n_tokens_out);
                        if (res != 0) {
                            SLT_ERR(slot, "failed to process image, res = %d\n", res);
                            send_error(slot, "failed to process image", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        slot.n_prompt_tokens_processed += n_tokens_out;

                        // add the image chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            slot.prompt.tokens.push_back(chunk.get()); // copy
                        }

                        has_mtmd = true;
                    }

                    const int32_t n_before_user = slot.task->params.n_before_user;
                    const bool n_before_user_known = n_before_user > 0;

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() &&
                           batch.n_tokens < n_batch &&
                           (size_t) (batch.n_tokens - n_tokens_prev) < prefill_quota) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        common_batch_add(batch,
                            cur_tok,
                            slot.prompt.tokens.pos_next(),
                            { slot.id },
                            slot.need_embd());
                        slot.prompt.tokens.push_back(cur_tok);

                        slot.n_prompt_tokens_processed++;

                        // stop the prompt batch exactly before the latest user input, so a checkpoint
                        // can be created after the previous messages
                        if (n_before_user_known &&
                            slot.prompt.n_tokens() == n_before_user) {
                            break;
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.n_tokens - n_tokens_prev;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        GGML_ASSERT(inference_engine.transition(
                                slot.lifecycle, SLOT_STATE_DONE_PROMPT));

                        GGML_ASSERT(batch.n_tokens > 0);

                        // extract the logits only for the last token
                        batch.logits[batch.n_tokens - 1] = true;

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.n_tokens - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints
                        if (!n_before_user_known && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // checkpoints are created before the current batch is decoded, so
                    // their token position is the batch start rather than the prompt end
                    const int32_t n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    {
                        const bool is_on_user =
                            n_before_user_known &&
                            n_tokens_start == n_before_user;

                        const bool is_after_user =
                            n_before_user_known &&
                            n_tokens_start > n_before_user;

                        const bool is_allowed =
                            !n_before_user_known ||
                            is_on_user ||
                            (is_after_user && near_prompt_end);

                        if (do_checkpoint && !is_allowed) {
                            do_checkpoint = false;
                        }
                    }

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together
                    do_checkpoint = do_checkpoint && (slot.prompt.checkpoints.empty() || n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }

                if (batch.n_tokens >= n_batch) {
                    break;
                }
            }
        }

        SRV_DBG("decoding batch, n_tokens = %d\n", batch.n_tokens);

        GGML_ASSERT(inference_engine.publish_plan(iteration, {
            decode_tokens_planned,
            prefill_plan.tokens_scheduled,
            (size_t) batch.n_tokens,
        }));
        inference_engine.execute(iteration, [&]() {

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        if (slot_batched) {
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        if (batch.n_tokens == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }
        } else {
            n_empty_consecutive = 0;
        }

        int32_t i_next = 0;

        // process the created batch of tokens
        std::vector<server_slot *> batch_owners(batch.n_tokens, nullptr);
        for (auto & slot : slots) {
            if (slot.i_batch >= 0 && slot.i_batch < batch.n_tokens) {
                batch_owners[slot.i_batch] = &slot;
            }
        }
        for (int32_t i = 0; i < batch.n_tokens; i = i_next) {
            const auto token_owner = [&](int32_t batch_index) {
                return batch_owners[batch_index];
            };
            const auto token_is_decode = [&](int32_t batch_index) {
                if (batch.n_seq_id[batch_index] != 1) return false;
                const auto owner = token_owner(batch_index);
                return owner != nullptr && owner->lifecycle.phase() == SLOT_STATE_GENERATING;
            };
            const auto token_requests_paged_decode = [&](int32_t batch_index) {
                const auto owner = token_owner(batch_index);
                return token_is_decode(batch_index) && owner != nullptr && owner->paged_decode_requested;
            };
            const int32_t maximum_tokens = std::min(n_batch, batch.n_tokens - i);
            const bool first_is_decode = token_is_decode(i);
            const bool first_requests_paged = token_requests_paged_decode(i);
            int32_t n_tokens = 1;
            // Use the same phase-homogeneous graph boundaries for every arm.
            // Otherwise only splitting Paged decode out of mixed prefill chunks
            // changes the reduction shape and confounds backend correctness/performance.
            while (n_tokens < maximum_tokens &&
                    token_is_decode(i + n_tokens) == first_is_decode &&
                    token_requests_paged_decode(i + n_tokens) == first_requests_paged) {
                ++n_tokens;
            }

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            enum llama_kv_execution_action llama_action = LLAMA_KV_EXECUTION_ACTION_DIRECT;
            // Paged decode is a graph-wide action. Admit it only when this
            // batch is a true decode batch: exactly one token for each unique
            // sequence and every token belongs to a slot whose policy chose
            // Paged. This prevents a mixed prefill/decode batch from being
            // relabeled while allowing continuous batching across sequences.
            bool paged_batch = first_requests_paged;
            std::vector<llama_seq_id> paged_sequences;
            paged_sequences.reserve(n_tokens);
            for (int32_t token = 0; token < n_tokens && paged_batch; ++token) {
                const int32_t batch_index = i + token;
                if (batch.n_seq_id[batch_index] != 1) {
                    paged_batch = false;
                    break;
                }
                const llama_seq_id seq_id = batch.seq_id[batch_index][0];
                if (std::find(paged_sequences.begin(), paged_sequences.end(), seq_id) !=
                        paged_sequences.end()) {
                    paged_batch = false;
                    break;
                }
                paged_sequences.push_back(seq_id);

                const auto owner = token_owner(batch_index);
                if (owner == nullptr || !owner->paged_decode_requested) {
                    paged_batch = false;
                }
            }
            for (const auto & slot : slots) {
                if (slot.i_batch < i || slot.i_batch >= i + n_tokens || !slot.has_pending_kv_action) continue;
                if (slot.pending_kv_action.action == server_kv_action::remap) {
                    llama_action = LLAMA_KV_EXECUTION_ACTION_REMAP;
                }
            }
            if (paged_batch) {
                llama_action = LLAMA_KV_EXECUTION_ACTION_PAGED;
            }
            llama_set_kv_execution_action(ctx_tgt, llama_action);
            const auto decode_result = inference_engine.runtime().decode(ctx_tgt, batch_view);
            const auto llama_action_status = llama_get_kv_execution_status(ctx_tgt);
            for (auto & slot : slots) {
                if (slot.i_batch < i || slot.i_batch >= i + n_tokens || !slot.has_pending_kv_action) continue;
                if (slot.pending_kv_action.action == server_kv_action::paged &&
                        llama_action_status.effective != LLAMA_KV_EXECUTION_ACTION_PAGED) {
                    slot.pending_kv_action_effective = server_kv_action::direct;
                }
            }
            const int ret = decode_result.code;

            metrics.on_decoded(slots);

            if (ret != 0) {
                {
                    std::string err;

                    if (n_batch == 1 && ret == 1) {
                        // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                        //       need to remove the tokens from the current batch too
                        err = "Context size has been exceeded.";
                    }

                    if (ret == -1) {
                        err = "Invalid input batch.";
                    }

                    if (ret < -1) {
                        // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                        err = "Compute error.";
                    }

                    // TODO: handle ret == 2 (abort) when we start aborting

                    if (!err.empty()) {
                        iteration.abort(err);
                        SRV_ERR("%s i = %d, n_batch = %d, ret = %d\n", err.c_str(), i, n_batch, ret);

                        for (auto & slot : slots) {
                            if (slot.is_processing()) {
                                if (slot.has_pending_kv_action) {
                                    inference_engine.kv_action_policy().observe(
                                            slot.pending_kv_action, {
                                                slot.pending_kv_action_effective,
                                                (ggml_time_us() - slot.pending_kv_action_started_us) / 1000.0,
                                                true,
                                            });
                                    slot.has_pending_kv_action = false;
                                }
                                send_error(slot, err);
                                slot.release();

                                // note: it's complicated to keep track of how much of the current batch has been
                                //       processed before the error occurred, so we simply clear the entire context
                                slot.prompt_clear(false);
                            }
                        }

                        break;
                    }
                }

                // retry with half the batch size to try to find a free slot in the KV cache
                if (!try_clear_idle_slots()) {
                    n_batch /= 2;
                }

                SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);

                continue; // continue loop of n_batch
            }

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            //       for now, always re-evaluate for simplicity
            //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
            //
            // | spec type   | need re-eval |
            // | ---         | ---          |
            // | draft model | no           | because the draft model does not use embeddings from the target
            // | MTP (std)   | yes          |
            // | MTP Gemma4  | no           | because the KV cache is shared
            // | Eagle3      | yes          |
            // | DFlash      | yes          | https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4405406982
            //
            // note: this logic is now moved in `common_speculative_process()`
            //       keeping the sketch here until for a bit, until the logic is finalized
            //
            //if (ctx_dft) {
            //    // TODO: update as needed for MTP, Eagle3, etc.
            //    const bool need_tgt_embd = false;

            //    if (need_tgt_embd) {
            //        llama_synchronize(ctx_tgt);
            //    }

            //    // the logic here varies depending on the speculative decoding method
            //    //  - some draft contexts require embeddings from the target context, others don't
            //    //  - some draft contexts involve an encoder step to transform the target embeddings to draft embeddings
            //    // TODO: extract this in a function ?
            //    {
            //        // TODO: hook the embeddings from the last target batch here
            //        if (llama_model_has_encoder(model_dft.get())) {
            //            //llama_encode(ctx_dft, ...);

            //            GGML_ABORT("not implemented yet\n");
            //        }

            //        const int ret = llama_decode(ctx_dft.get(), batch_view);

            //        if (ret != 0) {
            //            SRV_ERR("failed to decode draft batch, ret = %d\n", ret);

            //            // TODO: handle error
            //            break;
            //        }
            //    }
            //}
            if (!common_speculative_process(spec.get(), batch_view)) {
                SRV_ERR("%s", "failed to process speculative batch\n");
                iteration.abort("failed to process speculative batch");

                // TODO: handle error
                break;
            }

            // move the head of the batch forward with the number of tokens we just processed
            i_next = i + n_tokens;

            // on successful decode, restore the original batch size
            n_batch = llama_n_batch(ctx_tgt);

            // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
            for (auto & slot : slots) {
                if (slot.lifecycle.phase() == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                    std::vector<server_slot *> children;
                    for (auto & other : slots) {
                        if (other.lifecycle.phase() == SLOT_STATE_WAIT_OTHER &&
                                slot.task->id == other.task->id_parent) {
                            children.push_back(&other);
                        }
                    }

                    // all children slots should already launched by launch_slots_with_parent_task()
                    // copy state to the child slots
                    for (auto & child : children) {
                        SLT_INF(slot, " - copying state to child %d\n", child->id);

                        GGML_ASSERT(child->lifecycle.phase() == SLOT_STATE_WAIT_OTHER);

                        slot.copy_state_to(*child);
                        GGML_ASSERT(inference_engine.transition(
                                child->lifecycle, SLOT_STATE_DONE_PROMPT));
                    }
                }
            }

            for (auto & slot : slots) {
                // optionally send prompt processing progress
                if (slot.lifecycle.phase() == SLOT_STATE_PROCESSING_PROMPT ||
                        slot.lifecycle.phase() == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->params.stream && slot.task->params.return_progress) {
                        send_partial_response(slot, {}, true);
                    }
                }

                if (slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens)) {
                    continue; // continue loop of slots
                }

                // The action boundary closes only for the slot whose useful
                // target decode just completed. Other pending slots may have
                // been deferred by batching and must remain unobserved.
                if (slot.has_pending_kv_action) {
                    inference_engine.kv_action_policy().observe(slot.pending_kv_action, {
                        slot.pending_kv_action_effective,
                        (ggml_time_us() - slot.pending_kv_action_started_us) / 1000.0,
                        false,
                    });
                    slot.has_pending_kv_action = false;
                }

                if (slot.lifecycle.phase() == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                        // prompt evaluated for embedding
                        send_embedding(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                        send_rerank(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    GGML_ASSERT(slot.task->need_sampling());

                    // prompt evaluated for next-token prediction
                    GGML_ASSERT(inference_engine.transition(slot.lifecycle, SLOT_STATE_GENERATING));

                    if (slot.can_speculate()) {
                        common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                    }
                } else if (slot.lifecycle.phase() != SLOT_STATE_GENERATING) {
                    continue; // continue loop of slots
                }

                if (slot.can_speculate() && !slot.spec_draft.empty()) {
                    continue; // sample using speculative decoding
                }

                const int tok_idx = slot.i_batch - i;

                llama_token id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);

                slot.i_batch = -1;

                common_sampler_accept(slot.smpl.get(), id, true);

                // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
                const int64_t t_current = ggml_time_us();

                slot.n_decoded += 1;

                if (slot.n_decoded == 1) {
                    slot.t_start_generation = t_current;
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);
                }

                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                completion_token_output result;
                result.tok          = id;
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

                if (slot.task->params.sampling.n_probs > 0) {
                    populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
                }

                if (!process_token(result, slot)) {
                    // release slot because of stop condition
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();

                    continue;
                }

                slot.print_timings_tg();
            }

            // speculative decoding - main model sample and accept
            for (auto & slot : slots) {
                if (slot.lifecycle.phase() != SLOT_STATE_GENERATING ||
                        !slot.can_speculate() || slot.spec_draft.empty()) {
                    continue;
                }

                // save the original draft size
                const size_t n_draft = slot.spec_draft.size();

                GGML_ASSERT(n_draft > 0);

                // verify and try to accept the draft
                {
                    // save the sampler sampler state in case we need to restore it
                    common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                    GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                    auto accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                    slot.spec_i_batch.clear();

                    GGML_ASSERT(accepted.size() >= 1);

                    const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                    const bool use_ckpt_tgt =
                        ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                       (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));

                    // check for partial draft acceptance
                    if (n_rollback > 0) {
                        if (use_ckpt_tgt) {
                            if (trace > 0) {
                                SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                            }

                            // partial acceptance is not supported by the context -> truncate the draft and restore the state
                            slot.spec_draft = std::move(accepted);

                            const auto & ckpt = slot.spec_ckpt;

                            SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                            {
                                ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                common_context_seq_rm(slot.ctx_tgt, slot.id, ckpt.pos_max + 1, -1);
                            }

                            if (slot.ctx_dft) {
                                ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                common_context_seq_rm(slot.ctx_dft, slot.id, ckpt.pos_max + 1, -1);
                            }

                            slot.prompt.tokens.keep_first(ckpt.n_tokens);
                            slot.smpl = std::move(smpl_save);

                            continue;
                        }
                    }

                    if (trace > 0) {
                        SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                    }

                    common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);

                    slot.spec_draft = std::move(accepted);
                }

                const int64_t t_current = ggml_time_us();

                const auto ids = std::move(slot.spec_draft);

                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                // update how many tokens out of those tested were accepted
                slot.n_draft_accepted += ids.size() - 1;
                inference_engine.speculation().observe({
                    slot.id,
                    n_draft,
                    ids.size() - 1,
                    current_kv_pressure(),
                });
                metrics.on_speculation(n_draft, ids.size() - 1);

                // add accepted tokens to the prompt
                slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
                slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

                slot.sampled = ids.back(); // last accepted token
                SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

                common_context_seq_rm(slot.ctx_tgt, slot.id, slot.prompt.tokens.pos_next(), -1);
                if (slot.ctx_dft) {
                    common_context_seq_rm(slot.ctx_dft, slot.id, slot.prompt.tokens.pos_next(), -1);
                }

                for (size_t i = 0; i < ids.size(); ++i) {
                    completion_token_output result;

                    result.tok          = ids[i];
                    result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                    result.prob         = 1.0f; // set later

                    // TODO: set result.probs

                    slot.n_decoded += 1;

                    if (!process_token(result, slot)) {
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        slot.release();

                        break;
                    }
                }

                slot.print_timings_tg();

                SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) ids.size() - 1, (int) n_draft, slot.prompt.n_tokens());
            }
            GGML_ASSERT(iteration.record_executed((size_t) n_tokens));
        }
        });

        GGML_ASSERT(inference_engine.commit_with(iteration, [&]() {
            commit_iteration(decode_tokens_planned, prefill_plan, iteration_started_us,
                    benefit_decision_active, benefit_decision, iteration.is_aborted());
        }) || iteration.is_aborted());

        SRV_DBG("%s", "run slots completed\n");
    }

    void commit_iteration(
            size_t decode_tokens_planned,
            const server_prefill_plan & prefill_plan,
            int64_t iteration_started_us,
            bool benefit_decision_active,
            const server_benefit_decision & benefit_decision,
            bool execution_failed) {
        const double elapsed_ms = (ggml_time_us() - iteration_started_us) / 1000.0;
        inference_engine.scheduler().observe_iteration({
            decode_tokens_planned,
            prefill_plan.tokens_scheduled,
            elapsed_ms,
            prefill_plan.allocations.size(),
        });
        if (benefit_decision_active) {
            inference_engine.benefit_policy().observe(benefit_decision, {
                elapsed_ms,
                elapsed_ms > params_base.prefill_target_iteration_ms,
                execution_failed,
            });
        }
        synchronize_kv_runtime();
    }

    void update_slots() {
        inference_engine.step(
                [this]() { return prepare_iteration(); },
                [this](server_inference_iteration & iteration) {
                    plan_execute_iteration(iteration);
                    GGML_ASSERT(iteration.validate().empty());
                });
    }

    int get_slot_n_ctx() {
        return slots.back().n_ctx;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* has_inp_video          */ impl->chat_params.allow_video,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* json_webui_settings    */ impl->json_webui_settings,  // Deprecated
        /* slot_n_ctx             */ impl->get_slot_n_ctx(),
        /* pooling_type           */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
    };
}



// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_http_res {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::on_sleeping_changed(std::function<void(bool)> callback) {
    impl->queue_tasks.on_sleeping_state(std::move(callback));
}

// compute the number of tokens before the last user message in the prompt
static int32_t prompt_get_n_before_user(
        const json & message_spans,
        const std::string & prompt,
        const std::vector<raw_buffer> & files,
        const llama_vocab * vocab,
        mtmd_context * mctx) {
    int32_t result = -1;
    int32_t byte_pos = -1;

    for (const auto & span : message_spans) {
        const std::string role = json_value(span, "role", std::string());

        if (role == "user") {
            byte_pos = json_value(span, "pos", -1);
        }
    }

    if (byte_pos >= 0) {
        GGML_ASSERT((size_t) byte_pos <= prompt.size());

        const std::string prefix = prompt.substr(0, (size_t) byte_pos);

        const std::string marker = get_media_marker();
        size_t n_prefix_media = 0;
        for (size_t pos = 0; (pos = prefix.find(marker, pos)) != std::string::npos; pos += marker.size()) {
            n_prefix_media++;
        }

        GGML_ASSERT(n_prefix_media <= files.size());

        if (mctx != nullptr && n_prefix_media > 0) {
            // TODO: this makes a copy - avoid it
            std::vector<raw_buffer> prefix_files(files.begin(), files.begin() + n_prefix_media);

            result = (int32_t) process_mtmd_prompt(mctx, prefix, prefix_files).size();
        } else {
            result = (int32_t) tokenize_input_prompts(vocab, nullptr, prefix, true, true)[0].size();
        }

        SRV_TRC("message_spans: last user message: byte_pos=%d, media=%zu, n_before_user=%d\n",
                byte_pos, n_prefix_media, result);
    }

    return result;
}


//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_task::params_from_json_cmpl(
                    ctx_server.vocab,
                    params,
                    meta->slot_n_ctx,
                    meta->logit_bias_eog,
                    data);

            const auto message_spans = json_value(data, "message_spans", json::array());
            if (prompt.is_string() && message_spans.is_array()) {
                task.params.n_before_user =
                    prompt_get_n_before_user(
                        message_spans,
                        prompt.get<std::string>(),
                        files,
                        ctx_server.vocab,
                        ctx_server.mctx);
            }

            task.id_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->next = [res_this = res.get(), res_type, &req, &params](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            try {
                if (req.should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &req, &start_time, &params]() {
                    if (req.should_stop()) {
                        return true; // should_stop condition met
                    } else if (params.sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)params.sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(req.should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        };
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) res_task->n_prompt_tokens_processed_total}
            }, {
                    {"name",  "prompt_tokens_cached_total"},
                    {"help",  "Number of prompt tokens reused from the KV cache."},
                    {"value",  (uint64_t) res_task->n_prompt_tokens_cached_total}
            }, {
                    {"name",  "slot_cache_selections_total"},
                    {"help",  "Number of requests assigned by cache-aware slot scoring."},
                    {"value",  res_task->n_slot_cache_selections_total}
            }, {
                    {"name",  "slot_lru_selections_total"},
                    {"help",  "Number of requests assigned by the LRU fallback."},
                    {"value",  res_task->n_slot_lru_selections_total}
            }, {
                    {"name",  "slot_reused_tokens_total"},
                    {"help",  "Estimated prompt tokens preserved by cache-aware slot selection."},
                    {"value",  res_task->n_slot_reused_tokens_total}
            }, {
                    {"name",  "slot_evicted_tokens_total"},
                    {"help",  "Estimated cached tokens discarded by cache-aware slot selection."},
                    {"value",  res_task->n_slot_evicted_tokens_total}
            }, {
                    {"name",  "scheduler_iterations_total"},
                    {"help",  "Number of token-level inference scheduling iterations."},
                    {"value",  res_task->n_scheduler_iterations_total}
            }, {
                    {"name",  "decode_tokens_scheduled_total"},
                    {"help",  "Decode and speculative tokens admitted before prefill allocation."},
                    {"value",  res_task->n_decode_tokens_scheduled_total}
            }, {
                    {"name",  "prefill_tokens_scheduled_total"},
                    {"help",  "Prompt tokens admitted by the iteration-level prefill planner."},
                    {"value",  res_task->n_prefill_tokens_scheduled_total}
            }, {
                    {"name",  "prefill_chunks_scheduled_total"},
                    {"help",  "Prompt chunks admitted by the iteration-level prefill planner."},
                    {"value",  res_task->n_prefill_chunks_scheduled_total}
            }, {
                    {"name",  "kv_admission_pressure_total"},
                    {"help",  "Requests projected to exceed unified KV capacity before decode."},
                    {"value",  res_task->n_kv_admission_pressure_total}
            }, {
                    {"name",  "kv_proactive_evictions_total"},
                    {"help",  "Idle KV sequences proactively evicted before llama_decode."},
                    {"value",  res_task->n_kv_proactive_evictions_total}
            }, {
                    {"name",  "kv_proactive_reclaimed_tokens_total"},
                    {"help",  "Logical KV tokens proactively reclaimed by admission planning."},
                    {"value",  res_task->n_kv_proactive_reclaimed_tokens_total}
            }, {
                    {"name",  "kv_admission_failures_total"},
                    {"help",  "KV admission plans still over capacity after all idle victims."},
                    {"value",  res_task->n_kv_admission_failures_total}
            }, {
                    {"name",  "kv_copy_on_write_total"},
                    {"help",  "Logical shared tail blocks cloned before mutation."},
                    {"value",  res_task->kv_copy_on_write_total}
            }, {
                    {"name",  "kv_evicted_blocks_total"},
                    {"help",  "Logical KV blocks reclaimed proactively by admission control."},
                    {"value",  params.kv_block_size > 0
                            ? (res_task->n_kv_proactive_reclaimed_tokens_total + params.kv_block_size - 1) /
                                    params.kv_block_size
                            : 0}
            }, {
                    {"name",  "kv_swap_bytes_total"},
                    {"help",  "Cumulative bytes moved by CUDA and transactional host/file KV swap."},
                    {"value",  res_task->cuda_kv.swap_bytes +
                            res_task->kv_store_bytes_saved_total + res_task->kv_store_bytes_restored_total}
            }, {
                    {"name",  "kv_restore_seconds"},
                    {"help",  "Cumulative time spent restoring CUDA and transactional KV state."},
                    {"value",  (res_task->cuda_kv.swap_in_microseconds +
                            res_task->kv_store_restore_microseconds) / 1.e6}
            }, {
                    {"name",  "kv_store_bytes_current"},
                    {"help",  "Bytes currently resident in the transactional host/file KV store."},
                    {"value",  res_task->kv_store_bytes_current}
            }, {
                    {"name",  "kv_store_bytes_peak"},
                    {"help",  "Peak bytes resident in the transactional host/file KV store."},
                    {"value",  res_task->kv_store_bytes_peak}
            }, {
                    {"name",  "kv_store_failures_total"},
                    {"help",  "Transactional host/file KV save and restore failures."},
                    {"value",  res_task->kv_store_save_failures + res_task->kv_store_restore_failures}
            }, {
                    {"name",  "requests_preempted_total"},
                    {"help",  "Sequences selected for KV preemption."},
                    {"value",  res_task->n_requests_preempted_total}
            }, {
                    {"name",  "requests_swapped_total"},
                    {"help",  "Sequences physically swapped through CUDA or transactional host/file storage."},
                    {"value",  res_task->n_requests_swapped_total}
            }, {
                    {"name",  "requests_restored_total"},
                    {"help",  "Swapped sequences restored without full prefill recomputation."},
                    {"value",  res_task->n_requests_restored_total}
            }, {
                    {"name",  "requests_recomputed_total"},
                    {"help",  "Swapped sequences discarded because an incoming prompt did not match."},
                    {"value",  res_task->n_requests_recomputed_total}
            }, {
                    {"name",  "draft_tokens_total"},
                    {"help",  "Speculative draft tokens verified by the target model."},
                    {"value",  res_task->n_draft_tokens_total}
            }, {
                    {"name",  "draft_tokens_accepted_total"},
                    {"help",  "Speculative draft tokens accepted by the target model."},
                    {"value",  res_task->n_draft_tokens_accepted_total}
            }, {
                    {"name",  "cuda_kv_kernel_launches_total"},
                    {"help",  "Descriptor-driven real KV CUDA kernels launched."},
                    {"value",  res_task->cuda_kv.kernel_launches}
            }, {
                    {"name",  "cuda_kv_blocks_copied_total"},
                    {"help",  "Real KV tensor ranges copied by CacheFlow CUDA kernels."},
                    {"value",  res_task->cuda_kv.blocks_copied}
            }, {
                    {"name",  "cuda_kv_copy_on_write_total"},
                    {"help",  "Partial shared KV tails eagerly cloned by the production CUDA adapter."},
                    {"value",  res_task->cuda_kv.copy_on_write}
            }, {
                    {"name",  "cuda_kv_copy_bytes_total"},
                    {"help",  "Real KV bytes transported by CUDA copy and swap operations."},
                    {"value",  res_task->cuda_kv.copy_bytes}
            }, {
                    {"name",  "cuda_kv_remap_vectorized_bytes_total"},
                    {"help",  "Real KV remap bytes handled with aligned 128-bit CUDA loads and stores."},
                    {"value",  res_task->cuda_kv.remap_vectorized_bytes}
            }, {
                    {"name",  "cuda_kv_remap_scalar_bytes_total"},
                    {"help",  "Real KV remap bytes handled by the safe scalar alignment/tail fallback."},
                    {"value",  res_task->cuda_kv.remap_scalar_bytes}
            }, {
                    {"name",  "cuda_kv_events_waited_total"},
                    {"help",  "CUDA KV dependency events explicitly waited by the runtime."},
                    {"value",  res_task->cuda_kv.events_waited}
            }, {
                    {"name",  "cuda_kv_backend_errors_total"},
                    {"help",  "CUDA runtime failures observed by the real KV adapter."},
                    {"value",  res_task->cuda_kv.backend_errors}
            }, {
                    {"name",  "cuda_kv_copy_seconds"},
                    {"help",  "Cumulative CUDA-event time spent in descriptor KV copies."},
                    {"value",  res_task->cuda_kv.copy_microseconds / 1.e6}
            }, {
                    {"name",  "cuda_kv_swap_out_seconds"},
                    {"help",  "Cumulative CUDA-event time spent swapping KV to pinned host memory."},
                    {"value",  res_task->cuda_kv.swap_out_microseconds / 1.e6}
            }, {
                    {"name",  "cuda_kv_swap_in_seconds"},
                    {"help",  "Cumulative CUDA-event time spent restoring KV to device memory."},
                    {"value",  res_task->cuda_kv.swap_in_microseconds / 1.e6}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) res_task->t_prompt_processing_total / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) res_task->n_tokens_predicted_total}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) res_task->t_tokens_generation_total / 1.e3}
            }, {
                    {"name",  "n_decode_total"},
                    {"help",  "Total number of llama_decode() calls"},
                    {"value",  res_task->n_decode_total}
            }, {
                    {"name",  "n_tokens_max"},
                    {"help",  "Largest observed n_tokens."},
                    {"value",  res_task->n_tokens_max}
            }}},
            {"gauge", {{
                    {"name",  "cacheflow_scheduler_policy"},
                    {"help",  "Effective scheduler policy (0 = upstream, 1 = cacheflow)."},
                    {"value",  params.scheduler_policy == "cacheflow" ? 1 : 0}
            },{
                    {"name",  "cacheflow_prefill_policy"},
                    {"help",  "Effective prefill policy (0 = greedy, 1 = fixed, 2 = adaptive)."},
                    {"value",  params.prefill_policy == "adaptive" ? 2 : params.prefill_policy == "fixed" ? 1 : 0}
            },{
                    {"name",  "cacheflow_spec_policy"},
                    {"help",  "Effective speculation policy (0 = fixed, 1 = adaptive)."},
                    {"value",  params.spec_policy == "adaptive" ? 1 : 0}
            },{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  res_task->n_prompt_tokens_processed ? 1.e3 / res_task->t_prompt_processing * res_task->n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  res_task->n_tokens_predicted ? 1.e3 / res_task->t_tokens_generation * res_task->n_tokens_predicted : 0.}
            },{
                    {"name",  "prompt_cache_hit_ratio"},
                    {"help",  "Ratio of prompt tokens reused from the KV cache in the current metrics bucket."},
                    {"value",  res_task->n_prompt_tokens_cached + res_task->n_prompt_tokens_processed
                            ? (double) res_task->n_prompt_tokens_cached /
                                    (res_task->n_prompt_tokens_cached + res_task->n_prompt_tokens_processed)
                            : 0.}
            },{
                    {"name",  "kv_prefix_hit_ratio"},
                    {"help",  "Ratio of prompt tokens served from a resident KV prefix."},
                    {"value",  res_task->n_prompt_tokens_cached + res_task->n_prompt_tokens_processed
                            ? (double) res_task->n_prompt_tokens_cached /
                                    (res_task->n_prompt_tokens_cached + res_task->n_prompt_tokens_processed)
                            : 0.}
            },{
                    {"name",  "batch_tokens"},
                    {"help",  "Tokens admitted by the most recent scheduler iteration."},
                    {"value",  res_task->batch_tokens}
            },{
                    {"name",  "batch_sequences"},
                    {"help",  "Sequences admitted by the most recent scheduler iteration."},
                    {"value",  res_task->batch_sequences}
            },{
                    {"name",  "prefill_starvation_ms"},
                    {"help",  "Maximum queue age of a prefill sequence deferred by the latest plan."},
                    {"value",  res_task->prefill_starvation_ms}
            },{
                    {"name",  "kv_cache_tokens"},
                    {"help",  "Logical tokens currently occupying KV cache sequence spans."},
                    {"value",  res_task->kv_cache_tokens}
            },{
                    {"name",  "kv_blocks_used"},
                    {"help",  "Logical KV blocks currently allocated."},
                    {"value",  res_task->kv_blocks_used}
            },{
                    {"name",  "kv_blocks_free"},
                    {"help",  "Logical KV blocks available after reservations."},
                    {"value",  res_task->kv_blocks_free}
            },{
                    {"name",  "kv_shared_blocks"},
                    {"help",  "Logical KV blocks referenced by more than one sequence."},
                    {"value",  res_task->kv_shared_blocks}
            },{
                    {"name",  "adaptive_prefill_chunk"},
                    {"help",  "Current adaptive prefill chunk size in tokens."},
                    {"value",  res_task->adaptive_prefill_chunk}
            },{
                    {"name",  "adaptive_prefill_iteration_milliseconds"},
                    {"help",  "EWMA inference-iteration latency observed by adaptive prefill."},
                    {"value",  res_task->adaptive_prefill_iteration_ms}
            },{
                    {"name",  "adaptive_draft_length"},
                    {"help",  "Mean current draft length recommended across server slots."},
                    {"value",  res_task->adaptive_draft_length}
            },{
                    {"name",  "draft_acceptance_ratio"},
                    {"help",  "Cumulative fraction of speculative draft tokens accepted."},
                    {"value",  res_task->n_draft_tokens_total
                            ? (double) res_task->n_draft_tokens_accepted_total / res_task->n_draft_tokens_total
                            : 0.0}
            },{
                    {"name",  "speculation_net_saved_ms"},
                    {"help",  "Estimated accepted-target work saved minus observed draft wall time."},
                    {"value",  res_task->speculation_net_saved_ms}
            },{
                    {"name",  "cuda_kv_effective_bandwidth_bytes"},
                    {"help",  "Effective KV transport bytes per CUDA-event second."},
                    {"value",  res_task->cuda_kv.copy_microseconds +
                                    res_task->cuda_kv.swap_out_microseconds +
                                    res_task->cuda_kv.swap_in_microseconds
                            ? res_task->cuda_kv.copy_bytes * 1.e6 /
                                    (res_task->cuda_kv.copy_microseconds +
                                     res_task->cuda_kv.swap_out_microseconds +
                                     res_task->cuda_kv.swap_in_microseconds)
                            : 0.0}
            },{
                    {"name",  "cuda_kv_pinned_pool_bytes"},
                    {"help",  "Pinned host bytes currently retained by CUDA KV swap records."},
                    {"value",  res_task->cuda_kv.pinned_bytes_current}
            },{
                    {"name",  "cuda_kv_pinned_pool_peak_bytes"},
                    {"help",  "Peak pinned host bytes retained by CUDA KV swap records."},
                    {"value",  res_task->cuda_kv.pinned_bytes_peak}
            },{
                    {"name",  "kv_cache_capacity_tokens"},
                    {"help",  "Configured KV cache token capacity across all slots."},
                    {"value",  res_task->kv_cache_capacity_tokens}
            },{
                    {"name",  "kv_cache_usage_ratio"},
                    {"help",  "Logical KV cache token utilization ratio."},
                    {"value",  res_task->kv_cache_capacity_tokens
                            ? (double) res_task->kv_cache_tokens / res_task->kv_cache_capacity_tokens
                            : 0.}
            },{
                    {"name",  "memory_model_bytes"},
                    {"help",  "Model buffer bytes allocated by llama.cpp across all backends."},
                    {"value",  res_task->memory_model_bytes}
            },{
                    {"name",  "memory_context_bytes"},
                    {"help",  "Context and KV cache bytes allocated by llama.cpp across all backends."},
                    {"value",  res_task->memory_context_bytes}
            },{
                    {"name",  "memory_compute_bytes"},
                    {"help",  "Temporary compute buffer bytes allocated by llama.cpp across all backends."},
                    {"value",  res_task->memory_compute_bytes}
            },{
                    {"name",  "memory_total_bytes"},
                    {"help",  "Total model, context, and compute bytes allocated by llama.cpp."},
                    {"value",  res_task->memory_model_bytes + res_task->memory_context_bytes + res_task->memory_compute_bytes}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of requests processing."},
                    {"value",  (uint64_t) res_task->n_processing_slots}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of requests deferred."},
                    {"value",  (uint64_t) res_task->n_tasks_deferred}
            },{
                    {"name",  "n_busy_slots_per_decode"},
                    {"help",  "Average number of busy slots per llama_decode() call"},
                    {"value",  (float) res_task->n_busy_slots_total / std::max((float) res_task->n_decode_total, 1.f)}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        auto append_benefit = [&prometheus](const char * backend,
                const server_benefit_snapshot & value) {
            prometheus << "llamacpp:benefit_decisions_total{backend=\"" << backend
                       << "\",action=\"upstream\"} " << value.upstream_decisions << "\n"
                       << "llamacpp:benefit_decisions_total{backend=\"" << backend
                       << "\",action=\"cacheflow\"} " << value.cacheflow_decisions << "\n"
                       << "llamacpp:benefit_observations_total{backend=\"" << backend
                       << "\",action=\"upstream\"} " << value.upstream_observations << "\n"
                       << "llamacpp:benefit_observations_total{backend=\"" << backend
                       << "\",action=\"cacheflow\"} " << value.cacheflow_observations << "\n"
                       << "llamacpp:benefit_exploration_total{backend=\"" << backend << "\"} "
                       << value.exploration_decisions << "\n"
                       << "llamacpp:benefit_reason_total{backend=\"" << backend
                       << "\",reason=\"cold_start\"} " << value.cold_start_decisions << "\n"
                       << "llamacpp:benefit_reason_total{backend=\"" << backend
                       << "\",reason=\"positive_lower_bound\"} " << value.positive_lower_bound_decisions << "\n"
                       << "llamacpp:benefit_reason_total{backend=\"" << backend
                       << "\",reason=\"insufficient_evidence\"} " << value.insufficient_evidence_decisions << "\n"
                       << "llamacpp:benefit_safety_fallback_total{backend=\"" << backend << "\"} "
                       << value.safety_fallbacks << "\n"
                       << "llamacpp:benefit_drift_total{backend=\"" << backend << "\"} "
                       << value.drift_events << "\n"
                       << "llamacpp:benefit_cooldown_remaining{backend=\"" << backend << "\"} "
                       << value.cooldown_remaining << "\n"
                       << "llamacpp:benefit_predicted_benefit_ms{backend=\"" << backend << "\"} "
                       << value.last_predicted_benefit_ms << "\n"
                       << "llamacpp:benefit_uncertainty_ms{backend=\"" << backend << "\"} "
                       << value.last_uncertainty_ms << "\n";
            static constexpr const char * choose_duration_labels[] = {
                "1", "2", "5", "10", "20", "50", "100", "250", "+Inf",
            };
            uint64_t cumulative = 0;
            for (size_t i = 0; i < server_benefit_choose_duration_bucket_count; ++i) {
                cumulative += value.choose_duration_us_buckets[i];
                prometheus << "llamacpp:benefit_choose_duration_us_bucket{backend=\""
                           << backend << "\",le=\"" << choose_duration_labels[i]
                           << "\"} " << cumulative << "\n";
            }
            prometheus << "llamacpp:benefit_choose_duration_us_count{backend=\""
                       << backend << "\"} " << cumulative << "\n"
                       << "llamacpp:benefit_choose_duration_us_sum{backend=\""
                       << backend << "\"} " << value.choose_duration_us_sum << "\n";
        };
        prometheus << "# HELP llamacpp:benefit_decisions_total Conservative benefit-policy decisions.\n"
                   << "# TYPE llamacpp:benefit_decisions_total counter\n"
                   << "# HELP llamacpp:benefit_observations_total Executed decisions learned by the policy.\n"
                   << "# TYPE llamacpp:benefit_observations_total counter\n"
                   << "# HELP llamacpp:benefit_exploration_total Bounded CacheFlow cold-start probes.\n"
                   << "# TYPE llamacpp:benefit_exploration_total counter\n"
                   << "# HELP llamacpp:benefit_reason_total Explainable learned-policy decisions by reason.\n"
                   << "# TYPE llamacpp:benefit_reason_total counter\n"
                   << "# HELP llamacpp:benefit_safety_fallback_total High-pressure upstream fallbacks.\n"
                   << "# TYPE llamacpp:benefit_safety_fallback_total counter\n"
                   << "# HELP llamacpp:benefit_drift_total Model-drift cooldown entries.\n"
                   << "# TYPE llamacpp:benefit_drift_total counter\n"
                   << "# HELP llamacpp:benefit_cooldown_remaining Remaining safe-mode decisions.\n"
                   << "# TYPE llamacpp:benefit_cooldown_remaining gauge\n"
                   << "# HELP llamacpp:benefit_predicted_benefit_ms Last contextual upstream minus CacheFlow predicted cost.\n"
                   << "# TYPE llamacpp:benefit_predicted_benefit_ms gauge\n"
                   << "# HELP llamacpp:benefit_uncertainty_ms Last joint confidence radius and safety margin.\n"
                   << "# TYPE llamacpp:benefit_uncertainty_ms gauge\n"
                   << "# HELP llamacpp:benefit_choose_duration_us Online policy choose latency in microseconds.\n"
                   << "# TYPE llamacpp:benefit_choose_duration_us histogram\n";
        append_benefit("cpu", res_task->benefit_cpu);
        append_benefit("cuda", res_task->benefit_cuda);

        const auto & kv_actions = res_task->kv_action_policy;
        prometheus << "# HELP llamacpp:kv_action_decisions_total Unified KV execution decisions by selected action.\n"
                   << "# TYPE llamacpp:kv_action_decisions_total counter\n";
        for (size_t i = 0; i < server_kv_action_count; ++i) {
            prometheus << "llamacpp:kv_action_decisions_total{action=\""
                       << server_kv_action_name(static_cast<server_kv_action>(i))
                       << "\"} " << kv_actions.action_decisions[i] << "\n";
        }
        prometheus << "# HELP llamacpp:kv_action_decision_reasons_total Unified KV decisions by reason.\n"
                   << "# TYPE llamacpp:kv_action_decision_reasons_total counter\n";
        for (size_t i = 0; i < server_kv_action_reason_count; ++i) {
            prometheus << "llamacpp:kv_action_decision_reasons_total{reason=\""
                       << server_kv_action_reason_name(static_cast<server_kv_action_reason>(i))
                       << "\"} " << kv_actions.reason_decisions[i] << "\n";
        }
        prometheus << "# HELP llamacpp:kv_action_decisions_by_reason_total Unified KV decisions by action and reason.\n"
                   << "# TYPE llamacpp:kv_action_decisions_by_reason_total counter\n";
        for (size_t action = 0; action < server_kv_action_count; ++action) {
            for (size_t reason = 0; reason < server_kv_action_reason_count; ++reason) {
                prometheus << "llamacpp:kv_action_decisions_by_reason_total{action=\""
                           << server_kv_action_name(static_cast<server_kv_action>(action))
                           << "\",reason=\""
                           << server_kv_action_reason_name(static_cast<server_kv_action_reason>(reason))
                           << "\"} " << kv_actions.action_reason_decisions[action][reason] << "\n";
            }
        }
        prometheus << "# HELP llamacpp:kv_action_observations_total Completed action observations by executed action.\n"
                   << "# TYPE llamacpp:kv_action_observations_total counter\n";
        for (size_t i = 0; i < server_kv_action_count; ++i) {
            prometheus << "llamacpp:kv_action_observations_total{action=\""
                       << server_kv_action_name(static_cast<server_kv_action>(i))
                       << "\"} " << kv_actions.observations[i] << "\n";
        }
        prometheus << "# HELP llamacpp:kv_action_failures_total Failed executions by selected action.\n"
                   << "# TYPE llamacpp:kv_action_failures_total counter\n";
        for (size_t i = 0; i < server_kv_action_count; ++i) {
            prometheus << "llamacpp:kv_action_failures_total{action=\""
                       << server_kv_action_name(static_cast<server_kv_action>(i))
                       << "\"} " << kv_actions.action_failures[i] << "\n";
        }
        prometheus << "# HELP llamacpp:kv_action_observation_seconds_total Complete internal action cost by executed action.\n"
                   << "# TYPE llamacpp:kv_action_observation_seconds_total counter\n";
        for (size_t i = 0; i < server_kv_action_count; ++i) {
            prometheus << "llamacpp:kv_action_observation_seconds_total{action=\""
                       << server_kv_action_name(static_cast<server_kv_action>(i))
                       << "\"} " << kv_actions.observation_total_milliseconds[i] / 1.e3 << "\n";
        }
        prometheus << "# HELP llamacpp:kv_action_last_model_feature Last normalized production model feature.\n"
                   << "# TYPE llamacpp:kv_action_last_model_feature gauge\n";
        for (size_t i = 0; i < server_kv_action_feature_count; ++i) {
            prometheus << "llamacpp:kv_action_last_model_feature{index=\"" << i
                       << "\"} " << kv_actions.last_model_features[i] << "\n";
        }
        prometheus << "# HELP llamacpp:kv_action_safe_fallback_total Decisions conservatively executed through H0.\n"
                   << "# TYPE llamacpp:kv_action_safe_fallback_total counter\n"
                   << "llamacpp:kv_action_safe_fallback_total " << kv_actions.safe_fallbacks << "\n"
                   << "# HELP llamacpp:kv_action_invalid_features_total Invalid feature snapshots rejected before ranking.\n"
                   << "# TYPE llamacpp:kv_action_invalid_features_total counter\n"
                   << "llamacpp:kv_action_invalid_features_total " << kv_actions.invalid_feature_fallbacks << "\n"
                   << "# HELP llamacpp:kv_action_cold_start_total Decisions kept on H0 because action models lacked samples.\n"
                   << "# TYPE llamacpp:kv_action_cold_start_total counter\n"
                   << "llamacpp:kv_action_cold_start_total " << kv_actions.cold_start_fallbacks << "\n"
                   << "# HELP llamacpp:kv_action_uncertainty_fallback_total Decisions kept on H0 because confidence bounds overlapped.\n"
                   << "# TYPE llamacpp:kv_action_uncertainty_fallback_total counter\n"
                   << "llamacpp:kv_action_uncertainty_fallback_total " << kv_actions.uncertainty_fallbacks << "\n"
                   << "# HELP llamacpp:kv_action_shadow_total Learned recommendations observed while H0 remained executed.\n"
                   << "# TYPE llamacpp:kv_action_shadow_total counter\n"
                   << "llamacpp:kv_action_shadow_total " << kv_actions.shadow_decisions << "\n"
                   << "# HELP llamacpp:kv_action_decision_seconds_total CPU time spent in the decision module.\n"
                   << "# TYPE llamacpp:kv_action_decision_seconds_total counter\n"
                   << "llamacpp:kv_action_decision_seconds_total "
                   << kv_actions.total_decision_nanoseconds / 1.e9 << "\n"
                   << "# HELP llamacpp:kv_action_decision_seconds_max Maximum observed decision time.\n"
                   << "# TYPE llamacpp:kv_action_decision_seconds_max gauge\n"
                   << "llamacpp:kv_action_decision_seconds_max "
                   << kv_actions.maximum_decision_nanoseconds / 1.e9 << "\n";

        const auto & llama_kv_status = res_task->kv_execution_status;
        const auto cuda_paged_stats = server_cuda_paged_fattn_stats();
        prometheus << "# HELP llamacpp:paged_decode_calls_total Decode calls that entered the production paged graph.\n"
                   << "# TYPE llamacpp:paged_decode_calls_total counter\n"
                   << "llamacpp:paged_decode_calls_total " << llama_kv_status.paged_decode_calls << "\n"
                   << "# HELP llamacpp:paged_decode_fallbacks_total Paged requests rejected before KV mutation.\n"
                   << "# TYPE llamacpp:paged_decode_fallbacks_total counter\n"
                   << "llamacpp:paged_decode_fallbacks_total " << llama_kv_status.paged_decode_fallbacks << "\n"
                   << "# HELP llamacpp:paged_decode_sequences_total Sequences processed by production Paged graphs.\n"
                   << "# TYPE llamacpp:paged_decode_sequences_total counter\n"
                   << "llamacpp:paged_decode_sequences_total " << llama_kv_status.paged_decode_sequences << "\n"
                   << "# HELP llamacpp:paged_decode_max_batch Maximum sequence count in one production Paged graph.\n"
                   << "# TYPE llamacpp:paged_decode_max_batch gauge\n"
                   << "llamacpp:paged_decode_max_batch " << llama_kv_status.paged_decode_max_batch << "\n"
                   << "# HELP llamacpp:paged_decode_contiguous_fastpath_calls_total Paged actions executed by the upstream contiguous attention fast path.\n"
                   << "# TYPE llamacpp:paged_decode_contiguous_fastpath_calls_total counter\n"
                   << "llamacpp:paged_decode_contiguous_fastpath_calls_total "
                   << llama_kv_status.paged_decode_contiguous_fastpath_calls << "\n"
                   << "# HELP llamacpp:paged_decode_contiguous_fastpath_sequences_total Sequences executed by the Paged contiguous fast path.\n"
                   << "# TYPE llamacpp:paged_decode_contiguous_fastpath_sequences_total counter\n"
                   << "llamacpp:paged_decode_contiguous_fastpath_sequences_total "
                   << llama_kv_status.paged_decode_contiguous_fastpath_sequences << "\n"
                   << "# HELP llamacpp:paged_decode_last_fallback_reason Last fail-closed reason code (0 means none).\n"
                   << "# TYPE llamacpp:paged_decode_last_fallback_reason gauge\n"
                   << "llamacpp:paged_decode_last_fallback_reason "
                   << static_cast<int32_t>(llama_kv_status.fallback_reason) << "\n"
                   << "# HELP llamacpp:paged_decode_cuda_dispatches_total Paged Flash Attention layer operators dispatched by CUDA.\n"
                   << "# TYPE llamacpp:paged_decode_cuda_dispatches_total counter\n"
                   << "llamacpp:paged_decode_cuda_dispatches_total " << cuda_paged_stats.first << "\n"
                   << "# HELP llamacpp:paged_decode_cuda_sequences_total Sequence-layer pairs dispatched by CUDA Paged Flash Attention.\n"
                   << "# TYPE llamacpp:paged_decode_cuda_sequences_total counter\n"
                   << "llamacpp:paged_decode_cuda_sequences_total " << cuda_paged_stats.second << "\n";

        const auto & checkpoint = res_task->benefit_persistence;
        prometheus << "# HELP llamacpp:benefit_checkpoint_restore_total Benefit checkpoint startup outcomes.\n"
                   << "# TYPE llamacpp:benefit_checkpoint_restore_total counter\n"
                   << "llamacpp:benefit_checkpoint_restore_total{result=\"restored\"} " << checkpoint.restored << "\n"
                   << "llamacpp:benefit_checkpoint_restore_total{result=\"incompatible\"} " << checkpoint.incompatible << "\n"
                   << "llamacpp:benefit_checkpoint_restore_total{result=\"failed\"} " << checkpoint.restore_failures << "\n"
                   << "# HELP llamacpp:benefit_checkpoint_save_total Benefit checkpoint asynchronous save outcomes.\n"
                   << "# TYPE llamacpp:benefit_checkpoint_save_total counter\n"
                   << "llamacpp:benefit_checkpoint_save_total{result=\"completed\"} " << checkpoint.saves_completed << "\n"
                   << "llamacpp:benefit_checkpoint_save_total{result=\"failed\"} " << checkpoint.save_failures << "\n"
                   << "# HELP llamacpp:benefit_checkpoint_enqueued_total Policy snapshots submitted to the bounded writer.\n"
                   << "# TYPE llamacpp:benefit_checkpoint_enqueued_total counter\n"
                   << "llamacpp:benefit_checkpoint_enqueued_total " << checkpoint.checkpoints_enqueued << "\n"
                   << "# HELP llamacpp:benefit_checkpoint_coalesced_total Obsolete pending snapshots replaced by a newer state.\n"
                   << "# TYPE llamacpp:benefit_checkpoint_coalesced_total counter\n"
                   << "llamacpp:benefit_checkpoint_coalesced_total " << checkpoint.checkpoints_coalesced << "\n"
                   << "# HELP llamacpp:benefit_checkpoint_pending Checkpoint currently queued or being written.\n"
                   << "# TYPE llamacpp:benefit_checkpoint_pending gauge\n"
                   << "llamacpp:benefit_checkpoint_pending " << checkpoint.pending << "\n";

        prometheus << "# HELP llamacpp:speculation_disabled_total Adaptive speculation cooldown entries by reason.\n"
                   << "# TYPE llamacpp:speculation_disabled_total counter\n"
                   << "llamacpp:speculation_disabled_total{reason=\"low_acceptance\"} "
                   << res_task->n_speculation_disabled_low_acceptance_total << "\n";

        auto append_histogram = [&prometheus](const char * name, const char * help,
                const server_latency_histogram & histogram) {
            prometheus << "# HELP llamacpp:" << name << " " << help << "\n"
                       << "# TYPE llamacpp:" << name << " histogram\n";
            uint64_t cumulative = 0;
            for (size_t index = 0; index < histogram.buckets.size(); ++index) {
                cumulative += histogram.buckets[index];
                prometheus << "llamacpp:" << name << "_bucket{le=\"";
                if (index < server_latency_histogram::bounds.size()) {
                    prometheus << server_latency_histogram::bounds[index];
                } else {
                    prometheus << "+Inf";
                }
                prometheus << "\"} " << cumulative << "\n";
            }
            prometheus << "llamacpp:" << name << "_sum " << histogram.sum << "\n"
                       << "llamacpp:" << name << "_count " << histogram.count << "\n";
        };
        append_histogram("time_to_first_token_seconds",
                "Request queue plus prompt time to first sampled token.", res_task->time_to_first_token);
        append_histogram("time_per_output_token_seconds",
                "Mean inter-token generation time per completed request.", res_task->time_per_output_token);
        append_histogram("request_latency_seconds",
                "End-to-end server request latency.", res_task->request_latency);
        append_histogram("request_queue_seconds",
                "Time from enqueue until prompt processing starts.", res_task->request_queue);

        res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->t_start);
        res->content_type = "text/plain; version=0.0.4";
        res->status = 200;
        res->data = prometheus.str();
        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto * res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->slots_data);
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        task_params tparams;
        tparams.sampling = params.sampling;
        json default_generation_settings_for_props = json {
            { "params", tparams.to_json(true) },
            { "n_ctx",  meta->slot_n_ctx },
        };

        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        std::string tmpl_tools   = common_chat_templates_source(meta->chat_params.tmpls.get(), "tool_use");

        json props = {
            { "default_generation_settings", default_generation_settings_for_props },
            { "total_slots",                 params.n_parallel },
            { "model_alias",                 meta->model_name },
            { "model_path",                  meta->model_path },
            { "modalities",                  json {
                {"vision", meta->has_inp_image},
                {"video",  meta->has_inp_video},
                {"audio",  meta->has_inp_audio},
            } },
            { "media_marker",                get_media_marker() },
            { "endpoint_slots",              params.endpoint_slots },
            { "endpoint_props",              params.endpoint_props },
            { "endpoint_metrics",            params.endpoint_metrics },
            // New keys
            { "ui",                           params.ui },
            { "ui_settings",                  meta->json_ui_settings },
            // Deprecated: use ui/ui_settings instead (kept for backward compat)
            { "webui",                        params.webui },
            { "webui_settings",               meta->json_webui_settings },
            { "chat_template",               tmpl_default },
            { "chat_template_caps",          meta->chat_template_caps },
            { "bos_token",                   meta->bos_token_str },
            { "eos_token",                   meta->eos_token_str },
            { "build_info",                  meta->build_info },
            { "is_sleeping",                 queue_tasks.is_sleeping() },
            { "cors_proxy_enabled",          params.ui_mcp_proxy || params.webui_mcp_proxy },
        };
        if (params.use_jinja) {
            if (!tmpl_tools.empty()) {
                props["chat_template_tool_use"] = tmpl_tools;
            }
        }
        res->ok(props);
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        json models = {
            {"models", {
                {
                    {"name",  meta->model_name},
                    {"model", meta->model_name},
                    {"modified_at", ""},
                    {"size", ""},
                    {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                    {"type", "model"},
                    {"description", ""},
                    {"tags", {""}},
                    {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})},
                    {"parameters", ""},
                    {"details", {
                        {"parent_model", ""},
                        {"format", "gguf"},
                        {"family", ""},
                        {"families", {""}},
                        {"parameter_size", ""},
                        {"quantization_level", ""}
                    }}
                }
            }},
            {"object", "list"},
            {"data", {
                get_model_info(),
            }}
        };

        res->ok(models);
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i]);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

json server_routes::get_model_info() const {
    return json {
        {"id",       meta->model_name},
        {"aliases",  meta->model_aliases},
        {"tags",     meta->model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta->model_vocab_type},
            {"n_vocab",     meta->model_vocab_n_tokens},
            {"n_ctx",       meta->slot_n_ctx},
            {"n_ctx_train", meta->model_n_ctx_train},
            {"n_embd",      meta->model_n_embd_inp},
            {"n_params",    meta->model_n_params},
            {"size",        meta->model_size},
        }},
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(mctx, prompt.get<std::string>(), files, true).size();
    } else {
        n_tokens = tokenize_mixed(vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}
