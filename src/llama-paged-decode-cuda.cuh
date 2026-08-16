#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

struct llama_paged_decode_config {
    uint32_t batch_size = 0;
    uint32_t query_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t page_size = 0;
    uint32_t max_pages_per_sequence = 0;
    uint32_t physical_pages = 0;
    float scale = 0.0f;
};

struct llama_paged_decode_plan;

// Restricted prototype contract: FP16 Q/K/V, FP32 output, GQA, head_dim in
// {64,128}, page_size=16, and one decode token per sequence. Unsupported configurations
// fail closed; production fallback selection belongs to the serving runtime.
bool llama_paged_decode_supported(
        const llama_paged_decode_config & config,
        const char ** reason = nullptr);

cudaError_t llama_paged_decode_plan_create(
        const llama_paged_decode_config & config,
        const uint32_t * host_page_table,
        const uint32_t * host_context_lengths,
        llama_paged_decode_plan ** plan);

void llama_paged_decode_plan_destroy(llama_paged_decode_plan * plan);

cudaError_t llama_paged_decode_launch_paged(
        const llama_paged_decode_plan * plan,
        const uint16_t * query,
        const uint16_t * paged_k,
        const uint16_t * paged_v,
        float * output,
        cudaStream_t stream);

// K2 keeps the same contract and output layout as K1. The selected T2 tile
// owns two query heads, so each GQA KV vector is loaded once per tile (four
// times per seven-head group rather than seven times in K1).
cudaError_t llama_paged_decode_launch_paged_k2(
        const llama_paged_decode_plan * plan,
        const uint16_t * query,
        const uint16_t * paged_k,
        const uint16_t * paged_v,
        float * output,
        cudaStream_t stream);

// Development seam for the preregistered K2 tile ablation. Production uses
// the fixed tile selected before its confirmatory service experiment.
cudaError_t llama_paged_decode_launch_paged_k2_tile(
        const llama_paged_decode_plan * plan,
        const uint16_t * query,
        const uint16_t * paged_k,
        const uint16_t * paged_v,
        float * output,
        cudaStream_t stream,
        uint32_t query_head_tile);

cudaError_t llama_paged_decode_launch_contiguous(
        const llama_paged_decode_plan * plan,
        const uint16_t * query,
        const uint16_t * contiguous_k,
        const uint16_t * contiguous_v,
        float * output,
        cudaStream_t stream);

const llama_paged_decode_config & llama_paged_decode_plan_config(
        const llama_paged_decode_plan * plan);
