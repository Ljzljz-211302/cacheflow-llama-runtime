#include "llama-paged-decode-cuda.cuh"

#include <cuda_fp16.h>

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <new>

struct llama_paged_decode_plan {
    llama_paged_decode_config config{};
    uint32_t * page_table = nullptr;
    uint32_t * context_lengths = nullptr;
};

namespace {

constexpr uint32_t maximum_head_dim = 128;
constexpr uint32_t supported_page_size = 16;
constexpr uint32_t k2_gqa_group_size = 7;

bool reject(const char * message, const char ** reason) {
    if (reason) *reason = message;
    return false;
}

template<bool paged>
__global__ void paged_decode_attention(
        llama_paged_decode_config config,
        const uint32_t * page_table,
        const uint32_t * context_lengths,
        const __half * query,
        const __half * k,
        const __half * v,
        float * output) {
    const uint32_t sequence = blockIdx.x / config.query_heads;
    const uint32_t query_head = blockIdx.x % config.query_heads;
    const uint32_t dim = threadIdx.x;
    const uint32_t group_size = config.query_heads / config.kv_heads;
    const uint32_t kv_head = query_head / group_size;
    const uint32_t context = context_lengths[sequence];
    const size_t query_index = ((size_t) sequence * config.query_heads + query_head) *
            config.head_dim + dim;
    const float q = __half2float(query[query_index]);
    float accumulator = 0.0f;

    __shared__ float reduction[maximum_head_dim];
    // running max, running denominator, previous-scale alpha, new-score beta
    __shared__ float online_state[4];
    if (dim == 0) {
        online_state[0] = -FLT_MAX;
        online_state[1] = 0.0f;
    }
    __syncthreads();

    for (uint32_t token = 0; token < context; ++token) {
        size_t token_base = 0;
        if constexpr (paged) {
            const uint32_t logical_page = token / config.page_size;
            const uint32_t page_offset = token % config.page_size;
            const uint32_t physical_page = page_table[
                    (size_t) sequence * config.max_pages_per_sequence + logical_page];
            token_base = ((size_t) physical_page * config.page_size + page_offset) *
                    config.kv_heads * config.head_dim;
        } else {
            const uint32_t max_context = config.max_pages_per_sequence * config.page_size;
            token_base = ((size_t) sequence * max_context + token) *
                    config.kv_heads * config.head_dim;
        }
        const size_t kv_index = token_base + (size_t) kv_head * config.head_dim + dim;
        reduction[dim] = q * __half2float(k[kv_index]);
        __syncthreads();
        for (uint32_t stride = config.head_dim / 2; stride > 0; stride /= 2) {
            if (dim < stride) reduction[dim] += reduction[dim + stride];
            __syncthreads();
        }
        if (dim == 0) {
            const float score = reduction[0] * config.scale;
            const float next_max = fmaxf(online_state[0], score);
            const float alpha = online_state[1] == 0.0f ? 0.0f :
                    expf(online_state[0] - next_max);
            const float beta = expf(score - next_max);
            online_state[0] = next_max;
            online_state[1] = online_state[1] * alpha + beta;
            online_state[2] = alpha;
            online_state[3] = beta;
        }
        __syncthreads();
        accumulator = accumulator * online_state[2] +
                online_state[3] * __half2float(v[kv_index]);
        __syncthreads();
    }
    output[query_index] = accumulator / online_state[1];
}

template<uint32_t head_dim, uint32_t query_head_tile>
__global__ void paged_decode_attention_k2(
        llama_paged_decode_config config,
        const uint32_t * page_table,
        const uint32_t * context_lengths,
        const __half * query,
        const __half * k,
        const __half * v,
        float * output) {
    const uint32_t tiles_per_kv =
            (k2_gqa_group_size + query_head_tile - 1) / query_head_tile;
    const uint32_t tile = blockIdx.x % tiles_per_kv;
    const uint32_t sequence_kv = blockIdx.x / tiles_per_kv;
    const uint32_t sequence = sequence_kv / config.kv_heads;
    const uint32_t kv_head = sequence_kv % config.kv_heads;
    const uint32_t tile_head = threadIdx.x / head_dim;
    const uint32_t dim = threadIdx.x % head_dim;
    const uint32_t context = context_lengths[sequence];
    const uint32_t group_head_begin = tile * query_head_tile;
    const uint32_t active_heads = min(
            query_head_tile, k2_gqa_group_size - group_head_begin);
    const bool active = tile_head < active_heads;

    float q = 0.0f;
    float accumulator = 0.0f;
    if (active) {
        const uint32_t query_head =
                kv_head * k2_gqa_group_size + group_head_begin + tile_head;
        const size_t query_index = ((size_t) sequence * config.query_heads + query_head) *
                head_dim + dim;
        q = __half2float(query[query_index]);
    }

    __shared__ float reduction[query_head_tile * head_dim];
    __shared__ float key_tile[head_dim];
    __shared__ float value_tile[head_dim];
    // Per query head: running max, denominator, previous-scale alpha, score beta.
    __shared__ float online_state[query_head_tile * 4];
    if (active && dim == 0) {
        online_state[tile_head * 4] = -FLT_MAX;
        online_state[tile_head * 4 + 1] = 0.0f;
    }
    __syncthreads();

    for (uint32_t token = 0; token < context; ++token) {
        const uint32_t logical_page = token / config.page_size;
        const uint32_t page_offset = token % config.page_size;
        const uint32_t physical_page = page_table[
                (size_t) sequence * config.max_pages_per_sequence + logical_page];
        const size_t token_base = ((size_t) physical_page * config.page_size + page_offset) *
                config.kv_heads * head_dim;
        const size_t kv_index = token_base + (size_t) kv_head * head_dim + dim;
        if (threadIdx.x < head_dim) {
            key_tile[dim] = __half2float(k[kv_index]);
            value_tile[dim] = __half2float(v[kv_index]);
        }
        __syncthreads();
        if (active) {
            reduction[tile_head * head_dim + dim] = q * key_tile[dim];
        }
        __syncthreads();
        for (uint32_t stride = head_dim / 2; stride > 0; stride /= 2) {
            if (active && dim < stride) {
                reduction[tile_head * head_dim + dim] +=
                        reduction[tile_head * head_dim + dim + stride];
            }
            __syncthreads();
        }
        if (active && dim == 0) {
            float * state = online_state + tile_head * 4;
            const float score = reduction[tile_head * head_dim] * config.scale;
            const float next_max = fmaxf(state[0], score);
            const float alpha = state[1] == 0.0f ? 0.0f : expf(state[0] - next_max);
            const float beta = expf(score - next_max);
            state[0] = next_max;
            state[1] = state[1] * alpha + beta;
            state[2] = alpha;
            state[3] = beta;
        }
        __syncthreads();
        if (active) {
            const float * state = online_state + tile_head * 4;
            accumulator = accumulator * state[2] + state[3] * value_tile[dim];
        }
        __syncthreads();
    }

    if (active) {
        const uint32_t query_head =
                kv_head * k2_gqa_group_size + group_head_begin + tile_head;
        const size_t output_index = ((size_t) sequence * config.query_heads + query_head) *
                head_dim + dim;
        output[output_index] = accumulator / online_state[tile_head * 4 + 1];
    }
}

template<uint32_t head_dim, uint32_t query_head_tile>
cudaError_t launch_k2(
        const llama_paged_decode_plan * plan,
        const uint16_t * query,
        const uint16_t * paged_k,
        const uint16_t * paged_v,
        float * output,
        cudaStream_t stream) {
    const unsigned tiles_per_kv =
            (k2_gqa_group_size + query_head_tile - 1) / query_head_tile;
    const unsigned blocks =
            plan->config.batch_size * plan->config.kv_heads * tiles_per_kv;
    paged_decode_attention_k2<head_dim, query_head_tile><<<
            blocks, head_dim * query_head_tile, 0, stream>>>(
            plan->config, plan->page_table, plan->context_lengths,
            reinterpret_cast<const __half *>(query),
            reinterpret_cast<const __half *>(paged_k),
            reinterpret_cast<const __half *>(paged_v), output);
    return cudaGetLastError();
}

cudaError_t launch(
        const llama_paged_decode_plan * plan,
        const uint16_t * query,
        const uint16_t * k,
        const uint16_t * v,
        float * output,
        cudaStream_t stream,
        bool paged) {
    if (!plan || !query || !k || !v || !output) return cudaErrorInvalidValue;
    const unsigned blocks = plan->config.batch_size * plan->config.query_heads;
    if (paged) {
        paged_decode_attention<true><<<blocks, plan->config.head_dim, 0, stream>>>(
                plan->config, plan->page_table, plan->context_lengths,
                reinterpret_cast<const __half *>(query),
                reinterpret_cast<const __half *>(k),
                reinterpret_cast<const __half *>(v), output);
    } else {
        paged_decode_attention<false><<<blocks, plan->config.head_dim, 0, stream>>>(
                plan->config, plan->page_table, plan->context_lengths,
                reinterpret_cast<const __half *>(query),
                reinterpret_cast<const __half *>(k),
                reinterpret_cast<const __half *>(v), output);
    }
    return cudaGetLastError();
}

} // namespace

bool llama_paged_decode_supported(
        const llama_paged_decode_config & config,
        const char ** reason) {
    if (reason) *reason = nullptr;
    if (config.batch_size == 0) return reject("batch_size must be positive", reason);
    if (config.query_heads == 0 || config.kv_heads == 0 ||
            config.query_heads % config.kv_heads != 0) {
        return reject("query_heads must be divisible by kv_heads", reason);
    }
    if (config.head_dim != 64 && config.head_dim != 128) {
        return reject("only head_dim in {64,128} is supported", reason);
    }
    if (config.page_size != supported_page_size) {
        return reject("only page_size=16 is supported", reason);
    }
    if (config.max_pages_per_sequence == 0 || config.physical_pages == 0) {
        return reject("page counts must be positive", reason);
    }
    if (!std::isfinite(config.scale) || config.scale <= 0.0f) {
        return reject("attention scale must be finite and positive", reason);
    }
    return true;
}

cudaError_t llama_paged_decode_plan_create(
        const llama_paged_decode_config & config,
        const uint32_t * host_page_table,
        const uint32_t * host_context_lengths,
        llama_paged_decode_plan ** output) {
    if (!output) return cudaErrorInvalidValue;
    *output = nullptr;
    if (!host_page_table || !host_context_lengths ||
            !llama_paged_decode_supported(config)) {
        return cudaErrorInvalidValue;
    }
    for (uint32_t sequence = 0; sequence < config.batch_size; ++sequence) {
        const uint32_t context = host_context_lengths[sequence];
        if (context == 0 || context > config.max_pages_per_sequence * config.page_size) {
            return cudaErrorInvalidValue;
        }
        const uint32_t used_pages = (context + config.page_size - 1) / config.page_size;
        for (uint32_t page = 0; page < used_pages; ++page) {
            if (host_page_table[(size_t) sequence * config.max_pages_per_sequence + page] >=
                    config.physical_pages) {
                return cudaErrorInvalidValue;
            }
        }
    }
    auto * plan = new (std::nothrow) llama_paged_decode_plan;
    if (!plan) return cudaErrorMemoryAllocation;
    plan->config = config;
    const size_t table_bytes = (size_t) config.batch_size *
            config.max_pages_per_sequence * sizeof(uint32_t);
    const size_t context_bytes = (size_t) config.batch_size * sizeof(uint32_t);
    cudaError_t status = cudaMalloc((void **) &plan->page_table, table_bytes);
    if (status == cudaSuccess) {
        status = cudaMalloc((void **) &plan->context_lengths, context_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(plan->page_table, host_page_table, table_bytes,
                cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(plan->context_lengths, host_context_lengths, context_bytes,
                cudaMemcpyHostToDevice);
    }
    if (status != cudaSuccess) {
        if (plan->context_lengths) cudaFree(plan->context_lengths);
        if (plan->page_table) cudaFree(plan->page_table);
        delete plan;
        return status;
    }
    *output = plan;
    return cudaSuccess;
}

void llama_paged_decode_plan_destroy(llama_paged_decode_plan * plan) {
    if (!plan) return;
    if (plan->context_lengths) cudaFree(plan->context_lengths);
    if (plan->page_table) cudaFree(plan->page_table);
    delete plan;
}

cudaError_t llama_paged_decode_launch_paged(
        const llama_paged_decode_plan * plan,
        const uint16_t * query,
        const uint16_t * paged_k,
        const uint16_t * paged_v,
        float * output,
        cudaStream_t stream) {
    return launch(plan, query, paged_k, paged_v, output, stream, true);
}

cudaError_t llama_paged_decode_launch_paged_k2(
        const llama_paged_decode_plan * plan,
        const uint16_t * query,
        const uint16_t * paged_k,
        const uint16_t * paged_v,
        float * output,
        cudaStream_t stream) {
    return llama_paged_decode_launch_paged_k2_tile(
            plan, query, paged_k, paged_v, output, stream, 2);
}

cudaError_t llama_paged_decode_launch_paged_k2_tile(
        const llama_paged_decode_plan * plan,
        const uint16_t * query,
        const uint16_t * paged_k,
        const uint16_t * paged_v,
        float * output,
        cudaStream_t stream,
        uint32_t query_head_tile) {
    if (!plan || !query || !paged_k || !paged_v || !output) return cudaErrorInvalidValue;
    if (plan->config.query_heads / plan->config.kv_heads != k2_gqa_group_size) {
        return cudaErrorInvalidValue;
    }
#define CACHEFLOW_K2_TILE(D, T) \
    launch_k2<D, T>(plan, query, paged_k, paged_v, output, stream)
    if (plan->config.head_dim == 64) {
        if (query_head_tile == 2) return CACHEFLOW_K2_TILE(64, 2);
        if (query_head_tile == 4) return CACHEFLOW_K2_TILE(64, 4);
        if (query_head_tile == 7) return CACHEFLOW_K2_TILE(64, 7);
    } else if (plan->config.head_dim == 128) {
        if (query_head_tile == 2) return CACHEFLOW_K2_TILE(128, 2);
        if (query_head_tile == 4) return CACHEFLOW_K2_TILE(128, 4);
        if (query_head_tile == 7) return CACHEFLOW_K2_TILE(128, 7);
    }
#undef CACHEFLOW_K2_TILE
    return cudaErrorInvalidValue;
}

cudaError_t llama_paged_decode_launch_contiguous(
        const llama_paged_decode_plan * plan,
        const uint16_t * query,
        const uint16_t * contiguous_k,
        const uint16_t * contiguous_v,
        float * output,
        cudaStream_t stream) {
    return launch(plan, query, contiguous_k, contiguous_v, output, stream, false);
}

const llama_paged_decode_config & llama_paged_decode_plan_config(
        const llama_paged_decode_plan * plan) {
    static const llama_paged_decode_config empty{};
    return plan ? plan->config : empty;
}
