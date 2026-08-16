#pragma once

#include "common.cuh"

#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <mma.h>
#include <type_traits>

static __device__ unsigned long long cacheflow_paged_dispatches = 0;
static __device__ unsigned long long cacheflow_paged_dispatched_sequences = 0;

void ggml_cuda_get_paged_fattn_stats(uint64_t * dispatches, uint64_t * sequences) {
    unsigned long long device_dispatches = 0;
    unsigned long long device_sequences = 0;
    GGML_ASSERT(cudaMemcpyFromSymbol(&device_dispatches, cacheflow_paged_dispatches,
            sizeof(device_dispatches)) == cudaSuccess);
    GGML_ASSERT(cudaMemcpyFromSymbol(&device_sequences, cacheflow_paged_dispatched_sequences,
            sizeof(device_sequences)) == cudaSuccess);
    *dispatches = device_dispatches;
    *sequences = device_sequences;
}

static __device__ __forceinline__ void cacheflow_record_paged_dispatch(int32_t batch_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(&cacheflow_paged_dispatches, 1ULL);
        atomicAdd(&cacheflow_paged_dispatched_sequences, (unsigned long long) batch_size);
    }
}

template<int head_dim, typename query_type>
static __global__ void cacheflow_paged_decode_fattn_k1(
        const char * query,
        const char * key,
        const char * value,
        const int32_t * block_table,
        const int32_t * context_lengths,
        char * output,
        uint64_t q_sequence_stride,
        uint64_t q_head_stride,
        uint64_t output_sequence_stride,
        uint64_t output_head_stride,
        uint64_t k_token_stride,
        uint64_t k_head_stride,
        uint64_t k_stream_stride,
        uint64_t v_token_stride,
        uint64_t v_head_stride,
        uint64_t v_stream_stride,
        int32_t query_heads,
        int32_t kv_heads,
        int32_t max_blocks,
        int32_t batch_size,
        int32_t page_size,
        float scale) {
    cacheflow_record_paged_dispatch(batch_size);
    const int32_t sequence = blockIdx.x / query_heads;
    const int32_t query_head = blockIdx.x % query_heads;
    if (sequence >= batch_size) return;
    const int32_t dim = threadIdx.x;
    const int32_t kv_head = query_head / (query_heads / kv_heads);
    const int32_t context = context_lengths[sequence];
    const int32_t * sequence_table = block_table + (uint64_t) sequence * max_blocks;
    const query_type * q = reinterpret_cast<const query_type *>(
            query + (uint64_t) sequence * q_sequence_stride + (uint64_t) query_head * q_head_stride);
    float accumulator = 0.0f;

    __shared__ float reduction[head_dim];
    __shared__ float online[4];
    if (dim == 0) {
        online[0] = -FLT_MAX;
        online[1] = 0.0f;
    }
    __syncthreads();

    for (int32_t token = 0; token < context; ++token) {
        const int32_t logical_block = token / page_size;
        if (logical_block >= max_blocks) return;
        const int32_t cell = sequence_table[logical_block] + token % page_size;
        const __half * k = reinterpret_cast<const __half *>(
                key + (uint64_t) cell * k_token_stride + (uint64_t) kv_head * k_head_stride +
                (uint64_t) sequence * k_stream_stride);
        const __half * v = reinterpret_cast<const __half *>(
                value + (uint64_t) cell * v_token_stride + (uint64_t) kv_head * v_head_stride +
                (uint64_t) sequence * v_stream_stride);
        float query_value;
        if constexpr (std::is_same<query_type, __half>::value) {
            query_value = __half2float(q[dim]);
        } else {
            query_value = q[dim];
        }
        reduction[dim] = query_value * __half2float(k[dim]);
        __syncthreads();
        for (int32_t stride = head_dim / 2; stride > 0; stride /= 2) {
            if (dim < stride) reduction[dim] += reduction[dim + stride];
            __syncthreads();
        }
        if (dim == 0) {
            const float score = reduction[0] * scale;
            const float next_max = fmaxf(online[0], score);
            const float alpha = online[1] == 0.0f ? 0.0f : expf(online[0] - next_max);
            const float beta = expf(score - next_max);
            online[0] = next_max;
            online[1] = online[1] * alpha + beta;
            online[2] = alpha;
            online[3] = beta;
        }
        __syncthreads();
        accumulator = accumulator * online[2] + online[3] * __half2float(v[dim]);
        __syncthreads();
    }
    *reinterpret_cast<float *>(output + (uint64_t) sequence * output_sequence_stride +
            (uint64_t) query_head * output_head_stride + (uint64_t) dim * sizeof(float)) =
            accumulator / online[1];
}

template<int head_dim, typename query_type>
static __global__ void cacheflow_paged_decode_fattn_k2_t2(
        const char * query,
        const char * key,
        const char * value,
        const int32_t * block_table,
        const int32_t * context_lengths,
        char * output,
        uint64_t q_sequence_stride,
        uint64_t q_head_stride,
        uint64_t output_sequence_stride,
        uint64_t output_head_stride,
        uint64_t k_token_stride,
        uint64_t k_head_stride,
        uint64_t k_stream_stride,
        uint64_t v_token_stride,
        uint64_t v_head_stride,
        uint64_t v_stream_stride,
        int32_t query_heads,
        int32_t kv_heads,
        int32_t max_blocks,
        int32_t batch_size,
        int32_t page_size,
        float scale,
        int32_t partition_tokens,
        int32_t partitions,
        float * partial) {
    cacheflow_record_paged_dispatch(batch_size);
    constexpr int32_t query_head_tile = 2;
    const int32_t group_size = query_heads / kv_heads;
    const int32_t tiles_per_kv = (group_size + query_head_tile - 1) / query_head_tile;
    const int32_t blocks_per_sequence = kv_heads * tiles_per_kv * partitions;
    const int32_t sequence = blockIdx.x / blocks_per_sequence;
    const int32_t sequence_block = blockIdx.x % blocks_per_sequence;
    const int32_t partition = sequence_block % partitions;
    const int32_t head_tile_index = sequence_block / partitions;
    if (sequence >= batch_size) return;
    const int32_t tile = head_tile_index % tiles_per_kv;
    const int32_t kv_head = head_tile_index / tiles_per_kv;
    static_assert(head_dim == 64, "K2-T2 one-warp heads currently target D64");
    const int32_t tile_head = threadIdx.x / 32;
    const int32_t lane = threadIdx.x % 32;
    const int32_t group_head_begin = tile * query_head_tile;
    const int32_t active_heads = min(query_head_tile, group_size - group_head_begin);
    const bool active = tile_head < active_heads;
    const int32_t query_head = kv_head * group_size + group_head_begin + tile_head;
    const int32_t context = context_lengths[sequence];
    const int32_t * sequence_table = block_table + (uint64_t) sequence * max_blocks;
    const int32_t partition_begin = partition * partition_tokens;
    const int32_t partition_end = min(context, partition_begin + partition_tokens);

    if (context <= 0 || (context + page_size - 1) / page_size > max_blocks) return;

    __shared__ float query_tile[query_head_tile][head_dim];
    // K is transposed in shared memory so a warp reading one dimension
    // across logical tokens avoids a 32-way bank conflict. V keeps the
    // token-major layout used by the output accumulation. These buffers are
    // a fixed 32-token tile: long contexts reuse them rather than scaling
    // shared memory with sequence length.
    __shared__ float key_by_dim[head_dim][32];
    __shared__ float value_by_token[32][head_dim];

    if (head_tile_index >= kv_heads * tiles_per_kv) return;

    if (partition_begin >= context) {
        if (active && partitions > 1) {
            const uint64_t partial_base =
                    (((uint64_t) sequence * query_heads + query_head) * partitions + partition) *
                    (head_dim + 2);
            if (lane == 0) {
                partial[partial_base] = -FLT_MAX;
                partial[partial_base + 1] = 0.0f;
            }
            partial[partial_base + 2 + lane] = 0.0f;
            partial[partial_base + 2 + lane + 32] = 0.0f;
        }
        return;
    }

    if (active) {
        const query_type * q = reinterpret_cast<const query_type *>(
                query + (uint64_t) sequence * q_sequence_stride +
                (uint64_t) query_head * q_head_stride);
        if constexpr (std::is_same<query_type, __half>::value) {
            query_tile[tile_head][lane] = __half2float(q[lane]);
            query_tile[tile_head][lane + 32] = __half2float(q[lane + 32]);
        } else {
            query_tile[tile_head][lane] = q[lane];
            query_tile[tile_head][lane + 32] = q[lane + 32];
        }
    }
    __syncthreads();

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float output_0 = 0.0f;
    float output_1 = 0.0f;

    for (int32_t tile_begin = partition_begin; tile_begin < partition_end; tile_begin += 32) {
        const int32_t tile_tokens = min(32, partition_end - tile_begin);
        for (int32_t item = threadIdx.x; item < tile_tokens * head_dim; item += blockDim.x) {
            const int32_t tile_token = item / head_dim;
            const int32_t dim = item % head_dim;
            const int32_t token = tile_begin + tile_token;
            const int32_t logical_block = token / page_size;
            const int32_t cell = sequence_table[logical_block] + token % page_size;
            const __half * k = reinterpret_cast<const __half *>(
                    key + (uint64_t) cell * k_token_stride +
                    (uint64_t) kv_head * k_head_stride +
                    (uint64_t) sequence * k_stream_stride);
            const __half * v = reinterpret_cast<const __half *>(
                    value + (uint64_t) cell * v_token_stride +
                    (uint64_t) kv_head * v_head_stride +
                    (uint64_t) sequence * v_stream_stride);
            key_by_dim[dim][tile_token] = __half2float(k[dim]);
            value_by_token[tile_token][dim] = __half2float(v[dim]);
        }
        __syncthreads();

        if (active) {
            float logit = -FLT_MAX;
            if (lane < tile_tokens) {
                logit = 0.0f;
#pragma unroll
                for (int32_t dim = 0; dim < head_dim; ++dim) {
                    logit += query_tile[tile_head][dim] * key_by_dim[dim][lane];
                }
                logit *= scale;
            }
            float tile_max = logit;
#pragma unroll
            for (int32_t offset = 16; offset > 0; offset /= 2) {
                tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
            }
            tile_max = __shfl_sync(0xffffffff, tile_max, 0);
            const float weight = lane < tile_tokens ? expf(logit - tile_max) : 0.0f;
            float tile_sum = weight;
#pragma unroll
            for (int32_t offset = 16; offset > 0; offset /= 2) {
                tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);
            }
            tile_sum = __shfl_sync(0xffffffff, tile_sum, 0);

            float tile_output_0 = 0.0f;
            float tile_output_1 = 0.0f;
            for (int32_t token = 0; token < tile_tokens; ++token) {
                const float token_weight = __shfl_sync(0xffffffff, weight, token);
                tile_output_0 += token_weight * value_by_token[token][lane];
                tile_output_1 += token_weight * value_by_token[token][lane + 32];
            }

            const float next_max = fmaxf(running_max, tile_max);
            const float previous_scale = running_sum == 0.0f ? 0.0f : expf(running_max - next_max);
            const float tile_scale = expf(tile_max - next_max);
            output_0 = output_0 * previous_scale + tile_output_0 * tile_scale;
            output_1 = output_1 * previous_scale + tile_output_1 * tile_scale;
            running_sum = running_sum * previous_scale + tile_sum * tile_scale;
            running_max = next_max;
        }
        __syncthreads();
    }
    if (active) {
        if (partitions == 1) {
            *reinterpret_cast<float *>(output + (uint64_t) sequence * output_sequence_stride +
                    (uint64_t) query_head * output_head_stride + (uint64_t) lane * sizeof(float)) =
                    output_0 / running_sum;
            *reinterpret_cast<float *>(output + (uint64_t) sequence * output_sequence_stride +
                    (uint64_t) query_head * output_head_stride +
                    (uint64_t) (lane + 32) * sizeof(float)) = output_1 / running_sum;
        } else {
            const uint64_t partial_base =
                    (((uint64_t) sequence * query_heads + query_head) * partitions + partition) *
                    (head_dim + 2);
            if (lane == 0) {
                partial[partial_base] = running_max;
                partial[partial_base + 1] = running_sum;
            }
            partial[partial_base + 2 + lane] = output_0;
            partial[partial_base + 2 + lane + 32] = output_1;
        }
    }
}

// K3 keeps K2's page-indirect, partitioned online-softmax contract, but
// replaces its scalar 32-token inner loop with the same basic execution
// shape used by a decode-optimized vector kernel: 128 threads cooperate on
// one KV head, two GQA query heads are evaluated together, and all D64 K/V
// traffic is moved as half2. Page translation remains outside the arithmetic
// loop, so non-contiguous physical pages do not force a contiguous remap.
template<typename query_type>
static __global__ void cacheflow_paged_decode_fattn_k3_vec_t2(
        const char * query,
        const char * key,
        const char * value,
        const int32_t * block_table,
        const int32_t * context_lengths,
        char * output,
        uint64_t q_sequence_stride,
        uint64_t q_head_stride,
        uint64_t output_sequence_stride,
        uint64_t output_head_stride,
        uint64_t k_token_stride,
        uint64_t k_head_stride,
        uint64_t k_stream_stride,
        uint64_t v_token_stride,
        uint64_t v_head_stride,
        uint64_t v_stream_stride,
        int32_t query_heads,
        int32_t kv_heads,
        int32_t max_blocks,
        int32_t batch_size,
        int32_t page_size,
        float scale,
        int32_t partition_tokens,
        int32_t partitions,
        float * partial) {
    cacheflow_record_paged_dispatch(batch_size);
    constexpr int32_t head_dim = 64;
    constexpr int32_t half2_dim = head_dim / 2;
    constexpr int32_t query_head_tile = 2;
    constexpr int32_t token_tile = 64;

    const int32_t group_size = query_heads / kv_heads;
    const int32_t tiles_per_kv = (group_size + query_head_tile - 1) / query_head_tile;
    const int32_t blocks_per_sequence = kv_heads * tiles_per_kv * partitions;
    const int32_t sequence = blockIdx.x / blocks_per_sequence;
    const int32_t sequence_block = blockIdx.x % blocks_per_sequence;
    const int32_t partition = sequence_block % partitions;
    const int32_t head_tile_index = sequence_block / partitions;
    if (sequence >= batch_size) return;
    if (head_tile_index >= kv_heads * tiles_per_kv) return;

    const int32_t tile = head_tile_index % tiles_per_kv;
    const int32_t kv_head = head_tile_index / tiles_per_kv;
    const int32_t head_slot = threadIdx.x / token_tile;
    const int32_t local_thread = threadIdx.x % token_tile;
    const int32_t lane = local_thread % 32;
    const int32_t group_head_begin = tile * query_head_tile;
    const int32_t active_heads = min(query_head_tile, group_size - group_head_begin);
    const bool active_head = head_slot < active_heads;
    const int32_t query_head = kv_head * group_size + group_head_begin + head_slot;
    const int32_t context = context_lengths[sequence];
    const int32_t * sequence_table = block_table + (uint64_t) sequence * max_blocks;
    const int32_t partition_begin = partition * partition_tokens;
    const int32_t partition_end = min(context, partition_begin + partition_tokens);

    if (context <= 0 || (context + page_size - 1) / page_size > max_blocks) return;

    __shared__ float2 query_pair[query_head_tile][half2_dim];
    __shared__ __half2 key_pair[half2_dim][token_tile];
    __shared__ __half2 value_pair[token_tile][half2_dim];
    __shared__ float logits[query_head_tile][token_tile];

    if (partition_begin >= context) {
        if (active_head && local_thread < half2_dim && partitions > 1) {
            const uint64_t partial_base =
                    (((uint64_t) sequence * query_heads + query_head) * partitions + partition) *
                    (head_dim + 2);
            if (local_thread == 0) {
                partial[partial_base] = -FLT_MAX;
                partial[partial_base + 1] = 0.0f;
            }
            partial[partial_base + 2 + 2 * local_thread] = 0.0f;
            partial[partial_base + 3 + 2 * local_thread] = 0.0f;
        }
        return;
    }

    if (active_head && local_thread < half2_dim) {
        const query_type * q = reinterpret_cast<const query_type *>(
                query + (uint64_t) sequence * q_sequence_stride +
                (uint64_t) query_head * q_head_stride);
        if constexpr (std::is_same<query_type, __half>::value) {
            query_pair[head_slot][local_thread] = __half22float2(
                    reinterpret_cast<const __half2 *>(q)[local_thread]);
        } else {
            query_pair[head_slot][local_thread] = make_float2(
                    q[2 * local_thread], q[2 * local_thread + 1]);
        }
    }
    __syncthreads();

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float2 output_pair = make_float2(0.0f, 0.0f);

    for (int32_t tile_begin = partition_begin; tile_begin < partition_end; tile_begin += token_tile) {
        const int32_t tile_tokens = min(token_tile, partition_end - tile_begin);
        for (int32_t item = threadIdx.x; item < tile_tokens * half2_dim; item += blockDim.x) {
            const int32_t tile_token = item / half2_dim;
            const int32_t dim_pair = item % half2_dim;
            const int32_t token = tile_begin + tile_token;
            const int32_t logical_block = token / page_size;
            const int32_t cell = sequence_table[logical_block] + token % page_size;
            const __half2 * k = reinterpret_cast<const __half2 *>(
                    key + (uint64_t) cell * k_token_stride +
                    (uint64_t) kv_head * k_head_stride +
                    (uint64_t) sequence * k_stream_stride);
            const __half2 * v = reinterpret_cast<const __half2 *>(
                    value + (uint64_t) cell * v_token_stride +
                    (uint64_t) kv_head * v_head_stride +
                    (uint64_t) sequence * v_stream_stride);
            key_pair[dim_pair][tile_token] = k[dim_pair];
            value_pair[tile_token][dim_pair] = v[dim_pair];
        }
        __syncthreads();

        if (active_head) {
            float logit = -FLT_MAX;
            if (local_thread < tile_tokens) {
                logit = 0.0f;
#pragma unroll
                for (int32_t dim_pair = 0; dim_pair < half2_dim; ++dim_pair) {
                    const float2 q2 = query_pair[head_slot][dim_pair];
                    const float2 k2 = __half22float2(key_pair[dim_pair][local_thread]);
                    logit += q2.x * k2.x + q2.y * k2.y;
                }
                logit *= scale;
            }
            logits[head_slot][local_thread] = logit;
        }
        __syncthreads();

        if (active_head && local_thread < half2_dim) {
            float tile_max = fmaxf(logits[head_slot][lane], logits[head_slot][lane + 32]);
#pragma unroll
            for (int32_t offset = 16; offset > 0; offset /= 2) {
                tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
            }
            tile_max = __shfl_sync(0xffffffff, tile_max, 0);

            const float weight_0 = lane < tile_tokens ?
                    expf(logits[head_slot][lane] - tile_max) : 0.0f;
            const float weight_1 = lane + 32 < tile_tokens ?
                    expf(logits[head_slot][lane + 32] - tile_max) : 0.0f;
            float tile_sum = weight_0 + weight_1;
#pragma unroll
            for (int32_t offset = 16; offset > 0; offset /= 2) {
                tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);
            }
            tile_sum = __shfl_sync(0xffffffff, tile_sum, 0);

            float2 tile_output = make_float2(0.0f, 0.0f);
            for (int32_t token = 0; token < tile_tokens; ++token) {
                const float weight = token < 32 ?
                        __shfl_sync(0xffffffff, weight_0, token) :
                        __shfl_sync(0xffffffff, weight_1, token - 32);
                const float2 v2 = __half22float2(value_pair[token][local_thread]);
                tile_output.x += weight * v2.x;
                tile_output.y += weight * v2.y;
            }

            const float next_max = fmaxf(running_max, tile_max);
            const float previous_scale = running_sum == 0.0f ? 0.0f : expf(running_max - next_max);
            const float tile_scale = expf(tile_max - next_max);
            output_pair.x = output_pair.x * previous_scale + tile_output.x * tile_scale;
            output_pair.y = output_pair.y * previous_scale + tile_output.y * tile_scale;
            running_sum = running_sum * previous_scale + tile_sum * tile_scale;
            running_max = next_max;
        }
        __syncthreads();
    }

    if (active_head && local_thread < half2_dim) {
        if (partitions == 1) {
            *reinterpret_cast<float *>(output + (uint64_t) sequence * output_sequence_stride +
                    (uint64_t) query_head * output_head_stride +
                    (uint64_t) (2 * local_thread) * sizeof(float)) = output_pair.x / running_sum;
            *reinterpret_cast<float *>(output + (uint64_t) sequence * output_sequence_stride +
                    (uint64_t) query_head * output_head_stride +
                    (uint64_t) (2 * local_thread + 1) * sizeof(float)) = output_pair.y / running_sum;
        } else {
            const uint64_t partial_base =
                    (((uint64_t) sequence * query_heads + query_head) * partitions + partition) *
                    (head_dim + 2);
            if (local_thread == 0) {
                partial[partial_base] = running_max;
                partial[partial_base + 1] = running_sum;
            }
            partial[partial_base + 2 + 2 * local_thread] = output_pair.x;
            partial[partial_base + 3 + 2 * local_thread] = output_pair.y;
        }
    }
}

// K4 maps one complete 7:1 GQA group to one CTA. Seven warps evaluate the
// seven query heads while sharing a single page-indirect K/V tile; the eighth
// warp participates in memory movement. This removes K3's fourfold K/V reload
// per KV head and makes the execution decomposition follow the model geometry
// instead of an arbitrary two-head tile.
template<typename query_type>
static __global__ void cacheflow_paged_decode_fattn_k4_gqa7(
        const char * query,
        const char * key,
        const char * value,
        const int32_t * block_table,
        const int32_t * context_lengths,
        char * output,
        uint64_t q_sequence_stride,
        uint64_t q_head_stride,
        uint64_t output_sequence_stride,
        uint64_t output_head_stride,
        uint64_t k_token_stride,
        uint64_t k_head_stride,
        uint64_t k_stream_stride,
        uint64_t v_token_stride,
        uint64_t v_head_stride,
        uint64_t v_stream_stride,
        int32_t query_heads,
        int32_t kv_heads,
        int32_t max_blocks,
        int32_t batch_size,
        int32_t page_size,
        float scale,
        int32_t partition_tokens,
        int32_t partitions,
        float * partial) {
    cacheflow_record_paged_dispatch(batch_size);
    constexpr int32_t head_dim = 64;
    constexpr int32_t half2_dim = head_dim / 2;
    constexpr int32_t group_size = 7;
    constexpr int32_t token_tile = 32;

    const int32_t blocks_per_sequence = kv_heads * partitions;
    const int32_t sequence = blockIdx.x / blocks_per_sequence;
    const int32_t sequence_block = blockIdx.x % blocks_per_sequence;
    const int32_t partition = sequence_block % partitions;
    const int32_t kv_head = sequence_block / partitions;
    const int32_t warp = threadIdx.x / 32;
    const int32_t lane = threadIdx.x % 32;
    const bool active_head = warp < group_size;
    const int32_t query_head = kv_head * group_size + warp;
    if (sequence >= batch_size) return;
    const int32_t context = context_lengths[sequence];
    const int32_t * sequence_table = block_table + (uint64_t) sequence * max_blocks;
    const int32_t effective_partitions =
            (context + partition_tokens - 1) / partition_tokens;
    const int32_t partition_begin = partition * partition_tokens;
    const int32_t partition_end = min(context, partition_begin + partition_tokens);

    if (kv_head >= kv_heads || query_heads != kv_heads * group_size || context <= 0 ||
            (context + page_size - 1) / page_size > max_blocks) return;
    if (partition >= effective_partitions) return;

    __shared__ float2 query_pair[group_size][half2_dim];
    __shared__ __half2 key_pair[half2_dim][token_tile];
    __shared__ __half2 value_pair[token_tile][half2_dim];

    if (partition_begin >= context) {
        if (active_head && partitions > 1) {
            const uint64_t partial_base =
                    (((uint64_t) sequence * query_heads + query_head) * partitions + partition) *
                    (head_dim + 2);
            if (lane == 0) {
                partial[partial_base] = -FLT_MAX;
                partial[partial_base + 1] = 0.0f;
            }
            partial[partial_base + 2 + 2 * lane] = 0.0f;
            partial[partial_base + 3 + 2 * lane] = 0.0f;
        }
        return;
    }

    if (active_head) {
        const query_type * q = reinterpret_cast<const query_type *>(
                query + (uint64_t) sequence * q_sequence_stride +
                (uint64_t) query_head * q_head_stride);
        if constexpr (std::is_same<query_type, __half>::value) {
            query_pair[warp][lane] = __half22float2(
                    reinterpret_cast<const __half2 *>(q)[lane]);
        } else {
            query_pair[warp][lane] = make_float2(q[2 * lane], q[2 * lane + 1]);
        }
    }
    __syncthreads();

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float2 output_pair = make_float2(0.0f, 0.0f);

    for (int32_t tile_begin = partition_begin; tile_begin < partition_end; tile_begin += token_tile) {
        const int32_t tile_tokens = min(token_tile, partition_end - tile_begin);
        for (int32_t item = threadIdx.x; item < tile_tokens * half2_dim; item += blockDim.x) {
            const int32_t tile_token = item / half2_dim;
            const int32_t dim_pair = item % half2_dim;
            const int32_t token = tile_begin + tile_token;
            const int32_t logical_block = token / page_size;
            const int32_t cell = sequence_table[logical_block] + token % page_size;
            const __half2 * k = reinterpret_cast<const __half2 *>(
                    key + (uint64_t) cell * k_token_stride +
                    (uint64_t) kv_head * k_head_stride +
                    (uint64_t) sequence * k_stream_stride);
            const __half2 * v = reinterpret_cast<const __half2 *>(
                    value + (uint64_t) cell * v_token_stride +
                    (uint64_t) kv_head * v_head_stride +
                    (uint64_t) sequence * v_stream_stride);
            key_pair[dim_pair][tile_token] = k[dim_pair];
            value_pair[tile_token][dim_pair] = v[dim_pair];
        }
        __syncthreads();

        if (active_head) {
            float logit = -FLT_MAX;
            if (lane < tile_tokens) {
                logit = 0.0f;
#pragma unroll
                for (int32_t dim_pair = 0; dim_pair < half2_dim; ++dim_pair) {
                    const float2 q2 = query_pair[warp][dim_pair];
                    const float2 k2 = __half22float2(key_pair[dim_pair][lane]);
                    logit += q2.x * k2.x + q2.y * k2.y;
                }
                logit *= scale;
            }
            float tile_max = logit;
#pragma unroll
            for (int32_t offset = 16; offset > 0; offset /= 2) {
                tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
            }
            tile_max = __shfl_sync(0xffffffff, tile_max, 0);
            const float weight = lane < tile_tokens ? expf(logit - tile_max) : 0.0f;
            float tile_sum = weight;
#pragma unroll
            for (int32_t offset = 16; offset > 0; offset /= 2) {
                tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);
            }
            tile_sum = __shfl_sync(0xffffffff, tile_sum, 0);

            float2 tile_output = make_float2(0.0f, 0.0f);
            for (int32_t token = 0; token < tile_tokens; ++token) {
                const float token_weight = __shfl_sync(0xffffffff, weight, token);
                const float2 v2 = __half22float2(value_pair[token][lane]);
                tile_output.x += token_weight * v2.x;
                tile_output.y += token_weight * v2.y;
            }

            const float next_max = fmaxf(running_max, tile_max);
            const float previous_scale = running_sum == 0.0f ? 0.0f : expf(running_max - next_max);
            const float tile_scale = expf(tile_max - next_max);
            output_pair.x = output_pair.x * previous_scale + tile_output.x * tile_scale;
            output_pair.y = output_pair.y * previous_scale + tile_output.y * tile_scale;
            running_sum = running_sum * previous_scale + tile_sum * tile_scale;
            running_max = next_max;
        }
        __syncthreads();
    }

    if (active_head) {
        if (partitions == 1) {
            *reinterpret_cast<float *>(output + (uint64_t) sequence * output_sequence_stride +
                    (uint64_t) query_head * output_head_stride +
                    (uint64_t) (2 * lane) * sizeof(float)) = output_pair.x / running_sum;
            *reinterpret_cast<float *>(output + (uint64_t) sequence * output_sequence_stride +
                    (uint64_t) query_head * output_head_stride +
                    (uint64_t) (2 * lane + 1) * sizeof(float)) = output_pair.y / running_sum;
        } else {
            const uint64_t partial_base =
                    (((uint64_t) sequence * query_heads + query_head) * partitions + partition) *
                    (head_dim + 2);
            if (lane == 0) {
                partial[partial_base] = running_max;
                partial[partial_base + 1] = running_sum;
            }
            partial[partial_base + 2 + 2 * lane] = output_pair.x;
            partial[partial_base + 3 + 2 * lane] = output_pair.y;
        }
    }
}

template<int head_dim>
static __global__ void cacheflow_paged_decode_merge_partitions_k4(
        const float * partial,
        const int32_t * context_lengths,
        char * output,
        uint64_t output_sequence_stride,
        uint64_t output_head_stride,
        int32_t query_heads,
        int32_t partition_tokens,
        int32_t partition_stride) {
    const int32_t sequence = blockIdx.x / query_heads;
    const int32_t query_head = blockIdx.x % query_heads;
    const int32_t dim = threadIdx.x;
    const int32_t context = context_lengths[sequence];
    const int32_t effective_partitions = (context + partition_tokens - 1) / partition_tokens;
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float accumulator = 0.0f;
    for (int32_t partition = 0; partition < effective_partitions; ++partition) {
        const uint64_t base = (((uint64_t) sequence * query_heads + query_head) *
                partition_stride + partition) * (head_dim + 2);
        const float partition_max = partial[base];
        const float partition_sum = partial[base + 1];
        const float next_max = fmaxf(running_max, partition_max);
        const float previous_scale = running_sum == 0.0f ? 0.0f : expf(running_max - next_max);
        const float partition_scale = expf(partition_max - next_max);
        accumulator = accumulator * previous_scale + partial[base + 2 + dim] * partition_scale;
        running_sum = running_sum * previous_scale + partition_sum * partition_scale;
        running_max = next_max;
    }
    *reinterpret_cast<float *>(output + (uint64_t) sequence * output_sequence_stride +
            (uint64_t) query_head * output_head_stride + (uint64_t) dim * sizeof(float)) =
            accumulator / running_sum;
}

// K5 retains K4's one-CTA-per-GQA-group/page-partition decomposition, but
// computes Q*K^T and P*V as 16x16 WMMA tiles.  Rows 0..6 contain the model's
// seven query heads and the remaining rows are zero padding.  The online
// softmax recurrence stays in FP32, so page indirection changes data movement
// without weakening the numerical contract.
template<typename query_type>
static __global__ void cacheflow_paged_decode_fattn_k5_wmma_gqa7(
        const char * query,
        const char * key,
        const char * value,
        const int32_t * block_table,
        const int32_t * context_lengths,
        char * output,
        uint64_t q_sequence_stride,
        uint64_t q_head_stride,
        uint64_t output_sequence_stride,
        uint64_t output_head_stride,
        uint64_t k_token_stride,
        uint64_t k_head_stride,
        uint64_t k_stream_stride,
        uint64_t v_token_stride,
        uint64_t v_head_stride,
        uint64_t v_stream_stride,
        int32_t query_heads,
        int32_t kv_heads,
        int32_t max_blocks,
        int32_t batch_size,
        int32_t page_size,
        float scale,
        int32_t partition_tokens,
        int32_t partitions,
        float * partial) {
    cacheflow_record_paged_dispatch(batch_size);
    constexpr int32_t head_dim = 64;
    constexpr int32_t group_size = 7;
    constexpr int32_t tile_tokens = 32;
    constexpr int32_t padded_heads = 16;

    const int32_t blocks_per_sequence = kv_heads * partitions;
    const int32_t sequence = blockIdx.x / blocks_per_sequence;
    const int32_t sequence_block = blockIdx.x % blocks_per_sequence;
    const int32_t partition = sequence_block % partitions;
    const int32_t kv_head = sequence_block / partitions;
    const int32_t warp = threadIdx.x / 32;
    const int32_t lane = threadIdx.x % 32;
    if (sequence >= batch_size || kv_head >= kv_heads || query_heads != kv_heads * group_size) return;

    const int32_t context = context_lengths[sequence];
    const int32_t effective_partitions = (context + partition_tokens - 1) / partition_tokens;
    if (context <= 0 || (context + page_size - 1) / page_size > max_blocks ||
            partition >= effective_partitions) return;
    const int32_t partition_begin = partition * partition_tokens;
    const int32_t partition_end = min(context, partition_begin + partition_tokens);
    const int32_t * sequence_table = block_table + (uint64_t) sequence * max_blocks;

    __shared__ __half q_tile[padded_heads][head_dim];
    __shared__ __half k_tile[tile_tokens][head_dim];
    __shared__ __half v_tile[tile_tokens][head_dim];
    __shared__ float logits[padded_heads][tile_tokens];
    __shared__ __half weights[padded_heads][tile_tokens];
    __shared__ float tile_output[padded_heads][head_dim];
    __shared__ float output_accumulator[group_size][head_dim];
    __shared__ float running_max[group_size];
    __shared__ float running_sum[group_size];
    __shared__ float tile_maximum[group_size];
    __shared__ float tile_sum[group_size];

    for (int32_t item = threadIdx.x; item < padded_heads * head_dim; item += blockDim.x) {
        const int32_t head = item / head_dim;
        const int32_t dim = item % head_dim;
        if (head < group_size) {
            const query_type * q = reinterpret_cast<const query_type *>(
                    query + (uint64_t) sequence * q_sequence_stride +
                    (uint64_t) (kv_head * group_size + head) * q_head_stride);
            q_tile[head][dim] = __float2half_rn((float) q[dim] * scale);
        } else {
            q_tile[head][dim] = __float2half_rn(0.0f);
        }
    }
    for (int32_t item = threadIdx.x; item < group_size * head_dim; item += blockDim.x) {
        output_accumulator[item / head_dim][item % head_dim] = 0.0f;
    }
    if (threadIdx.x < group_size) {
        running_max[threadIdx.x] = -FLT_MAX;
        running_sum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int32_t tile_begin = partition_begin; tile_begin < partition_end;
            tile_begin += tile_tokens) {
        const int32_t valid_tokens = min(tile_tokens, partition_end - tile_begin);
        for (int32_t item = threadIdx.x; item < tile_tokens * head_dim; item += blockDim.x) {
            const int32_t token_in_tile = item / head_dim;
            const int32_t dim = item % head_dim;
            if (token_in_tile < valid_tokens) {
                const int32_t token = tile_begin + token_in_tile;
                const int32_t logical_block = token / page_size;
                const int32_t cell = sequence_table[logical_block] + token % page_size;
                const __half * k = reinterpret_cast<const __half *>(
                        key + (uint64_t) cell * k_token_stride +
                        (uint64_t) kv_head * k_head_stride +
                        (uint64_t) sequence * k_stream_stride);
                const __half * v = reinterpret_cast<const __half *>(
                        value + (uint64_t) cell * v_token_stride +
                        (uint64_t) kv_head * v_head_stride +
                        (uint64_t) sequence * v_stream_stride);
                k_tile[token_in_tile][dim] = k[dim];
                v_tile[token_in_tile][dim] = v[dim];
            } else {
                k_tile[token_in_tile][dim] = __float2half_rn(0.0f);
                v_tile[token_in_tile][dim] = __float2half_rn(0.0f);
            }
        }
        __syncthreads();

        if (warp < 2) {
            using namespace nvcuda;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
            wmma::fill_fragment(accumulator, 0.0f);
#pragma unroll
            for (int32_t k_begin = 0; k_begin < head_dim; k_begin += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_fragment;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_fragment;
                wmma::load_matrix_sync(q_fragment, &q_tile[0][k_begin], head_dim);
                wmma::load_matrix_sync(k_fragment, &k_tile[warp * 16][k_begin], head_dim);
                wmma::mma_sync(accumulator, q_fragment, k_fragment, accumulator);
            }
            wmma::store_matrix_sync(&logits[0][warp * 16], accumulator,
                    tile_tokens, wmma::mem_row_major);
        }
        __syncthreads();

        if (warp < group_size) {
            float score = lane < valid_tokens ? logits[warp][lane] : -FLT_MAX;
            float maximum = score;
#pragma unroll
            for (int32_t offset = 16; offset > 0; offset /= 2) {
                maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
            }
            maximum = __shfl_sync(0xffffffff, maximum, 0);
            const float weight = lane < valid_tokens ? expf(score - maximum) : 0.0f;
            float sum = weight;
#pragma unroll
            for (int32_t offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            sum = __shfl_sync(0xffffffff, sum, 0);
            weights[warp][lane] = __float2half_rn(weight);
            if (lane == 0) {
                tile_maximum[warp] = maximum;
                tile_sum[warp] = sum;
            }
        }
        for (int32_t item = threadIdx.x + group_size * tile_tokens;
                item < padded_heads * tile_tokens; item += blockDim.x) {
            weights[item / tile_tokens][item % tile_tokens] = __float2half_rn(0.0f);
        }
        __syncthreads();

        if (warp < 4) {
            using namespace nvcuda;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
            wmma::fill_fragment(accumulator, 0.0f);
#pragma unroll
            for (int32_t token_begin = 0; token_begin < tile_tokens; token_begin += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_fragment;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_fragment;
                wmma::load_matrix_sync(p_fragment, &weights[0][token_begin], tile_tokens);
                wmma::load_matrix_sync(v_fragment, &v_tile[token_begin][warp * 16], head_dim);
                wmma::mma_sync(accumulator, p_fragment, v_fragment, accumulator);
            }
            wmma::store_matrix_sync(&tile_output[0][warp * 16], accumulator,
                    head_dim, wmma::mem_row_major);
        }
        __syncthreads();

        if (warp < group_size) {
            const float next_maximum = fmaxf(running_max[warp], tile_maximum[warp]);
            const float previous_scale = running_sum[warp] == 0.0f ? 0.0f :
                    expf(running_max[warp] - next_maximum);
            const float tile_scale = expf(tile_maximum[warp] - next_maximum);
            for (int32_t dim = lane; dim < head_dim; dim += 32) {
                output_accumulator[warp][dim] =
                        output_accumulator[warp][dim] * previous_scale +
                        tile_output[warp][dim] * tile_scale;
            }
            if (lane == 0) {
                running_sum[warp] = running_sum[warp] * previous_scale +
                        tile_sum[warp] * tile_scale;
                running_max[warp] = next_maximum;
            }
        }
        __syncthreads();
    }

    if (warp < group_size) {
        const int32_t query_head = kv_head * group_size + warp;
        if (partitions == 1) {
            for (int32_t dim = lane; dim < head_dim; dim += 32) {
                *reinterpret_cast<float *>(output + (uint64_t) sequence * output_sequence_stride +
                        (uint64_t) query_head * output_head_stride +
                        (uint64_t) dim * sizeof(float)) =
                        output_accumulator[warp][dim] / running_sum[warp];
            }
        } else {
            const uint64_t base = (((uint64_t) sequence * query_heads + query_head) *
                    partitions + partition) * (head_dim + 2);
            if (lane == 0) {
                partial[base] = running_max[warp];
                partial[base + 1] = running_sum[warp];
            }
            for (int32_t dim = lane; dim < head_dim; dim += 32) {
                partial[base + 2 + dim] = output_accumulator[warp][dim];
            }
        }
    }
}

template<int head_dim>
static __global__ void cacheflow_paged_decode_merge_partitions(
        const float * partial,
        char * output,
        uint64_t output_sequence_stride,
        uint64_t output_head_stride,
        int32_t query_heads,
        int32_t partitions) {
    const int32_t sequence = blockIdx.x / query_heads;
    const int32_t query_head = blockIdx.x % query_heads;
    const int32_t dim = threadIdx.x;
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float accumulator = 0.0f;
    for (int32_t partition = 0; partition < partitions; ++partition) {
        const uint64_t base = (((uint64_t) sequence * query_heads + query_head) *
                partitions + partition) * (head_dim + 2);
        const float partition_max = partial[base];
        const float partition_sum = partial[base + 1];
        if (partition_sum == 0.0f) {
            continue;
        }
        const float next_max = fmaxf(running_max, partition_max);
        const float previous_scale = running_sum == 0.0f ? 0.0f : expf(running_max - next_max);
        const float partition_scale = expf(partition_max - next_max);
        accumulator = accumulator * previous_scale +
                partial[base + 2 + dim] * partition_scale;
        running_sum = running_sum * previous_scale + partition_sum * partition_scale;
        running_max = next_max;
    }
    *reinterpret_cast<float *>(output + (uint64_t) sequence * output_sequence_stride +
            (uint64_t) query_head * output_head_stride + (uint64_t) dim * sizeof(float)) =
            accumulator / running_sum;
}

static bool ggml_cuda_paged_fattn_supported(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * table = dst->src[5];
    const ggml_tensor * lengths = dst->src[6];
    if (!table) return false;
    const int32_t page_size = ggml_get_op_params_i32(dst, 4);
    float max_bias = 0.0f;
    float softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
    return lengths && (q->type == GGML_TYPE_F16 || q->type == GGML_TYPE_F32) &&
            k->type == GGML_TYPE_F16 &&
            v->type == GGML_TYPE_F16 && table->type == GGML_TYPE_I32 &&
            lengths->type == GGML_TYPE_I32 && (q->ne[0] == 64 || q->ne[0] == 128) &&
            q->ne[1] >= 1 && q->ne[3] >= 1 && (q->ne[1] == 1 || q->ne[3] == 1) &&
            table->ne[1] == q->ne[1] * q->ne[3] && lengths->ne[0] == table->ne[1] &&
            k->ne[3] == q->ne[3] && v->ne[3] == q->ne[3] &&
            q->ne[2] % k->ne[2] == 0 && k->ne[2] == v->ne[2] &&
            q->ne[0] == k->ne[0] && q->ne[0] == v->ne[0] && page_size == 16 &&
            max_bias == 0.0f && softcap == 0.0f && dst->src[4] == nullptr;
}

static void ggml_cuda_paged_fattn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    float scale = 0.0f;
    memcpy(&scale, dst->op_params, sizeof(float));
    const int32_t page_size = ggml_get_op_params_i32(dst, 4);
    const int32_t max_blocks = (int32_t) dst->src[5]->ne[0];
    const int32_t query_heads = (int32_t) q->ne[2];
    const int32_t kv_heads = (int32_t) k->ne[2];
    const int32_t batch_size = (int32_t) (q->ne[1] * q->ne[3]);
    const uint64_t q_sequence_stride = q->ne[1] > 1 ? q->nb[1] : q->nb[3];
    const uint64_t output_sequence_stride = q->ne[1] > 1 ? dst->nb[2] : dst->nb[3];
    const uint64_t k_stream_stride = q->ne[3] > 1 ? k->nb[3] : 0;
    const uint64_t v_stream_stride = q->ne[3] > 1 ? v->nb[3] : 0;
    // A process-scoped selector is intentionally retained for reproducible
    // incumbent/candidate service A/B. Production defaults to the GQA-aware
    // K4 implementation; K1/K2/K3 remain selectable as archived controls.
    static const int environment_variant = [] {
        const char * variant = std::getenv("LLAMA_CACHEFLOW_PAGED_KERNEL");
        if (variant && std::strcmp(variant, "K1") == 0) return 1;
        if (variant && std::strcmp(variant, "K2") == 0) return 2;
        if (variant && std::strcmp(variant, "K3") == 0) return 3;
        if (variant && std::strcmp(variant, "K5") == 0) return 5;
        return 4;
    }();
    // Disabled by default. A formal same-process experiment can point this at
    // a two-byte K1/K2 control file and switch only the kernel intervention
    // without restarting the serving process or disturbing its warm state.
    static const char * control_file = std::getenv("LLAMA_CACHEFLOW_PAGED_KERNEL_CONTROL_FILE");
    int control_variant = 0;
    if (control_file && control_file[0] != '\0') {
        FILE * file = std::fopen(control_file, "rb");
        GGML_ASSERT(file != nullptr);
        char selector[2] = {};
        const size_t count = std::fread(selector, 1, sizeof(selector), file);
        std::fclose(file);
        GGML_ASSERT(count == sizeof(selector));
        GGML_ASSERT((selector[0] == 'K' && selector[1] == '1') ||
                    (selector[0] == 'K' && selector[1] == '2'));
        control_variant = selector[1] == '1' ? 1 : 2;
    }
    const int requested_variant = control_variant == 0 ? environment_variant : control_variant;
    const bool specialized = query_heads / kv_heads == 7 && q->ne[0] == 64;
    const bool use_k2 = specialized && requested_variant == 2;
    const bool use_k3 = specialized && requested_variant == 3;
    const bool use_k4 = specialized && requested_variant == 4;
    const bool use_k5 = specialized && requested_variant == 5;
    constexpr int32_t k2_partition_tokens = 256;
    constexpr int32_t k3_partition_tokens = 256;
    // K4 uses one batch-wide partition geometry.  The block-table width is
    // the current batch's maximum logical page count, so it gives us a
    // synchronization-free upper bound for selecting the long-context tile.
    // Keeping this value common to the producer and merge kernels is also
    // required for ragged batches that straddle the 512-token boundary.
    static const int32_t k4_long_partition_tokens = [] {
        const char * value = std::getenv("LLAMA_CACHEFLOW_PAGED_K4_PARTITION_TOKENS");
        if (!value || value[0] == '\0') return 128;
        const int32_t parsed = std::atoi(value);
        GGML_ASSERT(parsed == 64 || parsed == 128 || parsed == 256 || parsed == 512);
        return parsed;
    }();
    const int32_t k4_partition_tokens =
            max_blocks * page_size <= 512 ? 64 : k4_long_partition_tokens;
    const int32_t partition_tokens = (use_k4 || use_k5) ? k4_partition_tokens :
            (use_k3 ? k3_partition_tokens : k2_partition_tokens);
    const int32_t partitions = (use_k2 || use_k3 || use_k4 || use_k5) ?
            (max_blocks * page_size + partition_tokens - 1) / partition_tokens : 1;
    ggml_cuda_pool_alloc<float> partial_alloc(ctx.pool());
    float * partial = partitions > 1 ? partial_alloc.alloc(
            (size_t) batch_size * query_heads * partitions * (q->ne[0] + 2)) : nullptr;
#define CACHEFLOW_PAGED_LAUNCH(D, Q) cacheflow_paged_decode_fattn_k1<D, Q><<<batch_size * q->ne[2], D, 0, ctx.stream()>>>( \
        (const char *) q->data, (const char *) k->data, (const char *) v->data, \
        (const int32_t *) dst->src[5]->data, (const int32_t *) dst->src[6]->data, \
        (char *) dst->data, q_sequence_stride, q->nb[2], output_sequence_stride, dst->nb[1], \
        k->nb[1], k->nb[2], k_stream_stride, v->nb[1], v->nb[2], v_stream_stride, \
        (int32_t) q->ne[2], (int32_t) k->ne[2], max_blocks, batch_size, page_size, scale)
#define CACHEFLOW_PAGED_K2_LAUNCH(D, Q) cacheflow_paged_decode_fattn_k2_t2<D, Q><<< \
        batch_size * k->ne[2] * 4 * partitions, 64, 0, ctx.stream()>>>( \
        (const char *) q->data, (const char *) k->data, (const char *) v->data, \
        (const int32_t *) dst->src[5]->data, (const int32_t *) dst->src[6]->data, \
        (char *) dst->data, q_sequence_stride, q->nb[2], output_sequence_stride, dst->nb[1], \
        k->nb[1], k->nb[2], k_stream_stride, v->nb[1], v->nb[2], v_stream_stride, \
        query_heads, kv_heads, max_blocks, batch_size, page_size, scale, partition_tokens, partitions, partial)
#define CACHEFLOW_PAGED_K3_LAUNCH(Q) cacheflow_paged_decode_fattn_k3_vec_t2<Q><<< \
        batch_size * k->ne[2] * 4 * partitions, 128, 0, ctx.stream()>>>( \
        (const char *) q->data, (const char *) k->data, (const char *) v->data, \
        (const int32_t *) dst->src[5]->data, (const int32_t *) dst->src[6]->data, \
        (char *) dst->data, q_sequence_stride, q->nb[2], output_sequence_stride, dst->nb[1], \
        k->nb[1], k->nb[2], k_stream_stride, v->nb[1], v->nb[2], v_stream_stride, \
        query_heads, kv_heads, max_blocks, batch_size, page_size, scale, partition_tokens, partitions, partial)
#define CACHEFLOW_PAGED_K4_LAUNCH(Q) cacheflow_paged_decode_fattn_k4_gqa7<Q><<< \
        batch_size * k->ne[2] * partitions, 256, 0, ctx.stream()>>>( \
        (const char *) q->data, (const char *) k->data, (const char *) v->data, \
        (const int32_t *) dst->src[5]->data, (const int32_t *) dst->src[6]->data, \
        (char *) dst->data, q_sequence_stride, q->nb[2], output_sequence_stride, dst->nb[1], \
        k->nb[1], k->nb[2], k_stream_stride, v->nb[1], v->nb[2], v_stream_stride, \
        query_heads, kv_heads, max_blocks, batch_size, page_size, scale, partition_tokens, partitions, partial)
#define CACHEFLOW_PAGED_K5_LAUNCH(Q) cacheflow_paged_decode_fattn_k5_wmma_gqa7<Q><<< \
        batch_size * k->ne[2] * partitions, 256, 0, ctx.stream()>>>( \
        (const char *) q->data, (const char *) k->data, (const char *) v->data, \
        (const int32_t *) dst->src[5]->data, (const int32_t *) dst->src[6]->data, \
        (char *) dst->data, q_sequence_stride, q->nb[2], output_sequence_stride, dst->nb[1], \
        k->nb[1], k->nb[2], k_stream_stride, v->nb[1], v->nb[2], v_stream_stride, \
        query_heads, kv_heads, max_blocks, batch_size, page_size, scale, partition_tokens, partitions, partial)
    if (q->type == GGML_TYPE_F16) {
        if (q->ne[0] == 64) {
            if (use_k5) CACHEFLOW_PAGED_K5_LAUNCH(__half);
            else if (use_k4) CACHEFLOW_PAGED_K4_LAUNCH(__half);
            else if (use_k3) CACHEFLOW_PAGED_K3_LAUNCH(__half);
            else if (use_k2) CACHEFLOW_PAGED_K2_LAUNCH(64, __half);
            else CACHEFLOW_PAGED_LAUNCH(64, __half);
        } else {
            CACHEFLOW_PAGED_LAUNCH(128, __half);
        }
    } else {
        if (q->ne[0] == 64) {
            if (use_k5) CACHEFLOW_PAGED_K5_LAUNCH(float);
            else if (use_k4) CACHEFLOW_PAGED_K4_LAUNCH(float);
            else if (use_k3) CACHEFLOW_PAGED_K3_LAUNCH(float);
            else if (use_k2) CACHEFLOW_PAGED_K2_LAUNCH(64, float);
            else CACHEFLOW_PAGED_LAUNCH(64, float);
        } else {
            CACHEFLOW_PAGED_LAUNCH(128, float);
        }
    }
    if ((use_k4 || use_k5) && partitions > 1) {
        cacheflow_paged_decode_merge_partitions_k4<64><<<batch_size * query_heads, 64, 0, ctx.stream()>>>(
                partial, (const int32_t *) dst->src[6]->data, (char *) dst->data,
                output_sequence_stride, dst->nb[1], query_heads, partition_tokens, partitions);
    } else if ((use_k2 || use_k3) && partitions > 1) {
        cacheflow_paged_decode_merge_partitions<64><<<batch_size * query_heads, 64, 0, ctx.stream()>>>(
                partial, (char *) dst->data, output_sequence_stride, dst->nb[1], query_heads, partitions);
    }
#undef CACHEFLOW_PAGED_K2_LAUNCH
#undef CACHEFLOW_PAGED_K3_LAUNCH
#undef CACHEFLOW_PAGED_K4_LAUNCH
#undef CACHEFLOW_PAGED_K5_LAUNCH
#undef CACHEFLOW_PAGED_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}
