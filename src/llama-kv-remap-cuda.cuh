#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

// One logical snapshot copy. `staging_offset` is measured in FP16 elements.
// The same descriptor is consumed by gather and scatter, so overlapping
// source/destination ranges retain deterministic snapshot semantics.
struct llama_kv_remap_copy {
    const uint16_t * source;
    uint16_t * destination;
    size_t staging_offset;
    size_t elements;
};

enum class llama_kv_remap_mode : uint8_t {
    scalar,
    vectorized,
};

struct llama_kv_remap_accounting {
    uint64_t vectorized_bytes = 0;
    uint64_t scalar_bytes = 0;
};

constexpr size_t LLAMA_KV_REMAP_VECTOR_ELEMENTS = sizeof(uint4) / sizeof(uint16_t);

inline bool llama_kv_remap_aligned(const void * pointer) {
    return (reinterpret_cast<uintptr_t>(pointer) & (alignof(uint4) - 1)) == 0;
}

inline llama_kv_remap_accounting llama_kv_remap_account(
        const std::vector<llama_kv_remap_copy> & copies,
        const uint16_t * staging) {
    llama_kv_remap_accounting result;
    for (const auto & copy : copies) {
        const bool aligned = llama_kv_remap_aligned(copy.source) &&
                llama_kv_remap_aligned(copy.destination) &&
                llama_kv_remap_aligned(staging + copy.staging_offset);
        const size_t vector_elements = aligned
                ? copy.elements / LLAMA_KV_REMAP_VECTOR_ELEMENTS * LLAMA_KV_REMAP_VECTOR_ELEMENTS
                : 0;
        result.vectorized_bytes += vector_elements * sizeof(uint16_t);
        result.scalar_bytes += (copy.elements - vector_elements) * sizeof(uint16_t);
    }
    return result;
}

static __global__ void llama_kv_remap_gather_scalar_kernel(
        const llama_kv_remap_copy * copies,
        uint16_t * staging,
        size_t copy_count,
        size_t max_elements) {
    const size_t element = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t copy_index = blockIdx.y;
    if (copy_index >= copy_count || element >= max_elements) return;
    const auto copy = copies[copy_index];
    if (element < copy.elements) staging[copy.staging_offset + element] = copy.source[element];
}

static __global__ void llama_kv_remap_scatter_scalar_kernel(
        const llama_kv_remap_copy * copies,
        const uint16_t * staging,
        size_t copy_count,
        size_t max_elements) {
    const size_t element = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t copy_index = blockIdx.y;
    if (copy_index >= copy_count || element >= max_elements) return;
    const auto copy = copies[copy_index];
    if (element < copy.elements) copy.destination[element] = staging[copy.staging_offset + element];
}

static __device__ bool llama_kv_remap_device_aligned(const void * pointer) {
    return (reinterpret_cast<uintptr_t>(pointer) & (alignof(uint4) - 1)) == 0;
}

static __global__ void llama_kv_remap_gather_vector_kernel(
        const llama_kv_remap_copy * copies,
        uint16_t * staging,
        size_t copy_count,
        size_t max_elements) {
    const size_t unit = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t copy_index = blockIdx.y;
    const size_t first = unit * LLAMA_KV_REMAP_VECTOR_ELEMENTS;
    if (copy_index >= copy_count || first >= max_elements) return;
    const auto copy = copies[copy_index];
    if (first >= copy.elements) return;
    auto * staged = staging + copy.staging_offset;
    const bool aligned = llama_kv_remap_device_aligned(copy.source) &&
            llama_kv_remap_device_aligned(staged);
    if (aligned && first + LLAMA_KV_REMAP_VECTOR_ELEMENTS <= copy.elements) {
        reinterpret_cast<uint4 *>(staged)[unit] = reinterpret_cast<const uint4 *>(copy.source)[unit];
        return;
    }
    #pragma unroll
    for (size_t lane = 0; lane < LLAMA_KV_REMAP_VECTOR_ELEMENTS; ++lane) {
        const size_t element = first + lane;
        if (element < copy.elements) staged[element] = copy.source[element];
    }
}

static __global__ void llama_kv_remap_scatter_vector_kernel(
        const llama_kv_remap_copy * copies,
        const uint16_t * staging,
        size_t copy_count,
        size_t max_elements) {
    const size_t unit = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t copy_index = blockIdx.y;
    const size_t first = unit * LLAMA_KV_REMAP_VECTOR_ELEMENTS;
    if (copy_index >= copy_count || first >= max_elements) return;
    const auto copy = copies[copy_index];
    if (first >= copy.elements) return;
    const auto * staged = staging + copy.staging_offset;
    const bool aligned = llama_kv_remap_device_aligned(copy.destination) &&
            llama_kv_remap_device_aligned(staged);
    if (aligned && first + LLAMA_KV_REMAP_VECTOR_ELEMENTS <= copy.elements) {
        reinterpret_cast<uint4 *>(copy.destination)[unit] = reinterpret_cast<const uint4 *>(staged)[unit];
        return;
    }
    #pragma unroll
    for (size_t lane = 0; lane < LLAMA_KV_REMAP_VECTOR_ELEMENTS; ++lane) {
        const size_t element = first + lane;
        if (element < copy.elements) copy.destination[element] = staged[element];
    }
}

inline cudaError_t llama_kv_remap_make_grid(
        size_t copy_count,
        size_t max_elements,
        llama_kv_remap_mode mode,
        dim3 & grid) {
    constexpr size_t threads = 256;
    constexpr size_t maximum_grid_x = 2147483647;
    constexpr size_t maximum_grid_y = 65535;
    if (copy_count > maximum_grid_y) return cudaErrorInvalidValue;
    const size_t elements_per_work = mode == llama_kv_remap_mode::vectorized
            ? LLAMA_KV_REMAP_VECTOR_ELEMENTS
            : 1;
    const size_t work = max_elements / elements_per_work +
            (max_elements % elements_per_work != 0);
    const size_t grid_x = work / threads + (work % threads != 0);
    if (grid_x == 0 || grid_x > maximum_grid_x) {
        return cudaErrorInvalidValue;
    }
    grid = dim3((unsigned) grid_x, (unsigned) copy_count);
    return cudaSuccess;
}

inline cudaError_t llama_kv_remap_launch_gather(
        const llama_kv_remap_copy * copies,
        uint16_t * staging,
        size_t copy_count,
        size_t max_elements,
        cudaStream_t stream,
        llama_kv_remap_mode mode) {
    if (copy_count == 0 || max_elements == 0) return cudaSuccess;
    constexpr int threads = 256;
    dim3 grid;
    const cudaError_t grid_status = llama_kv_remap_make_grid(copy_count, max_elements, mode, grid);
    if (grid_status != cudaSuccess) return grid_status;
    if (mode == llama_kv_remap_mode::vectorized) {
        llama_kv_remap_gather_vector_kernel<<<grid, threads, 0, stream>>>(
                copies, staging, copy_count, max_elements);
    } else {
        llama_kv_remap_gather_scalar_kernel<<<grid, threads, 0, stream>>>(
                copies, staging, copy_count, max_elements);
    }
    return cudaGetLastError();
}

inline cudaError_t llama_kv_remap_launch_scatter(
        const llama_kv_remap_copy * copies,
        const uint16_t * staging,
        size_t copy_count,
        size_t max_elements,
        cudaStream_t stream,
        llama_kv_remap_mode mode) {
    if (copy_count == 0 || max_elements == 0) return cudaSuccess;
    constexpr int threads = 256;
    dim3 grid;
    const cudaError_t grid_status = llama_kv_remap_make_grid(copy_count, max_elements, mode, grid);
    if (grid_status != cudaSuccess) return grid_status;
    if (mode == llama_kv_remap_mode::vectorized) {
        llama_kv_remap_scatter_vector_kernel<<<grid, threads, 0, stream>>>(
                copies, staging, copy_count, max_elements);
    } else {
        llama_kv_remap_scatter_scalar_kernel<<<grid, threads, 0, stream>>>(
                copies, staging, copy_count, max_elements);
    }
    return cudaGetLastError();
}
