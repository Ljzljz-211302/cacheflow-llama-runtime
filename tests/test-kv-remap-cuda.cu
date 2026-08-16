#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-kv-remap-cuda.cuh"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

static void require_equal(const std::vector<uint16_t> & expected, const uint16_t * device, size_t size) {
    std::vector<uint16_t> observed(size);
    check(cudaMemcpy(observed.data(), device, size * sizeof(uint16_t), cudaMemcpyDeviceToHost),
            "read remap result");
    assert(observed == expected);
}

static void test_aligned_overlap_uses_snapshot_and_vector_path() {
    constexpr size_t elements_per_block = 64;
    constexpr size_t blocks = 4;
    std::vector<uint16_t> initial(blocks * elements_per_block);
    for (size_t i = 0; i < initial.size(); ++i) initial[i] = (uint16_t) (17 + i * 13);
    auto expected = initial;
    std::copy_n(initial.begin(), elements_per_block,
            expected.begin() + elements_per_block);
    std::copy_n(initial.begin() + elements_per_block, elements_per_block,
            expected.begin());

    uint16_t * data = nullptr;
    uint16_t * staging = nullptr;
    llama_kv_remap_copy * device_copies = nullptr;
    check(cudaMalloc((void **) &data, initial.size() * sizeof(uint16_t)), "allocate remap data");
    check(cudaMalloc((void **) &staging, 2 * elements_per_block * sizeof(uint16_t)), "allocate staging");
    check(cudaMalloc((void **) &device_copies, 2 * sizeof(llama_kv_remap_copy)), "allocate descriptors");
    check(cudaMemcpy(data, initial.data(), initial.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
            "initialize remap data");
    const std::vector<llama_kv_remap_copy> copies = {
        { data, data + elements_per_block, 0, elements_per_block },
        { data + elements_per_block, data, elements_per_block, elements_per_block },
    };
    check(cudaMemcpy(device_copies, copies.data(), copies.size() * sizeof(llama_kv_remap_copy),
            cudaMemcpyHostToDevice), "upload descriptors");
    const auto accounting = llama_kv_remap_account(copies, staging);
    assert(accounting.vectorized_bytes == 2 * elements_per_block * sizeof(uint16_t));
    assert(accounting.scalar_bytes == 0);
    check(llama_kv_remap_launch_gather(device_copies, staging, copies.size(), elements_per_block,
            nullptr, llama_kv_remap_mode::vectorized), "launch vector gather");
    check(llama_kv_remap_launch_scatter(device_copies, staging, copies.size(), elements_per_block,
            nullptr, llama_kv_remap_mode::vectorized), "launch vector scatter");
    check(cudaDeviceSynchronize(), "wait vector remap");
    require_equal(expected, data, initial.size());

    // The benchmark compares the vectorized path with this scalar reference
    // implementation, so validate both modes against the same independent
    // snapshot oracle before using either result in a performance claim.
    check(cudaMemcpy(data, initial.data(), initial.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
            "reset scalar remap data");
    check(llama_kv_remap_launch_gather(device_copies, staging, copies.size(), elements_per_block,
            nullptr, llama_kv_remap_mode::scalar), "launch scalar gather");
    check(llama_kv_remap_launch_scatter(device_copies, staging, copies.size(), elements_per_block,
            nullptr, llama_kv_remap_mode::scalar), "launch scalar scatter");
    check(cudaDeviceSynchronize(), "wait scalar remap");
    require_equal(expected, data, initial.size());
    cudaFree(device_copies);
    cudaFree(staging);
    cudaFree(data);
}

static void test_unaligned_tail_falls_back_without_touching_guards() {
    constexpr size_t allocation_elements = 128;
    constexpr size_t elements = 19;
    std::vector<uint16_t> initial(allocation_elements, 0xa5a5);
    for (size_t i = 0; i < elements; ++i) initial[1 + i] = (uint16_t) (1000 + i);
    auto expected = initial;
    std::copy_n(initial.begin() + 1, elements, expected.begin() + 65);

    uint16_t * data = nullptr;
    uint16_t * staging = nullptr;
    llama_kv_remap_copy * device_copy = nullptr;
    check(cudaMalloc((void **) &data, initial.size() * sizeof(uint16_t)), "allocate tail data");
    check(cudaMalloc((void **) &staging, elements * sizeof(uint16_t)), "allocate tail staging");
    check(cudaMalloc((void **) &device_copy, sizeof(llama_kv_remap_copy)), "allocate tail descriptor");
    check(cudaMemcpy(data, initial.data(), initial.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
            "initialize tail data");
    const std::vector<llama_kv_remap_copy> copies = {{ data + 1, data + 65, 0, elements }};
    check(cudaMemcpy(device_copy, copies.data(), sizeof(llama_kv_remap_copy), cudaMemcpyHostToDevice),
            "upload tail descriptor");
    const auto accounting = llama_kv_remap_account(copies, staging);
    assert(accounting.vectorized_bytes == 0);
    assert(accounting.scalar_bytes == elements * sizeof(uint16_t));
    check(llama_kv_remap_launch_gather(device_copy, staging, 1, elements, nullptr,
            llama_kv_remap_mode::vectorized), "launch tail gather");
    check(llama_kv_remap_launch_scatter(device_copy, staging, 1, elements, nullptr,
            llama_kv_remap_mode::vectorized), "launch tail scatter");
    check(cudaDeviceSynchronize(), "wait tail remap");
    require_equal(expected, data, initial.size());
    cudaFree(device_copy);
    cudaFree(staging);
    cudaFree(data);
}

static void test_invalid_grid_shape_is_rejected_before_launch() {
    assert(llama_kv_remap_launch_gather(nullptr, nullptr, 65536, 1, nullptr,
            llama_kv_remap_mode::vectorized) == cudaErrorInvalidValue);
    assert(llama_kv_remap_launch_scatter(nullptr, nullptr, 65536, 1, nullptr,
            llama_kv_remap_mode::scalar) == cudaErrorInvalidValue);
    assert(llama_kv_remap_launch_gather(nullptr, nullptr, 1,
            std::numeric_limits<size_t>::max(), nullptr,
            llama_kv_remap_mode::vectorized) == cudaErrorInvalidValue);
    assert(llama_kv_remap_launch_scatter(nullptr, nullptr, 1,
            std::numeric_limits<size_t>::max(), nullptr,
            llama_kv_remap_mode::scalar) == cudaErrorInvalidValue);
}

int main() {
    test_aligned_overlap_uses_snapshot_and_vector_path();
    test_unaligned_tail_falls_back_without_touching_guards();
    test_invalid_grid_shape_is_rejected_before_launch();
    return 0;
}
