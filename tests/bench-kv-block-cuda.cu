#include "server-kv-block-backend.h"
#include "llama-kv-remap-cuda.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Some redistributable CUDA developer packages omit cuda_profiler_api.h even
// though cudart exports these stable APIs. Keep the declarations local so the
// profiling-only benchmark mode also builds in that supported package layout.
extern "C" cudaError_t CUDARTAPI cudaProfilerStart(void);
extern "C" cudaError_t CUDARTAPI cudaProfilerStop(void);

namespace {

void check(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

struct timings {
    double host_enqueue_ms;
    float gpu_ms;
    double end_to_end_ms;
};

timings measure_batch(
        uint16_t * staging,
        llama_kv_remap_copy * device_copies,
        const std::vector<llama_kv_remap_copy> & copies,
        size_t max_elements,
        cudaStream_t stream,
        llama_kv_remap_mode mode) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaEventCreate(&start), "create batch start event");
    check(cudaEventCreate(&stop), "create batch stop event");
    const auto wall_start = std::chrono::steady_clock::now();
    check(cudaMemcpyAsync(device_copies, copies.data(),
            copies.size() * sizeof(llama_kv_remap_copy), cudaMemcpyHostToDevice, stream),
            "upload descriptors");
    check(cudaEventRecord(start, stream), "record batch start");
    check(llama_kv_remap_launch_gather(device_copies, staging, copies.size(), max_elements,
            stream, mode), "launch KV gather");
    check(llama_kv_remap_launch_scatter(device_copies, staging, copies.size(), max_elements,
            stream, mode), "launch KV scatter");
    check(cudaEventRecord(stop, stream), "record batch stop");
    const auto enqueue_stop = std::chrono::steady_clock::now();
    check(cudaEventSynchronize(stop), "wait batch stop");
    const auto wall_stop = std::chrono::steady_clock::now();
    float gpu_ms = 0.0f;
    check(cudaEventElapsedTime(&gpu_ms, start, stop), "measure batch events");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return {
        std::chrono::duration<double, std::milli>(enqueue_stop - wall_start).count(),
        gpu_ms,
        std::chrono::duration<double, std::milli>(wall_stop - wall_start).count() };
}

} // namespace

int main(int argc, char ** argv) {
    bool profile = false;
    size_t profile_blocks = 0;
    int profile_repetitions = 5;
    bool profile_paired = false;
    bool profile_misaligned = false;
    llama_kv_remap_mode profile_mode = llama_kv_remap_mode::vectorized;
    const char * profile_method = "vectorized_gather_scatter";
    for (int argument = 1; argument < argc; ++argument) {
        if (std::strcmp(argv[argument], "--profile") == 0) {
            profile = true;
        } else if (std::strcmp(argv[argument], "--blocks") == 0 && argument + 1 < argc) {
            profile_blocks = std::strtoull(argv[++argument], nullptr, 10);
        } else if (std::strcmp(argv[argument], "--repetitions") == 0 && argument + 1 < argc) {
            profile_repetitions = std::atoi(argv[++argument]);
        } else if (std::strcmp(argv[argument], "--method") == 0 && argument + 1 < argc) {
            const char * method = argv[++argument];
            if (std::strcmp(method, "scalar") == 0) {
                profile_mode = llama_kv_remap_mode::scalar;
                profile_method = "scalar_gather_scatter";
            } else if (std::strcmp(method, "paired") == 0) {
                profile_paired = true;
            } else if (std::strcmp(method, "vectorized") != 0) {
                std::fprintf(stderr, "unsupported profile method: %s\n", method);
                return 2;
            }
        } else if (std::strcmp(argv[argument], "--layout") == 0 && argument + 1 < argc) {
            const char * layout_name = argv[++argument];
            if (std::strcmp(layout_name, "misaligned") == 0) {
                profile_misaligned = true;
            } else if (std::strcmp(layout_name, "aligned") != 0) {
                std::fprintf(stderr, "unsupported profile layout: %s\n", layout_name);
                return 2;
            }
        } else {
            std::fprintf(stderr, "unsupported benchmark argument: %s\n", argv[argument]);
            return 2;
        }
    }
    if (profile &&
            (profile_blocks != 1 && profile_blocks != 4 &&
             profile_blocks != 16 && profile_blocks != 32)) {
        std::fprintf(stderr, "--profile requires --blocks in {1,4,16,32}\n");
        return 2;
    }
    if (profile && profile_repetitions < 1) {
        std::fprintf(stderr, "--repetitions must be positive\n");
        return 2;
    }

    constexpr size_t capacity = 96;
    const server_kv_tensor_layout layout = {
        32, 8, 128, 16, server_kv_element_type::fp16,
        server_kv_memory_layout::separate_k_v_planes,
    };
    const size_t elements = layout.elements_per_plane_block();
    const size_t plane_elements = capacity * elements;
    uint16_t * k = nullptr;
    uint16_t * v = nullptr;
    uint16_t * staging = nullptr;
    llama_kv_remap_copy * device_copies = nullptr;
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create benchmark stream");
    check(cudaMalloc((void **) &k, plane_elements * sizeof(uint16_t)), "allocate K");
    check(cudaMalloc((void **) &v, plane_elements * sizeof(uint16_t)), "allocate V");
    check(cudaMalloc((void **) &staging,
            (64 * elements + LLAMA_KV_REMAP_VECTOR_ELEMENTS) * sizeof(uint16_t)),
            "allocate staging");
    check(cudaMalloc((void **) &device_copies, 64 * sizeof(llama_kv_remap_copy)), "allocate descriptors");
    check(cudaMemset(k, 0x2a, plane_elements * sizeof(uint16_t)), "initialize K");
    check(cudaMemset(v, 0x5c, plane_elements * sizeof(uint16_t)), "initialize V");

    if (profile) {
        std::vector<llama_kv_remap_copy> copies;
        for (size_t i = 0; i < profile_blocks; ++i) {
            const size_t source = i;
            const size_t destination = 48 + i;
            const size_t offset = profile_misaligned ? 1 : 0;
            const size_t copied_elements = profile_misaligned ? elements - 1 : elements;
            copies.push_back({ k + source * elements + offset,
                    k + destination * elements + offset,
                    2 * i * elements + offset, copied_elements });
            copies.push_back({ v + source * elements + offset,
                    v + destination * elements + offset,
                    (2 * i + 1) * elements + offset, copied_elements });
        }
        // Warm-up is deliberately outside cudaProfilerStart/Stop so NCU/NSYS capture
        // only the named mechanism sample. These replay-instrumented timings are not
        // used as the no-profiler latency result.
        if (profile_paired) {
            (void) measure_batch(staging, device_copies, copies, elements, stream,
                    llama_kv_remap_mode::scalar);
            (void) measure_batch(staging, device_copies, copies, elements, stream,
                    llama_kv_remap_mode::vectorized);
        } else {
            (void) measure_batch(staging, device_copies, copies, elements, stream, profile_mode);
        }
        check(cudaProfilerStart(), "start profiler capture range");
        std::puts("phase,method,blocks,trial,order_in_pair,random_seed,host_enqueue_ms,gpu_ms,end_to_end_ms,bytes");
        std::mt19937 profile_order_rng(20260806 + (unsigned) profile_blocks +
                (profile_misaligned ? 1000u : 0u));
        std::bernoulli_distribution profile_scalar_first(0.5);
        for (int trial = 0; trial < profile_repetitions; ++trial) {
            if (profile_paired) {
                timings scalar{};
                timings vectorized{};
                const bool scalar_first = profile_scalar_first(profile_order_rng);
                if (scalar_first) {
                    scalar = measure_batch(staging, device_copies, copies, elements, stream,
                            llama_kv_remap_mode::scalar);
                    vectorized = measure_batch(staging, device_copies, copies, elements, stream,
                            llama_kv_remap_mode::vectorized);
                } else {
                    vectorized = measure_batch(staging, device_copies, copies, elements, stream,
                            llama_kv_remap_mode::vectorized);
                    scalar = measure_batch(staging, device_copies, copies, elements, stream,
                            llama_kv_remap_mode::scalar);
                }
                const size_t bytes = profile_blocks * layout.bytes_per_block() -
                        (profile_misaligned ? 4 * profile_blocks : 0);
                const unsigned random_seed = 20260806 + (unsigned) profile_blocks +
                        (profile_misaligned ? 1000u : 0u);
                std::printf("profile,scalar_gather_scatter,%zu,%d,%d,%u,%.6f,%.6f,%.6f,%zu\n",
                        profile_blocks, trial, scalar_first ? 0 : 1, random_seed,
                        scalar.host_enqueue_ms, scalar.gpu_ms, scalar.end_to_end_ms, bytes);
                std::printf("profile,vectorized_gather_scatter,%zu,%d,%d,%u,%.6f,%.6f,%.6f,%zu\n",
                        profile_blocks, trial, scalar_first ? 1 : 0, random_seed,
                        vectorized.host_enqueue_ms, vectorized.gpu_ms,
                        vectorized.end_to_end_ms, bytes);
                continue;
            }
            const timings measured = measure_batch(
                    staging, device_copies, copies, elements, stream, profile_mode);
            std::printf("profile,%s,%zu,%d,0,20260806,%.6f,%.6f,%.6f,%zu\n",
                    profile_method, profile_blocks, trial,
                    measured.host_enqueue_ms, measured.gpu_ms, measured.end_to_end_ms,
                    profile_blocks * layout.bytes_per_block());
        }
        check(cudaProfilerStop(), "stop profiler capture range");
        cudaFree(device_copies);
        cudaFree(staging);
        cudaFree(v);
        cudaFree(k);
        cudaStreamDestroy(stream);
        return 0;
    }

    std::mt19937 order_rng(20260806);
    std::bernoulli_distribution scalar_first_distribution(0.5);
    std::puts("phase,method,blocks,trial,order_in_pair,random_seed,host_enqueue_ms,gpu_ms,end_to_end_ms,bytes");
    for (const size_t count : { 1u, 4u, 16u, 32u }) {
        std::vector<server_kv_block_copy> mapping;
        for (size_t i = 0; i < count; ++i) mapping.push_back({ (uint32_t) i, (uint32_t) (48 + i) });
        std::vector<llama_kv_remap_copy> copies;
        for (size_t i = 0; i < mapping.size(); ++i) {
            const auto & copy = mapping[i];
            copies.push_back({ k + (size_t) copy.source * elements,
                    k + (size_t) copy.destination * elements, 2 * i * elements, elements });
            copies.push_back({ v + (size_t) copy.source * elements,
                    v + (size_t) copy.destination * elements, (2 * i + 1) * elements, elements });
        }
        timings warmup_scalar{};
        timings warmup_vectorized{};
        const bool warmup_scalar_first = scalar_first_distribution(order_rng);
        if (warmup_scalar_first) {
            warmup_scalar = measure_batch(staging, device_copies, copies, elements,
                    stream, llama_kv_remap_mode::scalar);
            warmup_vectorized = measure_batch(staging, device_copies, copies, elements,
                    stream, llama_kv_remap_mode::vectorized);
        } else {
            warmup_vectorized = measure_batch(staging, device_copies, copies, elements,
                    stream, llama_kv_remap_mode::vectorized);
            warmup_scalar = measure_batch(staging, device_copies, copies, elements,
                    stream, llama_kv_remap_mode::scalar);
        }
        std::printf("warmup,scalar_gather_scatter,%zu,-1,%d,20260806,%.6f,%.6f,%.6f,%zu\n", count,
                warmup_scalar_first ? 0 : 1,
                warmup_scalar.host_enqueue_ms, warmup_scalar.gpu_ms, warmup_scalar.end_to_end_ms,
                count * layout.bytes_per_block());
        std::printf("warmup,vectorized_gather_scatter,%zu,-1,%d,20260806,%.6f,%.6f,%.6f,%zu\n", count,
                warmup_scalar_first ? 1 : 0,
                warmup_vectorized.host_enqueue_ms, warmup_vectorized.gpu_ms,
                warmup_vectorized.end_to_end_ms, count * layout.bytes_per_block());
        for (int trial = 0; trial < 20; ++trial) {
            timings scalar{};
            timings vectorized{};
            const bool scalar_first = scalar_first_distribution(order_rng);
            if (scalar_first) {
                scalar = measure_batch(staging, device_copies, copies, elements, stream,
                        llama_kv_remap_mode::scalar);
                vectorized = measure_batch(staging, device_copies, copies, elements, stream,
                        llama_kv_remap_mode::vectorized);
            } else {
                vectorized = measure_batch(staging, device_copies, copies, elements, stream,
                        llama_kv_remap_mode::vectorized);
                scalar = measure_batch(staging, device_copies, copies, elements, stream,
                        llama_kv_remap_mode::scalar);
            }
            std::printf("confirmatory,scalar_gather_scatter,%zu,%d,%d,20260806,%.6f,%.6f,%.6f,%zu\n",
                    count, trial, scalar_first ? 0 : 1,
                    scalar.host_enqueue_ms, scalar.gpu_ms, scalar.end_to_end_ms,
                    count * layout.bytes_per_block());
            std::printf("confirmatory,vectorized_gather_scatter,%zu,%d,%d,20260806,%.6f,%.6f,%.6f,%zu\n",
                    count, trial, scalar_first ? 1 : 0,
                    vectorized.host_enqueue_ms, vectorized.gpu_ms, vectorized.end_to_end_ms,
                    count * layout.bytes_per_block());
        }
    }
    cudaFree(device_copies);
    cudaFree(staging);
    cudaFree(v);
    cudaFree(k);
    cudaStreamDestroy(stream);
    return 0;
}
