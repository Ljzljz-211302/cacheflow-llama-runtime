#include "llama-paged-decode-cuda.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" cudaError_t CUDARTAPI cudaProfilerStart(void);
extern "C" cudaError_t CUDARTAPI cudaProfilerStop(void);

namespace {

void check(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

uint16_t fp16(float value) { return __half_as_ushort(__float2half(value)); }
float fp32(uint16_t value) { return __half2float(__ushort_as_half(value)); }
enum class method { contiguous, paged_k1, paged_k2, paged_k2_t4, paged_k2_full };

const char * method_name(method selected) {
    switch (selected) {
        case method::contiguous: return "contiguous";
        case method::paged_k1: return "paged-k1";
        case method::paged_k2: return "paged-k2-t2";
        case method::paged_k2_t4: return "paged-k2-t4";
        case method::paged_k2_full: return "paged-k2-full";
    }
    return "unknown";
}

struct measured {
    double enqueue_ms = 0.0;
    float gpu_ms = 0.0f;
    double end_to_end_ms = 0.0;
};

struct regime {
    uint32_t context = 0;
    uint32_t batch = 0;
    bool fragmented = false;
    bool qwen7b_shape = false;
};

std::vector<float> cpu_paged_reference(
        const llama_paged_decode_config & config,
        const std::vector<uint32_t> & page_table,
        const std::vector<uint32_t> & contexts,
        const std::vector<uint16_t> & query,
        const std::vector<uint16_t> & k,
        const std::vector<uint16_t> & v) {
    std::vector<float> output(
            (size_t) config.batch_size * config.query_heads * config.head_dim);
    for (uint32_t sequence = 0; sequence < config.batch_size; ++sequence) {
        for (uint32_t query_head = 0; query_head < config.query_heads; ++query_head) {
            const uint32_t kv_head = query_head /
                    (config.query_heads / config.kv_heads);
            std::vector<float> scores(contexts[sequence]);
            float maximum = -INFINITY;
            for (uint32_t token = 0; token < contexts[sequence]; ++token) {
                const uint32_t physical = page_table[
                        (size_t) sequence * config.max_pages_per_sequence +
                        token / config.page_size];
                const size_t token_base = ((size_t) physical * config.page_size +
                        token % config.page_size) * config.kv_heads * config.head_dim;
                float dot = 0.0f;
                for (uint32_t dim = 0; dim < config.head_dim; ++dim) {
                    const size_t q_index = ((size_t) sequence * config.query_heads +
                            query_head) * config.head_dim + dim;
                    dot += fp32(query[q_index]) * fp32(k[
                            token_base + (size_t) kv_head * config.head_dim + dim]);
                }
                scores[token] = dot * config.scale;
                maximum = std::max(maximum, scores[token]);
            }
            float denominator = 0.0f;
            for (float score : scores) denominator += std::exp(score - maximum);
            for (uint32_t dim = 0; dim < config.head_dim; ++dim) {
                float numerator = 0.0f;
                for (uint32_t token = 0; token < contexts[sequence]; ++token) {
                    const uint32_t physical = page_table[
                            (size_t) sequence * config.max_pages_per_sequence +
                            token / config.page_size];
                    const size_t token_base = ((size_t) physical * config.page_size +
                            token % config.page_size) * config.kv_heads * config.head_dim;
                    numerator += std::exp(scores[token] - maximum) * fp32(v[
                            token_base + (size_t) kv_head * config.head_dim + dim]);
                }
                output[((size_t) sequence * config.query_heads + query_head) *
                        config.head_dim + dim] = numerator / denominator;
            }
        }
    }
    return output;
}

float require_cpu_oracle(
        const std::vector<float> & expected,
        const std::vector<float> & actual,
        uint32_t seed,
        const char * implementation) {
    float maximum_error = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float error = std::fabs(expected[i] - actual[i]);
        const float tolerance = 1e-3f + 1e-3f * std::fabs(expected[i]);
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i]) || error > tolerance) {
            std::fprintf(stderr,
                    "%s CPU-oracle mismatch seed=%u index=%zu expected=%.9f actual=%.9f\n",
                    implementation, seed, i, expected[i], actual[i]);
            std::exit(1);
        }
        maximum_error = std::max(maximum_error, error);
    }
    return maximum_error;
}

measured measure(method selected, const llama_paged_decode_plan * plan,
        const uint16_t * query, const uint16_t * paged_k, const uint16_t * paged_v,
        const uint16_t * contiguous_k, const uint16_t * contiguous_v, float * output,
        cudaStream_t stream) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaEventCreate(&start), "create start event");
    check(cudaEventCreate(&stop), "create stop event");
    const auto wall_start = std::chrono::steady_clock::now();
    check(cudaEventRecord(start, stream), "record start");
    if (selected == method::paged_k1) {
        check(llama_paged_decode_launch_paged(plan, query, paged_k, paged_v, output, stream),
                "launch paged K1");
    } else if (selected == method::paged_k2) {
        check(llama_paged_decode_launch_paged_k2(
                plan, query, paged_k, paged_v, output, stream), "launch paged K2");
    } else if (selected == method::paged_k2_t4) {
        check(llama_paged_decode_launch_paged_k2_tile(
                plan, query, paged_k, paged_v, output, stream, 4), "launch paged K2-T4");
    } else if (selected == method::paged_k2_full) {
        check(llama_paged_decode_launch_paged_k2_tile(
                plan, query, paged_k, paged_v, output, stream, 7), "launch paged K2-full");
    } else {
        check(llama_paged_decode_launch_contiguous(
                plan, query, contiguous_k, contiguous_v, output, stream),
                "launch contiguous");
    }
    check(cudaEventRecord(stop, stream), "record stop");
    const auto enqueue_stop = std::chrono::steady_clock::now();
    check(cudaEventSynchronize(stop), "wait stop");
    const auto wall_stop = std::chrono::steady_clock::now();
    measured result;
    result.enqueue_ms = std::chrono::duration<double, std::milli>(
            enqueue_stop - wall_start).count();
    result.end_to_end_ms = std::chrono::duration<double, std::milli>(
            wall_stop - wall_start).count();
    check(cudaEventElapsedTime(&result.gpu_ms, start, stop), "elapsed time");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return result;
}

void run_regime(const regime & selected, int repetitions, bool profile, bool paired,
        bool compare_k1_k2, method k2_method, method single_method, uint32_t seed_base) {
    const uint32_t query_heads = selected.qwen7b_shape ? 28u : 14u;
    const uint32_t kv_heads = selected.qwen7b_shape ? 4u : 2u;
    const uint32_t head_dim = selected.qwen7b_shape ? 128u : 64u;
    constexpr uint32_t page_size = 16;
    const uint32_t pages = (selected.context + page_size - 1) / page_size;
    const uint32_t physical_pages = pages * selected.batch;
    const llama_paged_decode_config config = {
        selected.batch, query_heads, kv_heads, head_dim, page_size, pages,
        physical_pages, 1.0f / std::sqrt((float) head_dim),
    };
    const uint32_t seed = seed_base + selected.context * 17u + selected.batch * 101u +
            (selected.fragmented ? 1009u : 0u) + (selected.qwen7b_shape ? 10007u : 0u);
    std::vector<uint32_t> physical_order(physical_pages);
    for (uint32_t i = 0; i < physical_pages; ++i) physical_order[i] = i;
    if (selected.fragmented) {
        std::mt19937 shuffle_rng(seed);
        std::shuffle(physical_order.begin(), physical_order.end(), shuffle_rng);
    }
    std::vector<uint32_t> page_table((size_t) selected.batch * pages);
    for (size_t i = 0; i < page_table.size(); ++i) page_table[i] = physical_order[i];
    std::vector<uint32_t> contexts(selected.batch, selected.context);
    llama_paged_decode_plan * plan = nullptr;
    check(llama_paged_decode_plan_create(
            config, page_table.data(), contexts.data(), &plan), "create benchmark plan");

    const uint32_t max_context = pages * page_size;
    const size_t query_elements = (size_t) selected.batch * query_heads * head_dim;
    const size_t paged_elements = (size_t) physical_pages * page_size * kv_heads * head_dim;
    const size_t contiguous_elements = (size_t) selected.batch * max_context * kv_heads * head_dim;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> values(-0.25f, 0.25f);
    std::vector<uint16_t> query(query_elements);
    std::vector<uint16_t> paged_k(paged_elements);
    std::vector<uint16_t> paged_v(paged_elements);
    for (auto & value : query) value = fp16(values(rng));
    for (auto & value : paged_k) value = fp16(values(rng));
    for (auto & value : paged_v) value = fp16(values(rng));
    std::vector<uint16_t> contiguous_k(contiguous_elements, fp16(0.0f));
    std::vector<uint16_t> contiguous_v(contiguous_elements, fp16(0.0f));
    for (uint32_t sequence = 0; sequence < selected.batch; ++sequence) {
        for (uint32_t token = 0; token < selected.context; ++token) {
            const uint32_t physical = page_table[(size_t) sequence * pages + token / page_size];
            const size_t source = ((size_t) physical * page_size + token % page_size) *
                    kv_heads * head_dim;
            const size_t destination = ((size_t) sequence * max_context + token) *
                    kv_heads * head_dim;
            std::copy_n(paged_k.begin() + source, kv_heads * head_dim,
                    contiguous_k.begin() + destination);
            std::copy_n(paged_v.begin() + source, kv_heads * head_dim,
                    contiguous_v.begin() + destination);
        }
    }
    const std::vector<float> cpu_output = cpu_paged_reference(
            config, page_table, contexts, query, paged_k, paged_v);

    uint16_t * d_query = nullptr;
    uint16_t * d_paged_k = nullptr;
    uint16_t * d_paged_v = nullptr;
    uint16_t * d_contiguous_k = nullptr;
    uint16_t * d_contiguous_v = nullptr;
    float * d_output = nullptr;
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    check(cudaMalloc((void **) &d_query, query_elements * sizeof(uint16_t)), "allocate Q");
    check(cudaMalloc((void **) &d_paged_k, paged_elements * sizeof(uint16_t)), "allocate paged K");
    check(cudaMalloc((void **) &d_paged_v, paged_elements * sizeof(uint16_t)), "allocate paged V");
    check(cudaMalloc((void **) &d_contiguous_k, contiguous_elements * sizeof(uint16_t)),
            "allocate contiguous K");
    check(cudaMalloc((void **) &d_contiguous_v, contiguous_elements * sizeof(uint16_t)),
            "allocate contiguous V");
    check(cudaMalloc((void **) &d_output, query_elements * sizeof(float)), "allocate output");
    check(cudaMemcpy(d_query, query.data(), query_elements * sizeof(uint16_t),
            cudaMemcpyHostToDevice), "upload Q");
    check(cudaMemcpy(d_paged_k, paged_k.data(), paged_elements * sizeof(uint16_t),
            cudaMemcpyHostToDevice), "upload paged K");
    check(cudaMemcpy(d_paged_v, paged_v.data(), paged_elements * sizeof(uint16_t),
            cudaMemcpyHostToDevice), "upload paged V");
    check(cudaMemcpy(d_contiguous_k, contiguous_k.data(), contiguous_elements * sizeof(uint16_t),
            cudaMemcpyHostToDevice), "upload contiguous K");
    check(cudaMemcpy(d_contiguous_v, contiguous_v.data(), contiguous_elements * sizeof(uint16_t),
            cudaMemcpyHostToDevice), "upload contiguous V");

    (void) measure(method::contiguous, plan, d_query, d_paged_k, d_paged_v,
            d_contiguous_k, d_contiguous_v, d_output, stream);
    std::vector<float> contiguous_output(query_elements);
    check(cudaMemcpy(contiguous_output.data(), d_output, query_elements * sizeof(float),
            cudaMemcpyDeviceToHost), "read contiguous reference");
    (void) measure(method::paged_k1, plan, d_query, d_paged_k, d_paged_v,
            d_contiguous_k, d_contiguous_v, d_output, stream);
    std::vector<float> paged_output(query_elements);
    check(cudaMemcpy(paged_output.data(), d_output, query_elements * sizeof(float),
            cudaMemcpyDeviceToHost), "read paged output");
    float maximum_error = std::max(
            require_cpu_oracle(cpu_output, contiguous_output, seed, "contiguous"),
            require_cpu_oracle(cpu_output, paged_output, seed, "paged"));
    (void) measure(method::paged_k2, plan, d_query, d_paged_k, d_paged_v,
            d_contiguous_k, d_contiguous_v, d_output, stream);
    check(cudaMemcpy(paged_output.data(), d_output, query_elements * sizeof(float),
            cudaMemcpyDeviceToHost), "read paged K2 output");
    maximum_error = std::max(
            maximum_error, require_cpu_oracle(cpu_output, paged_output, seed, "paged-k2"));

    if (profile) check(cudaProfilerStart(), "start profiler");
    std::mt19937 order_rng(seed);
    std::bernoulli_distribution contiguous_first(0.5);
    for (int trial = 0; trial < repetitions; ++trial) {
        const bool first_contiguous = contiguous_first(order_rng);
        const size_t bytes = (size_t) selected.batch * selected.context * kv_heads *
                head_dim * sizeof(uint16_t) * 2;
        if (paired) {
            const method baseline_method =
                    compare_k1_k2 ? method::paged_k1 : method::contiguous;
            const method candidate_method =
                    compare_k1_k2 ? k2_method : method::paged_k1;
            measured baseline{};
            measured candidate{};
            if (first_contiguous) {
                baseline = measure(baseline_method, plan, d_query, d_paged_k, d_paged_v,
                        d_contiguous_k, d_contiguous_v, d_output, stream);
                candidate = measure(candidate_method, plan, d_query, d_paged_k, d_paged_v,
                        d_contiguous_k, d_contiguous_v, d_output, stream);
            } else {
                candidate = measure(candidate_method, plan, d_query, d_paged_k, d_paged_v,
                        d_contiguous_k, d_contiguous_v, d_output, stream);
                baseline = measure(baseline_method, plan, d_query, d_paged_k, d_paged_v,
                        d_contiguous_k, d_contiguous_v, d_output, stream);
            }
            std::printf("confirmatory,%s,%s,%u,%u,%s,%d,%d,%u,%.6f,%.6f,%.6f,%.9f,%zu\n",
                    method_name(baseline_method),
                    selected.qwen7b_shape ? "qwen2.5-7b-shape" : "qwen2.5-0.5b",
                    selected.context, selected.batch, selected.fragmented ? "fragmented" : "identity",
                    trial, first_contiguous ? 0 : 1, seed, baseline.enqueue_ms,
                    baseline.gpu_ms, baseline.end_to_end_ms, maximum_error, bytes);
            std::printf("confirmatory,%s,%s,%u,%u,%s,%d,%d,%u,%.6f,%.6f,%.6f,%.9f,%zu\n",
                    method_name(candidate_method),
                    selected.qwen7b_shape ? "qwen2.5-7b-shape" : "qwen2.5-0.5b",
                    selected.context, selected.batch, selected.fragmented ? "fragmented" : "identity",
                    trial, first_contiguous ? 1 : 0, seed, candidate.enqueue_ms,
                    candidate.gpu_ms, candidate.end_to_end_ms, maximum_error, bytes);
        } else {
            const measured result = measure(single_method, plan, d_query, d_paged_k, d_paged_v,
                    d_contiguous_k, d_contiguous_v, d_output, stream);
            std::printf("profile,%s,%s,%u,%u,%s,%d,0,%u,%.6f,%.6f,%.6f,%.9f,%zu\n",
                    method_name(single_method),
                    selected.qwen7b_shape ? "qwen2.5-7b-shape" : "qwen2.5-0.5b",
                    selected.context, selected.batch,
                    selected.fragmented ? "fragmented" : "identity", trial, seed,
                    result.enqueue_ms, result.gpu_ms, result.end_to_end_ms,
                    maximum_error, bytes);
        }
    }
    if (profile) check(cudaProfilerStop(), "stop profiler");
    cudaFree(d_output);
    cudaFree(d_contiguous_v);
    cudaFree(d_contiguous_k);
    cudaFree(d_paged_v);
    cudaFree(d_paged_k);
    cudaFree(d_query);
    cudaStreamDestroy(stream);
    llama_paged_decode_plan_destroy(plan);
}

} // namespace

int main(int argc, char ** argv) {
    bool profile = false;
    bool paired = true;
    regime selected{ 0, 0, false, false };
    int repetitions = 10;
    uint32_t seed_base = 20260807u;
    bool compare_k1_k2 = false;
    method k2_method = method::paged_k2;
    method single_method = method::paged_k1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--profile") == 0) profile = true;
        else if (std::strcmp(argv[i], "--context") == 0 && i + 1 < argc)
            selected.context = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc)
            selected.batch = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--repetitions") == 0 && i + 1 < argc)
            repetitions = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--seed-base") == 0 && i + 1 < argc)
            seed_base = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--shape") == 0 && i + 1 < argc) {
            const char * value = argv[++i];
            if (std::strcmp(value, "qwen2.5-7b-shape") == 0) selected.qwen7b_shape = true;
            else if (std::strcmp(value, "qwen2.5-0.5b") != 0) return 2;
        }
        else if (std::strcmp(argv[i], "--layout") == 0 && i + 1 < argc) {
            const char * value = argv[++i];
            if (std::strcmp(value, "fragmented") == 0) selected.fragmented = true;
            else if (std::strcmp(value, "identity") != 0) return 2;
        } else if (std::strcmp(argv[i], "--method") == 0 && i + 1 < argc) {
            const char * value = argv[++i];
            if (std::strcmp(value, "paired") == 0) {
                paired = true;
                compare_k1_k2 = false;
            } else if (std::strcmp(value, "k1-k2") == 0) {
                paired = true;
                compare_k1_k2 = true;
                k2_method = method::paged_k2;
            } else if (std::strcmp(value, "k1-k2-t4") == 0) {
                paired = true;
                compare_k1_k2 = true;
                k2_method = method::paged_k2_t4;
            } else if (std::strcmp(value, "k1-k2-full") == 0) {
                paired = true;
                compare_k1_k2 = true;
                k2_method = method::paged_k2_full;
            } else if (std::strcmp(value, "paged") == 0 ||
                    std::strcmp(value, "paged-k1") == 0) {
                paired = false; single_method = method::paged_k1;
            } else if (std::strcmp(value, "paged-k2") == 0) {
                paired = false; single_method = method::paged_k2;
            } else if (std::strcmp(value, "paged-k2-t4") == 0) {
                paired = false; single_method = method::paged_k2_t4;
            } else if (std::strcmp(value, "paged-k2-full") == 0) {
                paired = false; single_method = method::paged_k2_full;
            }
            else if (std::strcmp(value, "contiguous") == 0) {
                paired = false; single_method = method::contiguous;
            } else return 2;
        } else return 2;
    }
    if (repetitions < 1) return 2;
    std::puts("phase,method,shape,context,batch,layout,trial,order_in_pair,random_seed,host_enqueue_ms,gpu_ms,end_to_end_ms,max_abs_error,logical_kv_bytes");
    if (selected.context != 0 || selected.batch != 0) {
        if (selected.context == 0 || selected.batch == 0) return 2;
        run_regime(selected, repetitions, profile, paired, compare_k1_k2, k2_method,
                single_method, seed_base);
        return 0;
    }
    if (profile) return 2;
    for (bool qwen7b_shape : { false, true }) {
        for (uint32_t context : { 16u, 64u, 256u, 1024u }) {
            for (uint32_t batch : { 1u, 4u }) {
                for (bool fragmented : { false, true }) {
                    run_regime({ context, batch, fragmented, qwen7b_shape }, repetitions,
                            false, true, false, method::paged_k2,
                            method::paged_k1, seed_base);
                }
            }
        }
    }
    return 0;
}
