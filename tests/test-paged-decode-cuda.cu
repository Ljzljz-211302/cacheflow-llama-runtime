#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-paged-decode-cuda.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

namespace {

void check(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

uint16_t fp16(float value) {
    return __half_as_ushort(__float2half(value));
}

float fp32(uint16_t value) {
    return __half2float(__ushort_as_half(value));
}

std::vector<float> cpu_reference(
        const llama_paged_decode_config & config,
        const std::vector<uint32_t> & page_table,
        const std::vector<uint32_t> & contexts,
        const std::vector<uint16_t> & query,
        const std::vector<uint16_t> & k,
        const std::vector<uint16_t> & v,
        bool paged) {
    const uint32_t max_context = config.max_pages_per_sequence * config.page_size;
    std::vector<float> output(
            (size_t) config.batch_size * config.query_heads * config.head_dim);
    for (uint32_t sequence = 0; sequence < config.batch_size; ++sequence) {
        for (uint32_t query_head = 0; query_head < config.query_heads; ++query_head) {
            const uint32_t kv_head = query_head /
                    (config.query_heads / config.kv_heads);
            std::vector<float> scores(contexts[sequence]);
            float maximum = -std::numeric_limits<float>::infinity();
            for (uint32_t token = 0; token < contexts[sequence]; ++token) {
                size_t token_base = 0;
                if (paged) {
                    const uint32_t physical = page_table[
                            (size_t) sequence * config.max_pages_per_sequence +
                            token / config.page_size];
                    token_base = ((size_t) physical * config.page_size +
                            token % config.page_size) * config.kv_heads * config.head_dim;
                } else {
                    token_base = ((size_t) sequence * max_context + token) *
                            config.kv_heads * config.head_dim;
                }
                float dot = 0.0f;
                for (uint32_t dim = 0; dim < config.head_dim; ++dim) {
                    const size_t q_index = ((size_t) sequence * config.query_heads +
                            query_head) * config.head_dim + dim;
                    dot += fp32(query[q_index]) *
                            fp32(k[token_base + (size_t) kv_head * config.head_dim + dim]);
                }
                scores[token] = dot * config.scale;
                maximum = std::max(maximum, scores[token]);
            }
            float denominator = 0.0f;
            for (float score : scores) denominator += std::exp(score - maximum);
            for (uint32_t dim = 0; dim < config.head_dim; ++dim) {
                float value = 0.0f;
                for (uint32_t token = 0; token < contexts[sequence]; ++token) {
                    size_t token_base = 0;
                    if (paged) {
                        const uint32_t physical = page_table[
                                (size_t) sequence * config.max_pages_per_sequence +
                                token / config.page_size];
                        token_base = ((size_t) physical * config.page_size +
                                token % config.page_size) * config.kv_heads * config.head_dim;
                    } else {
                        token_base = ((size_t) sequence * max_context + token) *
                                config.kv_heads * config.head_dim;
                    }
                    value += std::exp(scores[token] - maximum) *
                            fp32(v[token_base + (size_t) kv_head * config.head_dim + dim]);
                }
                output[((size_t) sequence * config.query_heads + query_head) *
                        config.head_dim + dim] = value / denominator;
            }
        }
    }
    return output;
}

void require_close(const std::vector<float> & expected, const std::vector<float> & actual,
        uint32_t seed, const char * implementation) {
    assert(expected.size() == actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        const float tolerance = 1e-3f + 1e-3f * std::fabs(expected[i]);
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i]) ||
                std::fabs(expected[i] - actual[i]) > tolerance) {
            std::fprintf(stderr,
                    "%s mismatch seed=%u index=%zu: expected %.8f actual %.8f\n",
                    implementation, seed, i, expected[i], actual[i]);
            std::abort();
        }
    }
}

void test_non_contiguous_boundary_computes_attention_output() {
    const llama_paged_decode_config config = {
        1, 14, 2, 64, 16, 2, 4, 1.0f / std::sqrt(64.0f),
    };
    const uint32_t page_table[] = { 3, 1 };
    const uint32_t context_lengths[] = { 17 };
    llama_paged_decode_plan * plan = nullptr;
    check(llama_paged_decode_plan_create(
            config, page_table, context_lengths, &plan), "create plan");
    assert(plan != nullptr);

    const size_t query_elements = config.query_heads * config.head_dim;
    const size_t plane_elements = (size_t) config.physical_pages * config.page_size *
            config.kv_heads * config.head_dim;
    std::vector<uint16_t> query(query_elements, fp16(0.0f));
    std::vector<uint16_t> k(plane_elements, fp16(0.0f));
    std::vector<uint16_t> v(plane_elements, fp16(-100.0f));
    for (uint32_t token = 0; token < 17; ++token) {
        const uint32_t page = page_table[token / config.page_size];
        const uint32_t offset = token % config.page_size;
        for (uint32_t kv_head = 0; kv_head < config.kv_heads; ++kv_head) {
            for (uint32_t dim = 0; dim < config.head_dim; ++dim) {
                const size_t index = (((size_t) page * config.page_size + offset) *
                        config.kv_heads + kv_head) * config.head_dim + dim;
                v[index] = fp16((float) token + 1.0f);
            }
        }
    }

    uint16_t * device_query = nullptr;
    uint16_t * device_k = nullptr;
    uint16_t * device_v = nullptr;
    float * device_output = nullptr;
    check(cudaMalloc((void **) &device_query, query.size() * sizeof(uint16_t)), "alloc Q");
    check(cudaMalloc((void **) &device_k, k.size() * sizeof(uint16_t)), "alloc K");
    check(cudaMalloc((void **) &device_v, v.size() * sizeof(uint16_t)), "alloc V");
    check(cudaMalloc((void **) &device_output, query_elements * sizeof(float)), "alloc output");
    check(cudaMemcpy(device_query, query.data(), query.size() * sizeof(uint16_t),
            cudaMemcpyHostToDevice), "upload Q");
    check(cudaMemcpy(device_k, k.data(), k.size() * sizeof(uint16_t),
            cudaMemcpyHostToDevice), "upload K");
    check(cudaMemcpy(device_v, v.data(), v.size() * sizeof(uint16_t),
            cudaMemcpyHostToDevice), "upload V");
    check(llama_paged_decode_launch_paged(
            plan, device_query, device_k, device_v, device_output, nullptr), "launch paged");
    check(cudaDeviceSynchronize(), "wait paged");
    std::vector<float> output(query_elements);
    check(cudaMemcpy(output.data(), device_output, output.size() * sizeof(float),
            cudaMemcpyDeviceToHost), "read output");
    for (float value : output) assert(std::fabs(value - 9.0f) < 1e-5f);
    check(llama_paged_decode_launch_paged_k2(
            plan, device_query, device_k, device_v, device_output, nullptr), "launch paged K2");
    check(cudaDeviceSynchronize(), "wait paged K2");
    check(cudaMemcpy(output.data(), device_output, output.size() * sizeof(float),
            cudaMemcpyDeviceToHost), "read K2 output");
    for (float value : output) assert(std::fabs(value - 9.0f) < 1e-5f);

    cudaFree(device_output);
    cudaFree(device_v);
    cudaFree(device_k);
    cudaFree(device_query);
    llama_paged_decode_plan_destroy(plan);
}

void run_randomized_gqa_matches_independent_contiguous_oracle(
        const llama_paged_decode_config & config,
        const std::vector<uint32_t> & page_table,
        const std::vector<uint32_t> & contexts,
        uint32_t seed) {
    const uint32_t max_context = config.max_pages_per_sequence * config.page_size;
    const size_t query_elements = (size_t) config.batch_size * config.query_heads *
            config.head_dim;
    const size_t paged_elements = (size_t) config.physical_pages * config.page_size *
            config.kv_heads * config.head_dim;
    const size_t contiguous_elements = (size_t) config.batch_size * max_context *
            config.kv_heads * config.head_dim;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> values(-1.0f, 1.0f);
    std::vector<uint16_t> query(query_elements);
    std::vector<uint16_t> paged_k(paged_elements);
    std::vector<uint16_t> paged_v(paged_elements);
    for (auto & item : query) item = fp16(values(rng));
    for (auto & item : paged_k) item = fp16(values(rng));
    for (auto & item : paged_v) item = fp16(values(rng));
    std::vector<uint16_t> contiguous_k(contiguous_elements, fp16(0.0f));
    std::vector<uint16_t> contiguous_v(contiguous_elements, fp16(0.0f));
    for (uint32_t sequence = 0; sequence < config.batch_size; ++sequence) {
        for (uint32_t token = 0; token < contexts[sequence]; ++token) {
            const uint32_t physical = page_table[
                    (size_t) sequence * config.max_pages_per_sequence +
                    token / config.page_size];
            const size_t source = ((size_t) physical * config.page_size +
                    token % config.page_size) * config.kv_heads * config.head_dim;
            const size_t destination = ((size_t) sequence * max_context + token) *
                    config.kv_heads * config.head_dim;
            std::copy_n(paged_k.begin() + source,
                    config.kv_heads * config.head_dim, contiguous_k.begin() + destination);
            std::copy_n(paged_v.begin() + source,
                    config.kv_heads * config.head_dim, contiguous_v.begin() + destination);
        }
    }
    const auto expected = cpu_reference(
            config, page_table, contexts, query, contiguous_k, contiguous_v, false);

    llama_paged_decode_plan * plan = nullptr;
    check(llama_paged_decode_plan_create(config, page_table.data(), contexts.data(), &plan),
            "create randomized plan");
    uint16_t * device_query = nullptr;
    uint16_t * device_paged_k = nullptr;
    uint16_t * device_paged_v = nullptr;
    uint16_t * device_contiguous_k = nullptr;
    uint16_t * device_contiguous_v = nullptr;
    float * device_output_allocation = nullptr;
    check(cudaMalloc((void **) &device_query, query_elements * sizeof(uint16_t)), "alloc random Q");
    check(cudaMalloc((void **) &device_paged_k, paged_elements * sizeof(uint16_t)), "alloc paged K");
    check(cudaMalloc((void **) &device_paged_v, paged_elements * sizeof(uint16_t)), "alloc paged V");
    check(cudaMalloc((void **) &device_contiguous_k,
            contiguous_elements * sizeof(uint16_t)), "alloc contiguous K");
    check(cudaMalloc((void **) &device_contiguous_v,
            contiguous_elements * sizeof(uint16_t)), "alloc contiguous V");
    check(cudaMalloc((void **) &device_output_allocation,
            (query_elements + 2) * sizeof(float)), "alloc guarded output");
    check(cudaMemcpy(device_query, query.data(), query_elements * sizeof(uint16_t),
            cudaMemcpyHostToDevice), "upload random Q");
    check(cudaMemcpy(device_paged_k, paged_k.data(), paged_elements * sizeof(uint16_t),
            cudaMemcpyHostToDevice), "upload paged K");
    check(cudaMemcpy(device_paged_v, paged_v.data(), paged_elements * sizeof(uint16_t),
            cudaMemcpyHostToDevice), "upload paged V");
    check(cudaMemcpy(device_contiguous_k, contiguous_k.data(),
            contiguous_elements * sizeof(uint16_t), cudaMemcpyHostToDevice),
            "upload contiguous K");
    check(cudaMemcpy(device_contiguous_v, contiguous_v.data(),
            contiguous_elements * sizeof(uint16_t), cudaMemcpyHostToDevice),
            "upload contiguous V");
    const float guard = 123456.0f;
    check(cudaMemcpy(device_output_allocation, &guard, sizeof(float),
            cudaMemcpyHostToDevice), "write prefix guard");
    check(cudaMemcpy(device_output_allocation + query_elements + 1, &guard, sizeof(float),
            cudaMemcpyHostToDevice), "write suffix guard");
    float * device_output = device_output_allocation + 1;
    std::vector<float> observed(query_elements);
    check(llama_paged_decode_launch_paged(plan, device_query, device_paged_k,
            device_paged_v, device_output, nullptr), "launch randomized paged");
    check(cudaDeviceSynchronize(), "wait randomized paged");
    check(cudaMemcpy(observed.data(), device_output, query_elements * sizeof(float),
            cudaMemcpyDeviceToHost), "read randomized paged");
    require_close(expected, observed, seed, "paged");
    check(llama_paged_decode_launch_paged_k2(plan, device_query, device_paged_k,
            device_paged_v, device_output, nullptr), "launch randomized paged K2");
    check(cudaDeviceSynchronize(), "wait randomized paged K2");
    check(cudaMemcpy(observed.data(), device_output, query_elements * sizeof(float),
            cudaMemcpyDeviceToHost), "read randomized paged K2");
    require_close(expected, observed, seed, "paged-k2");
    for (uint32_t tile : { 4u, 7u }) {
        check(llama_paged_decode_launch_paged_k2_tile(plan, device_query, device_paged_k,
                device_paged_v, device_output, nullptr, tile), "launch K2 tile ablation");
        check(cudaDeviceSynchronize(), "wait K2 tile ablation");
        check(cudaMemcpy(observed.data(), device_output, query_elements * sizeof(float),
                cudaMemcpyDeviceToHost), "read K2 tile ablation");
        require_close(expected, observed, seed, tile == 4 ? "paged-k2-t4" : "paged-k2-full");
    }
    check(llama_paged_decode_launch_contiguous(plan, device_query, device_contiguous_k,
            device_contiguous_v, device_output, nullptr), "launch contiguous");
    check(cudaDeviceSynchronize(), "wait contiguous");
    check(cudaMemcpy(observed.data(), device_output, query_elements * sizeof(float),
            cudaMemcpyDeviceToHost), "read contiguous");
    require_close(expected, observed, seed, "contiguous");
    float prefix = 0.0f;
    float suffix = 0.0f;
    check(cudaMemcpy(&prefix, device_output_allocation, sizeof(float),
            cudaMemcpyDeviceToHost), "read prefix guard");
    check(cudaMemcpy(&suffix, device_output_allocation + query_elements + 1, sizeof(float),
            cudaMemcpyDeviceToHost), "read suffix guard");
    assert(prefix == guard && suffix == guard);

    cudaFree(device_output_allocation);
    cudaFree(device_contiguous_v);
    cudaFree(device_contiguous_k);
    cudaFree(device_paged_v);
    cudaFree(device_paged_k);
    cudaFree(device_query);
    llama_paged_decode_plan_destroy(plan);
}

void test_randomized_gqa_matches_independent_contiguous_oracle() {
    const std::vector<uint32_t> page_table = { 7, 2, 10, 1, 5, 9, 0, 11 };
    const std::vector<uint32_t> contexts = { 31, 49 };
    run_randomized_gqa_matches_independent_contiguous_oracle(
            { 2, 14, 2, 64, 16, 4, 12, 1.0f / std::sqrt(64.0f) },
            page_table, contexts, 20260807);
    run_randomized_gqa_matches_independent_contiguous_oracle(
            { 2, 28, 4, 128, 16, 4, 12, 1.0f / std::sqrt(128.0f) },
            page_table, contexts, 20260808);
}

void test_causal_tail_stops_at_every_page_boundary() {
    const llama_paged_decode_config config = {
        1, 14, 2, 64, 16, 2, 4, 1.0f / std::sqrt(64.0f),
    };
    const std::vector<uint32_t> page_table = { 2, 0 };
    const size_t query_elements = (size_t) config.query_heads * config.head_dim;
    const size_t plane_elements = (size_t) config.physical_pages * config.page_size *
            config.kv_heads * config.head_dim;
    std::vector<uint16_t> query(query_elements, fp16(0.0f));
    std::vector<uint16_t> k(plane_elements, fp16(0.0f));
    std::vector<uint16_t> v(plane_elements, fp16(1000.0f));
    for (uint32_t token = 0; token < 32; ++token) {
        const uint32_t physical = page_table[token / config.page_size];
        const uint32_t offset = token % config.page_size;
        for (uint32_t head = 0; head < config.kv_heads; ++head) {
            for (uint32_t dim = 0; dim < config.head_dim; ++dim) {
                const size_t index = (((size_t) physical * config.page_size + offset) *
                        config.kv_heads + head) * config.head_dim + dim;
                v[index] = fp16((float) token + 1.0f);
            }
        }
    }
    uint16_t * d_query = nullptr;
    uint16_t * d_k = nullptr;
    uint16_t * d_v = nullptr;
    float * d_output = nullptr;
    check(cudaMalloc((void **) &d_query, query_elements * sizeof(uint16_t)), "alloc tail Q");
    check(cudaMalloc((void **) &d_k, plane_elements * sizeof(uint16_t)), "alloc tail K");
    check(cudaMalloc((void **) &d_v, plane_elements * sizeof(uint16_t)), "alloc tail V");
    check(cudaMalloc((void **) &d_output, query_elements * sizeof(float)), "alloc tail output");
    check(cudaMemcpy(d_query, query.data(), query_elements * sizeof(uint16_t),
            cudaMemcpyHostToDevice), "upload tail Q");
    check(cudaMemcpy(d_k, k.data(), plane_elements * sizeof(uint16_t),
            cudaMemcpyHostToDevice), "upload tail K");
    check(cudaMemcpy(d_v, v.data(), plane_elements * sizeof(uint16_t),
            cudaMemcpyHostToDevice), "upload tail V");
    for (uint32_t context : { 1u, 15u, 16u, 17u, 31u, 32u }) {
        llama_paged_decode_plan * plan = nullptr;
        check(llama_paged_decode_plan_create(
                config, page_table.data(), &context, &plan), "create tail plan");
        check(llama_paged_decode_launch_paged(
                plan, d_query, d_k, d_v, d_output, nullptr), "launch tail case");
        check(cudaDeviceSynchronize(), "wait tail case");
        std::vector<float> output(query_elements);
        check(cudaMemcpy(output.data(), d_output, query_elements * sizeof(float),
                cudaMemcpyDeviceToHost), "read tail output");
        const float expected = ((float) context + 1.0f) / 2.0f;
        for (float value : output) assert(std::fabs(value - expected) < 1e-5f);
        check(llama_paged_decode_launch_paged_k2(
                plan, d_query, d_k, d_v, d_output, nullptr), "launch K2 tail case");
        check(cudaDeviceSynchronize(), "wait K2 tail case");
        check(cudaMemcpy(output.data(), d_output, query_elements * sizeof(float),
                cudaMemcpyDeviceToHost), "read K2 tail output");
        for (float value : output) assert(std::fabs(value - expected) < 1e-5f);
        llama_paged_decode_plan_destroy(plan);
    }
    cudaFree(d_output);
    cudaFree(d_v);
    cudaFree(d_k);
    cudaFree(d_query);
}

void test_unsupported_and_invalid_shapes_fail_closed() {
    llama_paged_decode_config config = {
        1, 8, 2, 128, 16, 2, 4, 1.0f / std::sqrt(128.0f),
    };
    const char * reason = nullptr;
    config.head_dim = 96;
    assert(!llama_paged_decode_supported(config, &reason));
    assert(reason != nullptr);
    config.head_dim = 128;
    config.page_size = 32;
    assert(!llama_paged_decode_supported(config, &reason));
    config.page_size = 16;
    config.query_heads = 7;
    assert(!llama_paged_decode_supported(config, &reason));
    config.query_heads = 8;

    llama_paged_decode_plan * plan = nullptr;
    const uint32_t invalid_table[] = { 0, 4 };
    const uint32_t full_context[] = { 32 };
    assert(llama_paged_decode_plan_create(config, invalid_table, full_context, &plan) ==
            cudaErrorInvalidValue);
    assert(plan == nullptr);
    const uint32_t valid_table[] = { 0, UINT32_MAX };
    const uint32_t short_context[] = { 1 };
    check(llama_paged_decode_plan_create(config, valid_table, short_context, &plan),
            "unused page-table entries are ignored");
    llama_paged_decode_plan_destroy(plan);
    plan = nullptr;
    const uint32_t empty_context[] = { 0 };
    assert(llama_paged_decode_plan_create(config, valid_table, empty_context, &plan) ==
            cudaErrorInvalidValue);
    assert(llama_paged_decode_launch_paged(
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == cudaErrorInvalidValue);
    assert(llama_paged_decode_launch_paged_k2_tile(
            plan, nullptr, nullptr, nullptr, nullptr, nullptr, 3) == cudaErrorInvalidValue);
}

} // namespace

int main() {
    test_non_contiguous_boundary_computes_attention_output();
    test_randomized_gqa_matches_independent_contiguous_oracle();
    test_causal_tail_stops_at_every_page_boundary();
    test_unsupported_and_invalid_shapes_fail_closed();
    return 0;
}
