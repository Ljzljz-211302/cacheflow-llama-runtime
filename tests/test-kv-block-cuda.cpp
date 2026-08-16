#ifdef NDEBUG
#undef NDEBUG
#endif

#include "server-kv-block-backend.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

static std::vector<uint16_t> payload(size_t size, uint16_t seed) {
    std::vector<uint16_t> result(size);
    for (size_t i = 0; i < size; ++i) result[i] = (uint16_t) (seed ^ (i * 131));
    return result;
}

static void require_equal(const server_kv_host_blocks & left, const server_kv_host_blocks & right) {
    assert(left.block_count == right.block_count);
    assert(left.k == right.k);
    assert(left.v == right.v);
}

static void set_fail_point(const char * value) {
#ifdef _WIN32
    _putenv_s("CACHEFLOW_TEST_CUDA_FAIL_POINT", value ? value : "");
#else
    if (value) setenv("CACHEFLOW_TEST_CUDA_FAIL_POINT", value, 1);
    else unsetenv("CACHEFLOW_TEST_CUDA_FAIL_POINT");
#endif
}

static void test_allocation_failures_release_partial_resources() {
    const server_kv_tensor_layout layout = {
        2, 2, 16, 8, server_kv_element_type::fp16,
        server_kv_memory_layout::separate_k_v_planes,
    };
    set_fail_point("constructor_after_k");
    bool constructor_failed = false;
    try { (void) server_kv_create_cuda_block_backend(layout, 16); }
    catch (const std::runtime_error &) { constructor_failed = true; }
    set_fail_point(nullptr);
    assert(constructor_failed);

    auto backend = server_kv_create_cuda_block_backend(layout, 16);
    const auto values = payload(layout.elements_per_plane_block(), 71);
    backend->write_block(0, values, values);
    backend->write_block(1, values, values);

    set_fail_point("device_staging");
    bool staging_failed = false;
    try { (void) backend->copy_blocks({ { 0, 1 }, { 1, 0 } }, layout); }
    catch (const std::runtime_error &) { staging_failed = true; }
    set_fail_point(nullptr);
    assert(staging_failed);
    assert(backend->verify_integrity());

    set_fail_point("pinned_planes");
    bool pinned_failed = false;
    server_kv_host_blocks host;
    try { (void) backend->swap_out({ 0, 1 }, host); }
    catch (const std::runtime_error &) { pinned_failed = true; }
    set_fail_point(nullptr);
    assert(pinned_failed);
    assert(backend->verify_integrity());
    const auto stats = backend->stats();
    assert(stats.backend_errors == 2);
    assert(stats.pinned_bytes_current == 0);

    backend->wait(backend->copy_blocks({ { 0, 2 } }, layout));
    assert(backend->verify_integrity());
}

static void run_layout_case(uint32_t block_tokens, uint32_t head_dim, uint32_t seed) {
    const server_kv_tensor_layout layout = {
        3, 4, head_dim, block_tokens, server_kv_element_type::fp16,
        server_kv_memory_layout::separate_k_v_planes,
    };
    constexpr uint32_t capacity = 48;
    auto cpu = server_kv_create_cpu_block_backend(layout, capacity);
    auto cuda = server_kv_create_cuda_block_backend(layout, capacity);
    const size_t elements = layout.elements_per_plane_block();
    for (uint32_t block = 0; block < capacity; ++block) {
        const auto k = payload(elements, (uint16_t) (seed + block));
        const auto v = payload(elements, (uint16_t) (seed + 1000 + block));
        cpu->write_block(block, k, v);
        cuda->write_block(block, k, v);
    }

    // Non-contiguous, overlapping and repeated sources exercise the batched
    // gather/scatter snapshot contract.
    const std::vector<server_kv_block_copy> mapping = {
        { 0, 13 }, { 7, 3 }, { 3, 20 }, { 7, 21 }, { 2, 0 },
    };
    cpu->wait(cpu->copy_blocks(mapping, layout));
    cuda->wait(cuda->copy_blocks(mapping, layout));
    require_equal(cpu->read_blocks({ 0, 3, 13, 20, 21 }),
            cuda->read_blocks({ 0, 3, 13, 20, 21 }));

    cpu->wait(cpu->clone_shared_tail(7, 25, layout));
    cuda->wait(cuda->clone_shared_tail(7, 25, layout));
    require_equal(cpu->read_blocks({ 25 }), cuda->read_blocks({ 25 }));

    server_kv_host_blocks cpu_host;
    server_kv_host_blocks cuda_host;
    cpu->wait(cpu->swap_out({ 0, 7, 25 }, cpu_host));
    cuda->wait(cuda->swap_out({ 0, 7, 25 }, cuda_host));
    require_equal(cpu_host, cuda_host);
    cpu->wait(cpu->swap_in(cpu_host, { 26, 27, 28 }));
    cuda->wait(cuda->swap_in(cuda_host, { 26, 27, 28 }));
    require_equal(cpu->read_blocks({ 26, 27, 28 }), cuda->read_blocks({ 26, 27, 28 }));

    // Property-style randomized batches cover non-contiguous mappings and
    // source/destination overlap across many generations.
    std::mt19937 rng(seed);
    for (size_t iteration = 0; iteration < 100; ++iteration) {
        const size_t count = 1 + rng() % 12;
        std::vector<server_kv_block_copy> copies;
        std::vector<uint32_t> destinations;
        while (copies.size() < count) {
            const uint32_t destination = rng() % capacity;
            if (std::find(destinations.begin(), destinations.end(), destination) != destinations.end()) {
                continue;
            }
            destinations.push_back(destination);
            copies.push_back({ rng() % capacity, destination });
        }
        cpu->wait(cpu->copy_blocks(copies, layout));
        cuda->wait(cuda->copy_blocks(copies, layout));
        require_equal(cpu->read_blocks(destinations), cuda->read_blocks(destinations));
    }

    bool cpu_rejected = false;
    bool cuda_rejected = false;
    try { cpu->copy_blocks({{ 0, 2 }, { 1, 2 }}, layout); }
    catch (const std::invalid_argument &) { cpu_rejected = true; }
    try { cuda->copy_blocks({{ 0, 2 }, { 1, 2 }}, layout); }
    catch (const std::invalid_argument &) { cuda_rejected = true; }
    assert(cpu_rejected && cuda_rejected);
    const auto stats = cuda->stats();
    assert(stats.kernel_launches > 0);
    assert(stats.blocks_copied > 0);
    assert(stats.copy_bytes >= stats.blocks_copied * layout.bytes_per_block());
    assert(stats.pinned_bytes_current == 0);
    assert(stats.pinned_bytes_peak > 0);
    assert(stats.direct_copy_batches > 0);
    assert(stats.staged_copy_batches > 0);
    assert(cpu->verify_integrity());
    assert(cuda->verify_integrity());
}

int main() {
    test_allocation_failures_release_partial_resources();
    run_layout_case(8, 16, 11);
    run_layout_case(16, 32, 22);
    run_layout_case(32, 64, 33);
    run_layout_case(64, 128, 44);
    return 0;
}
