#ifdef NDEBUG
#undef NDEBUG
#endif

#include "server-kv-block-backend.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

static server_kv_tensor_layout test_layout() {
    return { 2, 2, 3, 4, server_kv_element_type::fp16,
        server_kv_memory_layout::separate_k_v_planes };
}

static std::vector<uint16_t> payload(size_t size, uint16_t seed) {
    std::vector<uint16_t> result(size);
    for (size_t i = 0; i < size; ++i) result[i] = (uint16_t) (seed + i * 17);
    return result;
}

static void test_copy_and_cow_match_source() {
    const auto layout = test_layout();
    auto backend = server_kv_create_cpu_block_backend(layout, 8);
    const auto k = payload(layout.elements_per_plane_block(), 100);
    const auto v = payload(layout.elements_per_plane_block(), 900);
    backend->write_block(1, k, v);
    backend->wait(backend->copy_blocks({{ 1, 4 }}, layout));
    backend->wait(backend->clone_shared_tail(1, 5, layout));
    const auto result = backend->read_blocks({ 4, 5 });
    assert(result.block_count == 2);
    assert(std::vector<uint16_t>(result.k.begin(), result.k.begin() + k.size()) == k);
    assert(std::vector<uint16_t>(result.k.begin() + k.size(), result.k.end()) == k);
    assert(std::vector<uint16_t>(result.v.begin(), result.v.begin() + v.size()) == v);
    assert(std::vector<uint16_t>(result.v.begin() + v.size(), result.v.end()) == v);
    const auto stats = backend->stats();
    assert(stats.blocks_copied == 4); // copy + COW + two-block readback
    assert(stats.events_waited == 3);
}

static void test_overlapping_batch_uses_snapshot_semantics() {
    const auto layout = test_layout();
    auto backend = server_kv_create_cpu_block_backend(layout, 4);
    const size_t size = layout.elements_per_plane_block();
    const auto k0 = payload(size, 10);
    const auto v0 = payload(size, 20);
    const auto k1 = payload(size, 30);
    const auto v1 = payload(size, 40);
    backend->write_block(0, k0, v0);
    backend->write_block(1, k1, v1);
    backend->wait(backend->copy_blocks({{ 0, 1 }, { 1, 2 }}, layout));
    const auto result = backend->read_blocks({ 1, 2 });
    assert(std::vector<uint16_t>(result.k.begin(), result.k.begin() + size) == k0);
    assert(std::vector<uint16_t>(result.k.begin() + size, result.k.end()) == k1);
    assert(std::vector<uint16_t>(result.v.begin(), result.v.begin() + size) == v0);
    assert(std::vector<uint16_t>(result.v.begin() + size, result.v.end()) == v1);
}

static void test_swap_round_trip_and_validation() {
    const auto layout = test_layout();
    auto backend = server_kv_create_cpu_block_backend(layout, 6);
    const size_t size = layout.elements_per_plane_block();
    const auto k = payload(size, 0x1000);
    const auto v = payload(size, 0x2000);
    backend->write_block(2, k, v);
    server_kv_host_blocks host;
    backend->wait(backend->swap_out({ 2 }, host));
    backend->wait(backend->swap_in(host, { 5 }));
    const auto restored = backend->read_blocks({ 5 });
    assert(restored.k == k && restored.v == v);

    bool rejected = false;
    try {
        backend->copy_blocks({{ 2, 7 }}, layout);
    } catch (const std::out_of_range &) {
        rejected = true;
    }
    assert(rejected);

    rejected = false;
    try {
        backend->copy_blocks({{ 0, 3 }, { 1, 3 }}, layout);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
}

int main() {
    test_copy_and_cow_match_source();
    test_overlapping_batch_uses_snapshot_semantics();
    test_swap_round_trip_and_validation();
    return 0;
}
