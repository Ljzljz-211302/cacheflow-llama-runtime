#include "server-kv-block-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double elapsed_ms(const std::chrono::steady_clock::time_point & start) {
    return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
}

int main() {
    // Qwen2.5-0.5B KV geometry: 24 layers, 2 KV heads, 64 head dim.
    const server_kv_tensor_layout layout = {
        24, 2, 64, 16, server_kv_element_type::fp16,
        server_kv_memory_layout::separate_k_v_planes,
    };
    constexpr size_t sequence_blocks = 64;
    constexpr size_t capacity = sequence_blocks * 2 + 2;
    auto backend = server_kv_create_cuda_block_backend(layout, capacity);
    std::vector<uint16_t> k(layout.elements_per_plane_block(), 7);
    std::vector<uint16_t> v(layout.elements_per_plane_block(), 11);
    for (size_t block = 0; block < sequence_blocks; ++block) {
        k[0] = (uint16_t) block;
        v[0] = (uint16_t) (1000 + block);
        backend->write_block((uint32_t) block, k, v);
    }
    std::vector<server_kv_block_copy> full;
    for (size_t block = 0; block < sequence_blocks; ++block) {
        full.push_back({(uint32_t) block, (uint32_t) (sequence_blocks + block)});
    }
    // Warm both paths and allocator/event pools.
    backend->wait(backend->copy_blocks(full, layout));
    backend->wait(backend->clone_shared_tail(
            (uint32_t) (sequence_blocks - 1), (uint32_t) (capacity - 1), layout));

    std::puts("method,trial,end_to_end_ms,bytes,copy_launches,extra_device_bytes");
    for (int trial = 0; trial < 100; ++trial) {
        auto before = backend->stats();
        auto start = std::chrono::steady_clock::now();
        backend->wait(backend->copy_blocks(full, layout));
        const double baseline_ms = elapsed_ms(start);
        auto after = backend->stats();
        std::printf("whole_sequence_copy,%d,%.6f,%zu,%llu,%zu\n", trial, baseline_ms,
                sequence_blocks * layout.bytes_per_block(),
                (unsigned long long) ((after.direct_copy_batches - before.direct_copy_batches) *
                        sequence_blocks * 2),
                sequence_blocks * layout.bytes_per_block());

        before = backend->stats();
        start = std::chrono::steady_clock::now();
        backend->wait(backend->clone_shared_tail(
                (uint32_t) (sequence_blocks - 1), (uint32_t) (capacity - 1), layout));
        const double cow_ms = elapsed_ms(start);
        after = backend->stats();
        std::printf("tail_block_cow,%d,%.6f,%zu,%llu,%zu\n", trial, cow_ms,
                layout.bytes_per_block(),
                (unsigned long long) (after.kernel_launches - before.kernel_launches),
                layout.bytes_per_block());
    }
    if (!backend->verify_integrity()) return 1;
    return 0;
}
