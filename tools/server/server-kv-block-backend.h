#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

using server_kv_physical_block_id = uint32_t;

enum class server_kv_element_type : uint8_t {
    fp16,
};

enum class server_kv_memory_layout : uint8_t {
    separate_k_v_planes,
};

struct server_kv_tensor_layout {
    uint32_t layers = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t block_tokens = 0;
    server_kv_element_type element_type = server_kv_element_type::fp16;
    server_kv_memory_layout memory_layout = server_kv_memory_layout::separate_k_v_planes;

    size_t elements_per_plane_block() const;
    size_t bytes_per_block() const;
    void validate() const;
};

struct server_kv_block_copy {
    server_kv_physical_block_id source = 0;
    server_kv_physical_block_id destination = 0;
};

struct server_kv_backend_event {
    uint64_t value = 0;
};

struct server_kv_backend_stats {
    uint64_t kernel_launches = 0;
    uint64_t blocks_copied = 0;
    uint64_t copy_bytes = 0;
    uint64_t events_waited = 0;
    uint64_t pinned_bytes_current = 0;
    uint64_t pinned_bytes_peak = 0;
    uint64_t backend_errors = 0;
    uint64_t direct_copy_batches = 0;
    uint64_t staged_copy_batches = 0;
};

// Host representation used by swap. K and V use separate dense planes in the
// same block order as `blocks`/`destinations` passed to the operation.
struct server_kv_host_blocks {
    server_kv_tensor_layout layout;
    size_t block_count = 0;
    std::vector<uint16_t> k;
    std::vector<uint16_t> v;

    void resize(const server_kv_tensor_layout & value_layout, size_t value_block_count);
};

// Physical KV data mover. Policy, request state and logical reference counts
// intentionally do not cross this interface.
class server_kv_block_backend {
public:
    virtual ~server_kv_block_backend() = default;

    virtual size_t capacity_blocks() const = 0;
    virtual server_kv_backend_event copy_blocks(
            const std::vector<server_kv_block_copy> & mapping,
            const server_kv_tensor_layout & layout) = 0;
    virtual server_kv_backend_event clone_shared_tail(
            server_kv_physical_block_id source,
            server_kv_physical_block_id destination,
            const server_kv_tensor_layout & layout) = 0;
    virtual server_kv_backend_event swap_out(
            const std::vector<server_kv_physical_block_id> & blocks,
            server_kv_host_blocks & destination) = 0;
    virtual server_kv_backend_event swap_in(
            const server_kv_host_blocks & source,
            const std::vector<server_kv_physical_block_id> & destinations) = 0;
    virtual void wait(server_kv_backend_event event) = 0;
    virtual server_kv_backend_stats stats() const = 0;
    // Validates allocator guard regions when the backend provides them. This
    // is deliberately callable from long-running stress tests as a portable
    // alternative when Compute Sanitizer cannot attach to a WDDM device.
    virtual bool verify_integrity() const = 0;

    // Explicit transfer methods keep tests and model adapters independent of
    // backend allocation details. They are synchronous initialization/readback
    // boundaries, never called by the scheduler hot path.
    virtual void write_block(
            server_kv_physical_block_id block,
            const std::vector<uint16_t> & k,
            const std::vector<uint16_t> & v) = 0;
    virtual server_kv_host_blocks read_blocks(
            const std::vector<server_kv_physical_block_id> & blocks) = 0;
};

std::unique_ptr<server_kv_block_backend> server_kv_create_cpu_block_backend(
        server_kv_tensor_layout layout,
        size_t capacity_blocks);

#ifdef GGML_USE_CUDA
std::unique_ptr<server_kv_block_backend> server_kv_create_cuda_block_backend(
        server_kv_tensor_layout layout,
        size_t capacity_blocks,
        int device = 0);
#endif
