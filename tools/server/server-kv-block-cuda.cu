#include "server-kv-block-backend.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using u64 = uint64_t;

namespace {

void cuda_check(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

bool inject_failure(const char * point) {
    const char * configured = std::getenv("CACHEFLOW_TEST_CUDA_FAIL_POINT");
    return configured && std::string(configured) == point;
}

__global__ void gather_kv_blocks(
        const uint16_t * src_k,
        const uint16_t * src_v,
        uint16_t * staged_k,
        uint16_t * staged_v,
        const server_kv_block_copy * mapping,
        size_t mapping_count,
        size_t elements_per_block) {
    const size_t linear = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = mapping_count * elements_per_block;
    if (linear >= total) return;
    const size_t mapping_index = linear / elements_per_block;
    const size_t element = linear % elements_per_block;
    const size_t source = (size_t) mapping[mapping_index].source * elements_per_block + element;
    staged_k[linear] = src_k[source];
    staged_v[linear] = src_v[source];
}

__global__ void scatter_kv_blocks(
        const uint16_t * staged_k,
        const uint16_t * staged_v,
        uint16_t * dst_k,
        uint16_t * dst_v,
        const server_kv_block_copy * mapping,
        size_t mapping_count,
        size_t elements_per_block) {
    const size_t linear = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = mapping_count * elements_per_block;
    if (linear >= total) return;
    const size_t mapping_index = linear / elements_per_block;
    const size_t element = linear % elements_per_block;
    const size_t destination =
            (size_t) mapping[mapping_index].destination * elements_per_block + element;
    dst_k[destination] = staged_k[linear];
    dst_v[destination] = staged_v[linear];
}

__global__ void clone_shared_tail_block(
        const uint16_t * src_k,
        const uint16_t * src_v,
        uint16_t * dst_k,
        uint16_t * dst_v,
        server_kv_physical_block_id source_block,
        server_kv_physical_block_id destination_block,
        size_t elements_per_block) {
    const size_t element = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= elements_per_block) return;
    const size_t source = (size_t) source_block * elements_per_block + element;
    const size_t destination = (size_t) destination_block * elements_per_block + element;
    dst_k[destination] = src_k[source];
    dst_v[destination] = src_v[source];
}

class server_kv_cuda_block_backend final : public server_kv_block_backend {
public:
    server_kv_cuda_block_backend(
            server_kv_tensor_layout layout,
            size_t capacity_blocks,
            int device) :
            layout_(layout), capacity_blocks_(capacity_blocks), device_(device) {
        layout_.validate();
        if (capacity_blocks == 0) throw std::invalid_argument("KV backend capacity must be positive");
        cuda_check(cudaSetDevice(device_), "cudaSetDevice");
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, device_), "cudaGetDeviceProperties");
        int pools_supported = 0;
        cuda_check(cudaDeviceGetAttribute(
                &pools_supported, cudaDevAttrMemoryPoolsSupported, device_),
                "cudaDeviceGetAttribute(memory pools)");
        if (!pools_supported) {
            throw std::runtime_error("CUDA device does not support stream-ordered memory pools");
        }
        cuda_check(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking),
                "cudaStreamCreate(copy)");
        cuda_check(cudaStreamCreateWithFlags(&swap_stream_, cudaStreamNonBlocking),
                "cudaStreamCreate(swap)");
        const size_t bytes = capacity_blocks_ * layout_.elements_per_plane_block() * sizeof(uint16_t);
        const size_t allocation_bytes = bytes + 2 * guard_elements_ * sizeof(uint16_t);
        try {
            cuda_check(cudaMalloc((void **) &k_allocation_, allocation_bytes), "cudaMalloc(K plane)");
            if (inject_failure("constructor_after_k")) {
                throw std::runtime_error("injected CUDA allocation failure after K plane");
            }
            cuda_check(cudaMalloc((void **) &v_allocation_, allocation_bytes), "cudaMalloc(V plane)");
            k_ = k_allocation_ + guard_elements_;
            v_ = v_allocation_ + guard_elements_;
            cuda_check(cudaMemset(k_allocation_, guard_pattern_, allocation_bytes), "cudaMemset(K guards)");
            cuda_check(cudaMemset(v_allocation_, guard_pattern_, allocation_bytes), "cudaMemset(V guards)");
            cuda_check(cudaMemset(k_, 0, bytes), "cudaMemset(K plane)");
            cuda_check(cudaMemset(v_, 0, bytes), "cudaMemset(V plane)");
        } catch (...) {
            if (k_allocation_) cudaFree(k_allocation_);
            if (v_allocation_) cudaFree(v_allocation_);
            cudaStreamDestroy(copy_stream_);
            cudaStreamDestroy(swap_stream_);
            throw;
        }
    }

    ~server_kv_cuda_block_backend() override {
        // Destructors cannot report failures. Complete outstanding leases so
        // pinned/device staging is never released while a stream still uses it.
        std::vector<uint64_t> ids;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto & item : pending_) ids.push_back(item.first);
        }
        for (uint64_t id : ids) {
            try { wait({ id }); } catch (...) {}
        }
        cudaSetDevice(device_);
        if (k_allocation_) cudaFree(k_allocation_);
        if (v_allocation_) cudaFree(v_allocation_);
        if (copy_stream_) cudaStreamDestroy(copy_stream_);
        if (swap_stream_) cudaStreamDestroy(swap_stream_);
    }

    size_t capacity_blocks() const override { return capacity_blocks_; }

    server_kv_backend_event copy_blocks(
            const std::vector<server_kv_block_copy> & mapping,
            const server_kv_tensor_layout & layout) override {
        require_layout(layout);
        validate_mapping(mapping);
        cuda_check(cudaSetDevice(device_), "cudaSetDevice(copy)");
        if (!requires_snapshot(mapping)) {
            const size_t elements = layout_.elements_per_plane_block();
            const size_t bytes = elements * sizeof(uint16_t);
            for (const auto & copy : mapping) {
                cuda_check(cudaMemcpyAsync(k_ + (size_t) copy.destination * elements,
                        k_ + (size_t) copy.source * elements, bytes,
                        cudaMemcpyDeviceToDevice, copy_stream_), "cudaMemcpyAsync(direct K block)");
                cuda_check(cudaMemcpyAsync(v_ + (size_t) copy.destination * elements,
                        v_ + (size_t) copy.source * elements, bytes,
                        cudaMemcpyDeviceToDevice, copy_stream_), "cudaMemcpyAsync(direct V block)");
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stats_.blocks_copied += mapping.size();
                stats_.copy_bytes += mapping.size() * layout_.bytes_per_block();
                stats_.direct_copy_batches += !mapping.empty();
            }
            return record({}, copy_stream_);
        }
        pending_operation operation;
        try {
            allocate_pinned_mapping(operation, mapping);
            const size_t elements = mapping.size() * layout_.elements_per_plane_block();
            allocate_device_staging(operation, elements, copy_stream_);
            enqueue_mapping(operation, copy_stream_);
            launch_gather(operation, mapping.size(), copy_stream_);
            launch_scatter(operation, mapping.size(), copy_stream_);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stats_.kernel_launches += mapping.empty() ? 0 : 2;
                stats_.blocks_copied += mapping.size();
                stats_.copy_bytes += mapping.size() * layout_.bytes_per_block();
                stats_.staged_copy_batches += !mapping.empty();
            }
            release_device_staging_async(operation, copy_stream_);
            return record(std::move(operation), copy_stream_);
        } catch (...) {
            discard(operation, copy_stream_);
            note_error();
            throw;
        }
    }

    server_kv_backend_event clone_shared_tail(
            server_kv_physical_block_id source,
            server_kv_physical_block_id destination,
            const server_kv_tensor_layout & layout) override {
        require_layout(layout);
        validate_block(source);
        validate_block(destination);
        cuda_check(cudaSetDevice(device_), "cudaSetDevice(COW)");
        const size_t elements = layout_.elements_per_plane_block();
        const int threads = 256;
        clone_shared_tail_block<<<(unsigned) ((elements + threads - 1) / threads), threads, 0, copy_stream_>>>(
                k_, v_, k_, v_, source, destination, elements);
        cuda_check(cudaGetLastError(), "clone_shared_tail_block launch");
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stats_.kernel_launches++;
            stats_.blocks_copied++;
            stats_.copy_bytes += layout_.bytes_per_block();
        }
        return record({}, copy_stream_);
    }

    server_kv_backend_event swap_out(
            const std::vector<server_kv_physical_block_id> & blocks,
            server_kv_host_blocks & destination) override {
        validate_blocks(blocks);
        destination.resize(layout_, blocks.size());
        cuda_check(cudaSetDevice(device_), "cudaSetDevice(swap out)");
        pending_operation operation;
        std::vector<server_kv_block_copy> mapping;
        mapping.reserve(blocks.size());
        for (size_t i = 0; i < blocks.size(); ++i) mapping.push_back({ blocks[i], (uint32_t) i });
        try {
            allocate_pinned_mapping(operation, mapping);
            const size_t elements = blocks.size() * layout_.elements_per_plane_block();
            allocate_device_staging(operation, elements, swap_stream_);
            allocate_pinned_planes(operation, elements);
            operation.swap_out_destination = &destination;
            enqueue_mapping(operation, swap_stream_);
            launch_gather(operation, mapping.size(), swap_stream_);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stats_.kernel_launches += mapping.empty() ? 0 : 1;
                stats_.blocks_copied += blocks.size();
                stats_.copy_bytes += blocks.size() * layout_.bytes_per_block();
            }
            const size_t bytes = elements * sizeof(uint16_t);
            cuda_check(cudaMemcpyAsync(operation.pinned_k, operation.staged_k, bytes,
                    cudaMemcpyDeviceToHost, swap_stream_), "cudaMemcpyAsync(swap out K)");
            cuda_check(cudaMemcpyAsync(operation.pinned_v, operation.staged_v, bytes,
                    cudaMemcpyDeviceToHost, swap_stream_), "cudaMemcpyAsync(swap out V)");
            release_device_staging_async(operation, swap_stream_);
            return record(std::move(operation), swap_stream_);
        } catch (...) {
            discard(operation, swap_stream_);
            note_error();
            throw;
        }
    }

    server_kv_backend_event swap_in(
            const server_kv_host_blocks & source,
            const std::vector<server_kv_physical_block_id> & destinations) override {
        require_layout(source.layout);
        validate_blocks(destinations);
        if (source.block_count != destinations.size()) {
            throw std::invalid_argument("swap-in source and destination counts differ");
        }
        const size_t elements = source.block_count * layout_.elements_per_plane_block();
        if (source.k.size() != elements || source.v.size() != elements) {
            throw std::invalid_argument("swap-in host buffer has invalid size");
        }
        cuda_check(cudaSetDevice(device_), "cudaSetDevice(swap in)");
        pending_operation operation;
        std::vector<server_kv_block_copy> mapping;
        mapping.reserve(destinations.size());
        for (size_t i = 0; i < destinations.size(); ++i) {
            mapping.push_back({ (uint32_t) i, destinations[i] });
        }
        try {
            allocate_pinned_mapping(operation, mapping);
            allocate_device_staging(operation, elements, swap_stream_);
            allocate_pinned_planes(operation, elements);
            const size_t bytes = elements * sizeof(uint16_t);
            std::memcpy(operation.pinned_k, source.k.data(), bytes);
            std::memcpy(operation.pinned_v, source.v.data(), bytes);
            cuda_check(cudaMemcpyAsync(operation.staged_k, operation.pinned_k, bytes,
                    cudaMemcpyHostToDevice, swap_stream_), "cudaMemcpyAsync(swap in K)");
            cuda_check(cudaMemcpyAsync(operation.staged_v, operation.pinned_v, bytes,
                    cudaMemcpyHostToDevice, swap_stream_), "cudaMemcpyAsync(swap in V)");
            enqueue_mapping(operation, swap_stream_);
            launch_scatter(operation, mapping.size(), swap_stream_);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stats_.kernel_launches += mapping.empty() ? 0 : 1;
                stats_.blocks_copied += destinations.size();
                stats_.copy_bytes += destinations.size() * layout_.bytes_per_block();
            }
            release_device_staging_async(operation, swap_stream_);
            return record(std::move(operation), swap_stream_);
        } catch (...) {
            discard(operation, swap_stream_);
            note_error();
            throw;
        }
    }

    void wait(server_kv_backend_event event) override {
        pending_operation operation;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto it = pending_.find(event.value);
            if (it == pending_.end()) throw std::invalid_argument("unknown CUDA KV backend event");
            operation = std::move(it->second);
            pending_.erase(it);
        }
        cuda_check(cudaSetDevice(device_), "cudaSetDevice(wait)");
        const cudaError_t status = cudaEventSynchronize(operation.event);
        if (status == cudaSuccess && operation.swap_out_destination) {
            const size_t elements = operation.swap_out_destination->k.size();
            const size_t bytes = elements * sizeof(uint16_t);
            std::memcpy(operation.swap_out_destination->k.data(), operation.pinned_k, bytes);
            std::memcpy(operation.swap_out_destination->v.data(), operation.pinned_v, bytes);
        }
        cleanup(operation);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stats_.events_waited++;
            stats_.pinned_bytes_current -= operation.pinned_bytes;
        }
        cuda_check(status, "cudaEventSynchronize(KV backend)");
    }

    server_kv_backend_stats stats() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

    bool verify_integrity() const override {
        cuda_check(cudaSetDevice(device_), "cudaSetDevice(verify guards)");
        const size_t data_elements = capacity_blocks_ * layout_.elements_per_plane_block();
        std::vector<uint16_t> observed(4 * guard_elements_);
        cuda_check(cudaMemcpy(observed.data(), k_allocation_, guard_elements_ * sizeof(uint16_t),
                cudaMemcpyDeviceToHost), "read K prefix guard");
        cuda_check(cudaMemcpy(observed.data() + guard_elements_, k_ + data_elements,
                guard_elements_ * sizeof(uint16_t), cudaMemcpyDeviceToHost), "read K suffix guard");
        cuda_check(cudaMemcpy(observed.data() + 2 * guard_elements_, v_allocation_,
                guard_elements_ * sizeof(uint16_t), cudaMemcpyDeviceToHost), "read V prefix guard");
        cuda_check(cudaMemcpy(observed.data() + 3 * guard_elements_, v_ + data_elements,
                guard_elements_ * sizeof(uint16_t), cudaMemcpyDeviceToHost), "read V suffix guard");
        return std::all_of(observed.begin(), observed.end(), [](uint16_t value) {
            return value == 0xa5a5;
        });
    }

    void write_block(
            server_kv_physical_block_id block,
            const std::vector<uint16_t> & k,
            const std::vector<uint16_t> & v) override {
        validate_block(block);
        const size_t elements = layout_.elements_per_plane_block();
        if (k.size() != elements || v.size() != elements) {
            throw std::invalid_argument("block payload does not match KV layout");
        }
        cuda_check(cudaSetDevice(device_), "cudaSetDevice(write block)");
        const size_t offset = (size_t) block * elements;
        const size_t bytes = elements * sizeof(uint16_t);
        cuda_check(cudaMemcpy(k_ + offset, k.data(), bytes, cudaMemcpyHostToDevice),
                "cudaMemcpy(write K block)");
        cuda_check(cudaMemcpy(v_ + offset, v.data(), bytes, cudaMemcpyHostToDevice),
                "cudaMemcpy(write V block)");
    }

    server_kv_host_blocks read_blocks(
            const std::vector<server_kv_physical_block_id> & blocks) override {
        server_kv_host_blocks result;
        wait(swap_out(blocks, result));
        return result;
    }

private:
    struct pending_operation {
        cudaEvent_t event = nullptr;
        server_kv_block_copy * pinned_mapping = nullptr;
        server_kv_block_copy * device_mapping = nullptr;
        uint16_t * staged_k = nullptr;
        uint16_t * staged_v = nullptr;
        uint16_t * pinned_k = nullptr;
        uint16_t * pinned_v = nullptr;
        server_kv_host_blocks * swap_out_destination = nullptr;
        size_t mapping_count = 0;
        size_t pinned_bytes = 0;
    };

    server_kv_tensor_layout layout_;
    size_t capacity_blocks_;
    int device_;
    static constexpr size_t guard_elements_ = 4096;
    static constexpr int guard_pattern_ = 0xa5;
    uint16_t * k_allocation_ = nullptr;
    uint16_t * v_allocation_ = nullptr;
    uint16_t * k_ = nullptr;
    uint16_t * v_ = nullptr;
    cudaStream_t copy_stream_ = nullptr;
    cudaStream_t swap_stream_ = nullptr;
    uint64_t next_event_ = 1;
    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, pending_operation> pending_;
    server_kv_backend_stats stats_;

    void require_layout(const server_kv_tensor_layout & layout) const {
        if (layout.layers != layout_.layers || layout.kv_heads != layout_.kv_heads ||
                layout.head_dim != layout_.head_dim || layout.block_tokens != layout_.block_tokens ||
                layout.element_type != layout_.element_type ||
                layout.memory_layout != layout_.memory_layout) {
            throw std::invalid_argument("KV tensor layout does not match CUDA allocation");
        }
    }

    void validate_block(server_kv_physical_block_id block) const {
        if (block >= capacity_blocks_) throw std::out_of_range("physical KV block is out of range");
    }

    void validate_blocks(const std::vector<server_kv_physical_block_id> & blocks) const {
        for (auto block : blocks) validate_block(block);
    }

    void validate_mapping(const std::vector<server_kv_block_copy> & mapping) const {
        std::unordered_set<server_kv_physical_block_id> destinations;
        for (const auto & copy : mapping) {
            validate_block(copy.source);
            validate_block(copy.destination);
            if (!destinations.insert(copy.destination).second) {
                throw std::invalid_argument("KV block mapping contains duplicate destinations");
            }
        }
    }

    static bool requires_snapshot(const std::vector<server_kv_block_copy> & mapping) {
        std::unordered_set<server_kv_physical_block_id> destinations;
        for (const auto & copy : mapping) destinations.insert(copy.destination);
        return std::any_of(mapping.begin(), mapping.end(), [&](const server_kv_block_copy & copy) {
            return destinations.count(copy.source) != 0;
        });
    }

    static void allocate_pinned_mapping(
            pending_operation & operation,
            const std::vector<server_kv_block_copy> & mapping) {
        operation.mapping_count = mapping.size();
        if (mapping.empty()) return;
        const size_t bytes = mapping.size() * sizeof(server_kv_block_copy);
        cuda_check(cudaMallocHost((void **) &operation.pinned_mapping, bytes),
                "cudaMallocHost(block mapping)");
        operation.pinned_bytes += bytes;
        if (inject_failure("pinned_mapping")) {
            throw std::runtime_error("injected pinned mapping exhaustion");
        }
        std::memcpy(operation.pinned_mapping, mapping.data(), bytes);
    }

    static void allocate_device_staging(
            pending_operation & operation,
            size_t elements,
            cudaStream_t stream) {
        if (operation.mapping_count > 0) {
            cuda_check(cudaMallocAsync((void **) &operation.device_mapping,
                    operation.mapping_count * sizeof(server_kv_block_copy), stream),
                    "cudaMallocAsync(block mapping)");
            if (inject_failure("device_staging")) {
                throw std::runtime_error("injected CUDA staging allocation failure");
            }
        }
        if (elements > 0) {
            cuda_check(cudaMallocAsync((void **) &operation.staged_k,
                    elements * sizeof(uint16_t), stream), "cudaMallocAsync(staged K)");
            cuda_check(cudaMallocAsync((void **) &operation.staged_v,
                    elements * sizeof(uint16_t), stream), "cudaMallocAsync(staged V)");
        }
    }

    static void allocate_pinned_planes(pending_operation & operation, size_t elements) {
        if (elements == 0) return;
        cuda_check(cudaMallocHost((void **) &operation.pinned_k, elements * sizeof(uint16_t)),
                "cudaMallocHost(staged K)");
        operation.pinned_bytes += elements * sizeof(uint16_t);
        if (inject_failure("pinned_planes")) {
            throw std::runtime_error("injected pinned plane exhaustion");
        }
        cuda_check(cudaMallocHost((void **) &operation.pinned_v, elements * sizeof(uint16_t)),
                "cudaMallocHost(staged V)");
        operation.pinned_bytes += elements * sizeof(uint16_t);
    }

    static void enqueue_mapping(pending_operation & operation, cudaStream_t stream) {
        if (operation.mapping_count == 0) return;
        cuda_check(cudaMemcpyAsync(operation.device_mapping, operation.pinned_mapping,
                operation.mapping_count * sizeof(server_kv_block_copy),
                cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(block mapping)");
    }

    void launch_gather(pending_operation & operation, size_t mapping_count, cudaStream_t stream) {
        if (mapping_count == 0) return;
        const size_t total = mapping_count * layout_.elements_per_plane_block();
        const int threads = 256;
        gather_kv_blocks<<<(unsigned) ((total + threads - 1) / threads), threads, 0, stream>>>(
                k_, v_, operation.staged_k, operation.staged_v,
                operation.device_mapping, mapping_count, layout_.elements_per_plane_block());
        cuda_check(cudaGetLastError(), "gather_kv_blocks launch");
    }

    void launch_scatter(pending_operation & operation, size_t mapping_count, cudaStream_t stream) {
        if (mapping_count == 0) return;
        const size_t total = mapping_count * layout_.elements_per_plane_block();
        const int threads = 256;
        scatter_kv_blocks<<<(unsigned) ((total + threads - 1) / threads), threads, 0, stream>>>(
                operation.staged_k, operation.staged_v, k_, v_,
                operation.device_mapping, mapping_count, layout_.elements_per_plane_block());
        cuda_check(cudaGetLastError(), "scatter_kv_blocks launch");
    }

    static void release_device_staging_async(
            pending_operation & operation,
            cudaStream_t stream) {
        if (operation.device_mapping) cuda_check(cudaFreeAsync(operation.device_mapping, stream),
                "cudaFreeAsync(block mapping)");
        if (operation.staged_k) cuda_check(cudaFreeAsync(operation.staged_k, stream),
                "cudaFreeAsync(staged K)");
        if (operation.staged_v) cuda_check(cudaFreeAsync(operation.staged_v, stream),
                "cudaFreeAsync(staged V)");
        operation.device_mapping = nullptr;
        operation.staged_k = nullptr;
        operation.staged_v = nullptr;
    }

    server_kv_backend_event record(pending_operation operation, cudaStream_t stream) {
        cuda_check(cudaEventCreateWithFlags(&operation.event, cudaEventDisableTiming),
                "cudaEventCreate(KV backend)");
        cuda_check(cudaEventRecord(operation.event, stream), "cudaEventRecord(KV backend)");
        std::lock_guard<std::mutex> lock(mutex_);
        const uint64_t id = next_event_++;
        stats_.pinned_bytes_current += operation.pinned_bytes;
        stats_.pinned_bytes_peak = std::max(
                stats_.pinned_bytes_peak, stats_.pinned_bytes_current);
        pending_.emplace(id, std::move(operation));
        return { id };
    }

    static void cleanup(pending_operation & operation) {
        if (operation.event) cudaEventDestroy(operation.event);
        if (operation.pinned_mapping) cudaFreeHost(operation.pinned_mapping);
        if (operation.pinned_k) cudaFreeHost(operation.pinned_k);
        if (operation.pinned_v) cudaFreeHost(operation.pinned_v);
    }

    static void discard(pending_operation & operation, cudaStream_t stream) {
        cudaStreamSynchronize(stream);
        if (operation.device_mapping) cudaFree(operation.device_mapping);
        if (operation.staged_k) cudaFree(operation.staged_k);
        if (operation.staged_v) cudaFree(operation.staged_v);
        operation.device_mapping = nullptr;
        operation.staged_k = nullptr;
        operation.staged_v = nullptr;
        cleanup(operation);
    }

    void note_error() {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.backend_errors++;
    }
};

} // namespace

std::unique_ptr<server_kv_block_backend> server_kv_create_cuda_block_backend(
        server_kv_tensor_layout layout,
        size_t capacity_blocks,
        int device) {
    return std::make_unique<server_kv_cuda_block_backend>(layout, capacity_blocks, device);
}
