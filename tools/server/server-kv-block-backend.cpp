#include "server-kv-block-backend.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace {

size_t checked_multiply(size_t left, size_t right, const char * what) {
    if (right != 0 && left > std::numeric_limits<size_t>::max() / right) {
        throw std::overflow_error(std::string("KV layout overflow in ") + what);
    }
    return left * right;
}

void require_same_layout(
        const server_kv_tensor_layout & expected,
        const server_kv_tensor_layout & actual) {
    if (expected.layers != actual.layers || expected.kv_heads != actual.kv_heads ||
            expected.head_dim != actual.head_dim ||
            expected.block_tokens != actual.block_tokens ||
            expected.element_type != actual.element_type ||
            expected.memory_layout != actual.memory_layout) {
        throw std::invalid_argument("KV tensor layout does not match backend allocation");
    }
}

class server_kv_cpu_block_backend final : public server_kv_block_backend {
public:
    server_kv_cpu_block_backend(server_kv_tensor_layout layout, size_t capacity_blocks) :
            layout_(layout), capacity_blocks_(capacity_blocks) {
        layout_.validate();
        if (capacity_blocks == 0) {
            throw std::invalid_argument("KV backend capacity must be positive");
        }
        const size_t elements = checked_multiply(
                layout_.elements_per_plane_block(), capacity_blocks_, "backend capacity");
        k_.resize(elements);
        v_.resize(elements);
    }

    size_t capacity_blocks() const override { return capacity_blocks_; }

    server_kv_backend_event copy_blocks(
            const std::vector<server_kv_block_copy> & mapping,
            const server_kv_tensor_layout & layout) override {
        require_same_layout(layout_, layout);
        validate_mapping(mapping);
        const size_t stride = layout_.elements_per_plane_block();

        // Snapshot all sources before touching destinations. This defines
        // deterministic gather/scatter semantics for overlapping mappings.
        std::vector<uint16_t> staged_k(checked_multiply(stride, mapping.size(), "copy staging"));
        std::vector<uint16_t> staged_v(staged_k.size());
        for (size_t i = 0; i < mapping.size(); ++i) {
            copy_plane(k_, mapping[i].source, staged_k, i, stride);
            copy_plane(v_, mapping[i].source, staged_v, i, stride);
        }
        for (size_t i = 0; i < mapping.size(); ++i) {
            copy_plane(staged_k, i, k_, mapping[i].destination, stride);
            copy_plane(staged_v, i, v_, mapping[i].destination, stride);
        }
        stats_.blocks_copied += mapping.size();
        stats_.copy_bytes += mapping.size() * layout_.bytes_per_block();
        return complete_event();
    }

    server_kv_backend_event clone_shared_tail(
            server_kv_physical_block_id source,
            server_kv_physical_block_id destination,
            const server_kv_tensor_layout & layout) override {
        return copy_blocks({{ source, destination }}, layout);
    }

    server_kv_backend_event swap_out(
            const std::vector<server_kv_physical_block_id> & blocks,
            server_kv_host_blocks & destination) override {
        validate_blocks(blocks);
        destination.resize(layout_, blocks.size());
        const size_t stride = layout_.elements_per_plane_block();
        for (size_t i = 0; i < blocks.size(); ++i) {
            copy_plane(k_, blocks[i], destination.k, i, stride);
            copy_plane(v_, blocks[i], destination.v, i, stride);
        }
        stats_.blocks_copied += blocks.size();
        stats_.copy_bytes += blocks.size() * layout_.bytes_per_block();
        return complete_event();
    }

    server_kv_backend_event swap_in(
            const server_kv_host_blocks & source,
            const std::vector<server_kv_physical_block_id> & destinations) override {
        require_same_layout(layout_, source.layout);
        validate_blocks(destinations);
        if (source.block_count != destinations.size()) {
            throw std::invalid_argument("swap-in source and destination counts differ");
        }
        const size_t stride = layout_.elements_per_plane_block();
        const size_t expected = checked_multiply(stride, source.block_count, "swap-in buffer");
        if (source.k.size() != expected || source.v.size() != expected) {
            throw std::invalid_argument("swap-in host buffer has invalid size");
        }
        for (size_t i = 0; i < destinations.size(); ++i) {
            copy_plane(source.k, i, k_, destinations[i], stride);
            copy_plane(source.v, i, v_, destinations[i], stride);
        }
        stats_.blocks_copied += destinations.size();
        stats_.copy_bytes += destinations.size() * layout_.bytes_per_block();
        return complete_event();
    }

    void wait(server_kv_backend_event event) override {
        if (event.value == 0 || event.value > completed_event_) {
            throw std::invalid_argument("unknown CPU KV backend event");
        }
        stats_.events_waited++;
    }

    server_kv_backend_stats stats() const override { return stats_; }

    bool verify_integrity() const override { return true; }

    void write_block(
            server_kv_physical_block_id block,
            const std::vector<uint16_t> & k,
            const std::vector<uint16_t> & v) override {
        validate_block(block);
        const size_t stride = layout_.elements_per_plane_block();
        if (k.size() != stride || v.size() != stride) {
            throw std::invalid_argument("block payload does not match KV layout");
        }
        copy_plane(k, 0, k_, block, stride);
        copy_plane(v, 0, v_, block, stride);
    }

    server_kv_host_blocks read_blocks(
            const std::vector<server_kv_physical_block_id> & blocks) override {
        server_kv_host_blocks result;
        wait(swap_out(blocks, result));
        return result;
    }

private:
    server_kv_tensor_layout layout_;
    size_t capacity_blocks_;
    uint64_t completed_event_ = 0;
    std::vector<uint16_t> k_;
    std::vector<uint16_t> v_;
    server_kv_backend_stats stats_;

    server_kv_backend_event complete_event() { return { ++completed_event_ }; }

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

    static void copy_plane(
            const std::vector<uint16_t> & source,
            size_t source_block,
            std::vector<uint16_t> & destination,
            size_t destination_block,
            size_t stride) {
        std::copy_n(source.begin() + source_block * stride, stride,
                destination.begin() + destination_block * stride);
    }
};

} // namespace

size_t server_kv_tensor_layout::elements_per_plane_block() const {
    size_t result = layers;
    result = checked_multiply(result, kv_heads, "KV heads");
    result = checked_multiply(result, head_dim, "head dimension");
    return checked_multiply(result, block_tokens, "block tokens");
}

size_t server_kv_tensor_layout::bytes_per_block() const {
    // Two FP16 planes: K and V.
    return checked_multiply(elements_per_plane_block(), 2 * sizeof(uint16_t), "block bytes");
}

void server_kv_tensor_layout::validate() const {
    if (layers == 0 || kv_heads == 0 || head_dim == 0 || block_tokens == 0) {
        throw std::invalid_argument("KV tensor layout dimensions must be positive");
    }
    if (element_type != server_kv_element_type::fp16 ||
            memory_layout != server_kv_memory_layout::separate_k_v_planes) {
        throw std::invalid_argument("unsupported KV tensor representation");
    }
    (void) bytes_per_block();
}

void server_kv_host_blocks::resize(
        const server_kv_tensor_layout & value_layout,
        size_t value_block_count) {
    value_layout.validate();
    layout = value_layout;
    block_count = value_block_count;
    const size_t elements = checked_multiply(
            layout.elements_per_plane_block(), block_count, "host block buffer");
    k.resize(elements);
    v.resize(elements);
}

std::unique_ptr<server_kv_block_backend> server_kv_create_cpu_block_backend(
        server_kv_tensor_layout layout,
        size_t capacity_blocks) {
    return std::make_unique<server_kv_cpu_block_backend>(layout, capacity_blocks);
}
