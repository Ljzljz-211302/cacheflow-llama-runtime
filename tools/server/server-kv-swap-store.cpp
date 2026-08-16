#include "server-kv-swap-store.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace {

using swap_clock = std::chrono::steady_clock;

uint64_t elapsed_us(swap_clock::time_point start) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            swap_clock::now() - start).count();
}

constexpr uint64_t swap_magic = 0x31505753464b5643ULL; // "CVKFSWP1"

size_t payload_bytes(const server_kv_swap_payload & payload) {
    return payload.physical_blocks.size() * sizeof(server_kv_physical_block_id) +
            (payload.data.k.size() + payload.data.v.size()) * sizeof(uint16_t) +
            payload.opaque_state.size();
}

uint64_t checksum(const server_kv_swap_payload & payload) {
    uint64_t value = 1469598103934665603ULL;
    auto add = [&value](const void * ptr, size_t bytes) {
        const auto * data = static_cast<const uint8_t *>(ptr);
        for (size_t i = 0; i < bytes; ++i) {
            value ^= data[i];
            value *= 1099511628211ULL;
        }
    };
    add(&payload.sequence_id, sizeof(payload.sequence_id));
    if (!payload.physical_blocks.empty()) add(payload.physical_blocks.data(),
            payload.physical_blocks.size() * sizeof(payload.physical_blocks[0]));
    if (!payload.data.k.empty()) add(payload.data.k.data(), payload.data.k.size() * sizeof(uint16_t));
    if (!payload.data.v.empty()) add(payload.data.v.data(), payload.data.v.size() * sizeof(uint16_t));
    if (!payload.opaque_state.empty()) add(payload.opaque_state.data(), payload.opaque_state.size());
    return value;
}

void validate_payload(const server_kv_swap_payload & payload) {
    if (payload.sequence_id < 0) throw std::invalid_argument("invalid KV swap sequence");
    if (!payload.opaque_state.empty()) {
        if (!payload.physical_blocks.empty() || payload.data.block_count != 0 ||
                !payload.data.k.empty() || !payload.data.v.empty()) {
            throw std::invalid_argument("opaque and structured KV swap payloads cannot be mixed");
        }
        return;
    }
    payload.data.layout.validate();
    const size_t elements = payload.data.layout.elements_per_plane_block() * payload.data.block_count;
    if (payload.physical_blocks.size() != payload.data.block_count ||
            payload.data.k.size() != elements || payload.data.v.size() != elements) {
        throw std::invalid_argument("invalid KV swap payload");
    }
}

class host_swap_store final : public server_kv_swap_store {
public:
    explicit host_swap_store(size_t budget_bytes) : budget_bytes_(budget_bytes) {}

    server_kv_swap_result save(const server_kv_swap_payload & payload) override {
        const auto started = swap_clock::now();
        if (consume(server_kv_swap_fault::next_save)) return save_error("injected host save failure");
        try { validate_payload(payload); } catch (const std::exception & error) { return save_error(error.what()); }
        const size_t bytes = payload_bytes(payload);
        if (bytes > budget_bytes_ - std::min<size_t>(budget_bytes_, stats_.bytes_current)) {
            return save_error("host swap budget exhausted");
        }
        const auto handle = next_handle_++;
        records_.emplace(handle, payload);
        stats_.saves++;
        stats_.bytes_saved_total += bytes;
        stats_.save_microseconds += elapsed_us(started);
        stats_.bytes_current += bytes;
        stats_.bytes_peak = std::max(stats_.bytes_peak, stats_.bytes_current);
        return {true, handle, {}};
    }

    bool restore(server_kv_swap_handle handle, server_kv_swap_payload & payload, std::string * error) override {
        const auto started = swap_clock::now();
        if (consume(server_kv_swap_fault::next_restore)) return restore_error("injected host restore failure", error);
        const auto found = records_.find(handle);
        if (found == records_.end()) return restore_error("unknown host swap handle", error);
        payload = found->second;
        stats_.restores++;
        stats_.bytes_restored_total += payload_bytes(payload);
        stats_.restore_microseconds += elapsed_us(started);
        return true;
    }

    bool erase(server_kv_swap_handle handle) override {
        const auto found = records_.find(handle);
        if (found == records_.end()) return false;
        stats_.bytes_current -= payload_bytes(found->second);
        records_.erase(found);
        stats_.erases++;
        return true;
    }

    void inject(server_kv_swap_fault fault) override { fault_ = fault; }
    server_kv_swap_store_stats stats() const override { return stats_; }

private:
    size_t budget_bytes_;
    server_kv_swap_handle next_handle_ = 1;
    server_kv_swap_fault fault_ = server_kv_swap_fault::none;
    std::unordered_map<server_kv_swap_handle, server_kv_swap_payload> records_;
    server_kv_swap_store_stats stats_;

    bool consume(server_kv_swap_fault expected) {
        if (fault_ != expected) return false;
        fault_ = server_kv_swap_fault::none;
        return true;
    }
    server_kv_swap_result save_error(std::string error) {
        stats_.save_failures++;
        return {false, 0, std::move(error)};
    }
    bool restore_error(std::string message, std::string * error) {
        stats_.restore_failures++;
        if (error) *error = std::move(message);
        return false;
    }
};

struct file_header {
    uint64_t magic;
    uint64_t checksum;
    int64_t sequence_id;
    uint64_t physical_count;
    uint64_t block_count;
    uint64_t opaque_bytes;
    uint32_t layers;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t block_tokens;
    uint8_t element_type;
    uint8_t memory_layout;
    uint8_t reserved[6]{};
};

class file_swap_store final : public server_kv_swap_store {
public:
    file_swap_store(std::filesystem::path directory, size_t budget_bytes) :
            directory_(std::move(directory)), budget_bytes_(budget_bytes) {
        std::filesystem::create_directories(directory_);
    }

    server_kv_swap_result save(const server_kv_swap_payload & payload) override {
        const auto started = swap_clock::now();
        if (consume(server_kv_swap_fault::next_save)) return save_error("injected file save failure");
        try { validate_payload(payload); } catch (const std::exception & error) { return save_error(error.what()); }
        const size_t bytes = payload_bytes(payload);
        if (bytes > budget_bytes_ - std::min<size_t>(budget_bytes_, stats_.bytes_current)) {
            return save_error("file swap budget exhausted");
        }
        const auto handle = next_handle_++;
        const auto final_path = path(handle);
        const auto temporary_path = final_path.string() + ".tmp";
        file_header header {
            swap_magic, checksum(payload), payload.sequence_id,
            payload.physical_blocks.size(), payload.data.block_count,
            payload.opaque_state.size(),
            payload.data.layout.layers, payload.data.layout.kv_heads,
            payload.data.layout.head_dim, payload.data.layout.block_tokens,
            (uint8_t) payload.data.layout.element_type,
            (uint8_t) payload.data.layout.memory_layout,
        };
        try {
            std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
            output.exceptions(std::ios::badbit | std::ios::failbit);
            output.write(reinterpret_cast<const char *>(&header), sizeof(header));
            write_vector(output, payload.physical_blocks);
            write_vector(output, payload.data.k);
            write_vector(output, payload.data.v);
            write_vector(output, payload.opaque_state);
            output.close();
            std::filesystem::rename(temporary_path, final_path);
        } catch (const std::exception & error) {
            std::error_code ignored;
            std::filesystem::remove(temporary_path, ignored);
            return save_error(error.what());
        }
        sizes_[handle] = bytes;
        stats_.saves++;
        stats_.bytes_saved_total += bytes;
        stats_.save_microseconds += elapsed_us(started);
        stats_.bytes_current += bytes;
        stats_.bytes_peak = std::max(stats_.bytes_peak, stats_.bytes_current);
        return {true, handle, {}};
    }

    bool restore(server_kv_swap_handle handle, server_kv_swap_payload & payload, std::string * error) override {
        const auto started = swap_clock::now();
        if (consume(server_kv_swap_fault::next_restore)) return restore_error("injected file restore failure", error);
        if (sizes_.count(handle) == 0) return restore_error("unknown file swap handle", error);
        try {
            std::ifstream input(path(handle), std::ios::binary);
            input.exceptions(std::ios::badbit | std::ios::failbit);
            file_header header{};
            input.read(reinterpret_cast<char *>(&header), sizeof(header));
            if (header.magic != swap_magic || header.physical_count != header.block_count ||
                    header.block_count > std::numeric_limits<size_t>::max()) {
                throw std::runtime_error("invalid KV swap file header");
            }
            server_kv_swap_payload candidate;
            candidate.sequence_id = header.sequence_id;
            if (header.opaque_bytes > 0) {
                if (header.block_count != 0 || header.physical_count != 0 ||
                        header.opaque_bytes > std::numeric_limits<size_t>::max()) {
                    throw std::runtime_error("invalid opaque KV swap file header");
                }
                candidate.opaque_state.resize((size_t) header.opaque_bytes);
                read_vector(input, candidate.opaque_state);
            } else {
                candidate.data.layout = {
                    header.layers, header.kv_heads, header.head_dim, header.block_tokens,
                    (server_kv_element_type) header.element_type,
                    (server_kv_memory_layout) header.memory_layout,
                };
                candidate.data.layout.validate();
                candidate.data.block_count = (size_t) header.block_count;
                const size_t elements = candidate.data.layout.elements_per_plane_block() * candidate.data.block_count;
                candidate.physical_blocks.resize((size_t) header.physical_count);
                candidate.data.k.resize(elements);
                candidate.data.v.resize(elements);
                read_vector(input, candidate.physical_blocks);
                read_vector(input, candidate.data.k);
                read_vector(input, candidate.data.v);
            }
            input.peek();
            if (!input.eof() || checksum(candidate) != header.checksum) {
                throw std::runtime_error("KV swap checksum or length mismatch");
            }
            payload = std::move(candidate); // transactional publication
        } catch (const std::exception & exception) {
            return restore_error(exception.what(), error);
        }
        stats_.restores++;
        stats_.bytes_restored_total += payload_bytes(payload);
        stats_.restore_microseconds += elapsed_us(started);
        return true;
    }

    bool erase(server_kv_swap_handle handle) override {
        const auto found = sizes_.find(handle);
        if (found == sizes_.end()) return false;
        std::error_code error;
        if (!std::filesystem::remove(path(handle), error) || error) return false;
        stats_.bytes_current -= found->second;
        sizes_.erase(found);
        stats_.erases++;
        return true;
    }

    void inject(server_kv_swap_fault fault) override { fault_ = fault; }
    server_kv_swap_store_stats stats() const override { return stats_; }

private:
    std::filesystem::path directory_;
    size_t budget_bytes_;
    server_kv_swap_handle next_handle_ = 1;
    server_kv_swap_fault fault_ = server_kv_swap_fault::none;
    std::unordered_map<server_kv_swap_handle, size_t> sizes_;
    server_kv_swap_store_stats stats_;

    std::filesystem::path path(server_kv_swap_handle handle) const {
        return directory_ / ("sequence-" + std::to_string(handle) + ".cfswap");
    }
    bool consume(server_kv_swap_fault expected) {
        if (fault_ != expected) return false;
        fault_ = server_kv_swap_fault::none;
        return true;
    }
    server_kv_swap_result save_error(std::string error) {
        stats_.save_failures++;
        return {false, 0, std::move(error)};
    }
    bool restore_error(std::string message, std::string * error) {
        stats_.restore_failures++;
        if (error) *error = std::move(message);
        return false;
    }
    template <typename T> static void write_vector(std::ofstream & output, const std::vector<T> & values) {
        if (!values.empty()) output.write(reinterpret_cast<const char *>(values.data()),
                (std::streamsize) (values.size() * sizeof(T)));
    }
    template <typename T> static void read_vector(std::ifstream & input, std::vector<T> & values) {
        if (!values.empty()) input.read(reinterpret_cast<char *>(values.data()),
                (std::streamsize) (values.size() * sizeof(T)));
    }
};

} // namespace

std::unique_ptr<server_kv_swap_store> server_kv_create_host_swap_store(size_t budget_bytes) {
    return std::make_unique<host_swap_store>(budget_bytes);
}

std::unique_ptr<server_kv_swap_store> server_kv_create_file_swap_store(
        const std::filesystem::path & directory, size_t budget_bytes) {
    return std::make_unique<file_swap_store>(directory, budget_bytes);
}
