#pragma once

#include "server-kv-block-backend.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using server_kv_swap_handle = uint64_t;

struct server_kv_swap_payload {
    int64_t sequence_id = -1;
    std::vector<server_kv_physical_block_id> physical_blocks;
    server_kv_host_blocks data;
    // Serialized llama sequence state used by the production runtime adapter.
    // It is mutually exclusive with the structured physical-block payload.
    std::vector<uint8_t> opaque_state;
};

struct server_kv_swap_result {
    bool ok = false;
    server_kv_swap_handle handle = 0;
    std::string error;
};

struct server_kv_swap_store_stats {
    uint64_t saves = 0;
    uint64_t restores = 0;
    uint64_t erases = 0;
    uint64_t save_failures = 0;
    uint64_t restore_failures = 0;
    uint64_t bytes_saved_total = 0;
    uint64_t bytes_restored_total = 0;
    uint64_t save_microseconds = 0;
    uint64_t restore_microseconds = 0;
    uint64_t bytes_current = 0;
    uint64_t bytes_peak = 0;
};

enum class server_kv_swap_fault : uint8_t {
    none,
    next_save,
    next_restore,
};

// Durable seam between preemption policy and physical KV bytes. Implementations
// commit a complete payload or leave the previous handle untouched.
class server_kv_swap_store {
public:
    virtual ~server_kv_swap_store() = default;
    virtual server_kv_swap_result save(const server_kv_swap_payload & payload) = 0;
    virtual bool restore(server_kv_swap_handle handle, server_kv_swap_payload & payload,
            std::string * error = nullptr) = 0;
    virtual bool erase(server_kv_swap_handle handle) = 0;
    virtual void inject(server_kv_swap_fault fault) = 0;
    virtual server_kv_swap_store_stats stats() const = 0;
};

std::unique_ptr<server_kv_swap_store> server_kv_create_host_swap_store(size_t budget_bytes);
std::unique_ptr<server_kv_swap_store> server_kv_create_file_swap_store(
        const std::filesystem::path & directory, size_t budget_bytes);
