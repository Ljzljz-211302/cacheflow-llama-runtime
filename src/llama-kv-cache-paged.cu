#include "llama-kv-cache.h"

#include "ggml.h"
#include "llama-impl.h"
#include "llama-kv-remap-cuda.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct cacheflow_swap_piece {
    void * device = nullptr;
    void * pinned = nullptr;
    size_t bytes = 0;
};

struct cacheflow_swap_record {
    uint32_t stream_id = 0;
    llama_kv_cells cells;
    std::vector<cacheflow_swap_piece> pieces;
    cudaEvent_t ready = nullptr;
    cudaEvent_t started = nullptr;
    size_t bytes = 0;
};

struct cacheflow_swap_store {
    cudaStream_t stream = nullptr;
    std::unordered_map<llama_seq_id, std::unique_ptr<cacheflow_swap_record>> records;
};

std::vector<std::pair<uint32_t, uint32_t>> occupied_ranges(const llama_kv_cells & cells) {
    std::vector<std::pair<uint32_t, uint32_t>> result;
    uint32_t begin = 0;
    bool inside = false;
    for (uint32_t cell = 0; cell < cells.size(); ++cell) {
        if (!cells.is_empty(cell) && !inside) {
            begin = cell;
            inside = true;
        } else if (cells.is_empty(cell) && inside) {
            result.emplace_back(begin, cell);
            inside = false;
        }
    }
    if (inside) result.emplace_back(begin, cells.size());
    return result;
}

bool cuda_ok(cudaError_t status, const char * operation, llama_cacheflow_cuda_stats * stats = nullptr) {
    if (status == cudaSuccess) return true;
    if (stats) stats->backend_errors++;
    LLAMA_LOG_WARN("CacheFlow CUDA KV adapter %s failed: %s\n",
            operation, cudaGetErrorString(status));
    return false;
}

void release_record(cacheflow_swap_record & record, llama_cacheflow_cuda_stats * stats = nullptr) {
    if (record.ready) {
        cudaEventSynchronize(record.ready);
        cudaEventDestroy(record.ready);
    }
    if (record.started) cudaEventDestroy(record.started);
    for (auto & piece : record.pieces) {
        if (piece.pinned) cudaFreeHost(piece.pinned);
    }
    if (stats) stats->pinned_bytes_current -= std::min<uint64_t>(
            stats->pinned_bytes_current, record.bytes);
}

} // namespace

bool llama_kv_cache::seq_can_swap_cuda_impl(llama_seq_id seq_id) const {
    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size() || n_stream <= 1 || other) {
        return false;
    }
    const uint32_t stream_id = seq_to_stream[seq_id];
    // A physical stream may only be swapped independently when no other
    // sequence maps to it. Unified KV therefore falls back to recomputation.
    for (size_t id = 0; id < seq_to_stream.size(); ++id) {
        if ((llama_seq_id) id != seq_id && seq_to_stream[id] == stream_id) return false;
    }
    for (const auto & layer : layers) {
        const ggml_tensor * tensors[] = { layer.k_stream[stream_id], layer.v_stream[stream_id] };
        for (const auto * tensor : tensors) {
            if (!tensor) continue;
            cudaPointerAttributes attributes{};
            if (cudaPointerGetAttributes(&attributes, tensor->data) != cudaSuccess ||
                    attributes.type != cudaMemoryTypeDevice) {
                cudaGetLastError();
                return false;
            }
        }
    }
    return true;
}

bool llama_kv_cache::seq_swap_out_cuda_impl(llama_seq_id seq_id) {
    if (!seq_can_swap_cuda_impl(seq_id)) return false;
    auto * store = static_cast<cacheflow_swap_store *>(paged_cuda_swap_store_);
    if (!store) {
        auto owned = std::make_unique<cacheflow_swap_store>();
        if (!cuda_ok(cudaStreamCreateWithFlags(&owned->stream, cudaStreamNonBlocking),
                "swap stream create", &cuda_stats_)) return false;
        store = owned.release();
        paged_cuda_swap_store_ = store;
    }
    seq_swap_erase_cuda_impl(seq_id);
    auto record = std::make_unique<cacheflow_swap_record>();
    record->stream_id = seq_to_stream[seq_id];
    record->cells = v_cells[record->stream_id].cp(0, v_cells[record->stream_id].size());
    const auto ranges = occupied_ranges(v_cells[record->stream_id]);
    if (ranges.empty()) return false;
    bool enqueued = false;
    if (!cuda_ok(cudaEventCreate(&record->started), "swap start event create", &cuda_stats_) ||
            !cuda_ok(cudaEventRecord(record->started, store->stream), "swap start event record", &cuda_stats_)) {
        release_record(*record, &cuda_stats_);
        return false;
    }
    for (const auto & layer : layers) {
        ggml_tensor * tensors[] = {
            layer.k_stream[record->stream_id], layer.v_stream[record->stream_id],
        };
        for (auto * tensor : tensors) {
            if (!tensor) continue;
            for (const auto & range : ranges) {
                cacheflow_swap_piece piece;
                piece.device = static_cast<uint8_t *>(tensor->data) + (size_t) range.first * tensor->nb[1];
                piece.bytes = (size_t) (range.second - range.first) * tensor->nb[1];
                if (!cuda_ok(cudaMallocHost(&piece.pinned, piece.bytes), "swap pinned allocation", &cuda_stats_)) {
                    if (enqueued) cudaStreamSynchronize(store->stream);
                    release_record(*record, &cuda_stats_);
                    return false;
                }
                record->pieces.push_back(piece);
                record->bytes += piece.bytes;
                cuda_stats_.pinned_bytes_current += piece.bytes;
                cuda_stats_.pinned_bytes_peak = std::max(
                        cuda_stats_.pinned_bytes_peak, cuda_stats_.pinned_bytes_current);
                if (!cuda_ok(cudaMemcpyAsync(piece.pinned, piece.device, piece.bytes,
                        cudaMemcpyDeviceToHost, store->stream), "swap-out tensor copy", &cuda_stats_)) {
                    cudaStreamSynchronize(store->stream);
                    release_record(*record, &cuda_stats_);
                    return false;
                }
                enqueued = true;
            }
        }
    }
    if (record->pieces.empty() ||
            !cuda_ok(cudaEventCreate(&record->ready), "swap event create", &cuda_stats_) ||
            !cuda_ok(cudaEventRecord(record->ready, store->stream), "swap-out event record", &cuda_stats_)) {
        if (enqueued) cudaStreamSynchronize(store->stream);
        release_record(*record, &cuda_stats_);
        return false;
    }
    const size_t bytes = record->bytes;
    cuda_stats_.copy_bytes += bytes;
    cuda_stats_.swap_bytes += bytes;
    store->records.emplace(seq_id, std::move(record));
    LLAMA_LOG_INFO("CacheFlow CUDA swapped out sequence %d: stream %u, pinned bytes %zu\n",
            seq_id, seq_to_stream[seq_id], bytes);
    return true;
}

bool llama_kv_cache::seq_swap_in_cuda_impl(llama_seq_id seq_id) {
    auto * store = static_cast<cacheflow_swap_store *>(paged_cuda_swap_store_);
    if (!store) return false;
    const auto found = store->records.find(seq_id);
    if (found == store->records.end()) return false;
    auto & record = *found->second;
    if (!cuda_ok(cudaEventSynchronize(record.ready), "wait swap-out event", &cuda_stats_)) return false;
    cuda_stats_.events_waited++;
    float out_ms = 0.0f;
    if (record.started && cuda_ok(cudaEventElapsedTime(&out_ms, record.started, record.ready),
            "swap-out elapsed time", &cuda_stats_)) {
        cuda_stats_.swap_out_microseconds += (uint64_t) (out_ms * 1000.0f);
    }
    cudaEventDestroy(record.ready);
    record.ready = nullptr;
    if (record.started) {
        cudaEventDestroy(record.started);
        record.started = nullptr;
    }
    if (!cuda_ok(cudaEventCreate(&record.started), "restore start event create", &cuda_stats_) ||
            !cuda_ok(cudaEventRecord(record.started, store->stream), "restore start event record", &cuda_stats_)) {
        return false;
    }
    for (const auto & piece : record.pieces) {
        if (!cuda_ok(cudaMemcpyAsync(piece.device, piece.pinned, piece.bytes,
                cudaMemcpyHostToDevice, store->stream), "swap-in tensor copy", &cuda_stats_)) return false;
    }
    if (!cuda_ok(cudaEventCreate(&record.ready), "restore event create", &cuda_stats_) ||
            !cuda_ok(cudaEventRecord(record.ready, store->stream), "restore event record", &cuda_stats_) ||
            !cuda_ok(cudaEventSynchronize(record.ready), "wait restore event", &cuda_stats_)) return false;
    cuda_stats_.events_waited++;
    float in_ms = 0.0f;
    if (cuda_ok(cudaEventElapsedTime(&in_ms, record.started, record.ready),
            "swap-in elapsed time", &cuda_stats_)) {
        cuda_stats_.swap_in_microseconds += (uint64_t) (in_ms * 1000.0f);
    }
    v_cells[record.stream_id].set(0, record.cells);
    const size_t bytes = record.bytes;
    cuda_stats_.copy_bytes += bytes;
    cuda_stats_.swap_bytes += bytes;
    const uint32_t stream_id = record.stream_id;
    seq_swap_erase_cuda_impl(seq_id);
    LLAMA_LOG_INFO("CacheFlow CUDA restored sequence %d: stream %u, pinned bytes %zu\n",
            seq_id, stream_id, bytes);
    return true;
}

void llama_kv_cache::seq_swap_erase_cuda_impl(llama_seq_id seq_id) {
    auto * store = static_cast<cacheflow_swap_store *>(paged_cuda_swap_store_);
    if (!store) return;
    const auto found = store->records.find(seq_id);
    if (found == store->records.end()) return;
    release_record(*found->second, &cuda_stats_);
    store->records.erase(found);
}

void llama_kv_cache::destroy_paged_cuda_swap_store() {
    auto * store = static_cast<cacheflow_swap_store *>(paged_cuda_swap_store_);
    if (!store) return;
    for (auto & item : store->records) release_record(*item.second, &cuda_stats_);
    if (store->stream) cudaStreamDestroy(store->stream);
    delete store;
    paged_cuda_swap_store_ = nullptr;
}

bool llama_kv_cache::copy_streams_paged_cuda(const stream_copy_info & info) const {
    if (info.empty()) return true;
    std::vector<llama_kv_remap_copy> copies;
    size_t max_elements = 0;
    size_t total_elements = 0;
    int device = -1;
    size_t partial_tail_clones = 0;
    for (size_t mapping = 0; mapping < info.ssrc.size(); ++mapping) {
        const uint32_t source_stream = info.ssrc[mapping];
        const uint32_t destination_stream = info.sdst[mapping];
        if (source_stream >= n_stream || destination_stream >= n_stream ||
                source_stream == destination_stream) {
            return false;
        }
        if (cacheflow_block_tokens_ > 0 &&
                info.copied_tokens[mapping] % cacheflow_block_tokens_ != 0) {
            partial_tail_clones++;
        }
        const auto ranges = occupied_ranges(v_cells[destination_stream]);
        if (ranges.empty()) return false;
        for (const auto & layer : layers) {
            const ggml_tensor * planes[] = {
                layer.k_stream[source_stream], layer.v_stream[source_stream],
            };
            ggml_tensor * destinations[] = {
                layer.k_stream[destination_stream], layer.v_stream[destination_stream],
            };
            for (size_t plane = 0; plane < 2; ++plane) {
                const auto * source = planes[plane];
                auto * destination = destinations[plane];
                if (!source && !destination) continue;
                if (!source || !destination || source->type != GGML_TYPE_F16 ||
                        destination->type != GGML_TYPE_F16 ||
                        ggml_nbytes(source) != ggml_nbytes(destination)) {
                    return false;
                }
                cudaPointerAttributes source_attributes{};
                cudaPointerAttributes destination_attributes{};
                if (cudaPointerGetAttributes(&source_attributes, source->data) != cudaSuccess ||
                        cudaPointerGetAttributes(&destination_attributes, destination->data) != cudaSuccess ||
                        source_attributes.type != cudaMemoryTypeDevice ||
                        destination_attributes.type != cudaMemoryTypeDevice ||
                        source_attributes.device != destination_attributes.device) {
                    cudaGetLastError();
                    return false;
                }
                if (device < 0) device = source_attributes.device;
                if (source_attributes.device != device) return false; // multi-GPU falls back safely
                if (source->nb[1] != destination->nb[1] || source->nb[1] % sizeof(uint16_t) != 0) {
                    return false;
                }
                for (const auto & range : ranges) {
                    const size_t byte_offset = (size_t) range.first * source->nb[1];
                    const size_t elements = (size_t) (range.second - range.first) *
                            source->nb[1] / sizeof(uint16_t);
                    copies.push_back({
                        reinterpret_cast<const uint16_t *>(
                                static_cast<const uint8_t *>(source->data) + byte_offset),
                        reinterpret_cast<uint16_t *>(
                                static_cast<uint8_t *>(destination->data) + byte_offset),
                        total_elements,
                        elements,
                    });
                    total_elements += elements;
                    max_elements = std::max(max_elements, elements);
                }
            }
        }
    }
    if (copies.empty() || device < 0) return false;
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice", &cuda_stats_)) return false;

    cudaStream_t stream = nullptr;
    llama_kv_remap_copy * device_copies = nullptr;
    uint16_t * device_staging = nullptr;
    cudaEvent_t complete = nullptr;
    bool success = false;
    cudaEvent_t started = nullptr;
    llama_kv_remap_accounting remap_accounting = {};
    if (!cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create", &cuda_stats_)) goto cleanup;
    if (!cuda_ok(cudaEventCreate(&started), "copy start event create", &cuda_stats_) ||
            !cuda_ok(cudaEventRecord(started, stream), "copy start event record", &cuda_stats_)) goto cleanup;
    if (!cuda_ok(cudaMallocAsync((void **) &device_copies,
            copies.size() * sizeof(llama_kv_remap_copy), stream), "descriptor allocation", &cuda_stats_)) goto cleanup;
    if (!cuda_ok(cudaMallocAsync((void **) &device_staging,
            total_elements * sizeof(uint16_t), stream), "staging allocation", &cuda_stats_)) goto cleanup;
    if (!cuda_ok(cudaMemcpyAsync(device_copies, copies.data(),
            copies.size() * sizeof(llama_kv_remap_copy), cudaMemcpyHostToDevice, stream),
            "descriptor upload", &cuda_stats_)) goto cleanup;
    {
        remap_accounting = llama_kv_remap_account(copies, device_staging);
        if (!cuda_ok(llama_kv_remap_launch_gather(device_copies, device_staging,
                copies.size(), max_elements, stream, llama_kv_remap_mode::vectorized),
                "vectorized real KV tensor gather", &cuda_stats_)) goto cleanup;
        if (!cuda_ok(llama_kv_remap_launch_scatter(device_copies, device_staging,
                copies.size(), max_elements, stream, llama_kv_remap_mode::vectorized),
                "vectorized real KV tensor scatter", &cuda_stats_)) goto cleanup;
    }
    if (!cuda_ok(cudaFreeAsync(device_copies, stream), "descriptor release", &cuda_stats_)) goto cleanup;
    device_copies = nullptr;
    if (!cuda_ok(cudaFreeAsync(device_staging, stream), "staging release", &cuda_stats_)) goto cleanup;
    device_staging = nullptr;
    if (!cuda_ok(cudaEventCreate(&complete), "event create", &cuda_stats_)) goto cleanup;
    if (!cuda_ok(cudaEventRecord(complete, stream), "event record", &cuda_stats_)) goto cleanup;
    if (!cuda_ok(cudaEventSynchronize(complete), "event wait", &cuda_stats_)) goto cleanup;
    cuda_stats_.events_waited++;
    cuda_stats_.remap_vectorized_bytes += remap_accounting.vectorized_bytes;
    cuda_stats_.remap_scalar_bytes += remap_accounting.scalar_bytes;
    cuda_stats_.kernel_launches += 2;
    cuda_stats_.copy_on_write += partial_tail_clones;
    cuda_stats_.blocks_copied += copies.size();
    for (const auto & copy : copies) cuda_stats_.copy_bytes += copy.elements * sizeof(uint16_t);
    {
        float copy_ms = 0.0f;
        if (cuda_ok(cudaEventElapsedTime(&copy_ms, started, complete),
                "copy elapsed time", &cuda_stats_)) {
            cuda_stats_.copy_microseconds += (uint64_t) (copy_ms * 1000.0f);
        }
    }
    success = true;
    if (partial_tail_clones > 0) {
        LLAMA_LOG_INFO("CacheFlow CUDA eagerly cloned %zu partial shared KV tails before append\n",
                partial_tail_clones);
    }

cleanup:
    if (device_copies) cudaFree(device_copies);
    if (device_staging) cudaFree(device_staging);
    if (complete) cudaEventDestroy(complete);
    if (started) cudaEventDestroy(started);
    if (stream) cudaStreamDestroy(stream);
    return success;
}
