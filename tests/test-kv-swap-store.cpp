#ifdef NDEBUG
#undef NDEBUG
#endif

#include "server-kv-swap-store.h"

#include <cassert>
#include <filesystem>
#include <fstream>

static server_kv_swap_payload payload() {
    server_kv_swap_payload result;
    result.sequence_id = 17;
    result.physical_blocks = {3, 9};
    result.data.resize({2, 2, 4, 4}, 2);
    for (size_t i = 0; i < result.data.k.size(); ++i) {
        result.data.k[i] = (uint16_t) (i + 1);
        result.data.v[i] = (uint16_t) (1000 + i);
    }
    return result;
}

static server_kv_swap_payload opaque_payload() {
    server_kv_swap_payload result;
    result.sequence_id = 23;
    result.opaque_state = {0, 1, 2, 3, 5, 8, 13, 21};
    return result;
}

static void exercise_opaque(server_kv_swap_store & store) {
    const auto source = opaque_payload();
    const auto saved = store.save(source);
    assert(saved.ok);
    server_kv_swap_payload restored;
    assert(store.restore(saved.handle, restored));
    assert(restored.sequence_id == source.sequence_id);
    assert(restored.opaque_state == source.opaque_state);
    assert(restored.physical_blocks.empty() && restored.data.k.empty() && restored.data.v.empty());
    assert(store.erase(saved.handle));
}

static void exercise(server_kv_swap_store & store) {
    const auto source = payload();
    const auto saved = store.save(source);
    assert(saved.ok && saved.handle != 0);
    server_kv_swap_payload restored;
    assert(store.restore(saved.handle, restored));
    assert(restored.sequence_id == source.sequence_id);
    assert(restored.physical_blocks == source.physical_blocks);
    assert(restored.data.k == source.data.k && restored.data.v == source.data.v);

    store.inject(server_kv_swap_fault::next_restore);
    restored.sequence_id = 999;
    std::string error;
    assert(!store.restore(saved.handle, restored, &error));
    assert(!error.empty() && restored.sequence_id == 999); // no partial publication
    assert(store.erase(saved.handle));
    assert(!store.restore(saved.handle, restored));
    const auto stats = store.stats();
    assert(stats.saves == 1 && stats.restores == 1 && stats.erases == 1);
    assert(stats.restore_failures == 2 && stats.bytes_current == 0 && stats.bytes_peak > 0);
    assert(stats.bytes_saved_total > 0 && stats.bytes_restored_total == stats.bytes_saved_total);
}

int main() {
    auto host = server_kv_create_host_swap_store(1 << 20);
    exercise(*host);
    host->inject(server_kv_swap_fault::next_save);
    assert(!host->save(payload()).ok);
    auto tiny = server_kv_create_host_swap_store(1);
    assert(!tiny->save(payload()).ok);
    auto opaque_host = server_kv_create_host_swap_store(1 << 20);
    exercise_opaque(*opaque_host);

    const auto directory = std::filesystem::temp_directory_path() / "cacheflow-kv-swap-test";
    std::filesystem::remove_all(directory);
    auto file = server_kv_create_file_swap_store(directory, 1 << 20);
    exercise(*file);
    file->inject(server_kv_swap_fault::next_save);
    assert(!file->save(payload()).ok);

    const auto saved = file->save(payload());
    assert(saved.ok);
    const auto path = directory / ("sequence-" + std::to_string(saved.handle) + ".cfswap");
    std::ofstream corrupt(path, std::ios::binary | std::ios::app);
    corrupt.put('x');
    corrupt.close();
    server_kv_swap_payload untouched;
    untouched.sequence_id = 123;
    assert(!file->restore(saved.handle, untouched));
    assert(untouched.sequence_id == 123);
    assert(file->erase(saved.handle));
    auto opaque_file = server_kv_create_file_swap_store(directory, 1 << 20);
    exercise_opaque(*opaque_file);
    std::filesystem::remove_all(directory);
    return 0;
}
