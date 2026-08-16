#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-paged-decode-layout.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

int main() {
    std::vector<llama_paged_decode_cell> cells;
    for (uint32_t token = 0; token < 16; ++token) {
        cells.push_back({ token, 32 + token });
    }
    cells.push_back({ 16, 4 });

    llama_paged_decode_layout layout;
    auto status = llama_paged_decode_build_layout(cells, 16, layout);
    assert(status == llama_paged_decode_layout_status::success);
    assert((layout.context_lengths == std::vector<uint32_t>{ 17 }));
    assert(layout.max_blocks == 2);
    assert((layout.block_table == std::vector<uint32_t>{ 32, 4 }));
    assert(!layout.physically_contiguous);

    cells.clear();
    for (uint32_t token = 0; token < 33; ++token) {
        cells.push_back({ token, 48 + token });
    }
    status = llama_paged_decode_build_layout(cells, 16, layout);
    assert(status == llama_paged_decode_layout_status::success);
    assert(layout.physically_contiguous);
    std::reverse(cells.begin(), cells.end());
    status = llama_paged_decode_build_layout(cells, 16, layout);
    assert(status == llama_paged_decode_layout_status::success);
    assert(layout.physically_contiguous);
    std::reverse(cells.begin(), cells.end());

    // A block table stores one physical base per logical block.  A hole
    // inside a block must fail closed instead of silently becoming a
    // token-indirection table.
    cells[7].cell_index = 99;
    status = llama_paged_decode_build_layout(cells, 16, layout);
    assert(status == llama_paged_decode_layout_status::non_contiguous_block);
    assert(layout.block_table.empty());
    assert(layout.context_lengths.empty());
    assert(layout.max_blocks == 0);

    cells.clear();
    assert(llama_paged_decode_build_layout(cells, 16, layout) ==
            llama_paged_decode_layout_status::empty_sequence);
    cells = {{0, 4}};
    assert(llama_paged_decode_build_layout(cells, 0, layout) ==
            llama_paged_decode_layout_status::invalid_page_size);

    cells = {{0, 4}, {2, 6}};
    assert(llama_paged_decode_build_layout(cells, 16, layout) ==
            llama_paged_decode_layout_status::missing_position);
    assert(layout.block_table.empty());

    std::vector<std::vector<llama_paged_decode_cell>> sequences(3);
    for (uint32_t token = 0; token < 17; ++token) {
        sequences[0].push_back({ token, token < 16 ? 64 + token : 9 });
    }
    for (uint32_t token = 0; token < 33; ++token) {
        const uint32_t base = token < 16 ? 128 : (token < 32 ? 32 : 7);
        sequences[1].push_back({ token, base + token % 16 });
    }
    for (uint32_t token = 0; token < 8; ++token) {
        sequences[2].push_back({ token, 200 + token });
    }
    status = llama_paged_decode_build_batch_layout(sequences, 16, layout);
    assert(status == llama_paged_decode_layout_status::success);
    assert((layout.context_lengths == std::vector<uint32_t>{ 17, 33, 8 }));
    assert(layout.max_blocks == 3);
    assert((layout.block_table == std::vector<uint32_t>{
        64, 9, 0,
        128, 32, 7,
        200, 0, 0,
    }));
    assert(!layout.physically_contiguous);

    sequences[1][20].cell_index += 1;
    assert(llama_paged_decode_build_batch_layout(sequences, 16, layout) ==
            llama_paged_decode_layout_status::non_contiguous_block);
    assert(layout.block_table.empty());
    assert(layout.context_lengths.empty());

    cells = {{0, 4}, {0, 5}};
    assert(llama_paged_decode_build_layout(cells, 16, layout) ==
            llama_paged_decode_layout_status::duplicate_position);
    assert(layout.block_table.empty());

    return 0;
}
