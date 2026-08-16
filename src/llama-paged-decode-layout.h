#pragma once

#include <cstdint>
#include <vector>

struct llama_paged_decode_cell {
    uint32_t logical_position;
    uint32_t cell_index;
};

enum class llama_paged_decode_layout_status {
    success,
    invalid_page_size,
    empty_sequence,
    missing_position,
    duplicate_position,
    non_contiguous_block,
    unsupported_request,
};

struct llama_paged_decode_layout {
    // Row-major [sequence][logical block]. Shorter rows are padded with zero;
    // context_lengths determines which entries are live.
    std::vector<uint32_t> block_table;
    std::vector<uint32_t> context_lengths;
    uint32_t max_blocks = 0;
    // True when every live row is one physically consecutive token range.
    // Such a layout can reuse the upstream contiguous attention kernel while
    // retaining Paged as the selected high-level action.
    bool physically_contiguous = false;
};

// Converts the logical token order of one sequence into the physical KV block
// bases consumed by paged decode. Each logical block must be physically
// contiguous; otherwise the caller must use Remap or Direct.
llama_paged_decode_layout_status llama_paged_decode_build_layout(
        const std::vector<llama_paged_decode_cell> & cells,
        uint32_t page_size,
        llama_paged_decode_layout & output);

// Builds one independent page-table row per decode sequence. The row order is
// the query-token order used by the batched attention operator.
llama_paged_decode_layout_status llama_paged_decode_build_batch_layout(
        const std::vector<std::vector<llama_paged_decode_cell>> & sequences,
        uint32_t page_size,
        llama_paged_decode_layout & output);
