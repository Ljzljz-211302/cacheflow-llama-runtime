#include "llama-paged-decode-layout.h"

#include <algorithm>

llama_paged_decode_layout_status llama_paged_decode_build_layout(
        const std::vector<llama_paged_decode_cell> & cells,
        uint32_t page_size,
        llama_paged_decode_layout & output) {
    return llama_paged_decode_build_batch_layout({ cells }, page_size, output);
}

llama_paged_decode_layout_status llama_paged_decode_build_batch_layout(
        const std::vector<std::vector<llama_paged_decode_cell>> & sequences,
        uint32_t page_size,
        llama_paged_decode_layout & output) {
    output = {};
    if (page_size == 0) {
        return llama_paged_decode_layout_status::invalid_page_size;
    }
    if (sequences.empty()) {
        return llama_paged_decode_layout_status::empty_sequence;
    }

    std::vector<std::vector<uint32_t>> rows;
    rows.reserve(sequences.size());
    output.context_lengths.reserve(sequences.size());
    for (const auto & cells : sequences) {
        if (cells.empty()) {
            output = {};
            return llama_paged_decode_layout_status::empty_sequence;
        }

        const auto logical_less = [](const auto & lhs, const auto & rhs) {
            return lhs.logical_position < rhs.logical_position;
        };
        const std::vector<llama_paged_decode_cell> * ordered = &cells;
        std::vector<llama_paged_decode_cell> sorted_cells;
        if (!std::is_sorted(cells.begin(), cells.end(), logical_less)) {
            sorted_cells = cells;
            std::sort(sorted_cells.begin(), sorted_cells.end(), logical_less);
            ordered = &sorted_cells;
        }

        std::vector<uint32_t> row;
        row.reserve((ordered->size() + page_size - 1) / page_size);
        for (uint32_t position = 0; position < ordered->size(); ++position) {
            if ((*ordered)[position].logical_position < position) {
                output = {};
                return llama_paged_decode_layout_status::duplicate_position;
            }
            if ((*ordered)[position].logical_position != position) {
                output = {};
                return llama_paged_decode_layout_status::missing_position;
            }
            if (position % page_size == 0) {
                row.push_back((*ordered)[position].cell_index);
            } else if ((*ordered)[position].cell_index != row.back() + position % page_size) {
                output = {};
                return llama_paged_decode_layout_status::non_contiguous_block;
            }
        }
        output.context_lengths.push_back(static_cast<uint32_t>(ordered->size()));
        output.max_blocks = std::max(output.max_blocks, static_cast<uint32_t>(row.size()));
        rows.push_back(std::move(row));
    }

    output.block_table.assign(static_cast<size_t>(output.max_blocks) * rows.size(), 0);
    output.physically_contiguous = true;
    for (size_t sequence = 0; sequence < rows.size(); ++sequence) {
        std::copy(rows[sequence].begin(), rows[sequence].end(),
                output.block_table.begin() + sequence * output.max_blocks);
        for (size_t block = 1; block < rows[sequence].size(); ++block) {
            output.physically_contiguous = output.physically_contiguous &&
                    rows[sequence][block] == rows[sequence][0] + block * page_size;
        }
    }
    return llama_paged_decode_layout_status::success;
}
