#pragma once

#include <cstdint>
#include <vector>

namespace semgpu2::common::cu {
using Index = std::int64_t;

// Host-owned packed schedule, with the same fields and addressing as NB.
// Entity indices mean Space elements OR positions in a caller's domain.
struct BlockSchedule {
    std::vector<Index> thread_slots;
    std::vector<Index> slot_entity_indices;
    std::vector<Index> slot_thread_offsets;
    Index block_count = 0;
    int threads_per_block = 0;
    // Maximum occupied slots in this schedule; zero for an empty domain.
    int slots_per_block = 0;
};

// Pure host operations; no Python objects, CUDA allocations or solver codes.
Index validate_element_q_capacity(const std::vector<Index>& q_offsets, int threads_max);
BlockSchedule build_q_block_schedule(
    const std::vector<Index>& q_offsets, int threads_per_block);
}  // namespace semgpu2::common::cu
