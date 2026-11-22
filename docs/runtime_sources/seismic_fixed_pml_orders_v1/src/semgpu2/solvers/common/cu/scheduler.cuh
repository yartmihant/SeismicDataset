#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace semgpu2::common::cu {
struct QSlot {
    bool skip_thread;
    std::int64_t thread_index;
    std::int64_t element_index;
    std::int64_t slot_start;
    std::int64_t local_q;
};

// Reader only: address in a domain remains a domain address. No PET loading,
// shared memory or barriers. Only thread_index is meaningful when skipping.
__device__ inline QSlot resolve_q_slot(
    int slots_per_block, const std::int64_t* thread_slots,
    const std::int64_t* slot_entity_indices, const std::int64_t* slot_thread_offsets) {
    const std::int64_t tid = threadIdx.x;
    const std::int64_t block = blockIdx.x;
    const std::int64_t block_slot = thread_slots[block * blockDim.x + tid];
    if (block_slot < 0) {
        return {true, tid, 0, 0, 0};
    }
    const std::int64_t slot = block * slots_per_block + block_slot;
    const std::int64_t start = slot_thread_offsets[slot];
    return {false, tid, slot_entity_indices[slot], start, tid - start};
}
}  // namespace semgpu2::common::cu
