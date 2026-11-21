#include "scheduler.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace semgpu2::common::cu {
namespace {
std::size_t schedule_size(Index blocks, int width) {
    if (blocks > std::numeric_limits<Index>::max() / width) {
        throw std::overflow_error("Packed schedule index range overflow");
    }
    const Index entries = blocks * width;
    if (static_cast<std::uint64_t>(entries) > std::vector<Index>().max_size()) {
        throw std::overflow_error("Packed schedule allocation size overflow");
    }
    return static_cast<std::size_t>(entries);
}
}  // namespace

Index validate_element_q_capacity(const std::vector<Index>& q_offsets, int threads_max) {
    if (threads_max <= 0 || q_offsets.empty() || q_offsets.front() < 0) {
        throw std::invalid_argument("Q schedule requires positive capacity and nonnegative offsets");
    }
    if (q_offsets.size() - 1 > static_cast<std::uint64_t>(std::numeric_limits<Index>::max())) {
        throw std::overflow_error("Q schedule entity index overflow");
    }
    Index max_q = 0;
    for (std::size_t entity = 0; entity + 1 < q_offsets.size(); ++entity) {
        if (q_offsets[entity + 1] <= q_offsets[entity]) {
            throw std::invalid_argument("Every scheduled element must have a positive Q count");
        }
        const Index q_count = q_offsets[entity + 1] - q_offsets[entity];
        if (q_count > threads_max) {
            throw std::invalid_argument("Element " + std::to_string(entity) + " requires "
                + std::to_string(q_count) + " Q threads; capacity is " + std::to_string(threads_max));
        }
        max_q = std::max(max_q, q_count);
    }
    return max_q;
}

BlockSchedule build_q_block_schedule(
    const std::vector<Index>& q_offsets, int threads_per_block) {
    if (threads_per_block <= 0 || threads_per_block > 1024) {
        throw std::invalid_argument("Invalid packed schedule thread capacity");
    }
    validate_element_q_capacity(q_offsets, threads_per_block);
    BlockSchedule result;
    result.threads_per_block = threads_per_block;
    const Index entity_count = static_cast<Index>(q_offsets.size() - 1);
    if (entity_count == 0) {
        return result;
    }

    // Pack whole entities in order, limited only by available Q threads.
    // Count blocks and their maximum occupied slot count before allocating.
    result.block_count = 1;
    int used_slots = 0;
    Index used_threads = 0;
    for (Index entity = 0; entity < entity_count; ++entity) {
        const Index q_count = q_offsets[entity + 1] - q_offsets[entity];
        if (q_count > threads_per_block - used_threads) {
            ++result.block_count;
            used_slots = 0;
            used_threads = 0;
        }
        ++used_slots;
        result.slots_per_block = std::max(result.slots_per_block, used_slots);
        used_threads += q_count;
    }
    result.thread_slots.assign(schedule_size(result.block_count, threads_per_block), -1);
    result.slot_entity_indices.assign(schedule_size(result.block_count, result.slots_per_block), 0);
    result.slot_thread_offsets.assign(result.slot_entity_indices.size(), 0);

    Index block = 0;
    used_slots = 0;
    used_threads = 0;
    for (Index entity = 0; entity < entity_count; ++entity) {
        const Index q_count = q_offsets[entity + 1] - q_offsets[entity];
        if (q_count > threads_per_block - used_threads) {
            ++block;
            used_slots = 0;
            used_threads = 0;
        }
        const Index slot = block * result.slots_per_block + used_slots;
        result.slot_entity_indices[slot] = entity;
        result.slot_thread_offsets[slot] = used_threads;
        for (Index local_q = 0; local_q < q_count; ++local_q) {
            result.thread_slots[block * threads_per_block + used_threads + local_q] = used_slots;
        }
        ++used_slots;
        used_threads += q_count;
    }
    return result;
}
}  // namespace semgpu2::common::cu
