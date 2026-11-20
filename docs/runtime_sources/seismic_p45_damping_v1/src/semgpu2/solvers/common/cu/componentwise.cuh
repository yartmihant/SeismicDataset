#pragma once

// Include after the dimension-specific B field operators.
#include "scheduler.cuh"
#include <limits>

namespace semgpu2::common::cu {
struct ComponentFieldLaunch {
    PETView pets;
    const Code* pet_ids;
    const Index* pet_offsets;
    const Index* entities;
    const Index* p_offsets;
    const Index* q_offsets;
    const Index* thread_slots;
    const Index* slot_entities;
    const Index* slot_offsets;
    Index blocks, p_count, q_count;
    int slots_per_block;
};

template<int Factors, int Threads, bool Project>
__global__ void component_field_kernel(ComponentFieldLaunch data, const double* source,
    double* target, const double* inverses, const Index* inverse_offsets, Index component_base) {
    const auto slot = resolve_q_slot(data.slots_per_block, data.thread_slots, data.slot_entities, data.slot_offsets);
    const Index c = component_base + blockIdx.y;
    Code ids[Factors]{};
    int count = 0;
    if (!slot.skip_thread) {
        const Index element = data.entities[slot.element_index], begin = data.pet_offsets[element];
        count = static_cast<int>(data.pet_offsets[element + 1] - begin);
        for (int f = 0; f < count; ++f) ids[f] = data.pet_ids[begin + f];
    }
    const auto pq = slot.skip_thread ? PQLocal{false, 0, 0} : resolve_pq_local(data.pets, slot.local_q, ids, count);
    __shared__ double shared[1][Threads];
    shared[0][slot.thread_index] = 0.0;
    if constexpr (Project) {
        if (!slot.skip_thread) shared[0][slot.thread_index] =
            source[c * data.q_count + data.q_offsets[slot.element_index] + slot.local_q] *
            pet_q_weight(data.pets, slot.local_q, ids, count);
    } else if (pq.has_p) shared[0][slot.thread_index] =
        source[c * data.p_count + data.p_offsets[slot.element_index] + pq.p_local];
    __syncthreads();
    double value[1]{};
    if constexpr (!Project) {
        if (!slot.skip_thread) {
            local_q_value(data.pets, slot.local_q, slot.slot_start, shared, ids, count, value);
            target[c * data.q_count + data.q_offsets[slot.element_index] + slot.local_q] = value[0];
        }
    } else {
        if (pq.has_p) local_p_value(data.pets, slot.local_q, slot.slot_start, shared, ids, count, value);
        __syncthreads();
        shared[0][slot.thread_index] = value[0];
        __syncthreads();
        Index stride = 1;
        for (int f = 0; f < Factors; ++f) {
            double result = 0.0;
            if (pq.has_p && f < count) {
                const Code pet = ids[f];
                const Index pc = data.pets.p_node_counts[pet], qc = data.pets.q_node_counts[pet];
                const Index coordinate = (slot.local_q / stride) % qc;
                const Index origin = slot.slot_start + slot.local_q - coordinate * stride;
                const Index offset = inverse_offsets[pet] + coordinate * pc;
                for (Index other = 0; other < pc; ++other) result += inverses[offset + other] * shared[0][origin + other * stride];
            } else if (pq.has_p) result = shared[0][slot.thread_index];
            __syncthreads();
            shared[0][slot.thread_index] = result;
            __syncthreads();
            if (f < count) stride *= data.pets.q_node_counts[ids[f]];
        }
        if (pq.has_p) target[c * data.p_count + data.p_offsets[slot.element_index] + pq.p_local] = shared[0][slot.thread_index];
    }
}

// Nonaliasing SoA fields; all IDs and offsets address the caller's domain.
// Error/finite policy, allocation and stream ownership stay with the consumer.
template<int Factors, int Threads, bool Project>
inline cudaError_t launch_component_field(const ComponentFieldLaunch& data, const double* source,
    double* target, Index components, const double* inverses, const Index* inverse_offsets, cudaStream_t stream) {
    constexpr Index limit = std::numeric_limits<Index>::max() / sizeof(double);
    if (components < 0 || data.blocks < 0 || data.blocks > std::numeric_limits<int>::max() ||
        data.p_count < 0 || data.q_count < 0 ||
        (components && (data.p_count > limit / components || data.q_count > limit / components))) return cudaErrorInvalidValue;
    if (!components || !data.blocks || !data.p_count || !data.q_count) return cudaSuccess;
    if (!source || !target || source == target) return cudaErrorInvalidValue;
    for (Index base = 0; base < components;) {
        const unsigned count = static_cast<unsigned>(components - base > 65535 ? 65535 : components - base);
        component_field_kernel<Factors, Threads, Project><<<dim3(static_cast<unsigned>(data.blocks), count), Threads, 0, stream>>>(
            data, source, target, inverses, inverse_offsets, base);
        if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;
        base += count;
    }
    return cudaSuccess;
}
} // namespace semgpu2::common::cu
