#include "../geometry.hpp"
#include "../operators/pet.cuh"
#include <limits>

namespace semgpu2_seismic_cu {
namespace field = semgpu2::common::cu::tenzor;
namespace {
template<int Rank>
__global__ void assemble_q_geometry_kernel(GeometryLaunchData data,
    const Real* coordinates, Index n_count, Real* q_sizes, Real* q_difgeoms) {
    const auto slot = resolve_q_slot(data.slots_per_block, data.thread_slots,
        data.slot_entity_indices, data.slot_thread_offsets);

    __shared__ Real p_coordinates[2][THREADS_MAX];
    p_coordinates[0][slot.thread_index] = 0.0;
    p_coordinates[1][slot.thread_index] = 0.0;
    if (!slot.skip_thread) {

        const Index node = data.p_nodes[data.p_node_offsets[slot.element_index] + slot.local_q];
        p_coordinates[0][slot.thread_index] = coordinates[node];
        p_coordinates[1][slot.thread_index] = coordinates[n_count + node];
    }
    __syncthreads();
    if (slot.skip_thread) return;
    Real jacobian[2][Rank]{};
    field::local_q_grad(data.pets, slot.local_q, slot.slot_start, p_coordinates, jacobian);
    const Index qid = data.q_node_offsets[slot.element_index] + slot.local_q;
    const Real weight = field::pet_q_weight<Rank>(data.pets, slot.local_q);
    if constexpr (Rank == 2) {
        semgpu2::common::cu::reduce_full_rank_geometry(jacobian, q_difgeoms, q_sizes, qid, data.q_count, weight);
    } else {
        semgpu2::common::cu::reduce_embedded_geometry(jacobian, q_difgeoms, q_sizes, qid, data.q_count, weight);
    }
}

__global__ void assemble_nodal_measure_kernel(GeometryLaunchData data,
    const Real* q_sizes, Real* n_sizes) {
    const auto slot = resolve_q_slot(data.slots_per_block, data.thread_slots,
        data.slot_entity_indices, data.slot_thread_offsets);
    if (slot.skip_thread) return;
    const Index node = data.p_nodes[data.p_node_offsets[slot.element_index] + slot.local_q];
    atomicAdd(n_sizes + node, q_sizes[data.q_node_offsets[slot.element_index] + slot.local_q]);
}
}  // namespace

cudaError_t assemble_q_geometry(const GeometryLaunchData& data, int rank,
    const Real* coordinates, Index n_count, Real* q_sizes, Real* q_difgeoms, cudaStream_t stream) {
    if (data.block_count < 0 || data.block_count > std::numeric_limits<int>::max()) return cudaErrorInvalidConfiguration;
    if (rank != 1 && rank != 2) return cudaErrorInvalidValue;
    if (!data.block_count) return cudaSuccess;
    if (rank == 2) {
        assemble_q_geometry_kernel<2><<<static_cast<unsigned>(data.block_count), THREADS_MAX, 0, stream>>>(data, coordinates, n_count, q_sizes, q_difgeoms);
    } else {
        assemble_q_geometry_kernel<1><<<static_cast<unsigned>(data.block_count), THREADS_MAX, 0, stream>>>(data, coordinates, n_count, q_sizes, q_difgeoms);
    }
    return cudaGetLastError();
}

cudaError_t assemble_nodal_measure(const GeometryLaunchData& data,
    const Real* q_sizes, Real* n_sizes, cudaStream_t stream) {
    if (data.block_count < 0 || data.block_count > std::numeric_limits<int>::max()) return cudaErrorInvalidConfiguration;
    if (!data.block_count) return cudaSuccess;
    assemble_nodal_measure_kernel<<<static_cast<unsigned>(data.block_count), THREADS_MAX, 0, stream>>>(data, q_sizes, n_sizes);
    return cudaGetLastError();
}
}  // namespace semgpu2_seismic_cu
