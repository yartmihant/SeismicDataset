#pragma once

#include "data.hpp"
#include <cuda_runtime_api.h>

namespace semgpu2_seismic_cu {
// Borrowed device topology and schedule. Indices keep SolverAPI storage types.
struct GeometryLaunchData {
    PETView pets;

    const Index* p_nodes;
    const Index* p_node_offsets;
    const Index* q_node_offsets;
    const Index* thread_slots;
    const Index* slot_entity_indices;
    const Index* slot_thread_offsets;
    int slots_per_block;
    Index block_count;
    Index q_count;
};

cudaError_t assemble_q_geometry(const GeometryLaunchData& data, int rank,
    const Real* coordinates, Index n_count, Real* q_sizes, Real* q_difgeoms, cudaStream_t stream);
cudaError_t assemble_nodal_measure(const GeometryLaunchData& data,
    const Real* q_sizes, Real* n_sizes, cudaStream_t stream);
}  // namespace semgpu2_seismic_cu
