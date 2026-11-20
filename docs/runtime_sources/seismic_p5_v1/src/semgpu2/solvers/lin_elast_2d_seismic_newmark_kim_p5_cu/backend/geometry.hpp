#pragma once

#include "data.hpp"
#include <cuda_runtime_api.h>

namespace semgpu2_seismic_p5_cu {
// Borrowed device topology. Indices keep SolverAPI storage types.
struct GeometryLaunchData {
    const Index* p_nodes;
    int q_nodes_per_element;
    Index element_count;
    Index q_count;
};

cudaError_t assemble_q_geometry(const GeometryLaunchData& data, int rank,
    const Real* coordinates, Index n_count, Real* q_sizes, Real* q_difgeoms, cudaStream_t stream);
cudaError_t assemble_nodal_measure(const GeometryLaunchData& data,
    const Real* q_sizes, Real* n_sizes, cudaStream_t stream);
}  // namespace semgpu2_seismic_p5_cu
