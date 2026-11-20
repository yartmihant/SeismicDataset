#include "../geometry.hpp"
#include "../operators/pet.cuh"
#include <limits>

namespace semgpu2_seismic_p5_cu {
namespace field = semgpu2::common::cu::tenzor;
namespace {
template<int Rank>
__global__ void assemble_q_geometry_kernel(GeometryLaunchData data,
    const Real* coordinates, Index n_count, Real* q_sizes, Real* q_difgeoms) {
    const auto point = locate_element_point(data.element_count);
    const int thread_index = point.element_start + point.local_q;

    __shared__ Real p_coordinates[2][THREADS_MAX];
    p_coordinates[0][thread_index] = 0.0;
    p_coordinates[1][thread_index] = 0.0;
    if (point.active) {
        const Index node = data.p_nodes[point.element * data.q_nodes_per_element + point.local_q];
        p_coordinates[0][thread_index] = coordinates[node];
        p_coordinates[1][thread_index] = coordinates[n_count + node];
    }
    __syncthreads();
    if (!point.active) return;
    Real jacobian[2][Rank]{};
    p5_local_q_grad<2, THREADS_MAX, Rank>(point.local_q, point.element_start, p_coordinates, jacobian);
    const Index qid = point.element * data.q_nodes_per_element + point.local_q;
    const Real weight = p5_q_weight<Rank>(point.local_q);
    if constexpr (Rank == 2) {
        semgpu2::common::cu::reduce_full_rank_geometry(jacobian, q_difgeoms, q_sizes, qid, data.q_count, weight);
    } else {
        semgpu2::common::cu::reduce_embedded_geometry(jacobian, q_difgeoms, q_sizes, qid, data.q_count, weight);
    }
}

__global__ void assemble_nodal_measure_kernel(GeometryLaunchData data,
    const Real* q_sizes, Real* n_sizes) {
    const auto point = locate_element_point(data.element_count);
    if (!point.active) return;
    const Index node = data.p_nodes[point.element * data.q_nodes_per_element + point.local_q];
    atomicAdd(n_sizes + node, q_sizes[point.element * data.q_nodes_per_element + point.local_q]);
}
}  // namespace

cudaError_t assemble_q_geometry(const GeometryLaunchData& data, int rank,
    const Real* coordinates, Index n_count, Real* q_sizes, Real* q_difgeoms, cudaStream_t stream) {
    const dim3 block = element_block(data.q_nodes_per_element);
    const Index blocks = element_block_count(data.element_count, block);
    if (blocks < 0 || blocks > std::numeric_limits<int>::max()) return cudaErrorInvalidConfiguration;
    if (rank != 1 && rank != 2) return cudaErrorInvalidValue;
    if (!blocks) return cudaSuccess;
    if (rank == 2) {
        assemble_q_geometry_kernel<2><<<static_cast<unsigned>(blocks), block, 0, stream>>>(data, coordinates, n_count, q_sizes, q_difgeoms);
    } else {
        assemble_q_geometry_kernel<1><<<static_cast<unsigned>(blocks), block, 0, stream>>>(data, coordinates, n_count, q_sizes, q_difgeoms);
    }
    return cudaGetLastError();
}

cudaError_t assemble_nodal_measure(const GeometryLaunchData& data,
    const Real* q_sizes, Real* n_sizes, cudaStream_t stream) {
    const dim3 block = element_block(data.q_nodes_per_element);
    const Index blocks = element_block_count(data.element_count, block);
    if (blocks < 0 || blocks > std::numeric_limits<int>::max()) return cudaErrorInvalidConfiguration;
    if (!blocks) return cudaSuccess;
    assemble_nodal_measure_kernel<<<static_cast<unsigned>(blocks), block, 0, stream>>>(data, q_sizes, n_sizes);
    return cudaGetLastError();
}
}  // namespace semgpu2_seismic_p5_cu
