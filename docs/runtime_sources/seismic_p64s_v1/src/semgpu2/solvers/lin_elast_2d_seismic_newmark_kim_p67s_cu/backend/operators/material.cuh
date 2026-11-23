#pragma once

#include "pet.cuh"

namespace SEMGPU2_SEISMIC_NAMESPACE {
namespace field = semgpu2::common::cu::tenzor;

// Collective B_e action; caller fills shared P storage before entry.
template<int Threads>
__device__ inline void compute_strain_q_2d(
    bool active, int local_q, int element_start,
    const Real (&p_field)[2][Threads], Index qid, const Real* q_difgeoms, Index q_count, Real (&epsilon)[3], const Real* gradients) {
    __syncthreads();
    epsilon[0] = epsilon[1] = epsilon[2] = 0.0;
    if (active) {
        Real grad_ref[2][2]{};
        Real grad_u[2][2]{};
        factor_local_q_grad<2, Threads, 2>(local_q, element_start, p_field, grad_ref, gradients);
        field::ref_to_phys_grad_q(qid, q_difgeoms, q_count, grad_ref, grad_u);
        epsilon[0] = grad_u[0][0];
        epsilon[1] = grad_u[1][1];
        epsilon[2] = grad_u[0][1] + grad_u[1][0];
    }
}

}  // namespace SEMGPU2_SEISMIC_NAMESPACE
