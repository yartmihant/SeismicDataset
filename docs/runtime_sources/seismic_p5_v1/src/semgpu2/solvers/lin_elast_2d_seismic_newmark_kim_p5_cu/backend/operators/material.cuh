#pragma once

#include "pet.cuh"

namespace semgpu2_seismic_p5_cu {
namespace field = semgpu2::common::cu::tenzor;

// Collective B_e action; caller fills shared P storage before entry.
template<int Threads>
__device__ inline void compute_strain_q_2d(
    bool active, int local_q, int element_start,
    const Real (&p_field)[2][Threads], Index qid, const Real* q_difgeoms, Index q_count, Real (&epsilon)[3]) {
    __syncthreads();
    epsilon[0] = epsilon[1] = epsilon[2] = 0.0;
    if (active) {
        Real grad_ref[2][2]{};
        Real grad_u[2][2]{};
        p5_local_q_grad<2, Threads, 2>(local_q, element_start, p_field, grad_ref);
        field::ref_to_phys_grad_q(qid, q_difgeoms, q_count, grad_ref, grad_u);
        epsilon[0] = grad_u[0][0];
        epsilon[1] = grad_u[1][1];
        epsilon[2] = grad_u[0][1] + grad_u[1][0];
    }
}

// Positive B_e^T W_Q stress; preserves unweighted input stress. No N assembly.
template<int Threads>
__device__ inline void integrate_stress_p_2d(
    bool active, int local_q, int element_start,
    const Real (&stress)[3], Index qid, const Real* q_sizes, const Real* q_difgeoms, Index q_count,
    Real (&workspace)[2][2][Threads], Real (&result)[1][2]) {
    result[0][0] = result[0][1] = 0.0;
    const int tid = element_start + local_q;
    if (active) {
        Real weighted[3]{stress[0], stress[1], stress[2]};
        field::apply_q_measure(qid, q_sizes, weighted);
        Real physical[2][2]{{weighted[0], weighted[2]}, {weighted[2], weighted[1]}};
        Real reference[2][2];
        field::phys_to_ref_div_q(qid, q_difgeoms, q_count, physical, reference);
        for (int ref = 0; ref < 2; ++ref) {
            for (int component = 0; component < 2; ++component) {
                workspace[ref][component][tid] = reference[ref][component];
            }
        }
    } else {
        for (int ref = 0; ref < 2; ++ref) {
            for (int component = 0; component < 2; ++component) {
                workspace[ref][component][tid] = 0.0;
            }
        }
    }
    __syncthreads();
    if (active) {
        p5_local_p_div<2, Threads, 2>(local_q, element_start, workspace, result);
    }
}

}  // namespace semgpu2_seismic_p5_cu
