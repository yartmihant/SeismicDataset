#pragma once

#include "pet_data.hpp"
#include <cuda_runtime.h>

namespace semgpu2::common::cu {

namespace detail {

template<int Dimension>
__device__ inline double transform_pet_gradient(
    const double* representatives, Index direction_stride,
    const double* symmetries, Index symmetry, int direction) {
    static_assert(Dimension >= 1 && Dimension <= 3);
    const double* transform = symmetries + symmetry * Dimension * Dimension
        + direction * Dimension;
    double result = 0.0;
    for (int source_direction = 0; source_direction < Dimension; ++source_direction) {
        result += transform[source_direction]
            * representatives[source_direction * direction_stride];
    }
    return result;
}

}  // namespace detail

// common/nb/pet_coefficients.py::pet_grad_coeff, with the same summation order.
__device__ inline double pet_grad_coeff(
    const PETView& pets, Code type_id, Index q_idx, Index p_idx, int d_local) {
    const Index offset = pets.pq_grad_offsets[type_id];
    const Index p_nodes = pets.p_node_counts[type_id];
    const Index orbs = pets.orb_counts[type_id];
    if (!pets.respect_orbits && pets.symm_counts[type_id] == 1) {
        return pets.pq_grads[offset + d_local * orbs * p_nodes + q_idx * p_nodes + p_idx];
    }
    const Code q_orb = pets.q_node_orbs[pets.q_node_orb_offsets[type_id] + q_idx];
    const Code q_symm = pets.q_node_rep_symms[pets.q_node_rep_symm_offsets[type_id] + q_idx];
    const Index p_inv = pets.p_symm_inv_nodes[
        pets.p_symm_inv_node_offsets[type_id] + q_symm * p_nodes + p_idx];
    const int dim = pets.dims[type_id];
    const double* representatives = pets.pq_grads + offset + q_orb * p_nodes + p_inv;
    const double* symmetries = pets.symm_grad_ops + pets.symm_grad_op_offsets[type_id];
    const Index direction_stride = orbs * p_nodes;
    // Host validation admits only dimensions 1..3 (1..2 in 2D solvers).
    // Keep the reduction bound visible to nvcc: an arbitrary runtime bound
    // permits large unrolled paths whose registers affect every valid PET.
    if (dim == 1) {
        return detail::transform_pet_gradient<1>(
            representatives, direction_stride, symmetries, q_symm, d_local);
    }
    if (dim == 2) {
        return detail::transform_pet_gradient<2>(
            representatives, direction_stride, symmetries, q_symm, d_local);
    }
    return detail::transform_pet_gradient<3>(
        representatives, direction_stride, symmetries, q_symm, d_local);
}

// Field operators pass Code PET ids. Gram preparation also walks the complete
// catalog with an Index cursor; retain that width without widening field ids.
template<class PETIndex>
__device__ inline double pet_value_coeff(
    const PETView& pets, PETIndex type_id, Index q_idx, Index p_idx) {
    const Index offset = pets.pq_value_offsets[type_id];
    const Index p_nodes = pets.p_node_counts[type_id];
    if (!pets.respect_orbits && pets.symm_counts[type_id] == 1) {
        return pets.pq_values[offset + q_idx * p_nodes + p_idx];
    }
    const Code q_orb = pets.q_node_orbs[pets.q_node_orb_offsets[type_id] + q_idx];
    const Code q_symm = pets.q_node_rep_symms[pets.q_node_rep_symm_offsets[type_id] + q_idx];
    const Index p_inv = pets.p_symm_inv_nodes[
        pets.p_symm_inv_node_offsets[type_id] + q_symm * p_nodes + p_idx];
    return pets.pq_values[offset + q_orb * p_nodes + p_inv];
}

template<class PETIndex>
__device__ inline double pet_weight_coeff(const PETView& pets, PETIndex type_id, Index q_idx) {
    const Code q_orb = pets.q_node_orbs[pets.q_node_orb_offsets[type_id] + q_idx];
    return pets.q_weights[pets.q_weight_offsets[type_id] + q_orb];
}

__device__ inline double pet_q_weight(
    const PETView& pets, Index nu, const Code* element_types, int types_in_element) {
    double weight = 1.0;
    Index q_stride = 1;
    for (int factor = 0; factor < types_in_element; ++factor) {
        const Code type_id = element_types[factor];
        const Index q_nodes = pets.q_node_counts[type_id];
        const Index q_component = (nu / q_stride) % q_nodes;
        weight *= pet_weight_coeff(pets, type_id, q_component);
        q_stride *= q_nodes;
    }
    return weight;
}

}  // namespace semgpu2::common::cu
