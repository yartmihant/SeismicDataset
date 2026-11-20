#pragma once

#include "pet_coefficients.cuh"

namespace semgpu2::common::cu {

struct PQLocal {
    bool has_p;
    Index p_local;
    Index q_local;
};

__device__ inline PQLocal resolve_pq_local(
    const PETView& pets, Index nu, const Code* types, int type_count) {
    PQLocal result{true, 0, 0};
    Index p_stride = 1;
    Index q_stride = 1;
    for (int factor = 0; factor < type_count; ++factor) {
        const Index p_nodes = pets.p_node_counts[types[factor]];
        const Index q_nodes = pets.q_node_counts[types[factor]];
        const Index pq_nu = nu % q_nodes;
        if (pq_nu >= p_nodes) {
            result.has_p = false;
        }
        result.p_local += pq_nu * p_stride;
        result.q_local += pq_nu * q_stride;
        p_stride *= p_nodes;
        q_stride *= q_nodes;
        nu /= q_nodes;
    }
    return result;
}

namespace detail {
template<int Components, int Threads>
__device__ inline void apply_q_value_1d(
    const PETView& pets, Code type, Index q, Index p_count,
    const double (&value)[Components][Threads], Index base, Index stride,
    double (&result)[Components]) {
    for (Index p = 0; p < p_count; ++p) {
        const double coefficient = pet_value_coeff(pets, type, q, p);
        for (int component = 0; component < Components; ++component) {
            result[component] += coefficient * value[component][base + p * stride];
        }
    }
}

template<int Components>
__device__ inline void apply_q_grad_term(
    const PETView& pets, Code type, Index q, Index p, int dim,
    const double (&sample)[Components], double (&result)[Components][2],
    int dim_offset, double scale) {
    for (int direction = 0; direction < dim; ++direction) {
        const double coefficient = pet_grad_coeff(pets, type, q, p, direction);
        for (int component = 0; component < Components; ++component) {
            result[component][dim_offset + direction] += scale * coefficient * sample[component];
        }
    }
}

template<int Components, int Threads>
__device__ inline void apply_p_div_term(
    const PETView& pets, Code type, Index q, Index p, int dim,
    const double (&value)[2][Components][Threads], Index q_slot,
    double (&result)[1][Components], int dim_offset, double scale) {
    for (int direction = 0; direction < dim; ++direction) {
        const double coefficient = pet_grad_coeff(pets, type, q, p, direction);
        for (int component = 0; component < Components; ++component) {
            result[0][component] += scale * coefficient * value[dim_offset + direction][component][q_slot];
        }
    }
}
}  // namespace detail

// The four reference actions accumulate into caller-initialized results, as
// in NB. Slot storage uses Q strides even for P fields (P may be smaller).
// Templates fix component/storage dimensions at AOT build, never PET orders.
template<int Components, int Threads>
__device__ inline void local_q_value(
    const PETView& pets, Index nu, Index slot_start,
    const double (&value)[Components][Threads], const Code* types, int type_count,
    double (&result)[Components]) {
    const Code type_0 = types[0];
    const Index q_count_0 = pets.q_node_counts[type_0];
    if (type_count == 1) {
        detail::apply_q_value_1d(pets, type_0, nu % q_count_0,
            pets.p_node_counts[type_0], value, slot_start, 1, result);
        return;
    }
    const Code type_1 = types[1];
    const Index q_0 = nu % q_count_0;
    const Index q_1 = nu / q_count_0;
    if (q_1 >= pets.q_node_counts[type_1]) {
        return;
    }
    for (Index p_0 = 0; p_0 < pets.p_node_counts[type_0]; ++p_0) {
        const double coefficient_0 = pet_value_coeff(pets, type_0, q_0, p_0);
        for (Index p_1 = 0; p_1 < pets.p_node_counts[type_1]; ++p_1) {
            const double coefficient_1 = pet_value_coeff(pets, type_1, q_1, p_1);
            const Index local_slot = slot_start + p_0 + p_1 * q_count_0;
            for (int component = 0; component < Components; ++component) {
                result[component] += coefficient_0 * coefficient_1 * value[component][local_slot];
            }
        }
    }
}

template<int Components, int Threads>
__device__ inline void local_p_value(
    const PETView& pets, Index nu, Index slot_start,
    const double (&value)[Components][Threads], const Code* types, int type_count,
    double (&result)[Components]) {
    const Code type_0 = types[0];
    const Index q_count_0 = pets.q_node_counts[type_0];
    const Index p_0 = nu % q_count_0;
    if (type_count == 1) {
        if (p_0 < pets.p_node_counts[type_0]) {
            for (Index q = 0; q < q_count_0; ++q) {
                const double coefficient = pet_value_coeff(pets, type_0, q, p_0);
                for (int component = 0; component < Components; ++component) {
                    result[component] += coefficient * value[component][slot_start + q];
                }
            }
        }
        return;
    }
    const Code type_1 = types[1];
    const Index p_1 = nu / q_count_0;
    if (p_0 >= pets.p_node_counts[type_0] || p_1 >= pets.p_node_counts[type_1]) {
        return;
    }
    for (Index q_0 = 0; q_0 < q_count_0; ++q_0) {
        const double coefficient_0 = pet_value_coeff(pets, type_0, q_0, p_0);
        for (Index q_1 = 0; q_1 < pets.q_node_counts[type_1]; ++q_1) {
            const double coefficient_1 = pet_value_coeff(pets, type_1, q_1, p_1);
            const Index local_slot = slot_start + q_0 + q_1 * q_count_0;
            for (int component = 0; component < Components; ++component) {
                result[component] += coefficient_0 * coefficient_1 * value[component][local_slot];
            }
        }
    }
}

template<int Components, int Threads>
__device__ inline void local_q_grad(
    const PETView& pets, Index nu, Index slot_start,
    const double (&value)[Components][Threads], const Code* types, int type_count,
    double (&result)[Components][2]) {
    const Code type_0 = types[0];
    const Index q_count_0 = pets.q_node_counts[type_0];
    double sample[Components];
    if (type_count == 1) {
        for (Index p = 0; p < pets.p_node_counts[type_0]; ++p) {
            for (int component = 0; component < Components; ++component) {
                sample[component] = value[component][slot_start + p];
            }
            detail::apply_q_grad_term(pets, type_0, nu % q_count_0, p,
                pets.dims[type_0], sample, result, 0, 1.0);
        }
        return;
    }
    const Code type_1 = types[1];
    const Index q_0 = nu % q_count_0;
    const Index q_1 = nu / q_count_0;
    if (q_1 >= pets.q_node_counts[type_1]) {
        return;
    }
    for (Index p_0 = 0; p_0 < pets.p_node_counts[type_0]; ++p_0) {
        for (int component = 0; component < Components; ++component) {
            sample[component] = 0.0;
        }
        detail::apply_q_value_1d(pets, type_1, q_1, pets.p_node_counts[type_1],
            value, slot_start + p_0, q_count_0, sample);
        detail::apply_q_grad_term(pets, type_0, q_0, p_0,
            pets.dims[type_0], sample, result, 0, 1.0);
    }
    for (Index p_1 = 0; p_1 < pets.p_node_counts[type_1]; ++p_1) {
        for (int component = 0; component < Components; ++component) {
            sample[component] = 0.0;
        }
        detail::apply_q_value_1d(pets, type_0, q_0, pets.p_node_counts[type_0],
            value, slot_start + p_1 * q_count_0, 1, sample);
        detail::apply_q_grad_term(pets, type_1, q_1, p_1,
            pets.dims[type_1], sample, result, pets.dims[type_0], 1.0);
    }
}

template<int Components, int Threads>
__device__ inline void local_p_div(
    const PETView& pets, Index nu, Index slot_start,
    const double (&value)[2][Components][Threads], const Code* types, int type_count,
    double (&result)[1][Components]) {
    const Code type_0 = types[0];
    const Index q_count_0 = pets.q_node_counts[type_0];
    const Index p_0 = nu % q_count_0;
    if (type_count == 1) {
        if (p_0 >= pets.p_node_counts[type_0]) {
            return;
        }
        for (Index q = 0; q < q_count_0; ++q) {
            detail::apply_p_div_term(pets, type_0, q, p_0, pets.dims[type_0],
                value, slot_start + q, result, 0, 1.0);
        }
        return;
    }
    const Code type_1 = types[1];
    const Index p_1 = nu / q_count_0;
    if (p_0 >= pets.p_node_counts[type_0] || p_1 >= pets.p_node_counts[type_1]) {
        return;
    }
    for (Index q_0 = 0; q_0 < q_count_0; ++q_0) {
        const double coefficient_0 = pet_value_coeff(pets, type_0, q_0, p_0);
        for (Index q_1 = 0; q_1 < pets.q_node_counts[type_1]; ++q_1) {
            const double coefficient_1 = pet_value_coeff(pets, type_1, q_1, p_1);
            const Index slot = slot_start + q_0 + q_1 * q_count_0;
            detail::apply_p_div_term(pets, type_0, q_0, p_0, pets.dims[type_0],
                value, slot, result, 0, coefficient_1);
            detail::apply_p_div_term(pets, type_1, q_1, p_1, pets.dims[type_1],
                value, slot, result, pets.dims[type_0], coefficient_0);
        }
    }
}

template<int Components>
__device__ inline void ref_to_phys_grad_q(
    Index qid, const double* q_difgeoms, Index q_count,
    const double (&value)[Components][2], double (&result)[Components][2]) {
    for (int component = 0; component < Components; ++component) {
        result[component][0] += value[component][0] * q_difgeoms[qid]
            + value[component][1] * q_difgeoms[2 * q_count + qid];
        result[component][1] += value[component][0] * q_difgeoms[q_count + qid]
            + value[component][1] * q_difgeoms[3 * q_count + qid];
    }
}

template<int Components>
__device__ inline void phys_to_ref_div_q(
    Index qid, const double* q_difgeoms, Index q_count,
    const double (&value)[2][Components], double (&result)[2][Components]) {
    for (int component = 0; component < Components; ++component) {
        result[0][component] = q_difgeoms[qid] * value[0][component]
            + q_difgeoms[q_count + qid] * value[1][component];
        result[1][component] = q_difgeoms[2 * q_count + qid] * value[0][component]
            + q_difgeoms[3 * q_count + qid] * value[1][component];
    }
}

template<int Components>
__device__ inline void apply_q_measure(
    Index qid, const double* q_sizes, double (&value)[Components]) {
    const double measure = q_sizes[qid];
    for (int component = 0; component < Components; ++component) {
        value[component] = measure * value[component];
    }
}
}  // namespace semgpu2::common::cu
