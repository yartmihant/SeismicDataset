#pragma once
#include "pet_coefficients.cuh"

namespace semgpu2::common::cu {
struct PQLocal { bool has_p; Index p_local; Index q_local; };

__device__ inline PQLocal resolve_pq_local(const PETView& pets, Index nu, const Code* ids, int count) {
    PQLocal result{true, 0, nu};
    Index stride = 1;
    for (int factor = 0; factor < count; ++factor) {
        const Code pet = ids[factor];
        const Index p = pets.p_node_counts[pet], q = pets.q_node_counts[pet];
        const Index coordinate = nu % q;
        result.has_p = result.has_p && coordinate < p;
        result.p_local += coordinate * stride;
        stride *= p;
        nu /= q;
    }
    result.has_p = result.has_p && nu == 0;
    return result;
}

__device__ inline Index packed_p_address(const PETView& pets, Index p, const Code* ids, int count) {
    Index address = 0, stride = 1;
    for (int factor = 0; factor < count; ++factor) {
        const Code pet = ids[factor];
        address += (p % pets.p_node_counts[pet]) * stride;
        p /= pets.p_node_counts[pet];
        stride *= pets.q_node_counts[pet];
    }
    return address;
}

__device__ inline double product_coefficient(const PETView& pets, Index q, Index p,
    const Code* ids, int count, int direction) {
    double coefficient = 1.0;
    int offset = 0;
    for (int factor = 0; factor < count; ++factor) {
        const Code pet = ids[factor];
        const Index qi = q % pets.q_node_counts[pet], pi = p % pets.p_node_counts[pet];
        coefficient *= direction >= offset && direction < offset + pets.dims[pet]
            ? pet_grad_coeff(pets, pet, qi, pi, direction - offset) : pet_value_coeff(pets, pet, qi, pi);
        offset += pets.dims[pet];
        q /= pets.q_node_counts[pet];
        p /= pets.p_node_counts[pet];
    }
    return coefficient;
}

template<int Components, int Threads>
__device__ inline void local_q_value(const PETView& pets, Index nu, Index start,
    const double (&value)[Components][Threads], const Code* ids, int count, double (&result)[Components]) {
    Index total = 1;
    for (int f = 0; f < count; ++f) total *= pets.p_node_counts[ids[f]];
    for (Index p = 0; p < total; ++p) {
        const double coefficient = product_coefficient(pets, nu, p, ids, count, -1);
        const Index slot = start + packed_p_address(pets, p, ids, count);
        for (int c = 0; c < Components; ++c) result[c] += coefficient * value[c][slot];
    }
}

template<int Components, int Threads>
__device__ inline void local_p_value(const PETView& pets, Index nu, Index start,
    const double (&value)[Components][Threads], const Code* ids, int count, double (&result)[Components]) {
    const auto local = resolve_pq_local(pets, nu, ids, count);
    if (!local.has_p) return;
    Index total = 1;
    for (int f = 0; f < count; ++f) total *= pets.q_node_counts[ids[f]];
    for (Index q = 0; q < total; ++q) {
        const double coefficient = product_coefficient(pets, q, local.p_local, ids, count, -1);
        for (int c = 0; c < Components; ++c) result[c] += coefficient * value[c][start + q];
    }
}

template<int Components, int Threads, int Rank>
__device__ inline void local_q_grad(const PETView& pets, Index nu, Index start,
    const double (&value)[Components][Threads], const Code* ids, int count, double (&result)[Components][Rank]) {
    Index total = 1;
    int rank = 0;
    for (int f = 0; f < count; ++f) { total *= pets.p_node_counts[ids[f]]; rank += pets.dims[ids[f]]; }
    for (Index p = 0; p < total; ++p) {
        const Index slot = start + packed_p_address(pets, p, ids, count);
        for (int r = 0; r < rank; ++r) {
            const double coefficient = product_coefficient(pets, nu, p, ids, count, r);
            for (int c = 0; c < Components; ++c) result[c][r] += coefficient * value[c][slot];
        }
    }
}

template<int Components, int Threads, int Rank>
__device__ inline void local_p_div(const PETView& pets, Index nu, Index start,
    const double (&value)[Rank][Components][Threads], const Code* ids, int count, double (&result)[1][Components]) {
    const auto local = resolve_pq_local(pets, nu, ids, count);
    if (!local.has_p) return;
    Index total = 1;
    int rank = 0;
    for (int f = 0; f < count; ++f) { total *= pets.q_node_counts[ids[f]]; rank += pets.dims[ids[f]]; }
    for (Index q = 0; q < total; ++q) for (int r = 0; r < rank; ++r) {
        const double coefficient = product_coefficient(pets, q, local.p_local, ids, count, r);
        for (int c = 0; c < Components; ++c) result[0][c] += coefficient * value[r][c][start + q];
    }
}

template<int Components, int Rank, int Dimension>
__device__ inline void ref_to_phys_grad_q(Index qid, const double* mapping, Index q_count,
    const double (&value)[Components][Rank], double (&result)[Components][Dimension]) {
    for (int c = 0; c < Components; ++c) for (int d = 0; d < Dimension; ++d)
        for (int r = 0; r < Rank; ++r) result[c][d] += value[c][r] * mapping[(r * Dimension + d) * q_count + qid];
}

template<int Components, int Dimension, int Rank>
__device__ inline void phys_to_ref_div_q(Index qid, const double* mapping, Index q_count,
    const double (&value)[Dimension][Components], double (&result)[Rank][Components]) {
    for (int r = 0; r < Rank; ++r) for (int c = 0; c < Components; ++c) {
        result[r][c] = 0.0;
        for (int d = 0; d < Dimension; ++d) result[r][c] += mapping[(r * Dimension + d) * q_count + qid] * value[d][c];
    }
}

template<int Components>
__device__ inline void apply_q_measure(Index qid, const double* sizes, double (&value)[Components]) {
    for (int c = 0; c < Components; ++c) value[c] *= sizes[qid];
}
} // namespace semgpu2::common::cu
