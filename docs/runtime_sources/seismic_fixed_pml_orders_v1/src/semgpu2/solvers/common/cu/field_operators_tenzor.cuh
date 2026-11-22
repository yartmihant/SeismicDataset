#pragma once
#include "pet_data_tenzor.hpp"

namespace semgpu2::common::cu::tenzor {
__device__ inline double pet_grad_coeff(const PETView& pet, Index q, Index p) {
    return pet.pq_grads[q * pet.node_count + p];
}

template<int Rank>
__device__ inline double pet_q_weight(const PETView& pet, Index nu) {
    double weight = 1.0;
    for (int direction = 0; direction < Rank; ++direction) {
        weight *= pet.q_weights[nu % pet.node_count];
        nu /= pet.node_count;
    }
    return weight;
}

// These actions accumulate. The caller owns all shared publication/barriers.
// Factor zero varies fastest; differentiation changes only its own factor.
template<int Components, int Threads>
__device__ inline void local_q_value(const PETView&, Index nu, Index slot_start,
    const double (&value)[Components][Threads],
    double (&result)[Components]) {
    for (int c = 0; c < Components; ++c) result[c] += value[c][slot_start + nu];
}

template<int Components, int Threads>
__device__ inline void local_p_value(const PETView& pets, Index nu, Index slot_start,
    const double (&value)[Components][Threads],
    double (&result)[Components]) {
    local_q_value(pets, nu, slot_start, value, result);
}

template<int Components, int Threads, int Rank>
__device__ inline void local_q_grad(const PETView& pets, Index nu, Index slot_start,
    const double (&value)[Components][Threads],
    double (&result)[Components][Rank]) {
    static_assert(Rank >= 1 && Rank <= 3);
    Index stride = 1;
    const Index n = pets.node_count;
    for (int direction = 0; direction < Rank; ++direction) {
        const Index local = (nu / stride) % n;
        const Index base = slot_start + nu - local * stride;
        for (Index other = 0; other < n; ++other) {
            const double coefficient = pet_grad_coeff(pets, local, other);
            for (int c = 0; c < Components; ++c)
                result[c][direction] += coefficient * value[c][base + other * stride];
        }
        stride *= n;
    }
}

template<int Components, int Threads, int Rank>
__device__ inline void local_p_div(const PETView& pets, Index nu, Index slot_start,
    const double (&value)[Rank][Components][Threads],
    double (&result)[1][Components]) {
    static_assert(Rank >= 1 && Rank <= 3);
    Index stride = 1;
    const Index n = pets.node_count;
    for (int direction = 0; direction < Rank; ++direction) {
        const Index local = (nu / stride) % n;
        const Index base = slot_start + nu - local * stride;
        for (Index other = 0; other < n; ++other) {
            const double coefficient = pet_grad_coeff(pets, other, local);
            for (int c = 0; c < Components; ++c)
                result[0][c] += coefficient * value[direction][c][base + other * stride];
        }
        stride *= n;
    }
}

// H[reference, physical, Q] is the prepared full-rank or embedded mapping.
template<int Components, int Rank, int Dimension>
__device__ inline void ref_to_phys_grad_q(Index qid, const double* mapping, Index q_count,
    const double (&value)[Components][Rank], double (&result)[Components][Dimension]) {
    static_assert(1 <= Rank && Rank <= Dimension && Dimension <= 3);
    for (int c = 0; c < Components; ++c)
        for (int physical = 0; physical < Dimension; ++physical)
            for (int reference = 0; reference < Rank; ++reference)
                result[c][physical] += mapping[(reference * Dimension + physical) * q_count + qid] * value[c][reference];
}

template<int Components, int Dimension, int Rank>
__device__ inline void phys_to_ref_div_q(Index qid, const double* mapping, Index q_count,
    const double (&value)[Dimension][Components], double (&result)[Rank][Components]) {
    static_assert(1 <= Rank && Rank <= Dimension && Dimension <= 3);
    for (int reference = 0; reference < Rank; ++reference)
        for (int c = 0; c < Components; ++c) {
            result[reference][c] = 0.0;
            for (int physical = 0; physical < Dimension; ++physical)
                result[reference][c] += mapping[(reference * Dimension + physical) * q_count + qid] * value[physical][c];
        }
}

template<int Components>
__device__ inline void apply_q_measure(Index qid, const double* q_sizes, double (&value)[Components]) {
    for (int c = 0; c < Components; ++c) value[c] *= q_sizes[qid];
}
}  // namespace semgpu2::common::cu::tenzor
