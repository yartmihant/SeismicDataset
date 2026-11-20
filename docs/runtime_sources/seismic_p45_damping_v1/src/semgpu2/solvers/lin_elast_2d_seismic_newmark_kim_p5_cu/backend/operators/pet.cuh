#pragma once

#include "../constants.hpp"
#include "field_operators_tenzor.cuh"
#include "geometry_operators.cuh"

namespace SEMGPU2_SEISMIC_NAMESPACE {

namespace field = semgpu2::common::cu::tenzor;

struct FactorTables {
    Real weights[PET_N];
    Real values[PET_N * PET_N];
    Real gradients[PET_N * PET_N];
};

#include <factor_tables.cuh>

template<int Rank>
__device__ inline Real factor_q_weight(int local_q) {
    Real weight = 1.0;
#pragma unroll
    for (int direction = 0; direction < Rank; ++direction) {
        weight *= factor.weights[local_q % PET_N];
        local_q /= PET_N;
    }
    return weight;
}

template<int Components, int Threads, int Rank>
__device__ inline void factor_local_q_grad(int local_q, int element_start,
    const Real (&value)[Components][Threads], Real (&result)[Components][Rank]) {
    int stride = 1;
#pragma unroll
    for (int direction = 0; direction < Rank; ++direction) {
        const int local = (local_q / stride) % PET_N;
        const int base = element_start + local_q - local * stride;
#pragma unroll
        for (int other = 0; other < PET_N; ++other) {
            const Real coefficient = factor.gradients[local * PET_N + other];
#pragma unroll
            for (int component = 0; component < Components; ++component) {
                result[component][direction] += coefficient * value[component][base + other * stride];
            }
        }
        stride *= PET_N;
    }
}

template<int Components, int Threads, int Rank>
__device__ inline void factor_local_p_div(int local_q, int element_start,
    const Real (&value)[Rank][Components][Threads], Real (&result)[1][Components]) {
    int stride = 1;
#pragma unroll
    for (int direction = 0; direction < Rank; ++direction) {
        const int local = (local_q / stride) % PET_N;
        const int base = element_start + local_q - local * stride;
#pragma unroll
        for (int other = 0; other < PET_N; ++other) {
            const Real coefficient = factor.gradients[other * PET_N + local];
#pragma unroll
            for (int component = 0; component < Components; ++component) {
                result[0][component] += coefficient * value[direction][component][base + other * stride];
            }
        }
        stride *= PET_N;
    }
}

struct ElementPoint {
    bool active;
    Index element;
    int local_q;
    int element_start;
};

inline dim3 element_block(int width) {
    return dim3(width, THREADS_MAX / width);
}

inline Index element_block_count(Index count, dim3 block) {
    return count == 0 ? 0 : (count - 1) / block.y + 1;
}

__device__ inline ElementPoint locate_element_point(Index element_count) {
    const Index element = static_cast<Index>(blockIdx.x) * blockDim.y + threadIdx.y;
    return {element < element_count, element, static_cast<int>(threadIdx.x),
        static_cast<int>(threadIdx.y * blockDim.x)};
}

}  // namespace SEMGPU2_SEISMIC_NAMESPACE
