#pragma once

#include "../constants.hpp"
#include "field_operators_tenzor.cuh"
#include "geometry_operators.cuh"

namespace semgpu2_seismic_p5_cu {

namespace field = semgpu2::common::cu::tenzor;

// One constant object per CUDA translation unit, in the local ALine node order.
struct FactorTables {
    Real weights[PET_N];
    Real values[PET_N * PET_N];
    Real gradients[PET_N * PET_N];
};

// Collocation applies V by indexing. Keep its DTO bits alongside W and G.
static __constant__ const FactorTables factor = {
    {
        0x1.1111111111114p-4, 0x1.838ef05e9dfc4p-2, 0x1.1c1665ae8edfep-1,
        0x1.1c1665ae8edfap-1, 0x1.838ef05e9dfc4p-2, 0x1.111111111110cp-4,
    },
    {
        0x1.0000000000000p+0, 0x1.0000000000000p-53, 0x1.0000000000000p-54, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0,
        0x1.05b06c8674e0ap-58, 0x1.fffffffffffffp-1, 0x1.07b7b4f385d12p-54, -0x1.30d530b6df3d8p-54, 0x1.09a73de6e7158p-54, -0x1.779891cd752fbp-55,
        0x1.f008d8c9aa926p-55, 0x1.335cd42424964p-53, 0x1.fffffffffffffp-1, -0x1.eda2763473da0p-53, 0x1.655fc412db632p-53, 0x1.417057a455606p-55,
        -0x1.8046c64d54930p-58, 0x1.0ca32bdbdb69cp-53, 0x1.ac68880a307d8p-55, 0x1.ffffffffffffdp-1, 0x1.aa03bed249ce0p-57, 0x1.fd1f50b7553f4p-56,
        0x1.efa4f93798b1fp-54, 0x1.c11384471b522p-55, 0x1.7c2425863d177p-53, -0x1.33cab3d24830ap-52, 0x1.0000000000000p+0, -0x1.8219db8ca2b41p-53,
        -0x1.0000000000000p-54, 0x1.0000000000000p-54, 0x1.0000000000000p-54, 0x0.0p+0, 0x0.0p+0, 0x1.fffffffffffffp-1,
    },
    {
        -0x1.e000000000001p+2, 0x1.44867ab6b409cp+3, -0x1.0250e4699f8cep+2, 0x1.1f51d398dffcbp+1, -0x1.5993eb40e2140p+0, 0x1.0000000000000p-1,
        -0x1.c94f3697d64f1p+0, 0x1.4ff6e83595343p-50, 0x1.42ffa60d7480cp+1, -0x1.271fbf06d1d93p+0, 0x1.4e9dc765962cdp-1, -0x1.e6f9d1785f7eep-3,
        0x1.f09701eac49e8p-2, -0x1.b8a44bb06b20ap+0, -0x1.23dad7fcfe7dep-54, 0x1.c0c21d8cae8ccp+0, -0x1.929d577c61635p-1, 0x1.142c659cf077ap-2,
        -0x1.142c659cf0775p-2, 0x1.929d577c61632p-1, -0x1.c0c21d8cae8cap+0, -0x1.2396ab1964df6p-53, 0x1.b8a44bb06b20cp+0, -0x1.f09701eac49efp-2,
        0x1.e6f9d1785f7f4p-3, -0x1.4e9dc765962c8p-1, 0x1.271fbf06d1d91p+0, -0x1.42ffa60d7480cp+1, -0x1.14aa5734baa49p-50, 0x1.c94f3697d64f1p+0,
        -0x1.000000000000cp-1, 0x1.5993eb40e2140p+0, -0x1.1f51d398dffccp+1, 0x1.0250e4699f8d0p+2, -0x1.44867ab6b409ep+3, 0x1.e000000000002p+2,
    },
};

template<int Rank>
__device__ inline Real p5_q_weight(int local_q) {
    Real weight = 1.0;
#pragma unroll
    for (int direction = 0; direction < Rank; ++direction) {
        weight *= factor.weights[local_q % PET_N];
        local_q /= PET_N;
    }
    return weight;
}

template<int Components, int Threads, int Rank>
__device__ inline void p5_local_q_grad(int local_q, int element_start,
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
__device__ inline void p5_local_p_div(int local_q, int element_start,
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

// x traverses one element; y selects one of the complete elements in the block.
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

}  // namespace semgpu2_seismic_p5_cu
