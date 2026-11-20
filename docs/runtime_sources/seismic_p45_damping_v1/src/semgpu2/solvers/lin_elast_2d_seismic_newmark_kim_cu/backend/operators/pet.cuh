#pragma once

#include "../constants.hpp"
#include "field_operators_tenzor.cuh"
#include "geometry_operators.cuh"

namespace semgpu2_seismic_cu {

// The module composes common actions; it owns neither PET data nor topology.
using semgpu2::common::cu::tenzor::PETView;

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

}  // namespace semgpu2_seismic_cu
