#pragma once
#include <cstdint>

namespace semgpu2::common::cu::tenzor {
using Index = std::int64_t;
using Code = std::int32_t;

// Q-ordered factor tables; no values, orbits, symmetry or projection storage.
struct PETView {
    Index node_count;
    const double* pq_grads;
    const double* q_weights;
};
}  // namespace semgpu2::common::cu::tenzor
