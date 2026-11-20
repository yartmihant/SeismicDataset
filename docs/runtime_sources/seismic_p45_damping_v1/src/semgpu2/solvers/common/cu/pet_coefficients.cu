#include "pet_coefficients.cuh"

#include <limits>

namespace semgpu2::common::cu {
namespace {
__global__ void assemble_pet_gram_kernel(
    PETView pets, double* grams, const Index* offsets, Index pet_count) {
    const Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    Index pet_id = 0;
    while (pet_id < pet_count && index >= offsets[pet_id + 1]) {
        ++pet_id;
    }
    if (pet_id >= pet_count || index < offsets[pet_id]) {
        return;
    }
    const Index p_count = pets.p_node_counts[pet_id];
    const Index local = index - offsets[pet_id];
    const Index p_row = local / p_count;
    const Index p_col = local % p_count;
    double total = 0.0;
    for (Index q_local = 0; q_local < pets.q_node_counts[pet_id]; ++q_local) {
        total += pet_weight_coeff(pets, pet_id, q_local)
            * pet_value_coeff(pets, pet_id, q_local, p_row)
            * pet_value_coeff(pets, pet_id, q_local, p_col);
    }
    grams[index] = total;
}
}  // namespace

cudaError_t assemble_pet_gram(
    PETView pets, double* grams, const Index* offsets, Index pet_count,
    Index gram_entries, int threads_per_block, cudaStream_t stream) {
    if (pet_count < 0 || gram_entries < 0 || threads_per_block <= 0
        || threads_per_block > 1024) {
        return cudaErrorInvalidValue;
    }
    if (gram_entries == 0) {
        return cudaSuccess;
    }
    if (pet_count == 0) {
        return cudaErrorInvalidValue;
    }
    const Index blocks = (gram_entries - 1) / threads_per_block + 1;
    if (blocks > std::numeric_limits<int>::max()) {
        return cudaErrorInvalidConfiguration;
    }
    assemble_pet_gram_kernel<<<static_cast<unsigned>(blocks), threads_per_block, 0, stream>>>(
        pets, grams, offsets, pet_count);
    return cudaGetLastError();
}
}  // namespace semgpu2::common::cu
