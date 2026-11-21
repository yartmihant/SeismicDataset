#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace semgpu2::common::cu {

using Index = std::int64_t;
using Code = std::int32_t;

// Non-owning view of the SGPETData SoA arrays. Every pointer has the same
// address space; a device view is assembled after operation-owned uploads.
// No process-global catalog and no expanded tensor-product coefficients.
struct PETView {
    const Index* p_node_counts;
    const Index* q_node_counts;
    const Code* dims;
    const Index* orb_counts;
    const Index* symm_counts;
    const double* pq_values;
    const Index* pq_value_offsets;
    const double* pq_grads;
    const Index* pq_grad_offsets;
    const double* q_weights;
    const Index* q_weight_offsets;
    const Code* q_node_orbs;
    const Index* q_node_orb_offsets;
    const Code* q_node_rep_symms;
    const Index* q_node_rep_symm_offsets;
    const double* symm_grad_ops;
    const Index* symm_grad_op_offsets;
    const Index* p_symm_inv_nodes;
    const Index* p_symm_inv_node_offsets;
    bool respect_orbits = false;
};

// Offsets have pet_count + 1 entries. The caller owns all device arrays,
// validates sizes, and handles the returned launch error at the compile stage.
cudaError_t assemble_pet_gram(
    PETView pets, double* grams, const Index* offsets, Index pet_count,
    Index gram_entries, int threads_per_block, cudaStream_t stream);

}  // namespace semgpu2::common::cu
