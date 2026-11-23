#pragma once

#include "data.hpp"
#include "physics.hpp"

#include <vector>
#include <cuda_runtime_api.h>

namespace SEMGPU2_SEISMIC_NAMESPACE {

// The estimate is deliberately separate from the time-grid policy.  Runtime
// applies the same Kim/CFL formula as NB after it has these operator bounds.
struct SpectrumEstimate {
    std::vector<Real> by_element;
    Real stiffness = 0.0;
    Real damping = 0.0;
};

// Global K/C Lanczos on the assembled nodal mass.
SpectrumEstimate estimate_spectrum(
    const CompiledState& state, const PreparedPhysics& physics);

// Global recurrence vectors are intentionally flat [2,N] storage.  The
// physical K/C action remains in stepping, which owns the run device state.
cudaError_t initialize_lanczos_vectors(
    Real* current, Real* previous, Real* partials, Index n_count, Index blocks, cudaStream_t stream);
cudaError_t load_lanczos_probe(
    const Real* current, const Real* mass, Real* solution, Code solution_row,
    Index n_count, Index blocks, cudaStream_t stream);
cudaError_t assemble_lanczos_action(
    const Real* response, const Real* mass, const Real* current, const Real* previous,
    Real* residual, Real previous_beta, Code response_row, Real* partials,
    Index n_count, Index blocks, cudaStream_t stream);
cudaError_t subtract_lanczos_projection(
    Real* residual, const Real* current, Real alpha, Real* partials,
    Index scalar_count, Index blocks, cudaStream_t stream);
cudaError_t advance_lanczos_vectors(
    Real* previous, Real* current, const Real* residual, Real beta,
    Index scalar_count, Index blocks, cudaStream_t stream);
cudaError_t sum_lanczos_partials(
    const Real* partials, Index partial_count, Real* result, cudaStream_t stream);

}  // namespace SEMGPU2_SEISMIC_NAMESPACE
