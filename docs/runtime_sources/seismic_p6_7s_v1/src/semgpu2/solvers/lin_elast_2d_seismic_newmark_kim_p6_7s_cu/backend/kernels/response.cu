#include "../stepping.hpp"
#include "../operators/material.cuh"

#include <limits>

namespace SEMGPU2_SEISMIC_NAMESPACE {
namespace {

// Each block processes one shot and a group of elements belonging to one material.
// Geometry uses MAIN addressing. Material and history use the carrier's own Q index.
template<bool WithPml>
__global__ void material_response_kernel(ResponseLaunchData data, const Real* solution,
    const Real* measures, const Real* inverse_jacobians, Real* history, Real dt, Real* response) {
    const ElementPoint point = locate_element_point(data.entity_count);
    const Index local_element = point.element;
    const Index element = (point.active && data.field_entity_ids) ? data.field_entity_ids[local_element] : local_element;
    const Index q = element * data.space.nodes_per_element + point.local_q;
    const Index compact_q = local_element * data.space.nodes_per_element + point.local_q;
    const Index node = point.active ? data.space.p_nodes[q] : 0;
    const Index n = data.n_count, stride = data.field_q_count;
    const int lane = point.element_start + point.local_q;
    const Index shot = blockIdx.z;
    const Real* state = solution + (6 * shot) * n;
    Real* force = response + (2 * shot) * n;
    __shared__ Real displacement[2][THREADS_MAX];
    __shared__ Real reference_flux[2][2][THREADS_MAX];
    Real u[2]{}, velocity[2]{}, mid_gradient[2][2]{};

    for (int c = 0; c < 2; ++c) {
        if (point.active) u[c] = state[c * n + node];
        displacement[c][lane] = u[c];
    }
    if constexpr (WithPml) {
        // Reuse the divergence workspace for midpoint displacements. All readers
        // finish before any lane publishes the reference flux into this workspace.
        for (int c = 0; c < 2; ++c) {
            Real midpoint = 0.0;
            if (point.active) {
                velocity[c] = state[(2 + c) * n + node];
                const Real acceleration = state[(4 + c) * n + node];
                const Real previous = u[c] - dt * velocity[c] + 0.5 * dt * dt * acceleration;
                midpoint = 0.5 * (u[c] + previous);
            }
            reference_flux[0][c][lane] = midpoint;
        }
    }
    // compute_strain_q_2d starts with a block barrier after the two shared loads.
    Real strain[3]{};
    compute_strain_q_2d(point.active, point.local_q, point.element_start, displacement,
        q, inverse_jacobians, data.space.q_count, strain);
    if constexpr (WithPml) {
        if (point.active) {
            Real reference_gradient[2][2]{};
            factor_local_q_grad<2, THREADS_MAX, 2>(point.local_q, point.element_start,
                reference_flux[0], reference_gradient);
            field::ref_to_phys_grad_q(q, inverse_jacobians, data.space.q_count, reference_gradient, mid_gradient);
        }
        __syncthreads();
    }
    Real reference[2][2]{}, inertia[2]{};
    if (point.active) {
        constexpr Index elastic_start = WithPml ? pml::ROW_COUNT : 0;
        const Real* material = data.values + compact_q;
        const Real lambda = material[(elastic_start + 1) * stride];
        const Real mu = material[(elastic_start + 2) * stride];
        const Real pressure = lambda * (strain[0] + strain[1]);
        const Real shear = mu * strain[2];
        Real flux[2][2]{
            {pressure + 2.0 * mu * strain[0], shear},
            {shear, pressure + 2.0 * mu * strain[1]},
        };
        if constexpr (WithPml) {
            const Real dx = material[pml::DX * stride], dy = material[pml::DY * stride];
            const Real ax = material[pml::AX * stride], ay = material[pml::AY * stride];
            const Real bx = material[pml::BX * stride], by = material[pml::BY * stride];
            const Real mass = material[elastic_start * stride] * measures[q];
            Real* memory = history + (4 * shot) * stride + compact_q;
            for (int c = 0; c < 2; ++c) {
                Real& hx = memory[c * stride];
                Real& hy = memory[(2 + c) * stride];
                hx = ax * hx + bx * mid_gradient[c][0];
                hy = ay * hy + by * mid_gradient[c][1];
                flux[0][c] += (dy - dx) * (c == 0 ? lambda + 2.0 * mu : mu) * hx;
                flux[1][c] += (dx - dy) * (c == 1 ? lambda + 2.0 * mu : mu) * hy;
                inertia[c] = mass * ((dx + dy) * velocity[c] + dx * dy * u[c]);
            }
        }
        for (auto& direction : flux)
            for (Real& value : direction) value *= measures[q];
        field::phys_to_ref_div_q(q, inverse_jacobians, data.space.q_count, flux, reference);
    }
    for (int direction = 0; direction < 2; ++direction)
        for (int c = 0; c < 2; ++c) reference_flux[direction][c][lane] = reference[direction][c];
    __syncthreads();
    if (point.active) {
        Real divergence[1][2]{};
        factor_local_p_div<2, THREADS_MAX, 2>(point.local_q, point.element_start, reference_flux, divergence);
        for (int c = 0; c < 2; ++c)
            atomicAdd(force + c * n + node, -divergence[0][c] - inertia[c]);
    }
}

// The preceding material kernels complete in the same stream. Each thread
// owns a different shot's vertical response, even for coincident source nodes.
__global__ void prepared_source_kernel(const Real* sample, Index n, Real* response) {
    const int shot = threadIdx.x;
    response[(2 * shot + 1) * n + source_profile::node(shot)] += *sample;
}

template<bool WithPml>
cudaError_t launch_material(const ResponseLaunchData& data, const Real* solution, const Real* measures,
    const Real* inverse_jacobians, Real* history, Real dt, Real* response, Index shots, cudaStream_t stream) {
    const dim3 block = element_block(data.space.nodes_per_element);
    const Index blocks = element_block_count(data.entity_count, block);
    if (!blocks) return cudaSuccess;
    if (blocks < 0 || blocks > std::numeric_limits<int>::max()) return cudaErrorInvalidConfiguration;
    const dim3 grid(static_cast<unsigned>(blocks), 1, static_cast<unsigned>(shots));
    material_response_kernel<WithPml><<<grid, block, 0, stream>>>(data, solution, measures, inverse_jacobians,
        history, dt, response);
    return cudaGetLastError();
}
}  // namespace

cudaError_t assemble_material_response(const ResponseLaunchData& data, const Real* solution, const Real* measures,
    const Real* inverse_jacobians, Real* response, Index shots, cudaStream_t stream) {
    return launch_material<false>(data, solution, measures, inverse_jacobians, nullptr, 0., response, shots, stream);
}

cudaError_t assemble_pml_response(const ResponseLaunchData& data, const Real* solution, const Real* measures,
    const Real* inverse_jacobians, Real* history, Real dt, Real* response, Index shots, cudaStream_t stream) {
    return launch_material<true>(data, solution, measures, inverse_jacobians, history, dt, response, shots, stream);
}

cudaError_t assemble_prepared_source(const Real* sample, Index n, Real* response, cudaStream_t stream) {
    prepared_source_kernel<<<1, SHOT_COUNT, 0, stream>>>(sample, n, response);
    return cudaGetLastError();
}
}  // namespace SEMGPU2_SEISMIC_NAMESPACE
