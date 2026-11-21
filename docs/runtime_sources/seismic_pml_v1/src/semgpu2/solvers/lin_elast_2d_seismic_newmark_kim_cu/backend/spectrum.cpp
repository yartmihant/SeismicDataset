#include "spectrum.hpp"
#include "device_resources.hpp"
#include "stepping.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace semgpu2_seismic_cu {
namespace {
// A D2H target may be a stack scalar.  Keep it alive through a best-effort
// stream fence if the checked synchronization below throws.
struct StreamFence {
    CleanupStatus& cleanup;
    cudaStream_t stream;
    bool armed = true;
    ~StreamFence() { if (armed) cleanup.record(cudaStreamSynchronize(stream)); }
    void dismiss() { armed = false; }
};

Real reduce(CleanupStatus& cleanup, DeviceExecutionContext& run, DeviceBuffer<Real>& partials,
    DeviceBuffer<Real>& result, Index block_count) {
    Real host = 0.0;
    StreamFence fence{cleanup, run.stream()};
    cuda_check(sum_lanczos_partials(partials.get(), block_count, result.get(), run.stream()), "sum_lanczos_partials");
    result.download(&host, run.stream());
    run.synchronize();
    fence.dismiss();
    if (!std::isfinite(host)) throw std::runtime_error("nonfinite global Lanczos reduction");
    return host;
}

Real global_radius(DeviceExecutionContext& run, bool damping) {
    const Index n = run.n_count();
    const Index blocks = std::min<Index>((2 * n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, LANCZOS_MAX_BLOCKS);
    if (!blocks) return 0.0;
    return cuda_operation([&](CleanupStatus& cleanup) -> Real {
        DeviceBuffer<Real> current(cleanup, 2 * n), previous(cleanup, 2 * n), residual(cleanup, 2 * n);
        DeviceBuffer<Real> partials(cleanup, blocks), result(cleanup, 1);
        cuda_check(initialize_lanczos_vectors(current.get(), previous.get(), partials.get(), n, blocks, run.stream()), "initialize_lanczos_vectors");
        const Real norm = std::sqrt(reduce(cleanup, run, partials, result, blocks));
        if (!(norm > 0.0)) throw std::runtime_error("global Lanczos requires nonzero seed");
        cuda_check(advance_lanczos_vectors(previous.get(), current.get(), current.get(), norm, 2 * n, blocks, run.stream()), "normalize Lanczos seed");
        Real previous_beta = 0.0, scale = 0.0;
        std::vector<Real> alpha, beta;
        alpha.reserve(LANCZOS_STEPS); beta.reserve(LANCZOS_STEPS);
        const Code solution_row = damping ? 2 : 0;
        for (Index iteration = 0; iteration < std::min<Index>(LANCZOS_STEPS, 2 * n); ++iteration) {
            cuda_check(load_lanczos_probe(current.get(), run.nodal_mass(), run.nodal_solution(), solution_row, n, blocks, run.stream()), "load_lanczos_probe");
            run.assemble_force_response(0.0, damping ? 3 : 1);
            cuda_check(assemble_lanczos_action(run.nodal_response(), run.nodal_mass(), current.get(), previous.get(), residual.get(), previous_beta, 0, partials.get(), n, blocks, run.stream()), "assemble_lanczos_action");
            const Real a = reduce(cleanup, run, partials, result, blocks);
            alpha.push_back(a);
            cuda_check(subtract_lanczos_projection(residual.get(), current.get(), a, partials.get(), 2 * n, blocks, run.stream()), "subtract_lanczos_projection");
            const Real next_beta = std::sqrt(std::max(0.0, reduce(cleanup, run, partials, result, blocks)));
            scale = std::max({scale, std::abs(a), previous_beta});
            if (next_beta <= LANCZOS_BREAKDOWN_TOL * scale || iteration + 1 == std::min<Index>(LANCZOS_STEPS, 2 * n)) break;
            beta.push_back(next_beta);
            cuda_check(advance_lanczos_vectors(previous.get(), current.get(), residual.get(), next_beta, 2 * n, blocks, run.stream()), "advance_lanczos_vectors");
            previous_beta = next_beta;
        }
        // Symmetric tridiagonal QR is not needed for this small bounded matrix:
        // use Sturm bisection for its largest eigenvalue.
        Real upper = 0.0;
        for (Index i = 0; i < static_cast<Index>(alpha.size()); ++i) {
            Real radius = i ? std::abs(beta[static_cast<std::size_t>(i - 1)]) : 0.0;
            if (i < static_cast<Index>(beta.size())) radius += std::abs(beta[static_cast<std::size_t>(i)]);
            upper = std::max(upper, std::abs(alpha[static_cast<std::size_t>(i)]) + radius);
        }
        // The exact zero operator has zero spectral radius.  The Sturm guard
        // protects only nonzero tridiagonals from a divide-by-zero pivot; it
        // must not manufacture a positive CFL eigenvalue for a zero matrix.
        if (upper == 0.0) return 0.0;
        Real lo = 0.0, hi = upper;
        const Index count = static_cast<Index>(alpha.size());
        for (int step = 0; step < SPECTRUM_BISECTION_STEPS && count; ++step) {
            const Real middle = 0.5 * (lo + hi);
            Index negatives = 0;
            Real p = alpha[0] - middle;
            if (p <= 0.0) ++negatives;
            if (std::abs(p) < SPECTRUM_STURM_TOL) p = p < 0.0 ? -SPECTRUM_STURM_TOL : SPECTRUM_STURM_TOL;
            for (Index i = 1; i < count; ++i) {
                const Real off = beta[static_cast<std::size_t>(i - 1)];
                p = alpha[static_cast<std::size_t>(i)] - middle - off * off / p;
                if (p <= 0.0) ++negatives;
                if (std::abs(p) < SPECTRUM_STURM_TOL) p = p < 0.0 ? -SPECTRUM_STURM_TOL : SPECTRUM_STURM_TOL;
            }
            if (negatives < count) lo = middle; else hi = middle;
        }
        return std::max(0.0, hi);
    });
}
}  // namespace

SpectrumEstimate estimate_spectrum(const CompiledState& state, const PreparedPhysics& physics) {
    if (!state.meshed) throw std::runtime_error("spectrum requires meshed geometry");
    SpectrumEstimate estimate;
    estimate.by_element.assign(array_size<Real>(state.main.element_count()), 0.0);
    if (state.main.element_count() == 0) return estimate;
    return cuda_operation([&](CleanupStatus& cleanup) -> SpectrumEstimate {
        DeviceExecutionContext run(cleanup, state, physics.nodal_mass, physics.q_materials, physics.primary,
            {}, {}, {}, false);
        SpectrumEstimate result;
        result.by_element = estimate.by_element;
        result.stiffness = global_radius(run, false);
        bool has_damping = false;
        for (const auto& [id, field] : physics.primary)
            if (field.type_code / 10000 == 3) has_damping = true;
        result.damping = has_damping ? global_radius(run, true) : 0.0;
        return result;
    });
}
}  // namespace semgpu2_seismic_cu
