#include "runtime.hpp"
#include "materials.hpp"
#include "device_resources.hpp"

#include <algorithm>
#include <cmath>

namespace semgpu2_seismic_cu {
namespace {
Real select_step_bound(const SolverConfig& config, const SpectrumEstimate& spectrum) {
    const Real total = config.t_end - config.t_begin;
    const bool empty_time_config = !config.step_dt && !config.steps_count && !config.cfl_safety
        && !config.tact_dt && !config.tacts_count && !config.steps_per_tact
        && config.t_begin == 0.0 && config.t_end == 0.0;
    if (!empty_time_config && (!std::isfinite(config.kim_beta) || !std::isfinite(config.kim_gamma)
        || config.kim_beta < 0.0 || config.kim_gamma <= 0.0))
        throw std::invalid_argument("Invalid Newmark–Kim parameters");
    if (config.step_dt) return *config.step_dt;
    if (config.steps_count) {
        if (*config.steps_count <= 0) throw std::invalid_argument("steps_count must be positive");
        return total / static_cast<Real>(*config.steps_count);
    }
    if (!config.cfl_safety) return 1.0;
    if (!std::isfinite(*config.cfl_safety) || *config.cfl_safety <= 0.0)
        throw std::invalid_argument("Automatic step requires positive finite CFL safety");
    const Real k = spectrum.stiffness, c = spectrum.damping;
    if (!std::isfinite(k) || !std::isfinite(c) || k < 0.0 || c < 0.0 || std::max(k, c) <= 0.0)
        throw std::invalid_argument("Invalid native spectral radius");
    const Real kim = 2.0 / (2.0 * config.kim_gamma + 1.0);
    const Real scaled_c = c / kim;
    const Real bound = 2.0 * *config.cfl_safety * kim / (std::hypot(std::sqrt(k), scaled_c) + scaled_c);
    if (!std::isfinite(bound) || bound <= 0.0) throw std::invalid_argument("Step bound must be positive and finite");
    return bound;
}
}  // namespace

std::shared_ptr<PreparedState> prepare_model(std::shared_ptr<CompiledState> state,
    const std::vector<PhysicsData>& fields, const SolverConfig& config,
    const TimeGridBuilder& time_grid) {
    auto result = std::make_shared<PreparedState>();
    result->structure = std::move(state);
    result->config = config;
    result->generation = result->structure->generation;
    try {
        if (!result->structure->meshed) throw std::invalid_argument("mesh must precede prepare");
        result->physics = prepare_physics(*result->structure, fields);
        result->physics->nodal_mass = assemble_mass(*result->structure, result->physics->q_materials);
        for (Real mass : result->physics->nodal_mass)
            if (!std::isfinite(mass) || mass <= 0) throw std::invalid_argument("Invalid lumped mass");
    } catch (const CudaError& error) {
        throw PrepareError(SGPrepareErrorCodeEnum::DEVICE_FAILURE, error.what());
    } catch (const std::exception& error) {
        throw PrepareError(SGPrepareErrorCodeEnum::PHYSICS_FAILURE, error.what());
    }
    try {
        result->spectrum.by_element.assign(array_size<Real>(result->structure->main.element_count()), 0.0);
        const bool has_active_pml = std::any_of(result->physics->pml_values.begin(), result->physics->pml_values.end(),
            [](Real value) { return value != 0.0; });
        if (has_active_pml && config.cfl_safety.has_value())
            throw std::invalid_argument("PML requires explicit step_dt or steps_count");
        if (config.cfl_safety.has_value()) {
            result->spectrum = estimate_spectrum(*result->structure, *result->physics);
        }
        result->grid = time_grid(select_step_bound(config, result->spectrum));
    } catch (const CudaError& error) {
        throw PrepareError(SGPrepareErrorCodeEnum::DEVICE_FAILURE, error.what());
    } catch (const std::exception& error) {
        throw PrepareError(SGPrepareErrorCodeEnum::SPECTRUM_FAILURE, error.what());
    }
    auto& target = *result->structure;
    // Commit only after physical and spectral operations released resources.
    if (result->physics->nodal_mass.size() != target.materials[0].values.size()
        || result->physics->q_materials.size() != target.materials[1].values.size()) {
        throw PrepareError(SGPrepareErrorCodeEnum::PHYSICS_FAILURE, "Prepared material shape mismatch");
    }
    if (result->physics->pml_values.size() % 2 != 0) {
        throw PrepareError(SGPrepareErrorCodeEnum::PHYSICS_FAILURE, "Prepared PML component shape mismatch");
    }
    // Allocate all replacement runtime state before publishing any canonical
    // material row, so re-prepare remains atomic if allocation fails.
    std::vector<Index> pml_entity_ids = result->physics->pml_entity_ids;
    Field pml_history(4, static_cast<Index>(result->physics->pml_values.size() / 2));
    // Spectral preparation is complete; transfer the temporary assembled
    // fields to their single canonical host owners.  PreparedPhysics keeps
    // no second copy of the canonical material fields between runs.
    target.materials[0].values = std::move(result->physics->nodal_mass);
    target.materials[1].values = std::move(result->physics->q_materials);
    target.pml_entity_ids = std::move(pml_entity_ids);
    target.pml_history = std::move(pml_history);
    return result;
}
}  // namespace semgpu2_seismic_cu
