#pragma once

#include "device_data.hpp"
#include "geometry.hpp"
#include "physics.hpp"

#include <optional>
#include <vector>

namespace SEMGPU2_SEISMIC_NAMESPACE {

struct DeviceSpaceView {
    const Index* p_nodes;
    Index nodes_per_element;
    Index q_count;
};

// Owns a complete device execution context.  It borrows compiled/prepared
// host state but never publishes into it; root copies only requested frames.
class DeviceExecutionContext {
    CleanupStatus& cleanup_;
    const CompiledState& state_;
    Index shot_count_;
    CudaStream stream_;
    DeviceTopologyData main_;
    const Index pml_q_count_, ordinary_entity_count_;
    DeviceBuffer<Real> nodal_solution_, nodal_response_, nodal_mass_, ordinary_materials_, q_main_geometry_;
    DeviceBuffer<Real> pml_history_;
    DeviceBuffer<Real> pml_materials_;
    DeviceBuffer<Index> ordinary_entity_ids_;
    DeviceBuffer<Index> pml_entity_ids_;
    DeviceBuffer<Code> nonfinite_flag_;
    DeviceBuffer<Real> source_samples_;
public:
    DeviceExecutionContext(CleanupStatus&, const CompiledState&, const std::vector<Real>& nodal_mass,
        const std::vector<Real>& q_materials, const std::vector<Real>& source_samples,
        const std::optional<PreparedField>& pml_field, const std::vector<Real>& pml_history, bool check_finite,
        Index shot_count = SHOT_COUNT);
    ~DeviceExecutionContext() noexcept;
    DeviceExecutionContext(const DeviceExecutionContext&) = delete;
    DeviceExecutionContext& operator=(const DeviceExecutionContext&) = delete;

    cudaStream_t stream() const { return stream_.get(); }
    Index n_count() const { return state_.n_count; }
    Index shot_count() const { return shot_count_; }
    Real* nodal_solution() const { return nodal_solution_.get(); }
    Real* nodal_response() const { return nodal_response_.get(); }
    Real* nodal_mass() const { return nodal_mass_.get(); }
    const Real* q_main_measures() const { return q_main_geometry_.get(); }
    const Real* q_main_inv_jacobians() const { return q_main_geometry_.get() + state_.main.q_count(); }
    DeviceSpaceView main_space() const { return {main_.p_nodes.get(), state_.main.nodes_per_element, state_.main.q_count()}; }
    void synchronize();
    void copy_solution_to_host(Field&) const;
    void copy_pml_history_to_host(std::vector<Real>&) const;

    void assemble_stiffness_probe();
    void assemble_step_response(Index step_index, Real pml_dt);
    bool advance_internal_step(Index step_index, Real dt, Real beta, Real gamma, bool check_finite);
    bool check_finite_solution();
};

// CUDA launch boundary.  Each block densely packs whole elements of the
// single-PET space.
struct ResponseLaunchData {
    DeviceSpaceView space;
    Index n_count;
    const Index* field_entity_ids;
    const Real* values;
    Index field_q_count;
    Index entity_count;
};
cudaError_t assemble_material_response(const ResponseLaunchData&, const Real*, const Real*, const Real*, Real*, Index, cudaStream_t);
cudaError_t assemble_pml_response(const ResponseLaunchData&, const Real*, const Real*, const Real*, Real*, Real, Real*, Index, cudaStream_t);
cudaError_t assemble_prepared_source(const Real* sample, Index n, Real* response, cudaStream_t stream);
cudaError_t predict_nodal_state(Real*, Index, Real, Index, cudaStream_t);
cudaError_t correct_nodal_state(Real*, Real*, const Real*, Index, Real, Real, Real, Index, cudaStream_t);
cudaError_t flag_nonfinite_solution(const Real*, const Real*, Index, Index, Code*, cudaStream_t);
cudaError_t flag_nonfinite_values(const Real*, Index, Code*, cudaStream_t);
}
