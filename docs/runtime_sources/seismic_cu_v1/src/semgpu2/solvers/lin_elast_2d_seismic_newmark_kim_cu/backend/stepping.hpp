#pragma once

#include "device_data.hpp"
#include "geometry.hpp"
#include "physics.hpp"

#include <map>
#include <memory>
#include <optional>

namespace semgpu2_seismic_cu {

// Device copy of one prepared field.  `evaluated` deliberately owns the
// temporal interpolation result until the stream has consumed its upload.
struct DevicePreparedField {
    const PreparedField* host = nullptr;
    DeviceBuffer<Index> entity_ids, node_offsets;
    DeviceBuffer<Real> values;
    DeviceBuffer<Index> thread_slots, slot_entity_indices, slot_thread_offsets;
    std::vector<Real> evaluated;
    bool has_schedule = false;

    DevicePreparedField(CleanupStatus&, const PreparedField&, cudaStream_t);
    // Interpolation storage may be reused after the upload has completed.
    void upload_at(Real time, cudaStream_t);
};

struct DeviceSpaceView {
    PETView pets;

    const Index* p_nodes;
    const Index* p_node_offsets;
    const Index* q_node_offsets;
    Index q_count;
};

// Owns a complete device execution context.  It borrows compiled/prepared
// host state but never publishes into it; root copies only requested frames.
class DeviceExecutionContext {
    CleanupStatus& cleanup_;
    const CompiledState& state_;
    CudaStream stream_;
    DevicePETData pets_;
    DeviceSpaceData main_;
    DeviceTopologyData boundary_;
    DeviceBuffer<Real> nodal_solution_, nodal_response_, nodal_mass_, q_materials_, q_main_geometry_, q_boundary_geometry_;
    DeviceBuffer<Code> nonfinite_flag_;
    std::map<Index, std::unique_ptr<DevicePreparedField>> primary_;
public:
    DeviceExecutionContext(CleanupStatus&, const CompiledState&, const std::vector<Real>& nodal_mass,
        const std::vector<Real>& q_materials, const std::map<Index, PreparedField>& fields, bool check_finite);
    ~DeviceExecutionContext() noexcept;
    DeviceExecutionContext(const DeviceExecutionContext&) = delete;
    DeviceExecutionContext& operator=(const DeviceExecutionContext&) = delete;

    cudaStream_t stream() const { return stream_.get(); }
    const CompiledState& state() const { return state_; }
    Index n_count() const { return state_.n_count; }
    Real* nodal_solution() const { return nodal_solution_.get(); }
    Real* nodal_response() const { return nodal_response_.get(); }
    Real* nodal_mass() const { return nodal_mass_.get(); }
    const Real* q_main_measures() const { return q_main_geometry_.get(); }
    const Real* q_boundary_measures() const { return q_boundary_geometry_.get(); }
    const Real* q_main_inv_jacobians() const { return q_main_geometry_.get() + state_.main.q_count(); }
    const Real* q_boundary_normals() const { return state_.boundary.q_count() ? q_boundary_geometry_.get() + state_.boundary.q_count() : q_boundary_geometry_.get(); }
    DeviceSpaceView space(SGSpaceEnum) const;
    void synchronize();
    void copy_solution_to_host(Field&) const;

    // Optional leading physics category: 1 selects stiffness, 3 damping; nullopt selects all.
    void assemble_force_response(Real time,
        std::optional<Code> category = std::nullopt);
    bool advance_internal_step(Real time, Real dt, Real beta, Real gamma, bool check_finite);
    bool check_finite_solution();
};

// CUDA launch boundary.  Host code selects a material/condition variant once;
// device lanes execute only that AOT body and one packed-Q schedule.
struct ResponseLaunchData {
    DeviceSpaceView space;
    Index n_count;
    const Index* field_entity_ids;
    const Index* field_node_offsets;
    const Real* values;
    Index field_q_count;
    const Index* thread_slots;
    const Index* slot_entity_indices;
    const Index* slot_thread_offsets;
    int slots_per_block;
    Index block_count;
};
cudaError_t assemble_material_response(const ResponseLaunchData&, const Real*, const Real*, const Real*, Real*, cudaStream_t);
cudaError_t assemble_boundary_damping(const ResponseLaunchData&, const Real*, const Real*, const Real*, Real*, cudaStream_t);
cudaError_t assemble_nodal_force(const DevicePreparedField&, const Real*, Index, Real*, cudaStream_t);
cudaError_t assemble_nodal_damping(const DevicePreparedField&, const Real*, Index, const Real*, const Real*, Real*, cudaStream_t);
cudaError_t predict_nodal_state(Real*, Index, Real, cudaStream_t);
cudaError_t correct_nodal_state(Real*, Real*, const Real*, Index, Real, Real, Real, cudaStream_t);
cudaError_t flag_nonfinite_solution(const Real*, const Real*, Index, Code*, cudaStream_t);
}
