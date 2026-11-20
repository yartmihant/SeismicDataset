#pragma once

#include "data.hpp"

#include <functional>
#include <map>
#include <memory>

namespace SEMGPU2_SEISMIC_NAMESPACE {

// Binding-owned, shape-normalised input. Values are C-order
// [time, component, entity-local P node] for dynamic fields and
// [component, entity-local P node] for static fields.
struct PhysicsData {
    Index physics_id = 0;
    Code type_code = 0;
    Code space_id = 0;
    std::vector<Index> entity_ids;
    std::vector<Real> values;
    Index time_count = 0;
    Index component_count = 0;
    bool active = true;
    bool dynamic = false;
    Real t_begin = 0.0;
    Real t_end = 0.0;
    Code interp_code = 0;
};

// Prepared values are N fields or compact Q fields. For dynamic fields the
// time layer is the outermost contiguous dimension.
struct PreparedField {
    Code type_code = 0;
    Code space_id = 0;
    std::vector<Index> entity_ids;
    Index points_per_layer = 0;
    std::vector<Real> values;
    Index time_count = 1;
    Index component_count = 0;
    bool dynamic = false;
    Real t_begin = 0.0;
    Real t_end = 0.0;
    Code interp_code = 0;

    Index node_count() const { return points_per_layer; }
    Index layer_size() const { return checked_product(component_count, node_count()); }
    const Real* layer(Index index = 0) const { return values.data() + checked_product(index, layer_size()); }
};

struct PreparedPhysics {
    std::map<Index, PreparedField> primary;
    std::vector<Real> q_materials;
    std::vector<Real> nodal_mass;
};

std::shared_ptr<PreparedPhysics> prepare_physics(
    const CompiledState& state, const std::vector<PhysicsData>& fields);

// CPU temporal evaluation intentionally follows NB's layer-local Lagrange
// interpolation. Returned storage is caller-owned and empty outside support.
std::vector<Real> evaluate_physics_values(const PreparedField& field, Real time);

}  // namespace SEMGPU2_SEISMIC_NAMESPACE
