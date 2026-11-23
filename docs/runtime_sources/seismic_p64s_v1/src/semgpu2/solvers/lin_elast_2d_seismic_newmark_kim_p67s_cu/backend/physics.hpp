#pragma once

#include "data.hpp"

#include <memory>
#include <optional>

namespace SEMGPU2_SEISMIC_NAMESPACE {

// Static MAIN input, with a row per component and columns in element-local P order.
struct PhysicsData {
    Code type_code = 0;
    Code space_id = 0;
    Index component_count = 0;
    std::vector<Index> entity_ids;
    std::vector<Real> values;
};

// Compact MAIN-Q field of PML profiles and, after time selection, recurrence coefficients.
struct PreparedField {
    Code type_code = static_cast<Code>(SGSecondaryPhysicsTypeEnum::PML);
    Code space_id = static_cast<Code>(SGSpaceEnum::MAIN);
    std::vector<Index> entity_ids;
    Index points_per_layer = 0;
    Index component_count = 0;
    std::vector<Real> values;

    Index node_count() const { return points_per_layer; }
    Index layer_size() const { return checked_product(component_count, node_count()); }
};

struct PreparedPhysics {
    std::vector<Real> q_materials;
    std::vector<Real> nodal_mass;
    std::optional<PreparedField> pml;
};

std::shared_ptr<PreparedPhysics> prepare_physics(
    const CompiledState& state, std::vector<PhysicsData> fields);

void prepare_pml_recurrence(PreparedField& field, Real dt);

}  // namespace SEMGPU2_SEISMIC_NAMESPACE
