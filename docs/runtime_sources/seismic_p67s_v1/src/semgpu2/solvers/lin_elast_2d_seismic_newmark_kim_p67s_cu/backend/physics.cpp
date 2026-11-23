#include "physics.hpp"

#include <algorithm>
#include <cmath>
#include <map>

namespace SEMGPU2_SEISMIC_NAMESPACE {
namespace {
constexpr Code main_space = static_cast<Code>(SGSpaceEnum::MAIN);
constexpr Code P(SGPhysicsTypeEnum code) { return static_cast<Code>(code); }

Index components_for(Code code) {
    switch (static_cast<SGPhysicsTypeEnum>(code)) {
    case SGPhysicsTypeEnum::MATERIAL_ISO_PLANE_STRAIN: return 3;
    case SGPhysicsTypeEnum::PML: return 2;
    default: throw std::invalid_argument("Unsupported seismic physics type");
    }
}

Index validate_field(const CompiledState& state, const PhysicsData& field) {
    if (field.space_id != main_space || field.component_count != components_for(field.type_code))
        throw std::invalid_argument("Invalid MAIN physics carrier or component count");
    for (Index id : field.entity_ids) {
        if (id < 0 || id >= state.main.element_count()) throw std::invalid_argument("Physics entity outside MAIN");
    }
    const Index count = checked_product(static_cast<Index>(field.entity_ids.size()), state.main.nodes_per_element);
    if (static_cast<Index>(field.values.size()) != checked_product(count, field.component_count))
        throw std::invalid_argument("Physics value shape mismatch");
    for (Real value : field.values) {
        if (!std::isfinite(value)) throw std::invalid_argument("Physics values must be finite");
    }
    return count;
}
}  // namespace

std::shared_ptr<PreparedPhysics> prepare_physics(const CompiledState& state,
    std::vector<PhysicsData> fields) {
    auto physics = std::make_shared<PreparedPhysics>();
    std::vector<bool> covered(array_size<Index>(state.main.element_count()), false);
    const Index count = state.main.q_count();
    const Index main_width = state.main.nodes_per_element;
    physics->q_materials.resize(array_size<Real>(checked_product(3, count)));
    std::map<Index, std::vector<Real>> pml_by_element;
    for (const auto& field : fields) {
        const Index field_points = validate_field(state, field);
        if (field.type_code == P(SGPhysicsTypeEnum::PML)) {
            for (std::size_t i = 0; i < field.entity_ids.size(); ++i) {
                const Index element = field.entity_ids[i];
                const auto entry = pml_by_element.try_emplace(element,
                    std::vector<Real>(array_size<Real>(checked_product(2, main_width)), 0.0)).first;
                for (Index point = 0; point < main_width; ++point) {
                    const Index local = static_cast<Index>(i) * main_width + point;
                    const Real dx = field.values[local];
                    const Real dy = field.values[field_points + local];
                    if (dx < 0.0 || dy < 0.0)
                        throw std::invalid_argument("PML stretch coefficients must be nonnegative");
                    entry->second[point] += dx;
                    entry->second[main_width + point] += dy;
                    if (!std::isfinite(entry->second[point])
                        || !std::isfinite(entry->second[main_width + point])) {
                        throw std::invalid_argument("Summed PML stretch coefficients must be finite");
                    }
                }
            }
            continue;
        }
        for (std::size_t i = 0; i < field.entity_ids.size(); ++i) {
            const Index e = field.entity_ids[i];
            if (covered[e]) throw std::invalid_argument("Overlapping MAIN material domains");
            covered[e] = true;
            const int width = state.main.nodes_per_element;
            for (int point = 0; point < width; ++point) {
                const Index local = static_cast<Index>(i) * width + point;
                const Index q = e * width + point;
                const Real density = field.values[local];
                const Real young = field.values[field_points + local];
                const Real poisson = field.values[2 * field_points + local];
                const Real lambda = young * poisson / ((1 + poisson) * (1 - 2 * poisson));
                const Real mu = young / (2 * (1 + poisson));
                if (!(density > 0) || !std::isfinite(lambda) || !std::isfinite(mu))
                    throw std::invalid_argument("Invalid density or Lame parameters");
                physics->q_materials[q] = density;
                physics->q_materials[count + q] = lambda;
                physics->q_materials[2 * count + q] = mu;
            }
        }
    }
    if (std::find(covered.begin(), covered.end(), false) != covered.end() || covered.empty())
        throw std::invalid_argument("MAIN requires complete material coverage");
    if (!pml_by_element.empty()) {
        auto& pml_field = physics->pml.emplace();
        pml_field.component_count = 2;
        pml_field.points_per_layer = checked_product(static_cast<Index>(pml_by_element.size()), main_width);
        pml_field.entity_ids.reserve(pml_by_element.size());
        pml_field.values.resize(array_size<Real>(pml_field.layer_size()));
        for (const auto& [element, values] : pml_by_element) {
            const Index start = static_cast<Index>(pml_field.entity_ids.size()) * main_width;
            pml_field.entity_ids.push_back(element);
            for (Index component = 0; component < 2; ++component)
                std::copy_n(values.data() + component * main_width, main_width,
                    pml_field.values.data() + component * pml_field.node_count() + start);
        }
    }
    return physics;
}

void prepare_pml_recurrence(PreparedField& field, Real dt) {
    const Index count = field.node_count();
    field.component_count = pml::ROW_COUNT;
    field.values.resize(array_size<Real>(field.layer_size()));
    for (Index direction = 0; direction < 2; ++direction) {
        const Real* stretch = field.values.data() + (pml::DX + direction) * count;
        Real* decay = field.values.data() + (pml::AX + direction) * count;
        Real* weight = field.values.data() + (pml::BX + direction) * count;
        for (Index q = 0; q < count; ++q) {
            const Real z = -stretch[q] * dt;
            decay[q] = std::exp(z);
            weight[q] = decay[q] == 1.0 ? dt : -std::expm1(z) / stretch[q];
        }
    }
}

}  // namespace SEMGPU2_SEISMIC_NAMESPACE
