#include "physics.hpp"

#include <algorithm>
#include <cmath>

namespace semgpu2_seismic_p5_cu {
namespace {
constexpr Code n_space = static_cast<Code>(SGSpaceEnum::N);
constexpr Code main_space = static_cast<Code>(SGSpaceEnum::MAIN);
constexpr Code boundary_space = static_cast<Code>(SGSpaceEnum::BOUNDARY);
constexpr Code P(SGPhysicsTypeEnum code) { return static_cast<Code>(code); }

Index components_for(Code code) {
    switch (static_cast<SGPhysicsTypeEnum>(code)) {
    case SGPhysicsTypeEnum::MATERIAL_ISO_PLANE_STRAIN: return 3;
    case SGPhysicsTypeEnum::N_FORCE_DYNAMIC:
    case SGPhysicsTypeEnum::ABC: return 2;
    case SGPhysicsTypeEnum::N_MASS_DAMPING_STATIC: return 1;
    default: throw std::invalid_argument("Unsupported seismic physics type");
    }
}

const TenzorSpaceData& space_data(const CompiledState& state, Code space) {
    if (space == main_space) return state.main;
    if (space == boundary_space) return state.boundary;
    throw std::invalid_argument("Physics has an unsupported carrier space");
}
void finite(const std::vector<Real>& values, const char* message) {
    for (Real value : values) if (!std::isfinite(value)) throw std::invalid_argument(message);
}
PreparedField prepare_field(const CompiledState& state, const PhysicsData& input) {
    if (input.space_id != (input.type_code / 1000) % 10
        || input.dynamic != (input.type_code == P(SGPhysicsTypeEnum::N_FORCE_DYNAMIC)) || input.component_count != components_for(input.type_code)) {
        throw std::invalid_argument("Invalid physics code, carrier, time rank, or component count");
    }
    if (input.time_count < 1 || (!input.dynamic && input.time_count != 1)) throw std::invalid_argument("Invalid physics time layer count");
    if (input.dynamic && (!std::isfinite(input.t_begin) || !std::isfinite(input.t_end)
        || input.interp_code < 0 || input.t_end < input.t_begin
        || (input.time_count > 1 && input.t_end <= input.t_begin))) {
        throw std::invalid_argument("Invalid dynamic physics time grid");
    }
    finite(input.values, "Physics values must be finite");
    Index width = 1;
    Index entity_limit = state.n_count;
    if (input.space_id != n_space) {
        const auto& space = space_data(state, input.space_id);
        width = space.nodes_per_element;
        entity_limit = space.element_count();
    }
    for (Index id : input.entity_ids) {
        if (id < 0 || id >= entity_limit) throw std::invalid_argument("Physics entity outside its carrier");
    }
    PreparedField result;
    result.type_code = input.type_code;
    result.space_id = input.space_id;
    result.points_per_layer = checked_product(static_cast<Index>(input.entity_ids.size()), width);
    result.time_count = input.time_count;
    result.component_count = input.component_count;
    if (static_cast<Index>(input.values.size()) != checked_product(input.time_count, result.layer_size()))
        throw std::invalid_argument("Physics value shape mismatch");
    result.entity_ids = input.entity_ids;
    result.values = input.values;
    result.dynamic = input.dynamic;
    result.t_begin = input.t_begin;
    result.t_end = input.t_end;
    result.interp_code = input.interp_code;
    return result;
}

}  // namespace

std::vector<Real> evaluate_physics_values(const PreparedField& field, Real time) {
    if (!field.dynamic) return field.values;
    std::vector<Real> result(array_size<Real>(field.layer_size()), 0.0);
    if (time < field.t_begin || time > field.t_end) return result;
    if (field.time_count == 1) {
        std::copy_n(field.values.begin(), result.size(), result.begin());
        return result;
    }
    const Index degree = std::min<Index>(field.interp_code, field.time_count - 1);
    const Real coordinate = (time - field.t_begin) * static_cast<Real>(field.time_count - 1)
        / (field.t_end - field.t_begin);
    if (degree == 0) {
        const Index index = std::min<Index>(
            static_cast<Index>(std::floor(coordinate + 0.5)), field.time_count - 1);
        std::copy_n(field.layer(index), result.size(), result.begin());
        return result;
    }
    const Index start = std::min<Index>(
        std::max<Index>(static_cast<Index>(std::floor(coordinate)) - degree / 2, 0),
        field.time_count - degree - 1);
    for (Index i = 0; i <= degree; ++i) {
        Real weight = 1.0;
        for (Index j = 0; j <= degree; ++j) {
            if (i != j) {
                weight *= (coordinate - static_cast<Real>(start + j)) / static_cast<Real>(i - j);
            }
        }
        const Real* source = field.layer(start + i);
        for (Index k = 0; k < field.layer_size(); ++k) {
            result[k] += weight * source[k];
        }
    }
    return result;
}

std::shared_ptr<PreparedPhysics> prepare_physics(const CompiledState& state,
    const std::vector<PhysicsData>& fields) {
    auto physics = std::make_shared<PreparedPhysics>();
    std::vector<bool> covered(array_size<Index>(state.main.element_count()), false);
    const Index count = state.main.q_count();
    physics->q_materials.resize(array_size<Real>(checked_product(3, count)));
    for (const auto& input : fields) {
        if (!input.active) continue;
        auto field = prepare_field(state, input);
        if (field.type_code != P(SGPhysicsTypeEnum::MATERIAL_ISO_PLANE_STRAIN)) {
            if (!field.entity_ids.empty()) physics->primary.emplace(input.physics_id, std::move(field));
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
                const Real young = field.values[field.node_count() + local];
                const Real poisson = field.values[2 * field.node_count() + local];
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
    return physics;
}
}  // namespace semgpu2_seismic_p5_cu
