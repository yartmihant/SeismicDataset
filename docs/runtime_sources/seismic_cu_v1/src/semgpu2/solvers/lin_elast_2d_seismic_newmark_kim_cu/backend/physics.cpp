#include "physics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace semgpu2_seismic_cu {
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
PreparedField materialize_q_field(const PhysicsData& input, const std::vector<Index>& q_offsets) {
    PreparedField result;
    result.type_code = input.type_code;
    result.space_id = input.space_id;
    result.entity_ids = input.entity_ids;
    result.node_offsets = q_offsets;
    result.time_count = input.time_count;
    result.component_count = input.component_count;
    result.dynamic = input.dynamic;
    result.t_begin = input.t_begin; result.t_end = input.t_end; result.interp_code = input.interp_code;
    if (input.space_id == boundary_space) {
        result.schedule = semgpu2::common::cu::build_q_block_schedule(q_offsets, THREADS_MAX);
        result.has_schedule = true;
    }
    result.values = input.values;
    return result;
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
    if (input.space_id == n_space) {
        for (Index node : input.entity_ids) if (node < 0 || node >= state.n_count) throw std::invalid_argument("N physics entity outside N");
        const Index layer = checked_product(input.component_count, static_cast<Index>(input.entity_ids.size()));
        if (static_cast<Index>(input.values.size()) != checked_product(input.time_count, layer)) throw std::invalid_argument("N physics value shape mismatch");
        PreparedField result;
        result.type_code = input.type_code; result.space_id = n_space; result.entity_ids = input.entity_ids;
        result.node_offsets.resize(input.entity_ids.size() + 1); std::iota(result.node_offsets.begin(), result.node_offsets.end(), Index{0});
        result.values = input.values; result.time_count = input.time_count; result.component_count = input.component_count;
        result.dynamic = input.dynamic; result.t_begin = input.t_begin; result.t_end = input.t_end; result.interp_code = input.interp_code;
        return result;
    }
    const auto& space = space_data(state, input.space_id);
    std::vector<Index> p_offsets(input.entity_ids.size() + 1, 0), q_offsets(input.entity_ids.size() + 1, 0);
    for (std::size_t i = 0; i < input.entity_ids.size(); ++i) {
        const Index entity = input.entity_ids[i];
        if (entity < 0 || entity >= space.element_count()) throw std::invalid_argument("PQ physics entity outside its space");
        const Index p_count = space.p_node_offsets[entity + 1] - space.p_node_offsets[entity];
        const Index q_count = space.q_node_offsets[entity + 1] - space.q_node_offsets[entity];
        if (p_count < 0 || q_count < 0 || p_offsets[i] > std::numeric_limits<Index>::max() - p_count
            || q_offsets[i] > std::numeric_limits<Index>::max() - q_count) throw std::overflow_error("Physics compact domain exceeds int64 capacity");
        p_offsets[i + 1] = p_offsets[i] + p_count;
        q_offsets[i + 1] = q_offsets[i] + q_count;
    }
    const Index layer = checked_product(input.component_count, p_offsets.back());
    if (static_cast<Index>(input.values.size()) != checked_product(input.time_count, layer)) throw std::invalid_argument("P physics value shape mismatch");
    return materialize_q_field(input, q_offsets);
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
            for (Index local = field.node_offsets[i]; local < field.node_offsets[i+1]; ++local) {
                const Index q = state.main.q_node_offsets[e] + (local - field.node_offsets[i]);
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
}  // namespace semgpu2_seismic_cu
