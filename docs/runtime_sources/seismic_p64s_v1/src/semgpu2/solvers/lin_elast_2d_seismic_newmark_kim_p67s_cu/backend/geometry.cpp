#include "geometry.hpp"
#include "device_data.hpp"

#include <algorithm>
#include <cmath>

namespace SEMGPU2_SEISMIC_NAMESPACE {
namespace {
void validate_surface(const Real* xy, Index n_count) {
    using namespace source_profile;
    // Abscissae of the seven Legendre-Gauss-Lobatto nodes on [-1, 1].
    constexpr Real gll[PET_N]{-1., -.8302238962785669, -.4688487934707142, 0.,
                               .4688487934707142, .8302238962785669, 1.};
    for (int i = 0; i < surface_nodes; ++i) {
        const int cell = i / (PET_N - 1);
        const Real x = surface_x0 + element_width * (cell + 0.5 * (1. + gll[i % (PET_N - 1)]));
        if (!(std::abs(xy[i] - x) <= coordinate_tolerance &&
              std::abs(xy[n_count + i] - surface_y) <= coordinate_tolerance)) {
            throw std::invalid_argument("Unexpected coordinates of fixed upper-surface node " + std::to_string(i));
        }
    }
}
}  // namespace

void mesh_geometry(CompiledState& state, const Real* coordinates, Index coordinate_count) {
    state.meshed = false;
    try {
        if (coordinate_count != checked_product(2, state.n_count)) throw std::invalid_argument("Coordinates must have shape (2,N)");
        validate_surface(coordinates, state.n_count);
        auto result = cuda_operation([&](CleanupStatus& cleanup) {
            std::array<Field, 2> geometry;
            for (std::size_t id = 0; id < geometry.size(); ++id) {
                geometry[id] = Field(state.geometry[id].rows, state.geometry[id].columns);
            }
            std::copy_n(coordinates, array_size<Real>(coordinate_count), geometry[0].values.begin());
            CudaStream stream(cleanup);
            DeviceBuffer<Real> device_coordinates(cleanup, coordinate_count), device_n_sizes(cleanup, state.n_count);
            device_coordinates.upload(coordinates, stream.get());
            device_n_sizes.clear(stream.get());
            const auto& space = state.main;
            auto& output = geometry[static_cast<std::size_t>(SGSpaceEnum::MAIN)];
            const Index q_count = space.q_count();
            DeviceTopologyData topology(cleanup, space, stream.get());
            DeviceBuffer<Real> sizes(cleanup, q_count), difgeoms(cleanup, checked_product(4, q_count));
            GeometryLaunchData launch{topology.p_nodes.get(), space.nodes_per_element,
                space.element_count(), q_count};
            cuda_check(assemble_q_geometry(launch, device_coordinates.get(), state.n_count,
                sizes.get(), difgeoms.get(), stream.get()), "assemble_q_geometry");
            sizes.download(output.values.data(), stream.get());
            difgeoms.download(output.values.data() + q_count, stream.get());
            cuda_check(assemble_nodal_measure(launch, sizes.get(), device_n_sizes.get(), stream.get()), "assemble_nodal_measure");
            device_n_sizes.download(geometry[0].values.data() + 2 * state.n_count, stream.get());
            stream.synchronize();
            return geometry;
        });
        for (std::size_t id = 0; id < result.size(); ++id) {
            for (Real value : result[id].values) {
                if (!std::isfinite(value)) throw std::invalid_argument("Geometry contains a non-finite value");
            }
            if (id == static_cast<std::size_t>(SGSpaceEnum::MAIN)) {
                for (Index q = 0; q < result[id].columns; ++q) {
                    if (result[id].values[q] <= 0.0) throw std::invalid_argument("Geometry contains a non-positive Q measure");
                }
            }
        }
        // Fixed-size publication preserves pointers held by Python ndarray views.
        for (std::size_t id = 0; id < result.size(); ++id) {
            std::copy(result[id].values.begin(), result[id].values.end(), state.geometry[id].values.begin());
        }
        state.meshed = true;
    } catch (const CudaError& error) {
        throw MeshError(SGMeshErrorCodeEnum::DEVICE_FAILURE, error.what());
    } catch (const std::exception& error) {
        throw MeshError(SGMeshErrorCodeEnum::GEOMETRY_FAILURE, error.what());
    }
}
}  // namespace SEMGPU2_SEISMIC_NAMESPACE
