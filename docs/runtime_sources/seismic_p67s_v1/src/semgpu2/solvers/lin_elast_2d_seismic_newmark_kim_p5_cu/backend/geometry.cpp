#include "geometry.hpp"
#include "device_data.hpp"

#include <algorithm>
#include <cmath>

namespace SEMGPU2_SEISMIC_NAMESPACE {
void mesh_geometry(CompiledState& state, const Real* coordinates, Index coordinate_count) {
    state.meshed = false;
    try {
        if (coordinate_count != checked_product(2, state.n_count)) throw std::invalid_argument("Coordinates must have shape (2,N)");
        auto result = cuda_operation([&](CleanupStatus& cleanup) {
            std::array<Field, 3> geometry;
            for (std::size_t id = 0; id < geometry.size(); ++id) {
                geometry[id] = Field(state.geometry[id].rows, state.geometry[id].columns);
            }
            // Host download targets outlive the stream even on exceptional exits.
            std::vector<Real> boundary_difgeoms(array_size<Real>(checked_product(4, state.boundary.q_count())));
            std::copy_n(coordinates, array_size<Real>(coordinate_count), geometry[0].values.begin());
            CudaStream stream(cleanup);
            DeviceBuffer<Real> device_coordinates(cleanup, coordinate_count), device_n_sizes(cleanup, state.n_count);
            device_coordinates.upload(coordinates, stream.get());
            device_n_sizes.clear(stream.get());
            for (SGSpaceEnum id : {SGSpaceEnum::MAIN, SGSpaceEnum::BOUNDARY}) {
                const bool is_main = id == SGSpaceEnum::MAIN;
                const auto& space = is_main ? state.main : state.boundary;
                auto& output = geometry[static_cast<std::size_t>(id)];
                const Index q_count = space.q_count();
                const int q_nodes_per_element = space.nodes_per_element;
                DeviceTopologyData topology(cleanup, space, stream.get());
                DeviceBuffer<Real> sizes(cleanup, q_count), difgeoms(cleanup, checked_product(4, q_count));
                GeometryLaunchData launch{ topology.p_nodes.get(),
                    q_nodes_per_element, space.element_count(), q_count};
                cuda_check(assemble_q_geometry(launch, space.rank, device_coordinates.get(), state.n_count,
                    sizes.get(), difgeoms.get(), stream.get()), "assemble_q_geometry");
                sizes.download(output.values.data(), stream.get());
                if (is_main) {
                    difgeoms.download(output.values.data() + q_count, stream.get());
                } else {
                    difgeoms.download(boundary_difgeoms.data(), stream.get());
                }
                if (is_main) cuda_check(assemble_nodal_measure(launch, sizes.get(), device_n_sizes.get(), stream.get()), "assemble_nodal_measure");
                stream.synchronize();
                if (!is_main && q_count) {
                    std::copy_n(boundary_difgeoms.begin() + 2 * q_count,
                        array_size<Real>(2 * q_count), output.values.begin() + q_count);
                }
            }
            device_n_sizes.download(geometry[0].values.data() + 2 * state.n_count, stream.get());
            stream.synchronize();
            return geometry;
        });
        for (std::size_t id = 0; id < result.size(); ++id) {
            for (Real value : result[id].values) {
                if (!std::isfinite(value)) throw std::invalid_argument("Geometry contains a non-finite value");
            }
            if (id != static_cast<std::size_t>(SGSpaceEnum::N)) {
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
