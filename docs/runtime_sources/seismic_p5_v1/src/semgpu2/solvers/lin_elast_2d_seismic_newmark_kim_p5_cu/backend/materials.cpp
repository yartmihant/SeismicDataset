#include "materials.hpp"
#include "preparation.hpp"

namespace semgpu2_seismic_p5_cu {
std::vector<Real> assemble_mass(const CompiledState& state, const std::vector<Real>& materials) {
    const Index count = state.main.q_count();
    std::vector<Real> result(array_size<Real>(state.n_count));
    cuda_operation([&](CleanupStatus& cleanup) {
        CudaStream stream(cleanup);
        DeviceBuffer<Index> nodes(cleanup, count);
        DeviceBuffer<Real> density(cleanup, count), sizes(cleanup, count), mass(cleanup, state.n_count);
        nodes.upload(state.main.p_nodes.data(), stream.get());
        density.upload(materials.data(), stream.get());
        sizes.upload(state.geometry[1].values.data(), stream.get());
        mass.clear(stream.get());
        cuda_check(assemble_mass_2d(nodes.get(), density.get(), sizes.get(), count, mass.get(), stream.get()),
            "assemble lumped mass");
        mass.download(result.data(), stream.get());
        stream.synchronize();
        return 0;
    });
    return result;
}
}
