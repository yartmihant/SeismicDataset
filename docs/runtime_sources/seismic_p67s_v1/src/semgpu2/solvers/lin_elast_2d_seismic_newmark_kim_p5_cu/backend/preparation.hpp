#pragma once
#include "device_data.hpp"

namespace SEMGPU2_SEISMIC_NAMESPACE {
cudaError_t assemble_mass_2d(const Index* nodes, const Real* density, const Real* sizes,
    Index count, Real* mass, cudaStream_t stream);
}
