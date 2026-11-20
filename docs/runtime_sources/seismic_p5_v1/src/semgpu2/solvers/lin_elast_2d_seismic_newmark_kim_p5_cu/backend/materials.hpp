#pragma once
#include "data.hpp"

namespace semgpu2_seismic_p5_cu {
std::vector<Real> assemble_mass(const CompiledState&, const std::vector<Real>&);
}
