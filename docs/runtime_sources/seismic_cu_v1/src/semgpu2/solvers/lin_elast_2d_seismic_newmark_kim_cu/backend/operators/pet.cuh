#pragma once

#include "../constants.hpp"
#include "field_operators_tenzor.cuh"
#include "geometry_operators.cuh"
#include "scheduler.cuh"

namespace semgpu2_seismic_cu {

// The module composes common actions; it owns neither PET data nor schedules.
using semgpu2::common::cu::tenzor::PETView;
using semgpu2::common::cu::QSlot;
using semgpu2::common::cu::resolve_q_slot;

}  // namespace semgpu2_seismic_cu
