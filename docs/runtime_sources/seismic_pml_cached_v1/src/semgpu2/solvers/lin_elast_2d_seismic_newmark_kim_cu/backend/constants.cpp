#include "constants.hpp"
#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace semgpu2_seismic_cu {
void bind_constants(py::module_& module) {
    module.attr("SOLVER_ID") = SOLVER_ID;
    module.attr("THREADS_MIN") = THREADS_MIN;
    module.attr("THREADS_MAX") = THREADS_MAX;
    module.attr("THREADS_PER_BLOCK") = THREADS_PER_BLOCK;
    module.attr("LANCZOS_STEPS") = LANCZOS_STEPS;
    module.attr("LANCZOS_MAX_BLOCKS") = LANCZOS_MAX_BLOCKS;
    module.attr("SPECTRUM_BISECTION_STEPS") = SPECTRUM_BISECTION_STEPS;
    module.attr("LANCZOS_BREAKDOWN_TOL") = LANCZOS_BREAKDOWN_TOL;
    module.attr("SPECTRUM_STURM_TOL") = SPECTRUM_STURM_TOL;
    module.attr("DEFAULT_T_BEGIN") = DEFAULT_T_BEGIN;
    module.attr("DEFAULT_T_END") = DEFAULT_T_END;
    module.attr("DEFAULT_KIM_BETA") = DEFAULT_KIM_BETA;
    module.attr("DEFAULT_KIM_GAMMA") = DEFAULT_KIM_GAMMA;
    py::native_enum<SGSpaceEnum>(module, "SGSpaceEnum", "enum.IntEnum")
        .value("N", SGSpaceEnum::N)
        .value("MAIN", SGSpaceEnum::MAIN)
        .value("BOUNDARY", SGSpaceEnum::BOUNDARY)
        .finalize();
    py::native_enum<SGPhysicsTypeEnum>(module, "SGPhysicsTypeEnum", "enum.IntEnum")
        .value("MATERIAL_ISO_PLANE_STRAIN", SGPhysicsTypeEnum::MATERIAL_ISO_PLANE_STRAIN)
        .value("N_FORCE_DYNAMIC", SGPhysicsTypeEnum::N_FORCE_DYNAMIC)
        .value("N_MASS_DAMPING_STATIC", SGPhysicsTypeEnum::N_MASS_DAMPING_STATIC)
        .value("PML", SGPhysicsTypeEnum::PML)
        .value("ABC", SGPhysicsTypeEnum::ABC)
        .finalize();
    py::native_enum<SGReceiverComponentEnum>(module, "SGReceiverComponentEnum", "enum.IntEnum")
        .value("VX", SGReceiverComponentEnum::VX)
        .value("VY", SGReceiverComponentEnum::VY)
        .finalize();
    py::native_enum<SGNSolutionComponentEnum>(module, "SGNSolutionComponentEnum", "enum.IntEnum")
        .value("UX", SGNSolutionComponentEnum::UX)
        .value("UY", SGNSolutionComponentEnum::UY)
        .value("VX", SGNSolutionComponentEnum::VX)
        .value("VY", SGNSolutionComponentEnum::VY)
        .value("AX", SGNSolutionComponentEnum::AX)
        .value("AY", SGNSolutionComponentEnum::AY)
        .finalize();
    py::native_enum<SGNGeometryComponentEnum>(module, "SGNGeometryComponentEnum", "enum.IntEnum")
        .value("X", SGNGeometryComponentEnum::X)
        .value("Y", SGNGeometryComponentEnum::Y)
        .value("SIZE", SGNGeometryComponentEnum::SIZE)
        .finalize();
    py::native_enum<SGNMaterialsComponentEnum>(module, "SGNMaterialsComponentEnum", "enum.IntEnum")
        .value("MASS", SGNMaterialsComponentEnum::MASS)
        .finalize();
    py::native_enum<SGNResponseComponentEnum>(module, "SGNResponseComponentEnum", "enum.IntEnum")
        .value("FORCE_X", SGNResponseComponentEnum::FORCE_X)
        .value("FORCE_Y", SGNResponseComponentEnum::FORCE_Y)
        .finalize();
    py::native_enum<SGQMainGeometryComponentEnum>(module, "SGQMainGeometryComponentEnum", "enum.IntEnum")
        .value("SIZE", SGQMainGeometryComponentEnum::SIZE)
        .value("DIFGEOMETRY_XX", SGQMainGeometryComponentEnum::DIFGEOMETRY_XX)
        .value("DIFGEOMETRY_XY", SGQMainGeometryComponentEnum::DIFGEOMETRY_XY)
        .value("DIFGEOMETRY_YX", SGQMainGeometryComponentEnum::DIFGEOMETRY_YX)
        .value("DIFGEOMETRY_YY", SGQMainGeometryComponentEnum::DIFGEOMETRY_YY)
        .finalize();
    py::native_enum<SGQMainMaterialsComponentEnum>(module, "SGQMainMaterialsComponentEnum", "enum.IntEnum")
        .value("DENSITY", SGQMainMaterialsComponentEnum::DENSITY)
        .value("LAMBDA", SGQMainMaterialsComponentEnum::LAMBDA)
        .value("MU", SGQMainMaterialsComponentEnum::MU)
        .finalize();
    py::native_enum<SGQBoundaryGeometryComponentEnum>(module, "SGQBoundaryGeometryComponentEnum", "enum.IntEnum")
        .value("SIZE", SGQBoundaryGeometryComponentEnum::SIZE)
        .value("NORMAL_X", SGQBoundaryGeometryComponentEnum::NORMAL_X)
        .value("NORMAL_Y", SGQBoundaryGeometryComponentEnum::NORMAL_Y)
        .finalize();
    py::native_enum<SGCompileErrorCodeEnum>(module, "SGCompileErrorCodeEnum", "enum.IntEnum")
        .value("STRUCTURE_FAILURE", SGCompileErrorCodeEnum::STRUCTURE_FAILURE)
        .value("DEVICE_FAILURE", SGCompileErrorCodeEnum::DEVICE_FAILURE)
        .finalize();
    py::native_enum<SGMeshErrorCodeEnum>(module, "SGMeshErrorCodeEnum", "enum.IntEnum")
        .value("GEOMETRY_FAILURE", SGMeshErrorCodeEnum::GEOMETRY_FAILURE)
        .value("DEVICE_FAILURE", SGMeshErrorCodeEnum::DEVICE_FAILURE)
        .finalize();
    py::native_enum<SGPrepareErrorCodeEnum>(module, "SGPrepareErrorCodeEnum", "enum.IntEnum")
        .value("PHYSICS_FAILURE", SGPrepareErrorCodeEnum::PHYSICS_FAILURE)
        .value("SPECTRUM_FAILURE", SGPrepareErrorCodeEnum::SPECTRUM_FAILURE)
        .value("DEVICE_FAILURE", SGPrepareErrorCodeEnum::DEVICE_FAILURE)
        .finalize();
    py::native_enum<SGRunCodeEnum>(module, "SGRunCodeEnum", "enum.IntEnum")
        .value("TIME_END", SGRunCodeEnum::TIME_END)
        .value("UNDERCONFIGURED", SGRunCodeEnum::UNDERCONFIGURED)
        .value("NUMERICAL_DIVERGENCE", SGRunCodeEnum::NUMERICAL_DIVERGENCE)
        .value("DEVICE_FAILURE", SGRunCodeEnum::DEVICE_FAILURE)
        .value("HANDLER_FAILURE", SGRunCodeEnum::HANDLER_FAILURE)
        .value("STEP_LIMIT", SGRunCodeEnum::STEP_LIMIT)
        .finalize();
}
}  // namespace semgpu2_seismic_cu
