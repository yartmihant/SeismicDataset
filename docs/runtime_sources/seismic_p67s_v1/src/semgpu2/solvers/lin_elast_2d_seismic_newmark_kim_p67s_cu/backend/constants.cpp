#include "constants.hpp"
#include <initializer_list>
#include <string>
#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace SEMGPU2_SEISMIC_NAMESPACE {
namespace {
template<class Enum>
void bind_shot_components(py::native_enum<Enum>& catalog, std::initializer_list<const char*> names,
                          int stride, int first) {
    for (int shot = 0; shot < SHOT_COUNT; ++shot) {
        int code = shot * stride + first;
        for (const char* name : names) {
            std::string label(name);
            if (shot) label += "_S" + std::to_string(shot + 1);
            catalog.value(label.c_str(), static_cast<Enum>(code++));
        }
    }
    catalog.finalize();
}
}
void bind_constants(py::module_& module) {
    module.attr("SHOT_COUNT") = SHOT_COUNT;
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
        .finalize();
    py::native_enum<SGPhysicsTypeEnum>(module, "SGPhysicsTypeEnum", "enum.IntEnum")
        .value("MATERIAL_ISO_PLANE_STRAIN", SGPhysicsTypeEnum::MATERIAL_ISO_PLANE_STRAIN)
        .value("PML", SGPhysicsTypeEnum::PML)
        .finalize();
    auto velocity = py::native_enum<SGReceiverComponentEnum>(module, "SGReceiverComponentEnum", "enum.IntEnum");
    bind_shot_components(velocity, {"VX", "VY"}, 6, 4);
    auto solution = py::native_enum<SGNSolutionComponentEnum>(module, "SGNSolutionComponentEnum", "enum.IntEnum");
    bind_shot_components(solution, {"UX", "UY", "VX", "VY", "AX", "AY"}, 6, 0);
    py::native_enum<SGNGeometryComponentEnum>(module, "SGNGeometryComponentEnum", "enum.IntEnum")
        .value("X", SGNGeometryComponentEnum::X)
        .value("Y", SGNGeometryComponentEnum::Y)
        .value("SIZE", SGNGeometryComponentEnum::SIZE)
        .finalize();
    py::native_enum<SGNMaterialsComponentEnum>(module, "SGNMaterialsComponentEnum", "enum.IntEnum")
        .value("MASS", SGNMaterialsComponentEnum::MASS)
        .finalize();
    auto response = py::native_enum<SGNResponseComponentEnum>(module, "SGNResponseComponentEnum", "enum.IntEnum");
    bind_shot_components(response, {"FORCE_X", "FORCE_Y"}, 2, 0);
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
}  // namespace SEMGPU2_SEISMIC_NAMESPACE
