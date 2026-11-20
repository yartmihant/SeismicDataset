#pragma once

#include <cstdint>

// Single source of numeric solver codes and AOT execution capacities.
#define SEMGPU2_CU_THREADS_MIN 32
#define SEMGPU2_CU_THREADS_MAX 512
#define SEMGPU2_CU_THREADS_PER_BLOCK 128
#define SEMGPU2_CU_LANCZOS_STEPS 16
#define SEMGPU2_CU_LANCZOS_MAX_BLOCKS 1024
#define SEMGPU2_CU_SPECTRUM_BISECTION_STEPS 32
#define SEMGPU2_CU_LANCZOS_BREAKDOWN_TOL 1.0e-12
#define SEMGPU2_CU_SPECTRUM_STURM_TOL 1.0e-20
#define SEMGPU2_CU_DEFAULT_T_BEGIN 0.0
#define SEMGPU2_CU_DEFAULT_T_END 0.0
#define SEMGPU2_CU_DEFAULT_KIM_BETA 0.0
#define SEMGPU2_CU_DEFAULT_KIM_GAMMA 0.5

namespace semgpu2_seismic_p5_cu {

using Real = double;
using Index = std::int64_t;
using Code = std::int32_t;
inline constexpr const char* SOLVER_ID = "lin_elast_2d_seismic_newmark_kim_p5_cu";

inline constexpr int PET_N = 6;
static_assert(PET_N * PET_N <= SEMGPU2_CU_THREADS_MAX);

inline constexpr int THREADS_MIN = SEMGPU2_CU_THREADS_MIN;
inline constexpr int THREADS_MAX = SEMGPU2_CU_THREADS_MAX;
inline constexpr int THREADS_PER_BLOCK = SEMGPU2_CU_THREADS_PER_BLOCK;
inline constexpr int LANCZOS_STEPS = SEMGPU2_CU_LANCZOS_STEPS;
inline constexpr int LANCZOS_MAX_BLOCKS = SEMGPU2_CU_LANCZOS_MAX_BLOCKS;
inline constexpr int SPECTRUM_BISECTION_STEPS = SEMGPU2_CU_SPECTRUM_BISECTION_STEPS;
inline constexpr double LANCZOS_BREAKDOWN_TOL = SEMGPU2_CU_LANCZOS_BREAKDOWN_TOL;
inline constexpr double SPECTRUM_STURM_TOL = SEMGPU2_CU_SPECTRUM_STURM_TOL;
inline constexpr double DEFAULT_T_BEGIN = SEMGPU2_CU_DEFAULT_T_BEGIN;
inline constexpr double DEFAULT_T_END = SEMGPU2_CU_DEFAULT_T_END;
inline constexpr double DEFAULT_KIM_BETA = SEMGPU2_CU_DEFAULT_KIM_BETA;
inline constexpr double DEFAULT_KIM_GAMMA = SEMGPU2_CU_DEFAULT_KIM_GAMMA;

static_assert(THREADS_MAX <= 1024 && THREADS_MAX > 0);

#define SEMGPU2_CU_SGSpaceEnum_N 0
#define SEMGPU2_CU_SGSpaceEnum_MAIN 1
#define SEMGPU2_CU_SGSpaceEnum_BOUNDARY 2
enum class SGSpaceEnum : Code {
    N = SEMGPU2_CU_SGSpaceEnum_N,
    MAIN = SEMGPU2_CU_SGSpaceEnum_MAIN,
    BOUNDARY = SEMGPU2_CU_SGSpaceEnum_BOUNDARY,
};

enum class SGPhysicsTypeEnum : Code {
    MATERIAL_ISO_PLANE_STRAIN = 11000,
    N_FORCE_DYNAMIC = 20200,
    N_MASS_DAMPING_STATIC = 30001,
    ABC = 32000,
};


enum class SGReceiverComponentEnum : Code {
    VX = 4,
    VY = 5,
};

#define SEMGPU2_CU_SGNSolutionComponentEnum_UX 0
#define SEMGPU2_CU_SGNSolutionComponentEnum_UY 1
#define SEMGPU2_CU_SGNSolutionComponentEnum_VX 2
#define SEMGPU2_CU_SGNSolutionComponentEnum_VY 3
#define SEMGPU2_CU_SGNSolutionComponentEnum_AX 4
#define SEMGPU2_CU_SGNSolutionComponentEnum_AY 5
enum class SGNSolutionComponentEnum : Code {
    UX = SEMGPU2_CU_SGNSolutionComponentEnum_UX,
    UY = SEMGPU2_CU_SGNSolutionComponentEnum_UY,
    VX = SEMGPU2_CU_SGNSolutionComponentEnum_VX,
    VY = SEMGPU2_CU_SGNSolutionComponentEnum_VY,
    AX = SEMGPU2_CU_SGNSolutionComponentEnum_AX,
    AY = SEMGPU2_CU_SGNSolutionComponentEnum_AY,
};

#define SEMGPU2_CU_SGNGeometryComponentEnum_X 0
#define SEMGPU2_CU_SGNGeometryComponentEnum_Y 1
#define SEMGPU2_CU_SGNGeometryComponentEnum_SIZE 2
enum class SGNGeometryComponentEnum : Code {
    X = SEMGPU2_CU_SGNGeometryComponentEnum_X,
    Y = SEMGPU2_CU_SGNGeometryComponentEnum_Y,
    SIZE = SEMGPU2_CU_SGNGeometryComponentEnum_SIZE,
};

#define SEMGPU2_CU_SGNMaterialsComponentEnum_MASS 0
enum class SGNMaterialsComponentEnum : Code {
    MASS = SEMGPU2_CU_SGNMaterialsComponentEnum_MASS,
};

enum class SGNResponseComponentEnum : Code {
    FORCE_X = 0,
    FORCE_Y = 1,
};

#define SEMGPU2_CU_SGQMainGeometryComponentEnum_SIZE 0
#define SEMGPU2_CU_SGQMainGeometryComponentEnum_DIFGEOMETRY_XX 1
#define SEMGPU2_CU_SGQMainGeometryComponentEnum_DIFGEOMETRY_XY 2
#define SEMGPU2_CU_SGQMainGeometryComponentEnum_DIFGEOMETRY_YX 3
#define SEMGPU2_CU_SGQMainGeometryComponentEnum_DIFGEOMETRY_YY 4
enum class SGQMainGeometryComponentEnum : Code {
    SIZE = SEMGPU2_CU_SGQMainGeometryComponentEnum_SIZE,
    DIFGEOMETRY_XX = SEMGPU2_CU_SGQMainGeometryComponentEnum_DIFGEOMETRY_XX,
    DIFGEOMETRY_XY = SEMGPU2_CU_SGQMainGeometryComponentEnum_DIFGEOMETRY_XY,
    DIFGEOMETRY_YX = SEMGPU2_CU_SGQMainGeometryComponentEnum_DIFGEOMETRY_YX,
    DIFGEOMETRY_YY = SEMGPU2_CU_SGQMainGeometryComponentEnum_DIFGEOMETRY_YY,
};

enum class SGQMainMaterialsComponentEnum : Code {
    DENSITY = 0,
    LAMBDA = 1,
    MU = 2,
};


#define SEMGPU2_CU_SGQBoundaryGeometryComponentEnum_SIZE 0
#define SEMGPU2_CU_SGQBoundaryGeometryComponentEnum_NORMAL_X 1
#define SEMGPU2_CU_SGQBoundaryGeometryComponentEnum_NORMAL_Y 2
enum class SGQBoundaryGeometryComponentEnum : Code {
    SIZE = SEMGPU2_CU_SGQBoundaryGeometryComponentEnum_SIZE,
    NORMAL_X = SEMGPU2_CU_SGQBoundaryGeometryComponentEnum_NORMAL_X,
    NORMAL_Y = SEMGPU2_CU_SGQBoundaryGeometryComponentEnum_NORMAL_Y,
};

#define SEMGPU2_CU_SGCompileErrorCodeEnum_STRUCTURE_FAILURE 0
#define SEMGPU2_CU_SGCompileErrorCodeEnum_DEVICE_FAILURE 1
enum class SGCompileErrorCodeEnum : Code {
    STRUCTURE_FAILURE = SEMGPU2_CU_SGCompileErrorCodeEnum_STRUCTURE_FAILURE,
    DEVICE_FAILURE = SEMGPU2_CU_SGCompileErrorCodeEnum_DEVICE_FAILURE,
};

#define SEMGPU2_CU_SGMeshErrorCodeEnum_GEOMETRY_FAILURE 0
#define SEMGPU2_CU_SGMeshErrorCodeEnum_DEVICE_FAILURE 1
enum class SGMeshErrorCodeEnum : Code {
    GEOMETRY_FAILURE = SEMGPU2_CU_SGMeshErrorCodeEnum_GEOMETRY_FAILURE,
    DEVICE_FAILURE = SEMGPU2_CU_SGMeshErrorCodeEnum_DEVICE_FAILURE,
};

#define SEMGPU2_CU_SGPrepareErrorCodeEnum_PHYSICS_FAILURE 0
#define SEMGPU2_CU_SGPrepareErrorCodeEnum_SPECTRUM_FAILURE 1
#define SEMGPU2_CU_SGPrepareErrorCodeEnum_DEVICE_FAILURE 2
enum class SGPrepareErrorCodeEnum : Code {
    PHYSICS_FAILURE = SEMGPU2_CU_SGPrepareErrorCodeEnum_PHYSICS_FAILURE,
    SPECTRUM_FAILURE = SEMGPU2_CU_SGPrepareErrorCodeEnum_SPECTRUM_FAILURE,
    DEVICE_FAILURE = SEMGPU2_CU_SGPrepareErrorCodeEnum_DEVICE_FAILURE,
};

#define SEMGPU2_CU_SGRunCodeEnum_TIME_END 0
#define SEMGPU2_CU_SGRunCodeEnum_UNDERCONFIGURED 1
#define SEMGPU2_CU_SGRunCodeEnum_NUMERICAL_DIVERGENCE 2
#define SEMGPU2_CU_SGRunCodeEnum_DEVICE_FAILURE 3
#define SEMGPU2_CU_SGRunCodeEnum_HANDLER_FAILURE 4
#define SEMGPU2_CU_SGRunCodeEnum_STEP_LIMIT 5
enum class SGRunCodeEnum : Code {
    TIME_END = SEMGPU2_CU_SGRunCodeEnum_TIME_END,
    UNDERCONFIGURED = SEMGPU2_CU_SGRunCodeEnum_UNDERCONFIGURED,
    NUMERICAL_DIVERGENCE = SEMGPU2_CU_SGRunCodeEnum_NUMERICAL_DIVERGENCE,
    DEVICE_FAILURE = SEMGPU2_CU_SGRunCodeEnum_DEVICE_FAILURE,
    HANDLER_FAILURE = SEMGPU2_CU_SGRunCodeEnum_HANDLER_FAILURE,
    STEP_LIMIT = SEMGPU2_CU_SGRunCodeEnum_STEP_LIMIT,
};

}  // namespace semgpu2_seismic_p5_cu
