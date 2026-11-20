cmake_minimum_required(VERSION 3.27)
project(seismic_benchmark_native LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 3.1.0 EXACT CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
set(SEMGPU2_SEISMIC_CU_BACKEND_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/src/semgpu2/solvers/lin_elast_2d_seismic_newmark_kim_cu/backend"
)
set(SEMGPU2_SEISMIC_CU_BACKEND_SOURCES
    "${SEMGPU2_SEISMIC_CU_BACKEND_DIR}/bindings.cpp"
    "${SEMGPU2_SEISMIC_CU_BACKEND_DIR}/bindings_runtime.cpp"
    "${SEMGPU2_SEISMIC_CU_BACKEND_DIR}/constants.cpp"
    "${SEMGPU2_SEISMIC_CU_BACKEND_DIR}/solver.cpp"
    "${SEMGPU2_SEISMIC_CU_BACKEND_DIR}/geometry.cpp"
    "${SEMGPU2_SEISMIC_CU_BACKEND_DIR}/physics.cpp"
    "${SEMGPU2_SEISMIC_CU_BACKEND_DIR}/materials.cpp"
    "${SEMGPU2_SEISMIC_CU_BACKEND_DIR}/preparation.cpp"
    "${SEMGPU2_SEISMIC_CU_BACKEND_DIR}/stepping.cpp"
    "${SEMGPU2_SEISMIC_CU_BACKEND_DIR}/spectrum.cpp"
    "${SEMGPU2_SEISMIC_CU_BACKEND_DIR}/receivers.cpp"
    "${SEMGPU2_SEISMIC_CU_BACKEND_DIR}/run_control.cpp"
    "${SEMGPU2_SEISMIC_CU_BACKEND_DIR}/runtime.cpp"
    "${SEMGPU2_SEISMIC_CU_BACKEND_DIR}/kernels/geometry.cu"
    "${SEMGPU2_SEISMIC_CU_BACKEND_DIR}/kernels/preparation.cu"
    "${SEMGPU2_SEISMIC_CU_BACKEND_DIR}/kernels/response.cu"
    "${SEMGPU2_SEISMIC_CU_BACKEND_DIR}/kernels/step.cu"
    "${SEMGPU2_SEISMIC_CU_BACKEND_DIR}/kernels/global_spectrum.cu"
    "${SEMGPU2_SEISMIC_CU_BACKEND_DIR}/kernels/receiver.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/semgpu2/solvers/common/cu/scheduler.cpp"
)

pybind11_add_module(semgpu2_seismic_cu_native MODULE ${SEMGPU2_SEISMIC_CU_BACKEND_SOURCES})
# Both Python extensions are named _native, so their build outputs must differ.
set_target_properties(semgpu2_seismic_cu_native PROPERTIES
    OUTPUT_NAME "_native"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/seismic_cu"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/seismic_cu"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/seismic_cu"
)
target_include_directories(semgpu2_seismic_cu_native PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/semgpu2/solvers/common/cu"
    "${SEMGPU2_SEISMIC_CU_BACKEND_DIR}"
)
target_link_libraries(semgpu2_seismic_cu_native PRIVATE CUDA::cudart Threads::Threads)
target_compile_features(semgpu2_seismic_cu_native PRIVATE cxx_std_17 cuda_std_17)

if("$ENV{SEMGPU2_CUDA_TEST_HOOKS}" STREQUAL "ON")
    target_compile_definitions(semgpu2_seismic_cu_native PRIVATE SEMGPU2_CUDA_TEST_HOOKS=1)
endif()

install(TARGETS semgpu2_seismic_cu_native
    LIBRARY DESTINATION "semgpu2/solvers/lin_elast_2d_seismic_newmark_kim_cu"
    RUNTIME DESTINATION "semgpu2/solvers/lin_elast_2d_seismic_newmark_kim_cu"
)

