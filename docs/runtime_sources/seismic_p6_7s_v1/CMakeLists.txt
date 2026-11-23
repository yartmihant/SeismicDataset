cmake_minimum_required(VERSION 3.27)
project(seismic_benchmark_native LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 3.1.0 EXACT CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
function(add_seismic_solver solver_id output_dir install_component source_id shared_source_id)
    set(solvers "${CMAKE_CURRENT_SOURCE_DIR}/src/semgpu2/solvers")
    set(backend "${solvers}/${source_id}/backend")
    set(shared_backend "${solvers}/${shared_source_id}/backend")
    set(target "semgpu2_${output_dir}_native")
    pybind11_add_module(${target} MODULE
        "${backend}/bindings.cpp"
        "${backend}/bindings_runtime.cpp"
        "${backend}/constants.cpp"
        "${backend}/solver.cpp"
        "${backend}/geometry.cpp"
        "${backend}/physics.cpp"
        "${shared_backend}/materials.cpp"
        "${backend}/preparation.cpp"
        "${backend}/stepping.cpp"
        "${backend}/spectrum.cpp"
        "${backend}/receivers.cpp"
        "${shared_backend}/run_control.cpp"
        "${shared_backend}/runtime.cpp"
        "${backend}/kernels/geometry.cu"
        "${shared_backend}/kernels/preparation.cu"
        "${backend}/kernels/response.cu"
        "${backend}/kernels/step.cu"
        "${shared_backend}/kernels/global_spectrum.cu"
        "${backend}/kernels/receiver.cu"
    )
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME "_native"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${output_dir}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${output_dir}"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${output_dir}"
    )
    target_include_directories(${target} PRIVATE
        "${solvers}/${solver_id}/backend" "${backend}" "${solvers}/common/cu")
    target_link_libraries(${target} PRIVATE CUDA::cudart Threads::Threads)
    target_compile_features(${target} PRIVATE cxx_std_17 cuda_std_17)
    if("$ENV{SEMGPU2_CUDA_TEST_HOOKS}" STREQUAL "ON")
        target_compile_definitions(${target} PRIVATE SEMGPU2_CUDA_TEST_HOOKS=1)
    endif()
    install(TARGETS ${target}
        LIBRARY DESTINATION "semgpu2/solvers/${solver_id}" COMPONENT "${install_component}"
        RUNTIME DESTINATION "semgpu2/solvers/${solver_id}" COMPONENT "${install_component}"
    )
endfunction()

set(seismic_prefix lin_elast_2d_seismic_newmark_kim)


foreach(order RANGE 6 6)
    set(solver_id "${seismic_prefix}_p${order}_cu")
    set(target_prefix "semgpu2_seismic_p${order}_cu")
    add_seismic_solver(${solver_id} seismic_p${order}_cu ${target_prefix} ${seismic_prefix}_p5_cu ${seismic_prefix}_p5_cu)
    target_compile_definitions(${target_prefix}_native PRIVATE
        SEMGPU2_SEISMIC_ORDER=${order}
        SEMGPU2_SEISMIC_NAMESPACE=${target_prefix}
        SEMGPU2_SEISMIC_SOLVER_ID="${solver_id}"
        SEMGPU2_SEISMIC_CONSTANTS_MODULE="semgpu2.solvers.${solver_id}.constants"
    )
endforeach()

set(seven_shot_id "${seismic_prefix}_p6_7s_cu")
add_seismic_solver(${seven_shot_id} seismic_p6_7s_cu semgpu2_seismic_p6_7s_cu ${seven_shot_id} ${seismic_prefix}_p5_cu)
target_compile_definitions(semgpu2_seismic_p6_7s_cu_native PRIVATE
    SEMGPU2_SEISMIC_ORDER=6
    SEMGPU2_SEISMIC_NAMESPACE=semgpu2_seismic_p6_7s_cu
    SEMGPU2_SEISMIC_SOLVER_ID="${seven_shot_id}"
    SEMGPU2_SEISMIC_CONSTANTS_MODULE="semgpu2.solvers.${seven_shot_id}.constants"
)
target_include_directories(semgpu2_seismic_p6_7s_cu_native PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/semgpu2/solvers/${seismic_prefix}_p6_cu/backend"
)

