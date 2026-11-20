cmake_minimum_required(VERSION 3.27)
project(seismic_benchmark_native LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 3.1.0 EXACT CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
function(add_seismic_solver solver_id output_dir install_component)
    set(backend "${CMAKE_CURRENT_SOURCE_DIR}/src/semgpu2/solvers/${solver_id}/backend")
    set(target "semgpu2_${output_dir}_native")
    pybind11_add_module(${target} MODULE
        "${backend}/bindings.cpp"
        "${backend}/bindings_runtime.cpp"
        "${backend}/constants.cpp"
        "${backend}/solver.cpp"
        "${backend}/geometry.cpp"
        "${backend}/physics.cpp"
        "${backend}/materials.cpp"
        "${backend}/preparation.cpp"
        "${backend}/stepping.cpp"
        "${backend}/spectrum.cpp"
        "${backend}/receivers.cpp"
        "${backend}/run_control.cpp"
        "${backend}/runtime.cpp"
        "${backend}/kernels/geometry.cu"
        "${backend}/kernels/preparation.cu"
        "${backend}/kernels/response.cu"
        "${backend}/kernels/step.cu"
        "${backend}/kernels/global_spectrum.cu"
        "${backend}/kernels/receiver.cu"
    )
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME "_native"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${output_dir}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${output_dir}"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${output_dir}"
    )
    target_include_directories(${target} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/semgpu2/solvers/common/cu" "${backend}")
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

add_seismic_solver(lin_elast_2d_seismic_newmark_kim_cu seismic_cu Unspecified)
add_seismic_solver(lin_elast_2d_seismic_newmark_kim_p5_cu seismic_p5_cu semgpu2_seismic_p5_cu)
