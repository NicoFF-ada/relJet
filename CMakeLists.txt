cmake_minimum_required(VERSION 3.18)
project(RelJet VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(RELJET_ENABLE_OPENMP "Parallelise CPU cell loops with OpenMP" ON)
option(RELJET_ENABLE_CUDA "Build the CUDA backend" OFF)
option(RELJET_BUILD_TESTS "Build the unit tests" ON)

# ------------------------------------------------------------------ library --

set(RELJET_MODULES
    core mesh fields thermophysics srhd schemes boundaryConditions solver io
    problems verification)

add_library(relJetCore
    src/core/Dictionary.C
    src/mesh/Grid.C
    src/fields/Fields.C
    src/thermophysics/IdealGasEos.C
    src/schemes/Reconstruction.C
    src/boundaryConditions/Boundary.C
    src/solver/Solver.C
    src/io/CsvOutput.C
    src/io/VtkOutput.C
    src/problems/Jet.C
    src/verification/SmoothWave.C)

foreach(module IN LISTS RELJET_MODULES)
    target_include_directories(relJetCore PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/${module})
endforeach()
target_compile_options(relJetCore PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow>)

if(RELJET_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(relJetCore PUBLIC OpenMP::OpenMP_CXX)
endif()

if(RELJET_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)

    add_library(relJetCuda src/cuda/CudaSolver.cu)
    target_include_directories(relJetCuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)
    target_link_libraries(relJetCuda PUBLIC relJetCore CUDA::cudart)
    target_compile_options(relJetCuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_compile_definitions(relJetCuda PUBLIC RELJET_HAVE_CUDA)
endif()

# ------------------------------------------------------------- applications --

add_executable(relJet applications/solvers/relJet/relJet.C)
target_link_libraries(relJet PRIVATE relJetCore)
if(RELJET_ENABLE_CUDA)
    target_link_libraries(relJet PRIVATE relJetCuda)
endif()

add_executable(convergenceStudy
    applications/utilities/convergenceStudy/convergenceStudy.C)
target_link_libraries(convergenceStudy PRIVATE relJetCore)

# -------------------------------------------------------------------- tests --

if(RELJET_BUILD_TESTS)
    enable_testing()
    add_executable(relJetTests
        tests/testMain.C
        tests/testConfig.C
        tests/testGrid.C
        tests/testSrhd.C
        tests/testConsToPrim.C
        tests/testRiemann.C
        tests/testConsOps.C
        tests/testReconstruction.C
        tests/testSolver.C
        tests/testVerification.C
        tests/testMultiD.C
        tests/testJet.C)
    target_include_directories(relJetTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(relJetTests PRIVATE relJetCore)
    if(RELJET_ENABLE_CUDA)
        target_sources(relJetTests PRIVATE tests/testCuda.C)
        target_link_libraries(relJetTests PRIVATE relJetCuda)
    endif()
    add_test(NAME unit COMMAND relJetTests)
endif()
