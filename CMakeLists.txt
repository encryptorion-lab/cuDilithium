# deprecate old CMake versions
cmake_minimum_required(VERSION 3.20)

# set the project name, languages, version, and description
project(cuDilithium LANGUAGES C CXX CUDA VERSION 1.0.0 DESCRIPTION "Dilithium in CUDA")

# set language standards
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CUDA only: Enables separate compilation of device code
# enable separable compilation for all CUDA files in this project
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# deprecate old CUDA versions
if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    message(FATAL_ERROR "CUDA Toolkit 11.0 or higher is required")
endif()

# if CUDA architecture is not specified, use the following
if (CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 70 72 75 80 86)
endif()

# enable CTest testing
enable_testing()

# set the output directories, so that the executables are in bin/ and the libraries are in lib/
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# use OpenMP for CPU multi-threading
find_package(OpenMP)
if(OPENMP_C_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_C_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    link_libraries(${OpenMP_C_LIBRARIES})
endif()

# include headers in this project
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# add subdirectory of library source files
add_subdirectory(src)

# add subdirectory of test source files
add_subdirectory(tests)

# add subdirectory of benchmark source files
add_subdirectory(benches)
