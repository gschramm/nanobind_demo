cmake_minimum_required(VERSION 3.20)
project(vecadd LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Python + nanobind discovery
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ------------------------------------------------------------------
# 1. Build CUDA/OpenMP core as a shared library
# ------------------------------------------------------------------
add_library(vecadd_core SHARED vec_add.cu)

# Public include so bindings.cpp can #include "vec_add.h"
target_include_directories(vecadd_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(vecadd_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(vecadd_core
  PUBLIC
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

# Put the core lib next to the extension so the loader can find it easily
set_target_properties(vecadd_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ------------------------------------------------------------------
# 2. Build nanobind extension and link against vecadd_core
# ------------------------------------------------------------------
#execute_process(
#  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
#  OUTPUT_STRIP_TRAILING_WHITESPACE
#  OUTPUT_VARIABLE NB_DIR
#)
#list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
  vecadd_ext     # name = Python module name
  bindings.cpp
)

# Ensure it can see vec_add.h and link to core
target_include_directories(vecadd_ext
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(vecadd_ext
  PRIVATE
    vecadd_core
)

# (Optional) help runtime loader find libvecadd_core.so at import time
set_target_properties(vecadd_ext PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  INSTALL_RPATH "$ORIGIN"
)

