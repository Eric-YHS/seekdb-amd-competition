cmake_minimum_required(VERSION 3.20)
project(seekdb_hipvs_builder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(HIPVS_ROOT "/opt/hipvs" CACHE PATH "hipVS installation prefix")
set(ROCM_ROOT "/opt/rocm" CACHE PATH "ROCm installation prefix")
find_path(HIPVS_INCLUDE_DIR cuvs/neighbors/cagra.h HINTS "${HIPVS_ROOT}/include" REQUIRED)
find_path(HIP_INCLUDE_DIR hip/hip_runtime_api.h HINTS "${ROCM_ROOT}/include" REQUIRED)
find_library(HIPVS_C_LIBRARY cuvs_c HINTS "${HIPVS_ROOT}/lib" REQUIRED)
find_library(HIP_RUNTIME_LIBRARY amdhip64 HINTS "${ROCM_ROOT}/lib" REQUIRED)

add_executable(seekdb_hipvs_build hipvs_hnsw_export.cpp)
target_compile_definitions(seekdb_hipvs_build PRIVATE __HIP_PLATFORM_AMD__)
target_include_directories(seekdb_hipvs_build PRIVATE "${HIPVS_INCLUDE_DIR}" "${HIP_INCLUDE_DIR}")
target_link_libraries(seekdb_hipvs_build PRIVATE "${HIPVS_C_LIBRARY}" "${HIP_RUNTIME_LIBRARY}")
set_target_properties(seekdb_hipvs_build PROPERTIES
  BUILD_RPATH "${HIPVS_ROOT}/lib;${ROCM_ROOT}/lib"
  INSTALL_RPATH "${HIPVS_ROOT}/lib;${ROCM_ROOT}/lib")
install(TARGETS seekdb_hipvs_build RUNTIME DESTINATION bin)
