# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(libhrx VERSION ${HRX_VERSION} LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(LIBHRX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(LIBHRX_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

if(NOT TARGET iree_vm_vm)
  message(FATAL_ERROR
    "libhrx must be added after the reduced IREE runtime has been configured")
endif()

add_subdirectory(src/libhrx)
add_subdirectory(src/streaming)
add_subdirectory(src/binding/hip)

if(LIBHRX_BUILD_CUDA_BINDING)
  add_subdirectory(src/binding/cuda)
endif()

add_executable(hrx-info tools/hrx_info.c)
target_include_directories(hrx-info PRIVATE
  "${LIBHRX_SOURCE_DIR}/include"
  "${LIBHRX_SOURCE_DIR}/src/libhrx"
)
target_link_libraries(hrx-info PRIVATE
  hrx
  iree_base_tooling_flags
)
set_target_properties(hrx-info PROPERTIES
  INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
install(TARGETS hrx-info
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

if(LIBHRX_BUILD_CTS)
  enable_testing()
  add_subdirectory(cts)
endif()

if(LIBHRX_BUILD_PASSTHROUGH)
  add_subdirectory(src/passthrough)
endif()

add_library(hrx::hrx ALIAS hrx)

set(_hrx_cmake_package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/hrx")
set(_hrx_build_package_dir "${LIBHRX_BINARY_DIR}/cmake/hrx")

install(TARGETS hrx EXPORT hrx-targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hrx")
install(FILES
  include/hrx_runtime.h
  include/hrx_runtime_cxx.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hrx")
install(EXPORT hrx-targets
  NAMESPACE hrx::
  DESTINATION "${_hrx_cmake_package_dir}")

configure_package_config_file(
  cmake/hrx-config.cmake.in
  "${_hrx_build_package_dir}/hrx-config.cmake"
  INSTALL_DESTINATION "${_hrx_cmake_package_dir}")
write_basic_package_version_file(
  "${_hrx_build_package_dir}/hrx-config-version.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)
export(EXPORT hrx-targets
  NAMESPACE hrx::
  FILE "${_hrx_build_package_dir}/hrx-targets.cmake")

install(FILES
  "${_hrx_build_package_dir}/hrx-config.cmake"
  "${_hrx_build_package_dir}/hrx-config-version.cmake"
  DESTINATION "${_hrx_cmake_package_dir}")
