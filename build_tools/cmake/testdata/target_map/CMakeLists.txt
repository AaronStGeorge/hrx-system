# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26)
project(iree_cmake_target_map_test C)

include("${IREE_REPO_ROOT}/build_tools/cmake/iree_install_support.cmake")

set(FIXTURE_OUTPUT_NAME "named tool" CACHE STRING "Executable output name")
add_subdirectory(child)
iree_add_alias_executable(fixture::tool fixture_program)
add_library(headers INTERFACE)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated.h" "#pragma once\n")
add_library(generated_headers INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/generated.h")

function(create_deferred_target)
  add_custom_target(late_target)
endfunction()
cmake_language(DEFER CALL create_deferred_target)
cmake_language(DEFER CALL iree_write_target_map)
