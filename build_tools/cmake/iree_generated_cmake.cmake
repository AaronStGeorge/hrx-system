# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# CMake owns generation so direct configure, dev.py, and native regeneration
# consume the same declarations. Each HRX binary tree owns its package tree.
function(iree_generate_cmake)
  if(CMAKE_HOST_WIN32 AND CMAKE_VERSION VERSION_LESS 4.2)
    message(FATAL_ERROR
      "Windows HRX builds require CMake 4.2 or newer for short object names.")
  endif()
  if(NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "Configuring HRX requires a Python interpreter.")
  endif()
  set(IREE_GENERATED_CMAKE_DIR "${IREE_BINARY_DIR}/.iree/cmake")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../bazel_to_cmake/bazel_to_cmake.py"
      --output-dir "${IREE_GENERATED_CMAKE_DIR}"
    WORKING_DIRECTORY "${IREE_SOURCE_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
  )
  include("${IREE_GENERATED_CMAKE_DIR}/inputs.cmake")
  set(IREE_GENERATED_CMAKE_DIR "${IREE_GENERATED_CMAKE_DIR}" PARENT_SCOPE)
endfunction()
