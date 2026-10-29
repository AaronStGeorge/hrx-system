# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Enter each package in a real directory scope, with generated declarations in
# the private CMake tree and native artifacts at their original binary paths.
# A macro preserves add_subdirectory's parent variable scope for authored CMake.
macro(iree_add_subdirectory DIRECTORY)
  iree_package_source_dir(_IREE_PARENT_SOURCE_DIR)
  get_filename_component(_IREE_CHILD_SOURCE_DIR "${DIRECTORY}" ABSOLUTE
    BASE_DIR "${_IREE_PARENT_SOURCE_DIR}")
  file(RELATIVE_PATH _IREE_CHILD_PACKAGE "${IREE_SOURCE_DIR}"
    "${_IREE_CHILD_SOURCE_DIR}")
  if(IREE_GENERATED_CMAKE_DIR AND EXISTS
      "${IREE_GENERATED_CMAKE_DIR}/${_IREE_CHILD_PACKAGE}/CMakeLists.txt")
    add_subdirectory("${IREE_GENERATED_CMAKE_DIR}/${_IREE_CHILD_PACKAGE}"
      "${CMAKE_CURRENT_BINARY_DIR}/${DIRECTORY}" ${ARGN})
  else()
    add_subdirectory("${_IREE_CHILD_SOURCE_DIR}"
      "${CMAKE_CURRENT_BINARY_DIR}/${DIRECTORY}" ${ARGN})
  endif()
endmacro()

# Children may be source-authored projects or generated BUILD packages.
macro(iree_add_all_subdirs)
  iree_package_source_dir(_IREE_PARENT_SOURCE_DIR)
  file(GLOB _IREE_CHILDREN RELATIVE "${_IREE_PARENT_SOURCE_DIR}"
    "${_IREE_PARENT_SOURCE_DIR}/*")
  foreach(_IREE_CHILD IN LISTS _IREE_CHILDREN)
    file(RELATIVE_PATH _IREE_CHILD_PACKAGE "${IREE_SOURCE_DIR}"
      "${_IREE_PARENT_SOURCE_DIR}/${_IREE_CHILD}")
    if(IS_DIRECTORY "${_IREE_PARENT_SOURCE_DIR}/${_IREE_CHILD}" AND
        (EXISTS "${_IREE_PARENT_SOURCE_DIR}/${_IREE_CHILD}/CMakeLists.txt" OR
         EXISTS "${IREE_GENERATED_CMAKE_DIR}/${_IREE_CHILD_PACKAGE}/CMakeLists.txt"))
      iree_add_subdirectory("${_IREE_CHILD}")
    endif()
  endforeach()
endmacro()
