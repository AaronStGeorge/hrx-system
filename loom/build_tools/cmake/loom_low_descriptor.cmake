# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Loom target-low generated table build helpers.
#
# These helpers mirror loom/src/loom/build_defs.bzl for generated CMake.
# Target packages declare descriptor shards and target-info tables; this file
# owns the CMake mechanics for running generators into the binary tree and
# wrapping the outputs in iree_cc_library targets.

include(ExternalProject)

function(loom_low_descriptor_data_archive)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SOURCE_DIR;SHA256"
    "URLS"
    ${ARGN}
  )

  if(NOT _RULE_NAME)
    message(FATAL_ERROR "loom_low_descriptor_data_archive requires NAME")
  endif()
  if(NOT _RULE_SOURCE_DIR)
    message(FATAL_ERROR "loom_low_descriptor_data_archive requires SOURCE_DIR")
  endif()
  if(NOT _RULE_URLS)
    message(FATAL_ERROR "loom_low_descriptor_data_archive requires URLS")
  endif()
  if(NOT _RULE_SHA256)
    message(FATAL_ERROR "loom_low_descriptor_data_archive requires SHA256")
  endif()

  ExternalProject_Add("${_RULE_NAME}"
    URL
      ${_RULE_URLS}
    URL_HASH
      "SHA256=${_RULE_SHA256}"
    SOURCE_DIR
      "${_RULE_SOURCE_DIR}"
    EXCLUDE_FROM_ALL
      TRUE
    CONFIGURE_COMMAND
      ""
    BUILD_COMMAND
      ""
    INSTALL_COMMAND
      ""
    UPDATE_COMMAND
      ""
  )
endfunction()

function(loom_target_table_cc_library)
  cmake_parse_arguments(
    _RULE
    "EXCLUDE_FROM_ALL"
    "NAME;GENERATOR;SOURCE;HEADER"
    "ARGS;INPUTS;DEPS"
    ${ARGN}
  )

  if(NOT _RULE_NAME)
    message(FATAL_ERROR "loom_target_table_cc_library requires NAME")
  endif()
  if(NOT _RULE_GENERATOR)
    message(FATAL_ERROR "loom_target_table_cc_library requires GENERATOR")
  endif()
  if(NOT _RULE_SOURCE)
    set(_RULE_SOURCE "${_RULE_NAME}.c")
  endif()
  if(NOT _RULE_HEADER)
    set(_RULE_HEADER "${_RULE_NAME}.h")
  endif()

  set(_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_SOURCE}")
  set(_HEADER "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_HEADER}")
  iree_py_library_main(_GENERATOR "${_RULE_GENERATOR}")
  iree_py_library_collect_sources(_GENERATOR_INPUTS "${_RULE_GENERATOR}")

  add_custom_command(
    OUTPUT
      "${_SOURCE}"
      "${_HEADER}"
    COMMAND
      "${Python3_EXECUTABLE}"
      "${_GENERATOR}"
      ${_RULE_ARGS}
      "--source=${_SOURCE}"
      "--header=${_HEADER}"
    DEPENDS
      ${_GENERATOR_INPUTS}
      ${_RULE_INPUTS}
    COMMENT
      "Generating ${_RULE_NAME} target table"
    VERBATIM
  )

  iree_package_name(_PACKAGE_NAME)
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}_gen"
    DEPENDS
      "${_SOURCE}"
      "${_HEADER}"
  )

  loom_cc_library(
    NAME
      ${_RULE_NAME}
    HDRS
      "${_HEADER}"
    SRCS
      "${_SOURCE}"
    DEPS
      ${_RULE_DEPS}
    PUBLIC
  )

  if(_RULE_EXCLUDE_FROM_ALL)
    iree_cc_library_exclude_from_all(${_RULE_NAME} TRUE)
  endif()
endfunction()

function(loom_low_descriptor_cc_library)
  cmake_parse_arguments(
    _RULE
    "EXCLUDE_FROM_ALL"
    "NAME;HEADER"
    "DEPS"
    ${ARGN}
  )

  loom_target_table_cc_library(${ARGN})

  if(NOT _RULE_NAME)
    message(FATAL_ERROR "loom_low_descriptor_cc_library requires NAME")
  endif()
  if(NOT _RULE_HEADER)
    set(_RULE_HEADER "${_RULE_NAME}.h")
  endif()

  set(_HEADER "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_HEADER}")
  iree_package_name(_PACKAGE_NAME)
  loom_cc_library(
    NAME
      "${_RULE_NAME}_ids"
    HDRS
      "${_HEADER}"
    DEPS
      ${_RULE_DEPS}
    PUBLIC
  )
  add_dependencies(
    "${_PACKAGE_NAME}_${_RULE_NAME}_ids"
    "${_PACKAGE_NAME}_${_RULE_NAME}_gen"
  )
endfunction()

function(loom_low_descriptor_exclude_from_all)
  cmake_parse_arguments(
    _RULE
    ""
    ""
    "CC_LIBRARIES;TARGETS"
    ${ARGN}
  )

  iree_package_name(_PACKAGE_NAME)

  foreach(_CC_LIBRARY IN LISTS _RULE_CC_LIBRARIES)
    set(_NAME "${_PACKAGE_NAME}_${_CC_LIBRARY}")
    if(NOT TARGET "${_NAME}")
      message(FATAL_ERROR
        "Cannot exclude missing low descriptor library ${_CC_LIBRARY} from all")
    endif()
    iree_cc_library_exclude_from_all(${_CC_LIBRARY} TRUE)
  endforeach()

  foreach(_TARGET IN LISTS _RULE_TARGETS)
    set(_NAME "${_PACKAGE_NAME}_${_TARGET}")
    if(TARGET "${_NAME}")
      set_property(TARGET "${_NAME}" PROPERTY EXCLUDE_FROM_ALL TRUE)
    elseif(IREE_BUILD_TESTS)
      message(FATAL_ERROR
        "Cannot exclude missing low descriptor target ${_TARGET} from all")
    endif()
  endforeach()
endfunction()
