# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Loom generated C build helpers.
#
# These helpers mirror loom/src/loom/build_defs.bzl for generated CMake.
# Loom packages keep source-of-truth tables in Python and generate compact C
# data into the build tree.

function(loom_generated_cc_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;GENERATOR;SOURCE"
    "HDRS;ARGS;INPUTS;EXTRA_OUTPUT_FLAGS;EXTRA_OUTPUTS;DEPS"
    ${ARGN}
  )

  if(NOT _RULE_NAME)
    message(FATAL_ERROR "loom_generated_cc_library requires NAME")
  endif()
  if(NOT _RULE_GENERATOR)
    message(FATAL_ERROR "loom_generated_cc_library requires GENERATOR")
  endif()
  if(NOT _RULE_SOURCE)
    set(_RULE_SOURCE "${_RULE_NAME}.c")
  endif()

  list(LENGTH _RULE_EXTRA_OUTPUT_FLAGS _EXTRA_OUTPUT_FLAG_COUNT)
  list(LENGTH _RULE_EXTRA_OUTPUTS _EXTRA_OUTPUT_COUNT)
  if(NOT _EXTRA_OUTPUT_FLAG_COUNT EQUAL _EXTRA_OUTPUT_COUNT)
    message(FATAL_ERROR
      "loom_generated_cc_library requires paired EXTRA_OUTPUT_FLAGS and EXTRA_OUTPUTS")
  endif()

  set(_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_SOURCE}")
  set(_OUTPUTS "${_SOURCE}")
  set(_OUTPUT_ARGS "--source=${_SOURCE}")
  if(_EXTRA_OUTPUT_COUNT GREATER 0)
    math(EXPR _EXTRA_OUTPUT_LAST "${_EXTRA_OUTPUT_COUNT} - 1")
    foreach(_INDEX RANGE 0 ${_EXTRA_OUTPUT_LAST})
      list(GET _RULE_EXTRA_OUTPUT_FLAGS ${_INDEX} _OUTPUT_FLAG)
      list(GET _RULE_EXTRA_OUTPUTS ${_INDEX} _OUTPUT)
      set(_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${_OUTPUT}")
      list(APPEND _OUTPUTS "${_OUTPUT_PATH}")
      list(APPEND _OUTPUT_ARGS "${_OUTPUT_FLAG}=${_OUTPUT_PATH}")
    endforeach()
  endif()

  iree_py_library_main(_GENERATOR "${_RULE_GENERATOR}")
  iree_py_library_collect_sources(_GENERATOR_INPUTS "${_RULE_GENERATOR}")

  add_custom_command(
    OUTPUT
      ${_OUTPUTS}
    COMMAND
      "${Python3_EXECUTABLE}"
      "${_GENERATOR}"
      ${_RULE_ARGS}
      ${_OUTPUT_ARGS}
    DEPENDS
      ${_GENERATOR_INPUTS}
      ${_RULE_INPUTS}
    COMMENT
      "Generating ${_RULE_NAME} C table"
    VERBATIM
  )

  iree_package_name(_PACKAGE_NAME)
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}_gen"
    DEPENDS
      ${_OUTPUTS}
  )

  loom_cc_library(
    NAME
      ${_RULE_NAME}
    HDRS
      ${_RULE_HDRS}
    SRCS
      "${_SOURCE}"
    DEPS
      ${_RULE_DEPS}
    PUBLIC
  )
endfunction()
