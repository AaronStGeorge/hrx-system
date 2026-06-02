# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(FetchContent)

function(iree_fetch_content_assert_allowed dep_name)
  if(IREE_HERMETIC_BUILD)
    message(FATAL_ERROR
      "${dep_name} was not found via package discovery and IREE_HERMETIC_BUILD=ON "
      "forbids FetchContent. Provide the package via CMAKE_PREFIX_PATH or an "
      "equivalent CMake package path.")
  endif()
endfunction()

function(_iree_dependency_variable_name out_var dep_name property_name)
  string(TOUPPER "${dep_name}" _dependency_identifier)
  string(REGEX REPLACE "[^A-Z0-9]" "_" _dependency_identifier
    "${_dependency_identifier}")
  set(${out_var} "IREE_DEP_${_dependency_identifier}_${property_name}"
    PARENT_SCOPE)
endfunction()

function(iree_get_locked_dependency_property out_var dep_name property_name)
  _iree_dependency_variable_name(_variable_name "${dep_name}" "${property_name}")
  if(NOT DEFINED ${_variable_name})
    message(FATAL_ERROR
      "${dep_name} is missing ${property_name} in MODULE.cmake.lock. "
      "Regenerate the lock with build_tools/bazel_to_cmake/deps.py.")
  endif()
  set(${out_var} "${${_variable_name}}" PARENT_SCOPE)
endfunction()

function(iree_declare_locked_fetch_content dep_name)
  cmake_parse_arguments(PARSE_ARGV 1 _ARGS "" "FETCH_NAME" "")
  if(_ARGS_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "iree_declare_locked_fetch_content(${dep_name}) received unexpected "
      "arguments: ${_ARGS_UNPARSED_ARGUMENTS}")
  endif()
  if(_ARGS_FETCH_NAME)
    set(_fetch_name "${_ARGS_FETCH_NAME}")
  else()
    set(_fetch_name "${dep_name}")
  endif()

  iree_get_locked_dependency_property(_urls "${dep_name}" "URLS")
  iree_get_locked_dependency_property(_sha256 "${dep_name}" "SHA256")
  if(NOT _urls)
    message(FATAL_ERROR "${dep_name} has no locked source URLs")
  endif()
  if(NOT _sha256)
    message(FATAL_ERROR "${dep_name} has no locked SHA256")
  endif()

  FetchContent_Declare(
    ${_fetch_name}
    URL ${_urls}
    URL_HASH SHA256=${_sha256}
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    EXCLUDE_FROM_ALL
  )
endfunction()

function(iree_add_alias_interface alias_name)
  set(_deps ${ARGN})
  if(TARGET ${alias_name})
    return()
  endif()
  string(MAKE_C_IDENTIFIER "${alias_name}" _target_name)
  set(_target_name "iree_${_target_name}")
  add_library(${_target_name} INTERFACE)
  target_link_libraries(${_target_name} INTERFACE ${_deps})
  add_library(${alias_name} ALIAS ${_target_name})
endfunction()
