# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# A generator action with resolved output paths and command arguments. Python
# entry points retain their declared module closure; native tools retain their
# executable dependency. Both use a freshness stamp so content-stable outputs
# do not force their consumers to rebuild after an authority-only change.
function(iree_generated_files)
  cmake_parse_arguments(_RULE "USE_PARAM_FILE" "NAME;GENERATOR;COMMENT"
    "OUTPUTS;ARGS;INPUTS" ${ARGN})
  iree_package_name(_PACKAGE_NAME)
  iree_package_binary_dir(_IREE_PACKAGE_BINARY_DIR)
  set(_GEN_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}")
  set(_GEN_STAMP "${_IREE_PACKAGE_BINARY_DIR}/CMakeFiles/${_GEN_TARGET}.stamp")

  iree_package_target_name(_GENERATOR_TARGET "${_RULE_GENERATOR}")
  if(TARGET "${_GENERATOR_TARGET}")
    get_target_property(_GENERATOR_TYPE "${_GENERATOR_TARGET}" TYPE)
  endif()
  if(_GENERATOR_TYPE STREQUAL "UTILITY")
    iree_py_library_entrypoint(_ENTRYPOINT "${_RULE_GENERATOR}")
    iree_py_library_collect_sources(_GENERATOR_INPUTS "${_RULE_GENERATOR}")
    iree_py_library_collect_package_dirs(_PACKAGE_DIRS "${_RULE_GENERATOR}")
    list(APPEND _PACKAGE_DIRS "$ENV{PYTHONPATH}")
    if(CMAKE_HOST_WIN32)
      list(JOIN _PACKAGE_DIRS "\\;" _PYTHONPATH)
    else()
      list(JOIN _PACKAGE_DIRS ":" _PYTHONPATH)
    endif()
    set(_COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${_PYTHONPATH}"
      "PYTHONDONTWRITEBYTECODE=1" "${Python3_EXECUTABLE}" ${_ENTRYPOINT})
  else()
    # Executables may be declared after their consumers. Native target-file
    # expressions preserve their owning directory and configuration.
    iree_package_ns(_PACKAGE_NS)
    string(REGEX REPLACE "^::" "${_PACKAGE_NS}::"
      _GENERATOR_TARGET "${_RULE_GENERATOR}")
    set(_COMMAND "$<TARGET_FILE:${_GENERATOR_TARGET}>")
    set(_GENERATOR_INPUTS "${_GENERATOR_TARGET}")
  endif()

  set(_ARGS ${_RULE_ARGS})
  if(_RULE_USE_PARAM_FILE)
    set(_PARAM_FILE "${_IREE_PACKAGE_BINARY_DIR}/CMakeFiles/${_GEN_TARGET}-$<CONFIG>.args")
    list(JOIN _ARGS "\n" _PARAM_CONTENT)
    file(GENERATE OUTPUT "${_PARAM_FILE}" CONTENT "${_PARAM_CONTENT}\n")
    set(_ARGS "@${_PARAM_FILE}")
    list(APPEND _GENERATOR_INPUTS "${_PARAM_FILE}")
  endif()
  iree_resolve_source_files(_RULE_INPUTS ${_RULE_INPUTS})
  add_custom_command(
    OUTPUT "${_GEN_STAMP}"
    BYPRODUCTS ${_RULE_OUTPUTS}
    COMMAND ${_COMMAND} ${_ARGS}
    COMMAND "${CMAKE_COMMAND}" -E touch "${_GEN_STAMP}"
    DEPENDS ${_GENERATOR_INPUTS} ${_RULE_INPUTS}
    COMMENT "${_RULE_COMMENT}"
    VERBATIM
  )
  set_source_files_properties(${_RULE_OUTPUTS} PROPERTIES GENERATED TRUE)
  add_custom_target("${_GEN_TARGET}" DEPENDS "${_GEN_STAMP}")
  iree_register_generated_compile_input("${_GEN_TARGET}" OUTPUTS ${_RULE_OUTPUTS})
endfunction()
