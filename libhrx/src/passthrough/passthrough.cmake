# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(TARGET libhrx_src_passthrough_hip_intercept)
  set_target_properties(libhrx_src_passthrough_hip_intercept PROPERTIES
    OUTPUT_NAME hip_intercept
    VERSION 1.0.0
    SOVERSION 1
    C_VISIBILITY_PRESET default
    LINK_FLAGS "-Wl,--undefined-version -Wl,--version-script=${_IREE_PACKAGE_SOURCE_DIR}/passthrough_full.map"
  )
  set_target_properties(libhrx_src_passthrough_hip_logging PROPERTIES
    OUTPUT_NAME hip_logging
    VERSION 1.0.0
    SOVERSION 1
    C_VISIBILITY_PRESET default
  )
  set_target_properties(libhrx_src_passthrough_hip_noop PROPERTIES
    OUTPUT_NAME hip_noop
    VERSION 1.0.0
    SOVERSION 1
    C_VISIBILITY_PRESET default
  )
  set_target_properties(libhrx_src_passthrough_hip_buffer_tracer PROPERTIES
    OUTPUT_NAME hip_buffer_tracer
    VERSION 1.0.0
    SOVERSION 1
    C_VISIBILITY_PRESET default
  )
  target_compile_options(libhrx_src_passthrough_hip_intercept PRIVATE
    -fvisibility=default
  )
  target_compile_options(libhrx_src_passthrough_hip_logging PRIVATE
    -fvisibility=default
  )
  target_compile_options(libhrx_src_passthrough_hip_noop PRIVATE
    -fvisibility=default
  )
  target_compile_options(libhrx_src_passthrough_hip_buffer_tracer PRIVATE
    -fvisibility=default
  )
endif()
install(FILES "${_IREE_PACKAGE_SOURCE_DIR}/hip_function_table.h"
  DESTINATION include/passthrough
  COMPONENT HrxPassthroughDist
)
