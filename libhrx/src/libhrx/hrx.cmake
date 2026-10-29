# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(IREE_HAL_DRIVER_AMDGPU)
  target_link_libraries(libhrx_src_libhrx_hrx PRIVATE
    iree::hal::drivers::amdgpu::registration
  )
  target_compile_definitions(libhrx_src_libhrx_hrx PRIVATE
    HRX_HAS_IREE_AMDGPU_DRIVER
  )
endif()
foreach(_hrx_target
    libhrx_src_libhrx_hrx_static
    libhrx_src_libhrx_hrx)
  foreach(_hrx_generated_header_target
      op_table_gen
      encoding_table_gen)
    if(TARGET ${_hrx_generated_header_target})
      add_dependencies(${_hrx_target} ${_hrx_generated_header_target})
    endif()
  endforeach()
endforeach()
set_target_properties(libhrx_src_libhrx_hrx PROPERTIES
  OUTPUT_NAME hrx
  VERSION ${PROJECT_VERSION}
  SOVERSION 0
  EXPORT_NAME hrx
  POSITION_INDEPENDENT_CODE ON
)
if(APPLE)
  set_target_properties(libhrx_src_libhrx_hrx PROPERTIES
    BUILD_RPATH "@loader_path"
    INSTALL_RPATH "@loader_path"
    C_VISIBILITY_PRESET hidden
  )
  set(_hrx_export_file "${LIBHRX_SOURCE_DIR}/cmake/hrx_exports_macos.txt")
  target_link_options(libhrx_src_libhrx_hrx PRIVATE
    "-Wl,-exported_symbols_list,${_hrx_export_file}"
  )
  set_property(TARGET libhrx_src_libhrx_hrx APPEND PROPERTY
    LINK_DEPENDS "${_hrx_export_file}"
  )
elseif(UNIX)
  set_target_properties(libhrx_src_libhrx_hrx PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
    C_VISIBILITY_PRESET hidden
  )
  set(_hrx_export_file "${LIBHRX_SOURCE_DIR}/cmake/hrx_exports.lds")
  target_link_options(libhrx_src_libhrx_hrx PRIVATE
    "-Wl,--version-script=${_hrx_export_file}"
  )
  set_property(TARGET libhrx_src_libhrx_hrx APPEND PROPERTY
    LINK_DEPENDS "${_hrx_export_file}"
  )
endif()
set_property(TARGET libhrx_src_libhrx_hrx PROPERTY
  INTERFACE_LINK_LIBRARIES ""
)
if(NOT TARGET hrx)
  add_custom_target(hrx DEPENDS libhrx_src_libhrx_hrx)
endif()
if(NOT TARGET hrx::hrx)
  iree_add_alias_library(hrx::hrx libhrx_src_libhrx_hrx)
endif()
