# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Point HIP DSO tests at the shared object built in this tree so they dlopen()
# the exact artifact under test, and make sure that artifact is built first.
# The build-tree path is only valid for the in-tree CTest run.
if(TARGET libhrx_src_binding_hip_hip_dso_test_util)
  target_compile_definitions(libhrx_src_binding_hip_hip_dso_test_util
    PRIVATE
      "HRX_TEST_LIBAMDHIP64_PATH=\"$<TARGET_FILE:libhrx_src_binding_hip_amdhip64>\""
  )
  add_dependencies(libhrx_src_binding_hip_hip_dso_test_util
    libhrx_src_binding_hip_amdhip64
  )
endif()
foreach(_hrx_hip_dso_test
    libhrx_src_binding_hip_hip_api_name_test
    libhrx_src_binding_hip_hip_device_contracts_api_test
    libhrx_src_binding_hip_hip_device_attribute_no_device_api_test
    libhrx_src_binding_hip_hip_device_attribute_validation_api_test
    libhrx_src_binding_hip_hip_get_proc_address_test
    libhrx_src_binding_hip_hip_launch_validation_api_test
    libhrx_src_binding_hip_hip_unified_addressing_attribute_api_test)
  if(TARGET ${_hrx_hip_dso_test})
    target_compile_definitions(${_hrx_hip_dso_test}
      PRIVATE
        "HRX_TEST_LIBAMDHIP64_PATH=\"$<TARGET_FILE:libhrx_src_binding_hip_amdhip64>\""
    )
    add_dependencies(${_hrx_hip_dso_test}
      libhrx_src_binding_hip_amdhip64
    )
  endif()
endforeach()

# The array-copy suite requires its test-only lease observer even when invoked
# directly without the CTest-provided HRX_TEST_LIBAMDHIP64 environment variable.
if(TARGET libhrx_src_binding_hip_hip_array_copy_spt_api_test)
  target_compile_definitions(
    libhrx_src_binding_hip_hip_array_copy_spt_api_test
    PRIVATE
      "HRX_TEST_LIBAMDHIP64_PATH=\"$<TARGET_FILE:libhrx_src_binding_hip_amdhip64_array_lease_test>\""
  )
  add_dependencies(
    libhrx_src_binding_hip_hip_array_copy_spt_api_test
    libhrx_src_binding_hip_amdhip64_array_lease_test
  )
endif()

# BUILD.bazel's linkopts own the symbol version script and --undefined-version.
# A second version script in LINK_FLAGS would apply it twice.
if(TARGET libhrx_src_binding_hip_amdhip64)
  install(TARGETS libhrx_src_binding_hip_amdhip64
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}"
      NAMELINK_COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
      COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}")
endif()
