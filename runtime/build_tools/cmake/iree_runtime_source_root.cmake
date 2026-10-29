# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_setup_c_src_root(
  DEFAULT_EXPORT_SET Runtime
  DEFAULT_INSTALL_COMPONENT IREEDevLibraries-Runtime
)

# Native file requirements use the syscall metadata definitions independently
# of any device provider or hardware test configuration.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  include("${PROJECT_SOURCE_DIR}/build_tools/third_party/linux_uapi/linux_uapi.cmake")
  iree_configure_linux_io_uapi()
endif()

# Runtime coverage and dynamic-dispatch instrumentation policy.
if(IREE_ENABLE_RUNTIME_COVERAGE)
  message(WARNING
    "IREE_ENABLE_RUNTIME_COVERAGE enabling coverage in all runtime libraries. "
    "All runtime binaries are instrumented and should not be used for "
    "benchmarking."
  )
  add_compile_options(
    "-fprofile-instr-generate"
    "-fcoverage-mapping"
  )
  add_link_options(
    "-fprofile-instr-generate"
    "-fcoverage-mapping"
  )
endif(IREE_ENABLE_RUNTIME_COVERAGE)

if(IREE_ENABLE_UBSAN)
  # Type-erased dispatch uses compatible calling conventions without exact
  # function-type identity. Only Clang implements the function sanitizer;
  # other compilers must not receive its unsupported suppression option.
  add_compile_options(
    "$<$<COMPILE_LANG_AND_ID:C,Clang,AppleClang>:-fno-sanitize=function>"
    "$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-fno-sanitize=function>"
  )
endif(IREE_ENABLE_UBSAN)
