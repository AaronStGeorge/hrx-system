# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

function(iree_configure_google_benchmark)
  iree_dependency_package_discovery_allowed(_package_discovery_allowed)
  if(_package_discovery_allowed)
    find_package(benchmark CONFIG QUIET)
  endif()

  if(NOT TARGET benchmark AND NOT TARGET benchmark::benchmark)
    iree_dependency_require_pinned_source_allowed("google benchmark")
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    iree_declare_locked_fetch_content(google_benchmark)
    FetchContent_MakeAvailable(google_benchmark)
  endif()

  if(TARGET benchmark::benchmark)
    iree_add_alias_interface(benchmark benchmark::benchmark)
  endif()
  if(NOT TARGET benchmark)
    message(FATAL_ERROR "google benchmark did not provide a benchmark target")
  endif()

  iree_add_alias_interface(iree::third_party::google_benchmark benchmark)

  if(TARGET benchmark_main)
    iree_add_alias_interface(
      iree::third_party::google_benchmark_main benchmark_main)
  elseif(TARGET benchmark::benchmark_main)
    iree_add_alias_interface(
      iree::third_party::google_benchmark_main benchmark::benchmark_main)
  endif()
endfunction()
