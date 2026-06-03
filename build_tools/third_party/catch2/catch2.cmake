# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

function(iree_configure_catch2)
  iree_dependency_package_discovery_allowed(_package_discovery_allowed)
  if(_package_discovery_allowed)
    find_package(Catch2 3 CONFIG QUIET)
  endif()

  if(NOT TARGET Catch2::Catch2)
    iree_dependency_require_pinned_source_allowed("Catch2")
    set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
    set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "" FORCE)
    set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    iree_declare_locked_fetch_content(catch2)
    FetchContent_MakeAvailable(catch2)
  endif()

  if(NOT TARGET Catch2::Catch2)
    message(FATAL_ERROR "Catch2 did not provide a Catch2::Catch2 target")
  endif()
  iree_add_alias_interface(iree::third_party::catch2 Catch2::Catch2)
endfunction()
