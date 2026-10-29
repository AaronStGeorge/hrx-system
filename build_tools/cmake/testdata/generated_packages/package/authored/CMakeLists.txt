# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

project(authored_package C)
add_library(authored_usage INTERFACE IMPORTED)
set_target_properties(authored_usage PROPERTIES
  INTERFACE_COMPILE_DEFINITIONS "AUTHORED_VALUE=23")
iree_add_subdirectory(child)
return()
message(FATAL_ERROR "An authored package return must exit its directory")
