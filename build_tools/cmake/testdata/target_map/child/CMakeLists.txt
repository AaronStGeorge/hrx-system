# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_executable(fixture_program main.c)
set_target_properties(fixture_program PROPERTIES
  OUTPUT_NAME "${FIXTURE_OUTPUT_NAME}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin with spaces/$<CONFIG>")
