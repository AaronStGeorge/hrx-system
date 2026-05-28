# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_executable(iree-c-embed-data embed_data.cc)
set_target_properties(iree-c-embed-data PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  FOLDER "${HRX_IDE_FOLDER}/build_tools"
  RUNTIME_OUTPUT_DIRECTORY "${IREE_BINARY_DIR}/build_tools"
)
