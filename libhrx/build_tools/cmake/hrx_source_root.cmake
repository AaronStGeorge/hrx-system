# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(LIBHRX_SOURCE_DIR "${IREE_SOURCE_DIR}/libhrx")
set(LIBHRX_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
iree_setup_c_src_root(PACKAGE_ROOT_PREFIX libhrx)
