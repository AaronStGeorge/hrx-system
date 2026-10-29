# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_cc_library(
  NAME
    provider_set
  HDRS
    "provider_set.h"
  DEPS
    iree_defs
    loom_defs
    loom::binding::c::loomc
    loom::target::provider
  PUBLIC
)
