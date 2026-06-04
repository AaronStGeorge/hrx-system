# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_MODULE_VMVX)
  return()
endif()

iree_cc_library(
  NAME
    vmvx
  HDRS
    "module.h"
  TEXTUAL_HDRS
    "exports.inl"
  SRCS
    "module.c"
  DEFINES
    "IREE_HAVE_VMVX_MODULE"
  DEPS
    iree::base
    iree::vm
  PUBLIC
)
