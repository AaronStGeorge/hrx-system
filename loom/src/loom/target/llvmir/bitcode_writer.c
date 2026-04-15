// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/llvmir/bitcode_writer.h"

iree_status_t loom_llvmir_bitcode_write_module(
    const loom_llvmir_module_t* module, loom_output_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(stream);
  (void)module;
  (void)stream;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "LLVM bitcode writer is not implemented");
}
