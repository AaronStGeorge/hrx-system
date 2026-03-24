// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/func/ops.h"
#include "loom/testing/gen.h"

//===----------------------------------------------------------------------===//
// Hook table
//===----------------------------------------------------------------------===//

const loom_test_gen_op_hook_t* loom_test_gen_func_hooks(
    iree_host_size_t* out_hook_count) {
  *out_hook_count = 0;
  return NULL;
}
