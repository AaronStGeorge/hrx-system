// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nvfp4_matrix.h"

#include <loomcxx/target.h>

static loom::kernel::configuration configure_nvfp4_matrix() {
  return {{1, 1, 1}, {loom::target::subgroup::size(), 1, 1}};
}

[[loom::kernel(configure_nvfp4_matrix)]] void nvfp4_matrix(
    [[loom::noalias, loom::assume_aligned(8)]] const NvFp4Payload* lhs_payload,
    [[loom::noalias, loom::assume_aligned(4)]] const NvFp4Scale* lhs_scale,
    [[loom::noalias, loom::assume_aligned(8)]] const NvFp4Payload* rhs_payload,
    [[loom::noalias, loom::assume_aligned(4)]] const NvFp4Scale* rhs_scale,
    [[loom::noalias, loom::assume_aligned(64)]] float* output) {
  nvfp4_matrix_tile(*lhs_payload, *lhs_scale, *rhs_payload, *rhs_scale, output);
}
