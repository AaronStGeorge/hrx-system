// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TEST_CORPUS_AUTHORING_CXX_NVFP4_MATRIX_H_
#define LOOM_TEST_CORPUS_AUTHORING_CXX_NVFP4_MATRIX_H_

#include <loomcxx/kernel.h>

using NvFp4Payload = unsigned __attribute__((ext_vector_type(2)));
using NvFp4Scale = unsigned __attribute__((ext_vector_type(1)));

// Multiplies one logical 16x16x16 NVFP4 tile. The encoded values and scales
// form the stable family boundary; each target provider owns its physical
// operand and accumulator carriers.
LOOM_DEVICE LOOM_TEMPLATE_DECL(
    "model.nvfp4.matrix_tile") void nvfp4_matrix_tile(NvFp4Payload lhs_payload,
                                                      NvFp4Scale lhs_scale,
                                                      NvFp4Payload rhs_payload,
                                                      NvFp4Scale rhs_scale,
                                                      float* output);

#endif  // LOOM_TEST_CORPUS_AUTHORING_CXX_NVFP4_MATRIX_H_
