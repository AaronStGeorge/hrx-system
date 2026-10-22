// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_BINDING_C_BENCHMARK_IMPORT_CXX_INPUTS_Q8S32_SPECIALIZATION_H_
#define LOOM_BINDING_C_BENCHMARK_IMPORT_CXX_INPUTS_Q8S32_SPECIALIZATION_H_

#include <loomcxx/kernel.h>
#include <loomcxx/predicate.h>

#include <stdfloat>

using Q8S32Codes16 = signed char __attribute__((ext_vector_type(16)));
using Q8S32BFloat16 = std::bfloat16_t __attribute__((ext_vector_type(16)));

// Selects a target- and capacity-specific projection while preserving one
// stable kernel-facing signature. Each lane consumes one complete 32-weight
// scale block split into two native vector values.
LOOM_DEVICE LOOM_TEMPLATE_DECL("model.q8s32.project") float q8s32_project(
    unsigned input_capacity, Q8S32Codes16 weights_low,
    Q8S32Codes16 weights_high, Q8S32BFloat16 activations_low,
    Q8S32BFloat16 activations_high, std::bfloat16_t scale)
    [[loom::where(loom::predicate::range(input_capacity, 256u, 32768u) &&
                  loom::predicate::multiple_of(input_capacity, 256u))]];

#endif  // LOOM_BINDING_C_BENCHMARK_IMPORT_CXX_INPUTS_Q8S32_SPECIALIZATION_H_
