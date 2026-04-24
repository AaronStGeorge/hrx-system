// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vector op adapters for generic contract requests.

#ifndef LOOM_ANALYSIS_CONTRACT_VECTOR_H_
#define LOOM_ANALYSIS_CONTRACT_VECTOR_H_

#include "loom/analysis/contract.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maps vector dot ops to generic packed-vector contract requests.
//
// This helper preserves only target-independent structure: vector lane
// grouping, payload numeric facts, accumulator/result types, and lowering
// policy. Target feature bits and descriptor names remain target-owned
// projection outputs.
bool loom_contract_request_from_vector_dot_op(
    const loom_module_t* module, const loom_op_t* op,
    loom_lowering_policy_t policy, loom_contract_request_t* out_request,
    loom_contract_diagnostic_t* out_diagnostic);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_CONTRACT_VECTOR_H_
