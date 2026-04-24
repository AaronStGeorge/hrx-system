// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Storage-schema fact adapters for generic contract requests.

#ifndef LOOM_ANALYSIS_CONTRACT_STORAGE_H_
#define LOOM_ANALYSIS_CONTRACT_STORAGE_H_

#include "loom/analysis/contract.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maps a matrix storage-schema format to a generic contract numeric type.
bool loom_contract_numeric_type_from_matrix_format(
    loom_value_fact_matrix_format_t format,
    loom_contract_numeric_type_t* out_numeric_type);

// Maps a matrix storage-schema scale kind to a generic contract scale kind.
bool loom_contract_scale_kind_from_storage_schema(
    loom_value_fact_storage_schema_t schema,
    loom_contract_scale_kind_t* out_scale_kind);

// Returns generic contract capability facts proven by a storage schema.
loom_contract_capability_flags_t
loom_contract_capability_flags_from_storage_schema(
    loom_value_fact_storage_schema_t schema);

// Maps a matrix storage-schema payload to a generic contract operand.
bool loom_contract_operand_from_storage_schema(
    loom_contract_operand_role_t role, loom_value_fact_storage_schema_t schema,
    loom_contract_operand_t* out_operand);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_CONTRACT_STORAGE_H_
