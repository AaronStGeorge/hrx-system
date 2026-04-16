// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bridges Loom storage-schema facts to AMDGPU matrix contract requests.
//
// Encoding facts remain target-independent analysis payloads. This helper owns
// the AMDGPU interpretation: selector values, payload numeric types, explicit
// scale kinds, and which descriptor flag classes are proven available.

#ifndef LOOM_TARGET_AMDGPU_MATRIX_FACTS_H_
#define LOOM_TARGET_AMDGPU_MATRIX_FACTS_H_

#include "loom/target/amdgpu/matrix_contract.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maps a storage-schema matrix format to the AMDGPU matrix-format selector
// immediate used by scaled MFMA/WMMA selector-family contracts.
bool loom_amdgpu_matrix_format_selector_from_storage_schema(
    loom_value_fact_storage_schema_t schema,
    loom_amdgpu_matrix_format_selector_t* out_selector);

// Maps a storage-schema matrix payload to the exact numeric payload shape known
// before descriptor-family matching. Selector-family descriptors such as
// f8f6f4 match this exact payload when matrix-format facts are available.
bool loom_amdgpu_matrix_payload_from_storage_schema(
    loom_value_fact_storage_schema_t schema,
    loom_amdgpu_matrix_payload_shape_t* out_payload);

// Maps storage-schema scale facts to the AMDGPU contract scale kind.
bool loom_amdgpu_matrix_scale_kind_from_storage_schema(
    loom_value_fact_storage_schema_t schema,
    loom_amdgpu_matrix_scale_kind_t* out_scale_kind);

// Returns descriptor flag classes proven available by a storage schema.
loom_amdgpu_matrix_contract_flags_t
loom_amdgpu_matrix_available_flags_from_storage_schema(
    loom_value_fact_storage_schema_t schema);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_AMDGPU_MATRIX_FACTS_H_
