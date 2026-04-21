// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor-resolved builders for target-low packets.
//
// These helpers construct descriptor-backed low ops from durable descriptor
// IDs after target binding has selected a concrete descriptor set. Text parsing
// remains the only path that resolves descriptor names from source text.

#ifndef LOOM_CODEGEN_LOW_BUILDER_H_
#define LOOM_CODEGEN_LOW_BUILDER_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ops/low/ops.h"

#ifdef __cplusplus
extern "C" {
#endif

// Interns the descriptor-set register-class spelling in |module|.
//
// |reg_class_id| is descriptor-set-local, not a module string ID. The returned
// string ID is suitable for constructing reg<...> types while keeping lowering
// code keyed on generated register-class ID constants.
iree_status_t loom_low_build_register_class_string_id(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    uint16_t reg_class_id, loom_string_id_t* out_string_id);

// Creates a reg<...> type selected by descriptor-set register-class ID.
iree_status_t loom_low_build_register_type(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    uint16_t reg_class_id, uint32_t unit_count, loom_type_t* out_type);

// Emits a descriptor-backed low.op selected by stable descriptor ID.
//
// |descriptor_set| is the selected target-low descriptor set and is used to
// recover the packet's textual descriptor spelling for IR printing and
// diagnostics. Descriptor identity is still carried by the descriptor_id attr,
// so compiled lowerings never need to hash descriptor strings.
iree_status_t loom_low_build_descriptor_op(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint64_t descriptor_id, const loom_value_id_t* operands,
    iree_host_size_t operand_count, loom_named_attr_slice_t attrs,
    const loom_type_t* result_types, iree_host_size_t result_count,
    const loom_tied_result_t* tied_results, iree_host_size_t tied_result_count,
    loom_location_id_t location, loom_op_t** out_op);

// Emits a descriptor-backed low.const selected by stable descriptor ID.
iree_status_t loom_low_build_descriptor_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint64_t descriptor_id, loom_named_attr_slice_t attrs,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_BUILDER_H_
