// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Parser-facing low assembly interface.
//
// The text parser owns the surface syntax for low asm regions, but target
// descriptor tables and canonical low operation builders live outside the text
// package. This interface keeps that boundary explicit: parser/printer code
// receives packet shapes and build hooks from an environment supplied by the
// tool or compiler pipeline.

#ifndef LOOM_FORMAT_TEXT_LOW_ASM_H_
#define LOOM_FORMAT_TEXT_LOW_ASM_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_text_low_asm_packet_descriptor_t {
  // Opaque descriptor-set handle owned by the environment implementation.
  const void* descriptor_set;
  // Opaque asm-form handle owned by the environment implementation.
  const void* form;
  // Opaque low descriptor handle owned by the environment implementation.
  const void* descriptor;
  // Stable canonical opcode key to store on the emitted low operation.
  iree_string_view_t opcode_key;
  // Number of SSA results produced by this asm packet.
  uint16_t result_count;
  // Number of SSA operands consumed by this asm packet.
  uint16_t operand_count;
  // Number of immediate attributes parsed by this asm packet.
  uint16_t immediate_count;
  // True when at least one immediate requires named-dictionary syntax.
  bool has_named_immediates;
  // True when this packet should canonicalize to a low const operation.
  bool builds_as_const;
} loom_text_low_asm_packet_descriptor_t;

typedef struct loom_text_low_asm_immediate_descriptor_t {
  // Canonical low attribute field name stored in the emitted operation.
  iree_string_view_t field_name;
  // Surface spelling accepted by the asm parser for this immediate.
  iree_string_view_t spelling;
} loom_text_low_asm_immediate_descriptor_t;

typedef iree_status_t (*loom_text_low_asm_lookup_descriptor_set_fn_t)(
    void* user_data, iree_string_view_t key, const void** out_descriptor_set);

typedef iree_status_t (*loom_text_low_asm_lookup_packet_fn_t)(
    void* user_data, const void* descriptor_set, iree_string_view_t mnemonic,
    loom_text_low_asm_packet_descriptor_t* out_packet);

typedef iree_status_t (*loom_text_low_asm_infer_result_type_fn_t)(
    void* user_data, const loom_text_low_asm_packet_descriptor_t* packet,
    uint16_t result_index, loom_module_t* module, loom_type_t* out_type,
    iree_string_view_t* out_diagnostic_detail);

typedef iree_status_t (*loom_text_low_asm_immediate_descriptor_fn_t)(
    void* user_data, const loom_text_low_asm_packet_descriptor_t* packet,
    uint16_t immediate_index,
    loom_text_low_asm_immediate_descriptor_t* out_immediate);

typedef iree_status_t (*loom_text_low_asm_build_packet_fn_t)(
    void* user_data, loom_builder_t* builder,
    const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attributes, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_location_id_t location,
    loom_op_t** out_op);

typedef iree_status_t (*loom_text_low_asm_build_return_fn_t)(
    void* user_data, loom_builder_t* builder, const loom_value_id_t* values,
    iree_host_size_t value_count, loom_location_id_t location,
    loom_op_t** out_op);

typedef struct loom_text_low_asm_vtable_t {
  // Resolves an `asm<...>` descriptor-set key to an environment-owned handle.
  loom_text_low_asm_lookup_descriptor_set_fn_t lookup_descriptor_set;
  // Resolves a mnemonic within a descriptor-set handle to a packet descriptor.
  loom_text_low_asm_lookup_packet_fn_t lookup_packet;
  // Infers a result type when the asm packet omits explicit type annotations.
  loom_text_low_asm_infer_result_type_fn_t infer_result_type;
  // Returns canonical field and surface spelling metadata for one immediate.
  loom_text_low_asm_immediate_descriptor_fn_t immediate_descriptor;
  // Builds the canonical low operation for a parsed non-return asm packet.
  loom_text_low_asm_build_packet_fn_t build_packet;
  // Builds the canonical low return operation for an asm `return` packet.
  loom_text_low_asm_build_return_fn_t build_return;
} loom_text_low_asm_vtable_t;

typedef struct loom_text_low_asm_environment_t {
  // Function table implementing low asm lookup, type inference, and builders.
  const loom_text_low_asm_vtable_t* vtable;
  // Opaque environment state passed to every vtable callback.
  void* user_data;
} loom_text_low_asm_environment_t;

static inline bool loom_text_low_asm_environment_is_configured(
    const loom_text_low_asm_environment_t* environment) {
  return environment && environment->vtable &&
         environment->vtable->lookup_descriptor_set &&
         environment->vtable->lookup_packet &&
         environment->vtable->infer_result_type &&
         environment->vtable->immediate_descriptor &&
         environment->vtable->build_packet && environment->vtable->build_return;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_LOW_ASM_H_
