// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Text low assembly interface.
//
// The text parser owns the surface syntax for low asm regions, but target
// descriptor tables and canonical low operation builders live outside the text
// package. This interface keeps that boundary explicit: parser/printer code
// receives packet shapes, build hooks, and print hooks from an environment
// supplied by the tool or compiler pipeline.

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
  // Surface mnemonic emitted or parsed for this asm packet.
  iree_string_view_t mnemonic;
  // Number of SSA results produced by this asm packet.
  uint16_t result_count;
  // Number of SSA operands consumed by this asm packet.
  uint16_t operand_count;
  // Number of immediate attributes parsed by this asm packet.
  uint16_t immediate_count;
  // Operation attribute field index storing packet immediate attributes.
  uint16_t immediate_attribute_field_index;
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

typedef enum loom_text_low_asm_statement_kind_e {
  // Unknown or uninitialized statement kind.
  LOOM_TEXT_LOW_ASM_STATEMENT_UNKNOWN = 0,
  // Descriptor-backed packet printed as an instruction mnemonic.
  LOOM_TEXT_LOW_ASM_STATEMENT_PACKET = 1,
  // Low return packet printed as `return`.
  LOOM_TEXT_LOW_ASM_STATEMENT_RETURN = 2,
} loom_text_low_asm_statement_kind_t;

typedef struct loom_text_low_asm_statement_t {
  // Statement kind describing which fields below are meaningful.
  loom_text_low_asm_statement_kind_t kind;
  // Original canonical operation represented by this asm statement.
  const loom_op_t* op;
  // Packet descriptor for descriptor-backed packet statements.
  loom_text_low_asm_packet_descriptor_t packet;
  // SSA results defined by a packet statement.
  const loom_value_id_t* results;
  // Number of SSA results in |results|.
  uint16_t result_count;
  // SSA operands consumed by a packet statement, or return values for returns.
  const loom_value_id_t* operands;
  // Number of SSA values in |operands|.
  uint16_t operand_count;
  // Canonical immediate attributes stored on the packet operation.
  loom_named_attr_slice_t attributes;
  // True when immediate attributes correspond to a concrete op attr field.
  bool has_immediate_attribute_field;
  // Operation attribute field index storing packet immediate attributes.
  uint16_t immediate_attribute_field_index;
  // Source location attached to the represented operation.
  loom_location_id_t location;
} loom_text_low_asm_statement_t;

typedef iree_status_t (*loom_text_low_asm_lookup_descriptor_set_fn_t)(
    void* user_data, iree_string_view_t key, const void** out_descriptor_set);

typedef iree_status_t (*loom_text_low_asm_lookup_packet_fn_t)(
    void* user_data, const void* descriptor_set, iree_string_view_t mnemonic,
    loom_text_low_asm_packet_descriptor_t* out_packet);

typedef iree_status_t (*loom_text_low_asm_infer_result_type_fn_t)(
    void* user_data, const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    uint16_t result_index, loom_module_t* module, loom_type_t* out_type,
    iree_string_view_t* out_diagnostic_detail);

typedef iree_status_t (*loom_text_low_asm_validate_result_type_fn_t)(
    void* user_data, const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    uint16_t result_index, loom_module_t* module, loom_type_t type,
    iree_string_view_t* out_diagnostic_detail);

typedef iree_status_t (*loom_text_low_asm_result_type_annotation_required_fn_t)(
    void* user_data, const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    uint16_t result_index, const loom_module_t* module, loom_type_t type,
    bool* out_required, iree_string_view_t* out_diagnostic_detail);

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

typedef iree_status_t (*loom_text_low_asm_describe_operation_fn_t)(
    void* user_data, const void* descriptor_set, const loom_module_t* module,
    const loom_op_t* op, loom_text_low_asm_statement_t* out_statement);

typedef struct loom_text_low_asm_vtable_t {
  // Resolves an `asm<...>` descriptor-set key to an environment-owned handle.
  loom_text_low_asm_lookup_descriptor_set_fn_t lookup_descriptor_set;
  // Resolves a mnemonic within a descriptor-set handle to a packet descriptor.
  loom_text_low_asm_lookup_packet_fn_t lookup_packet;
  // Infers a result type when the asm packet omits explicit type annotations.
  loom_text_low_asm_infer_result_type_fn_t infer_result_type;
  // Validates an explicit asm result type annotation against the descriptor.
  loom_text_low_asm_validate_result_type_fn_t validate_result_type;
  // Returns whether printing must emit an explicit result type annotation.
  loom_text_low_asm_result_type_annotation_required_fn_t
      result_type_annotation_required;
  // Returns canonical field and surface spelling metadata for one immediate.
  loom_text_low_asm_immediate_descriptor_fn_t immediate_descriptor;
  // Builds the canonical low operation for a parsed non-return asm packet.
  loom_text_low_asm_build_packet_fn_t build_packet;
  // Builds the canonical low return operation for an asm `return` packet.
  loom_text_low_asm_build_return_fn_t build_return;
  // Describes a canonical operation as a printable low asm statement.
  loom_text_low_asm_describe_operation_fn_t describe_operation;
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
         environment->vtable->validate_result_type &&
         environment->vtable->immediate_descriptor &&
         environment->vtable->build_packet && environment->vtable->build_return;
}

static inline bool loom_text_low_asm_environment_supports_printing(
    const loom_text_low_asm_environment_t* environment) {
  return environment && environment->vtable &&
         environment->vtable->lookup_descriptor_set &&
         environment->vtable->result_type_annotation_required &&
         environment->vtable->immediate_descriptor &&
         environment->vtable->describe_operation;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_LOW_ASM_H_
