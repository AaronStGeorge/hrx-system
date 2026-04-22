// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Private AMDGPU source-to-target-low lowering helpers.
//
// This header is target-local glue between lower.c and focused leaf lowerers.
// Keep declarations here narrow: a new declaration should mean two AMDGPU
// lowering invariant clusters genuinely share one contract.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_INTERNAL_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_INTERNAL_H_

#include <stdint.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/low/ops.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES 4u

// Returns true when the source type is a scalar i32.
bool loom_amdgpu_type_is_i32(loom_type_t type);

// Returns true when the source type is an address-sized scalar lowered through
// the current 32-bit AMDGPU scalar path.
bool loom_amdgpu_type_is_address_scalar(loom_type_t type);

// Returns true when the source type is a scalar f32.
bool loom_amdgpu_type_is_f32(loom_type_t type);

// Returns a static rank-1 vector lane count for the requested element type, or
// zero when the type is not a supported static rank-1 vector.
uint32_t loom_amdgpu_static_vector_lane_count(loom_type_t type,
                                              loom_scalar_type_t element_type,
                                              uint32_t max_lane_count);

// Returns the lane count for a supported AMDGPU 32-bit vector payload, or zero
// when the source type is not representable as that payload.
uint32_t loom_amdgpu_vector_32bit_lane_count(loom_type_t type);

// Returns the i32 lane count for a supported AMDGPU 32-bit vector payload, or
// zero when the source type is not representable as that payload.
uint32_t loom_amdgpu_vector_i32_lane_count(loom_type_t type);

// Returns the f32 lane count for a supported AMDGPU 32-bit vector payload, or
// zero when the source type is not representable as that payload.
uint32_t loom_amdgpu_vector_f32_lane_count(loom_type_t type);

// Returns true when the source type is a 32-bit-element view that can map to an
// AMDGPU HAL/global buffer resource or LDS root.
bool loom_amdgpu_type_is_32bit_view(loom_type_t type);

// Returns true when a source value has scalar i32 type.
bool loom_amdgpu_value_is_i32(loom_low_lower_context_t* context,
                              loom_value_id_t value_id);

// Returns true when a source value has an address scalar type.
bool loom_amdgpu_value_is_address_scalar(loom_low_lower_context_t* context,
                                         loom_value_id_t value_id);

// Returns true when a source value has scalar f32 type.
bool loom_amdgpu_value_is_f32(loom_low_lower_context_t* context,
                              loom_value_id_t value_id);

// Returns true when a source value is a supported rank-1 i32/f32 vector
// payload.
bool loom_amdgpu_value_is_vector_32bit_lane_range(
    loom_low_lower_context_t* context, loom_value_id_t value_id);

// Returns true when a source value has a 32-bit-element view type.
bool loom_amdgpu_value_is_32bit_view(loom_low_lower_context_t* context,
                                     loom_value_id_t value_id);

// Builds a one-unit SGPR register type in the current lowering context.
iree_status_t loom_amdgpu_make_sgpr_type(loom_low_lower_context_t* context,
                                         loom_type_t* out_type);

// Builds a multi-unit SGPR register type in the current lowering context.
iree_status_t loom_amdgpu_make_sgpr_range_type(
    loom_low_lower_context_t* context, uint32_t unit_count,
    loom_type_t* out_type);

// Builds a one-unit VGPR register type in the current lowering context.
iree_status_t loom_amdgpu_make_vgpr_type(loom_low_lower_context_t* context,
                                         loom_type_t* out_type);

// Builds the register type for a descriptor's implicit resource operand.
iree_status_t loom_amdgpu_make_descriptor_implicit_resource_type(
    loom_low_lower_context_t* context, uint64_t descriptor_id,
    loom_type_t* out_type);

// Returns whether a low register type belongs to the requested AMDGPU register
// class.
iree_status_t loom_amdgpu_low_type_register_class_is(
    loom_low_lower_context_t* context, loom_type_t type, uint16_t reg_class_id,
    bool* out_match);

// Returns true when the source value should prefer a VGPR mapping even if its
// scalar type could otherwise map to an SGPR.
bool loom_amdgpu_value_prefers_vgpr(loom_low_lower_context_t* context,
                                    loom_value_id_t source_value_id);

// Maps a source type to the default AMDGPU low register type.
iree_status_t loom_amdgpu_map_type(void* user_data,
                                   loom_low_lower_context_t* context,
                                   const loom_op_t* source_op,
                                   loom_type_t source_type,
                                   loom_type_t* out_low_type);

// Maps a source value to the AMDGPU low register type selected for its
// placement-sensitive use.
iree_status_t loom_amdgpu_map_value(void* user_data,
                                    loom_low_lower_context_t* context,
                                    const loom_op_t* source_op,
                                    loom_value_id_t source_value_id,
                                    loom_type_t source_type,
                                    loom_type_t* out_low_type);

// Maps one source function argument to the low ABI representation selected for
// the active AMDGPU bundle.
iree_status_t loom_amdgpu_map_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument);

// Extracts an exact index constant from a module value.
bool loom_amdgpu_module_value_as_exact_index_constant(
    const loom_module_t* module, loom_value_id_t value_id, int64_t* out_value);

// Extracts an exact non-negative signed 64-bit integer from value facts.
bool loom_amdgpu_value_facts_as_exact_non_negative_i64(loom_value_facts_t facts,
                                                       int64_t* out_value);

// Returns true when an attribute can encode as a signed 32-bit immediate.
bool loom_amdgpu_attr_is_i32_immediate(loom_attribute_t value);

// Returns true when an attribute can encode as an f32 immediate payload.
bool loom_amdgpu_attr_is_f32_immediate(loom_attribute_t value);

// Returns the f32 bit pattern produced by narrowing an attribute.
uint32_t loom_amdgpu_attr_f32_bit_pattern(loom_attribute_t value);

// Extracts a source scalar i32 constant.
bool loom_amdgpu_value_as_i32_constant(loom_low_lower_context_t* context,
                                       loom_value_id_t value_id,
                                       int64_t* out_value);

// Returns true when a source scalar i32 value can be materialized as a VGPR
// operand for vector-style packets.
bool loom_amdgpu_value_can_materialize_as_vgpr_i32(
    loom_low_lower_context_t* context, loom_value_id_t value_id);

// Looks up a lowered i32 value and materializes exact source constants into
// VGPRs when a vector-style packet cannot consume the existing lowering.
iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_i32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value);

// Returns true when value is a non-zero power of two.
bool loom_amdgpu_u32_is_power_of_two(uint32_t value);

// Interns a low lowering helper string in the active module.
iree_status_t loom_amdgpu_intern(loom_low_lower_context_t* context,
                                 iree_string_view_t string,
                                 loom_string_id_t* out_string_id);

// Maps a source result to the low register type already selected by the active
// lowering policy and verifies that it is a register payload.
iree_status_t loom_amdgpu_low_result_type(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t source_result,
                                          loom_type_t* out_low_type);

// Emits one descriptor-backed low.op with source provenance.
iree_status_t loom_amdgpu_emit_low_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, const loom_value_id_t* operands,
    iree_host_size_t operand_count, loom_named_attr_slice_t attrs,
    const loom_type_t* result_types, iree_host_size_t result_count,
    loom_op_t** out_low_op);

// Emits one descriptor-backed low.const with an imm32 attribute.
iree_status_t loom_amdgpu_emit_const_u32(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         uint64_t descriptor_id, uint32_t value,
                                         loom_type_t result_type,
                                         loom_value_id_t* out_value_id);

// Emits a low.slice from a register range.
iree_status_t loom_amdgpu_emit_low_slice(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_value_id_t source,
                                         uint32_t lane_offset,
                                         loom_type_t result_type,
                                         loom_value_id_t* out_value);

// Finds the M0 live-in value emitted for special-register memory packets.
iree_status_t loom_amdgpu_lookup_m0_live_in(loom_low_lower_context_t* context,
                                            loom_value_id_t* out_value_id);

// Returns true when lowering |source_op| selects a descriptor that consumes M0.
bool loom_amdgpu_source_op_selects_m0_descriptor(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t* out_descriptor_id);

// Returns true when the target bundle belongs to an AMDGPU contract set.
bool loom_amdgpu_low_legality_bundle_is_amdgpu(
    const loom_target_bundle_t* bundle);

// Returns true for source vector-dot ops handled by the AMDGPU dot lowerer.
bool loom_amdgpu_op_is_vector_dot(loom_op_kind_t kind);

// Returns true when a source vector.dot4i op can lower under the active AMDGPU
// descriptor set.
bool loom_amdgpu_can_lower_vector_dot4i(loom_low_lower_context_t* context,
                                        const loom_op_t* source_op);

// Lowers a source vector.dot4i op to AMDGPU descriptor-backed low packets.
iree_status_t loom_amdgpu_lower_vector_dot4i(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op);

// Returns true when a source vector.reduce op can lower through the current
// AMDGPU source value placement and supported reduction descriptor family.
bool loom_amdgpu_can_lower_vector_reduce(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op);

// Lowers a source vector.reduce op to AMDGPU descriptor-backed low packets.
iree_status_t loom_amdgpu_lower_vector_reduce(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op);

// Verifies source vector-dot legality for AMDGPU target-low selection.
iree_status_t loom_amdgpu_low_legality_verify_vector_dot(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Returns true when a vector.load can lower to one supported AMDGPU memory
// packet form under the active descriptor set.
bool loom_amdgpu_can_lower_vector_load(loom_low_lower_context_t* context,
                                       const loom_op_t* source_op);

// Returns true when a vector.store can lower to one supported AMDGPU memory
// packet form under the active descriptor set.
bool loom_amdgpu_can_lower_vector_store(loom_low_lower_context_t* context,
                                        const loom_op_t* source_op);

// Lowers a source vector.load to an AMDGPU memory packet.
iree_status_t loom_amdgpu_lower_vector_load(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op);

// Lowers a source vector.store to an AMDGPU memory packet.
iree_status_t loom_amdgpu_lower_vector_store(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op);

// Verifies source vector memory legality for AMDGPU target-low selection.
iree_status_t loom_amdgpu_low_legality_verify_vector_memory(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_INTERNAL_H_
