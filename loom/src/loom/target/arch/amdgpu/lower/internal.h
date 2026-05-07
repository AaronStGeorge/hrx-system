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
#include "loom/target/arch/amdgpu/lower/materializers.h"
#include "loom/target/arch/amdgpu/lower/plan.h"
#include "loom/target/arch/amdgpu/target_refs.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when the source type is a scalar i32.
bool loom_amdgpu_type_is_i32(loom_type_t type);

// Returns true when the source type is a scalar i64.
bool loom_amdgpu_type_is_i64(loom_type_t type);

// Returns true when the source type is a scalar i1.
bool loom_amdgpu_type_is_i1(loom_type_t type);

// Returns true when the source type is an address-sized scalar lowered through
// the current 32-bit AMDGPU scalar path.
bool loom_amdgpu_type_is_address_scalar(loom_type_t type);

// Returns true when the source type is a scalar f32.
bool loom_amdgpu_type_is_f32(loom_type_t type);

// Returns true when the source type is a scalar f16 or bf16.
bool loom_amdgpu_type_is_16bit_float(loom_type_t type);

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

// Returns true when the source type can be loaded through scalar memory and
// left in SGPR form.
bool loom_amdgpu_type_is_i32_memory_payload(loom_type_t type);

// Returns the f32 lane count for a supported AMDGPU 32-bit vector payload, or
// zero when the source type is not representable as that payload.
uint32_t loom_amdgpu_vector_f32_lane_count(loom_type_t type);

// Returns the i1 lane count for a supported AMDGPU mask vector payload, or zero
// when the source type is not representable as that payload.
uint32_t loom_amdgpu_vector_i1_lane_count(loom_type_t type);

// Returns the i8 lane count for a supported AMDGPU packed byte payload, or zero
// when the source type is not representable as that payload.
uint32_t loom_amdgpu_vector_i8_lane_count(loom_type_t type);

// Returns true when the source type is an integer vector payload that can be
// stored in packed 32-bit registers. The payload occupies the low bits of the
// register range; unused high bits in the final register are unspecified.
bool loom_amdgpu_type_packed_integer_storage(loom_type_t type,
                                             uint32_t* out_payload_bit_count,
                                             uint32_t* out_register_count);

// Returns true when the source type is an f16/bf16 vector payload that can be
// stored in packed 32-bit registers. Odd lane counts occupy the low half of the
// final register; the high half is unspecified.
bool loom_amdgpu_type_packed_16bit_float_storage(
    loom_type_t type, uint32_t* out_payload_bit_count,
    uint32_t* out_register_count);

// Returns true when the source type is a byte-addressable view that can map to
// an AMDGPU HAL/global buffer resource or LDS root.
bool loom_amdgpu_type_is_byte_addressable_view(loom_type_t type);

// Returns true when a source value has scalar i32 type.
bool loom_amdgpu_value_is_i32(loom_low_lower_context_t* context,
                              loom_value_id_t value_id);

// Returns true when a source value has an address scalar type.
bool loom_amdgpu_value_is_address_scalar(loom_low_lower_context_t* context,
                                         loom_value_id_t value_id);

// Returns true when a source value has scalar f32 type.
bool loom_amdgpu_value_is_f32(loom_low_lower_context_t* context,
                              loom_value_id_t value_id);

// Returns true when a source value has scalar f16 or bf16 type.
bool loom_amdgpu_value_is_16bit_float(loom_low_lower_context_t* context,
                                      loom_value_id_t value_id);

// Returns true when a source value has a byte-addressable view type.
bool loom_amdgpu_value_is_byte_addressable_view(
    loom_low_lower_context_t* context, loom_value_id_t value_id);

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

// Builds the one-unit SCC register type in the current lowering context.
iree_status_t loom_amdgpu_make_scc_type(loom_low_lower_context_t* context,
                                        loom_type_t* out_type);

// Builds a multi-unit VGPR register type in the current lowering context.
iree_status_t loom_amdgpu_make_vgpr_range_type(
    loom_low_lower_context_t* context, uint32_t unit_count,
    loom_type_t* out_type);

// Builds the register type for a descriptor row's implicit resource operand.
iree_status_t loom_amdgpu_make_descriptor_row_implicit_resource_type(
    loom_low_lower_context_t* context, const loom_low_descriptor_t* descriptor,
    loom_type_t* out_type);

// Returns whether a low register type belongs to the requested AMDGPU register
// class.
iree_status_t loom_amdgpu_low_type_register_class_is(
    loom_low_lower_context_t* context, loom_type_t type, uint16_t reg_class_id,
    bool* out_match);

// Returns true when the module source value should prefer a VGPR mapping even
// if its scalar type could otherwise map to an SGPR.
bool loom_amdgpu_module_value_prefers_vgpr(const loom_module_t* module,
                                           loom_value_id_t source_value_id);

// Returns true when the module source value is an i1 represented by an
// EXEC-width native lane mask.
bool loom_amdgpu_module_value_is_native_i1_mask(
    const loom_module_t* module, loom_value_id_t source_value_id);

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

// Maps a source value to AMDGPU descriptor register metadata for read-only
// target contract queries.
iree_status_t loom_amdgpu_map_contract_value(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value);

// Maps one source function argument to the low ABI representation selected for
// the active AMDGPU bundle.
iree_status_t loom_amdgpu_map_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument);

// Plans divergent branch expansion before the source body is emitted.
iree_status_t loom_amdgpu_prepare_branch(void* user_data,
                                         loom_low_lower_context_t* context,
                                         const loom_op_t* source_terminator);

// Emits a conditional branch, using EXEC narrowing for divergent SGPR masks.
iree_status_t loom_amdgpu_emit_cond_branch(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_value_id_t low_condition,
                                           loom_block_t* low_true_dest,
                                           loom_block_t* low_false_dest);

// Extracts an exact index constant from a module value.
bool loom_amdgpu_module_value_as_exact_index_constant(
    const loom_module_t* module, loom_value_id_t value_id, int64_t* out_value);

// Extracts an exact scalar i32 constant from a module value.
bool loom_amdgpu_module_value_as_i32_constant(const loom_module_t* module,
                                              loom_value_id_t value_id,
                                              int64_t* out_value);

// Extracts an exact scalar f32 constant as a raw IEEE bit pattern.
bool loom_amdgpu_module_value_as_f32_constant(const loom_module_t* module,
                                              loom_value_id_t value_id,
                                              uint32_t* out_bit_pattern);

// Extracts an exact non-negative signed 64-bit integer from value facts.
bool loom_amdgpu_value_facts_as_exact_non_negative_i64(loom_value_facts_t facts,
                                                       int64_t* out_value);

// Resolves the source-order workgroup-storage base for a workgroup
// buffer.alloca root. The layout must match the low.storage.reserve order
// emitted by buffer lowering.
bool loom_amdgpu_source_lds_layout_lookup_root(
    const loom_value_fact_table_t* fact_table, loom_func_like_t source_function,
    loom_value_id_t root_value_id, uint64_t* out_byte_offset);

// Returns true when an attribute can encode as a signed 32-bit immediate.
bool loom_amdgpu_attr_is_i32_immediate(loom_attribute_t value);

// Returns true when an attribute can encode as an unsigned 32-bit address.
bool loom_amdgpu_attr_is_u32_address_immediate(loom_attribute_t value);

// Returns true when an attribute can encode as an f32 immediate payload.
bool loom_amdgpu_attr_is_f32_immediate(loom_attribute_t value);

// Returns the f32 bit pattern produced by narrowing an attribute.
uint32_t loom_amdgpu_attr_f32_bit_pattern(loom_attribute_t value);

// Returns true when an attribute can encode as an f16/bf16 immediate payload.
bool loom_amdgpu_attr_is_16bit_float_immediate(loom_attribute_t value);

// Returns the low-half f16/bf16 bit pattern produced by narrowing an attribute.
uint32_t loom_amdgpu_attr_16bit_float_bit_pattern(loom_scalar_type_t type,
                                                  loom_attribute_t value);

// Extracts a source scalar i32 constant.
bool loom_amdgpu_value_as_i32_constant(loom_low_lower_context_t* context,
                                       loom_value_id_t value_id,
                                       int64_t* out_value);

// Extracts a source scalar f32 constant as a raw IEEE bit pattern.
bool loom_amdgpu_value_as_f32_constant(loom_low_lower_context_t* context,
                                       loom_value_id_t value_id,
                                       uint32_t* out_bit_pattern);

// Extracts a source address scalar constant.
bool loom_amdgpu_value_as_address_constant(loom_low_lower_context_t* context,
                                           loom_value_id_t value_id,
                                           int64_t* out_value);

// Returns the exact wavefront size selected by the active target bundle.
iree_status_t loom_amdgpu_target_wavefront_size(
    const loom_target_bundle_t* bundle, uint32_t* out_wavefront_size);

// Returns the fixed flat workgroup size required by the source function or
// target ABI.
bool loom_amdgpu_required_flat_workgroup_size(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, uint32_t* out_flat_size);

enum {
  // target_key, export_name, config_key, function_name, and op_name.
  LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT = 5,
};

// Populates the common AMDGPU legality diagnostic context params.
void loom_amdgpu_low_legality_make_context_params(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_diagnostic_param_t* params);

// Emits ERR_AMDGPU_023 for a source-to-low legality constraint owned by the
// AMDGPU lowering provider.
iree_status_t loom_amdgpu_low_legality_reject(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t constraint_key);

// Emits the current invocation lane id within its subgroup as a VGPR value.
iree_status_t loom_amdgpu_emit_current_subgroup_lane_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t result_type, loom_value_id_t* out_lane_id);

// Selects a plan for value-construction source ops.
iree_status_t loom_amdgpu_select_value_plan(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_low_lower_plan_t* out_plan);

// Lowers a value-construction source op using its selected plan.
iree_status_t loom_amdgpu_lower_value_op(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t plan);

// Verifies AMDGPU low legality for vector coordinate construction source ops.
iree_status_t loom_amdgpu_low_legality_verify_vector_iota(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects a plan for kernel preamble source ops.
iree_status_t loom_amdgpu_select_preamble_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Emits preamble live-ins for the current low function.
iree_status_t loom_amdgpu_emit_preamble(void* user_data,
                                        loom_low_lower_context_t* context);

// Lowers a kernel preamble source op using its pre-bound live-in value.
iree_status_t loom_amdgpu_lower_preamble_op(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op);

// Verifies AMDGPU low legality for launch preamble query source ops.
iree_status_t loom_amdgpu_low_legality_verify_kernel_preamble(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects a workgroup barrier packet for a source kernel.barrier op.
iree_status_t loom_amdgpu_select_kernel_barrier_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Lowers a source kernel.barrier to an AMDGPU workgroup barrier packet.
iree_status_t loom_amdgpu_lower_kernel_barrier(
    loom_low_lower_context_t* context, const loom_op_t* source_op);

// Verifies source kernel.barrier legality for AMDGPU target-low selection.
iree_status_t loom_amdgpu_low_legality_verify_kernel_barrier(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects a native AMDGPU cross-lane packet for a source subgroup shuffle.
iree_status_t loom_amdgpu_select_kernel_subgroup_shuffle_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_shuffle_plan_t* out_plan, bool* out_selected);

// Lowers a source subgroup shuffle using one DS bpermute per 32-bit payload
// register.
iree_status_t loom_amdgpu_lower_kernel_subgroup_shuffle(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_shuffle_plan_t* plan);

// Verifies source subgroup shuffle legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_shuffle(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects native AMDGPU cross-lane packets for a source subgroup reduce.
iree_status_t loom_amdgpu_select_kernel_subgroup_reduce_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_reduce_plan_t* out_plan, bool* out_selected);

// Selects native AMDGPU cross-lane packets for a source workgroup reduce when
// the workgroup is exactly one subgroup.
iree_status_t loom_amdgpu_select_kernel_workgroup_reduce_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_reduce_plan_t* out_plan, bool* out_selected);

// Lowers a source subgroup reduce using DS bpermute tree steps and native VGPR
// combining packets.
iree_status_t loom_amdgpu_lower_kernel_subgroup_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan);

// Lowers a source workgroup reduce selected as a single-subgroup native reduce.
iree_status_t loom_amdgpu_lower_kernel_workgroup_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_reduce_plan_t* plan);

// Verifies source subgroup reduce legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_reduce(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies source workgroup reduce legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_workgroup_reduce(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects native AMDGPU cross-lane packets for a source subgroup scan.
iree_status_t loom_amdgpu_select_kernel_subgroup_scan_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_scan_plan_t* out_plan, bool* out_selected);

// Lowers a source subgroup scan using DS bpermute prefix steps, native VGPR
// combining packets, and per-step lane-bound masks.
iree_status_t loom_amdgpu_lower_kernel_subgroup_scan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_scan_plan_t* plan);

// Verifies source subgroup scan legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_scan(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects native AMDGPU EXEC-mask packets for source subgroup active.mask.
iree_status_t loom_amdgpu_select_kernel_subgroup_active_mask_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_active_mask_plan_t* out_plan, bool* out_selected);

// Lowers a source subgroup active.mask by reading EXEC.
iree_status_t loom_amdgpu_lower_kernel_subgroup_active_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_active_mask_plan_t* plan);

// Verifies source subgroup active.mask legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_active_mask(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects native AMDGPU EXEC-mask packets for source subgroup vote.ballot.
iree_status_t loom_amdgpu_select_kernel_subgroup_ballot_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_ballot_plan_t* out_plan, bool* out_selected);

// Lowers a source subgroup vote.ballot by exposing the native predicate mask.
iree_status_t loom_amdgpu_lower_kernel_subgroup_ballot(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_ballot_plan_t* plan);

// Verifies source subgroup vote.ballot legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_ballot(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects native AMDGPU SALU packets for source subgroup vote.any.
iree_status_t loom_amdgpu_select_kernel_subgroup_vote_any_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_vote_any_plan_t* out_plan, bool* out_selected);

// Lowers a source subgroup vote.any by comparing a predicate mask with zero.
iree_status_t loom_amdgpu_lower_kernel_subgroup_vote_any(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_vote_any_plan_t* plan);

// Verifies source subgroup vote.any legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_vote_any(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects native AMDGPU SALU packets for source subgroup vote.all.
iree_status_t loom_amdgpu_select_kernel_subgroup_vote_all_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_vote_all_plan_t* out_plan, bool* out_selected);

// Lowers a source subgroup vote.all by comparing predicate and EXEC masks.
iree_status_t loom_amdgpu_lower_kernel_subgroup_vote_all(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_vote_all_plan_t* plan);

// Verifies source subgroup vote.all legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_vote_all(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects a native AMDGPU cross-lane packet for a source subgroup broadcast.
iree_status_t loom_amdgpu_select_kernel_subgroup_broadcast_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_broadcast_plan_t* out_plan, bool* out_selected);

// Lowers a source subgroup broadcast using one DS bpermute per 32-bit payload
// register.
iree_status_t loom_amdgpu_lower_kernel_subgroup_broadcast(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_broadcast_plan_t* plan);

// Verifies source subgroup broadcast legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_broadcast(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects a native AMDGPU first-active lane read for a source subgroup
// broadcast.first.
iree_status_t loom_amdgpu_select_kernel_subgroup_broadcast_first_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_subgroup_broadcast_first_plan_t* out_plan, bool* out_selected);

// Lowers a source subgroup broadcast.first using one V_READFIRSTLANE per
// 32-bit payload register.
iree_status_t loom_amdgpu_lower_kernel_subgroup_broadcast_first(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_subgroup_broadcast_first_plan_t* plan);

// Verifies source subgroup broadcast.first legality for native AMDGPU lowering.
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_broadcast_first(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Rejects source subgroup match ops that require explicit target legalization
// before AMDGPU source-to-low packet selection.
iree_status_t loom_amdgpu_low_legality_verify_kernel_subgroup_match(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies AMDGPU low legality for collective source ops without selected
// packet lowering yet.
iree_status_t loom_amdgpu_low_legality_verify_kernel_collective(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies source kernel.async group/wait legality for AMDGPU target-low
// selection.
iree_status_t loom_amdgpu_low_legality_verify_kernel_async(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects an AMDGPU async gather packet for a source kernel.async.gather op.
iree_status_t loom_amdgpu_select_kernel_async_gather_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_async_gather_plan_t* out_plan, bool* out_selected);

// Selects an AMDGPU wait packet for a source kernel.async.wait op.
iree_status_t loom_amdgpu_select_kernel_async_wait_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_async_wait_plan_t* out_plan, bool* out_selected);

// Lowers a source kernel.async.gather to a global-to-LDS packet and elides its
// async token.
iree_status_t loom_amdgpu_lower_kernel_async_gather(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_async_gather_plan_t* plan);

// Lowers a source kernel.async.wait to an explicit wait packet when the source
// group contains target-visible async transfers.
iree_status_t loom_amdgpu_lower_kernel_async_wait(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_async_wait_plan_t* plan);

// Selects a plan for buffer construction source ops.
iree_status_t loom_amdgpu_select_buffer_plan(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t* out_plan);

// Lowers a buffer construction source op using its selected plan.
iree_status_t loom_amdgpu_lower_buffer_op(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_low_lower_plan_t plan);

// Returns true when value is a non-zero power of two.
bool loom_amdgpu_u32_is_power_of_two(uint32_t value);

// Interns a low lowering helper string in the active module.
iree_status_t loom_amdgpu_intern(loom_low_lower_context_t* context,
                                 iree_string_view_t string,
                                 loom_string_id_t* out_string_id);

// Appends one signed integer attribute to a descriptor-backed low op.
iree_status_t loom_amdgpu_append_i64_attr(loom_low_lower_context_t* context,
                                          iree_string_view_t name,
                                          int64_t value,
                                          loom_named_attr_t* attrs,
                                          iree_host_size_t attr_capacity,
                                          iree_host_size_t* inout_attr_count);

// Emits an SGPR byte offset from an optional source dynamic index plus a static
// byte offset.
iree_status_t loom_amdgpu_emit_sgpr_byte_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t dynamic_index, int64_t dynamic_index_byte_stride,
    uint32_t dynamic_index_byte_shift, uint32_t static_byte_offset,
    loom_value_id_t* out_low_offset);

// Emits an SGPR byte offset from the scalar-address terms selected for a source
// memory access plus a static byte offset.
iree_status_t loom_amdgpu_emit_sgpr_byte_offset_terms(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_memory_dynamic_index_kind_t* dynamic_term_kinds,
    uint32_t static_byte_offset, loom_value_id_t* out_low_offset);

// Maps a source result to the low register type already selected by the active
// lowering policy and verifies that it is a register payload.
iree_status_t loom_amdgpu_low_result_type(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t source_result,
                                          loom_type_t* out_low_type);

// Resolves an optional AMDGPU descriptor ref against the active descriptor set.
iree_status_t loom_amdgpu_resolve_descriptor_ref_if_present(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_low_lower_resolved_descriptor_t* out_descriptor, bool* out_present);

// Resolves a required AMDGPU descriptor ref against the active descriptor set.
iree_status_t loom_amdgpu_resolve_descriptor_ref(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_low_lower_resolved_descriptor_t* out_descriptor);

// Returns true when a descriptor row has any target-owned implicit operand.
bool loom_amdgpu_descriptor_has_implicit_operand(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns true when a descriptor row has an implicit resource operand.
bool loom_amdgpu_descriptor_has_implicit_resource_operand(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Resolves one optional explicit packet descriptor and its immediate names.
iree_status_t loom_amdgpu_resolve_explicit_packet_plan(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_amdgpu_explicit_packet_immediate_template_t* immediates,
    iree_host_size_t immediate_count,
    loom_amdgpu_explicit_packet_plan_t* out_plan, bool* out_present);

// Resolves one explicit packet descriptor row and its immediate names.
iree_status_t loom_amdgpu_resolve_explicit_packet_row_plan(
    loom_low_lower_context_t* context, const loom_low_descriptor_t* descriptor,
    const loom_amdgpu_explicit_packet_immediate_template_t* immediates,
    iree_host_size_t immediate_count,
    loom_amdgpu_explicit_packet_plan_t* out_plan);

// Emits one descriptor-backed low.op with source provenance.
iree_status_t loom_amdgpu_emit_low_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_op_t** out_low_op);

// Emits one explicit descriptor packet selected during source-to-low planning.
iree_status_t loom_amdgpu_emit_explicit_packet_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_explicit_packet_plan_t* plan);

// Emits one descriptor-backed low.const with an imm32 attribute.
iree_status_t loom_amdgpu_emit_const_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t value,
    loom_type_t result_type, loom_value_id_t* out_value_id);

// Emits one resolved descriptor-backed low.const with an imm32 attribute.
iree_status_t loom_amdgpu_emit_resolved_const_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_string_id_t imm32_attr_name_id, uint32_t value,
    loom_type_t result_type, loom_value_id_t* out_value_id);

// Emits a fresh VGPR carrying the same 32-bit bit payload as |low_source|.
iree_status_t loom_amdgpu_emit_vgpr_b32_copy(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_value_id_t low_source,
                                             loom_value_id_t* out_value);

// Returns |low_value| when it is already a one-unit VGPR, otherwise emits a
// fresh VGPR carrying the same 32-bit bit payload.
iree_status_t loom_amdgpu_materialize_low_vgpr_b32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_value);

// Emits one binary VGPR descriptor op.
iree_status_t loom_amdgpu_emit_vgpr_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t lane_type, loom_value_id_t* out_value);

// Emits one VGPR descriptor op with one VGPR operand and one imm32 literal.
iree_status_t loom_amdgpu_emit_vgpr_binary_literal(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t value,
    uint32_t literal, loom_type_t lane_type, loom_value_id_t* out_value);

// Emits one VGPR literal-shift descriptor op. If |shift| is zero, returns
// |value| unchanged.
iree_status_t loom_amdgpu_emit_vgpr_shift(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t literal_descriptor_ref, uint32_t shift,
    loom_value_id_t value, loom_type_t lane_type, loom_value_id_t* out_value);

typedef enum loom_amdgpu_vgpr_scale_u32_flag_bits_e {
  // No additional facts about the input value are known.
  LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE = 0u,
  // The input value is known to be in the unsigned 24-bit range.
  LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_VALUE_UNSIGNED_24 = 1u << 0,
} loom_amdgpu_vgpr_scale_u32_flag_bits_t;
typedef uint32_t loom_amdgpu_vgpr_scale_u32_flags_t;

// Emits |value| scaled by an unsigned 32-bit constant into a one-unit VGPR.
iree_status_t loom_amdgpu_emit_vgpr_scale_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value, uint32_t scale,
    loom_amdgpu_vgpr_scale_u32_flags_t flags, loom_type_t lane_type,
    loom_value_id_t* out_value);

// Emits a low.slice from a register range.
iree_status_t loom_amdgpu_emit_low_slice(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_value_id_t source,
                                         uint32_t lane_offset,
                                         loom_type_t result_type,
                                         loom_value_id_t* out_value);

// Emits a value into M0 for special-register packet operands.
iree_status_t loom_amdgpu_emit_m0_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* consumer_descriptor,
    uint32_t value, loom_value_id_t* out_value_id);

// Returns true when the target bundle belongs to an AMDGPU contract set.
bool loom_amdgpu_low_legality_bundle_is_amdgpu(
    const loom_target_bundle_t* bundle);

// Supplies AMDGPU options for shared descriptor-matrix source adapters.
iree_status_t loom_amdgpu_descriptor_matrix_options(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    loom_contract_vector_mma_options_t* out_options);

// Projects generic matrix contracts to AMDGPU matrix descriptors.
iree_status_t loom_amdgpu_descriptor_matrix_query(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op, const loom_contract_request_t* contract_request,
    loom_target_contract_query_result_t* out_result);

// Materializes AMDGPU matrix descriptor immediate attributes.
iree_status_t loom_amdgpu_descriptor_matrix_attrs(
    void* user_data, loom_low_lower_context_t* context,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_contract_request_t* contract_request,
    const loom_low_descriptor_t* descriptor,
    loom_named_attr_slice_t* out_attrs);

// Selects an AMDGPU register-table lookup plan.
iree_status_t loom_amdgpu_select_vector_table_lookup_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_table_lookup_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.table.lookup op to AMDGPU compare/select packets.
iree_status_t loom_amdgpu_lower_vector_table_lookup(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan);

// Verifies source vector table op legality for AMDGPU target-low selection.
iree_status_t loom_amdgpu_low_legality_verify_vector_table(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects an AMDGPU vector.bitfield.extract* plan.
iree_status_t loom_amdgpu_select_vector_bitfield_extract_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_bitfield_extract_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.bitfield.extract* op to AMDGPU descriptor-backed low
// packets.
iree_status_t loom_amdgpu_lower_vector_bitfield_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitfield_extract_plan_t* plan);

// Selects an AMDGPU vector.bitfield.insert plan.
iree_status_t loom_amdgpu_select_vector_bitfield_insert_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_bitfield_insert_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.bitfield.insert op to AMDGPU descriptor-backed low
// packets.
iree_status_t loom_amdgpu_lower_vector_bitfield_insert(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitfield_insert_plan_t* plan);

// Selects an AMDGPU vector.bitpack plan.
iree_status_t loom_amdgpu_select_vector_bitpack_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_bitpack_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.bitpack op to AMDGPU descriptor-backed low packets.
iree_status_t loom_amdgpu_lower_vector_bitpack(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitpack_plan_t* plan);

// Selects an AMDGPU vector.bitunpack plan.
iree_status_t loom_amdgpu_select_vector_bitunpack_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_bitunpack_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.bitunpack op to AMDGPU descriptor-backed low packets.
iree_status_t loom_amdgpu_lower_vector_bitunpack(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitunpack_plan_t* plan);

// Verifies source vector bitstream op legality for AMDGPU target-low
// selection.
iree_status_t loom_amdgpu_low_legality_verify_vector_bitstream(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects an AMDGPU vector.bitcast register reinterpretation plan.
iree_status_t loom_amdgpu_select_vector_bitcast_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_bitcast_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.bitcast op as an AMDGPU register reinterpretation.
iree_status_t loom_amdgpu_lower_vector_bitcast(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_bitcast_plan_t* plan);

// Selects an AMDGPU vector.slice register slicing plan.
iree_status_t loom_amdgpu_select_vector_slice_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_slice_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.slice op as AMDGPU register slicing.
iree_status_t loom_amdgpu_lower_vector_slice(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_slice_plan_t* plan);

// Selects an AMDGPU matrix-fragment load plan.
iree_status_t loom_amdgpu_select_vector_fragment_load_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_plan_t* out_plan, bool* out_selected);

// Selects an AMDGPU matrix-fragment store plan.
iree_status_t loom_amdgpu_select_vector_fragment_store_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.fragment.load op to lane-owned memory packets.
iree_status_t loom_amdgpu_lower_vector_fragment_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan);

// Lowers a source vector.fragment.store op to lane-owned memory packets.
iree_status_t loom_amdgpu_lower_vector_fragment_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan);

// Verifies source vector.fragment.load/store legality for AMDGPU target-low
// selection.
iree_status_t loom_amdgpu_low_legality_verify_vector_fragment_memory(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies source vector structural op legality for AMDGPU target-low
// selection.
iree_status_t loom_amdgpu_low_legality_verify_vector_structural(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Selects an AMDGPU source memory-load packet plan.
iree_status_t loom_amdgpu_select_memory_load_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_plan_t* out_plan, bool* out_selected);

// Selects an AMDGPU source memory-store packet plan.
iree_status_t loom_amdgpu_select_memory_store_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_plan_t* out_plan, bool* out_selected);

// Lowers a source memory-load op to an AMDGPU memory packet.
iree_status_t loom_amdgpu_lower_memory_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* plan);

// Lowers a source memory-store op to an AMDGPU memory packet.
iree_status_t loom_amdgpu_lower_memory_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* plan);

// Selects an AMDGPU LDS atomic packet plan.
iree_status_t loom_amdgpu_select_view_atomic_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_atomic_plan_t* out_plan, bool* out_selected);

// Lowers a source view.atomic.* op to an AMDGPU LDS atomic packet.
iree_status_t loom_amdgpu_lower_view_atomic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_atomic_plan_t* plan);

// Selects an AMDGPU scalar-buffer data prefetch plan.
iree_status_t loom_amdgpu_select_view_prefetch_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_prefetch_plan_t* out_plan, bool* out_selected);

// Lowers a source view.prefetch to an AMDGPU scalar-buffer data prefetch
// packet.
iree_status_t loom_amdgpu_lower_view_prefetch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_prefetch_plan_t* plan);

// Verifies source memory legality for AMDGPU target-low selection.
iree_status_t loom_amdgpu_low_legality_verify_memory(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies source view atomic legality for AMDGPU target-low selection.
iree_status_t loom_amdgpu_low_legality_verify_view_atomic(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_INTERNAL_H_
