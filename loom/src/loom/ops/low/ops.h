// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables
// clang-format off

#ifndef LOOM_OPS_LOW_OPS_H_
#define LOOM_OPS_LOW_OPS_H_

#include "loom/ops/op_defs.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_LOW_FUNC_DEF = LOOM_OP_KIND(LOOM_DIALECT_LOW, 0),
  LOOM_OP_LOW_KERNEL_DEF = LOOM_OP_KIND(LOOM_DIALECT_LOW, 1),
  LOOM_OP_LOW_FUNC_DECL = LOOM_OP_KIND(LOOM_DIALECT_LOW, 2),
  LOOM_OP_LOW_RETURN = LOOM_OP_KIND(LOOM_DIALECT_LOW, 3),
  LOOM_OP_LOW_FUNC_CALL = LOOM_OP_KIND(LOOM_DIALECT_LOW, 4),
  LOOM_OP_LOW_OP = LOOM_OP_KIND(LOOM_DIALECT_LOW, 5),
  LOOM_OP_LOW_CONST = LOOM_OP_KIND(LOOM_DIALECT_LOW, 6),
  LOOM_OP_LOW_COPY = LOOM_OP_KIND(LOOM_DIALECT_LOW, 7),
  LOOM_OP_LOW_SLICE = LOOM_OP_KIND(LOOM_DIALECT_LOW, 8),
  LOOM_OP_LOW_CONCAT = LOOM_OP_KIND(LOOM_DIALECT_LOW, 9),
  LOOM_OP_LOW_INVOKE = LOOM_OP_KIND(LOOM_DIALECT_LOW, 10),
  LOOM_OP_LOW_SLOT = LOOM_OP_KIND(LOOM_DIALECT_LOW, 11),
  LOOM_OP_LOW_SPILL = LOOM_OP_KIND(LOOM_DIALECT_LOW, 12),
  LOOM_OP_LOW_RELOAD = LOOM_OP_KIND(LOOM_DIALECT_LOW, 13),
  LOOM_OP_LOW_FRAME_INDEX = LOOM_OP_KIND(LOOM_DIALECT_LOW, 14),
  LOOM_OP_LOW_BR = LOOM_OP_KIND(LOOM_DIALECT_LOW, 15),
  LOOM_OP_LOW_COND_BR = LOOM_OP_KIND(LOOM_DIALECT_LOW, 16),
  LOOM_OP_LOW_RESOURCE = LOOM_OP_KIND(LOOM_DIALECT_LOW, 17),
  LOOM_OP_LOW_LIVE_IN = LOOM_OP_KIND(LOOM_DIALECT_LOW, 18),
  LOOM_OP_LOW_COUNT_ = 19,
};

// Function visibility. Absent (0) means private (module-internal).
typedef enum loom_low_visibility_e {
  LOOM_LOW_VISIBILITY_PUBLIC = 1,
  LOOM_LOW_VISIBILITY_COUNT_ = 2,
} loom_low_visibility_t;

// Function calling convention. Absent (0) means host.
typedef enum loom_low_cc_e {
  LOOM_LOW_CC_HOST = 1,
  LOOM_LOW_CC_DEVICE = 2,
  LOOM_LOW_CC_INITIALIZER = 3,
  LOOM_LOW_CC_DEINITIALIZER = 4,
  LOOM_LOW_CC_COUNT_ = 5,
} loom_low_cc_t;

// Function purity. Absent (0) means unspecified (conservative).
typedef enum loom_low_purity_e {
  LOOM_LOW_PURITY_PURE = 1,
  LOOM_LOW_PURITY_COUNT_ = 2,
} loom_low_purity_t;

// Register allocation exactness mode for a low function. Absent means virtual.
typedef enum loom_low_allocation_e {
  LOOM_LOW_ALLOCATION_VIRTUAL = 1,
  LOOM_LOW_ALLOCATION_ASSIGNED = 2,
  LOOM_LOW_ALLOCATION_FIXED = 3,
  LOOM_LOW_ALLOCATION_COUNT_ = 4,
} loom_low_allocation_t;

// Instruction scheduling exactness mode for a low function. Absent means free.
typedef enum loom_low_schedule_e {
  LOOM_LOW_SCHEDULE_FREE = 1,
  LOOM_LOW_SCHEDULE_CONSTRAINED = 2,
  LOOM_LOW_SCHEDULE_LOCKED = 3,
  LOOM_LOW_SCHEDULE_COUNT_ = 4,
} loom_low_schedule_t;

// External code source kind for an imported low function declaration.
typedef enum loom_low_func_decl_import_kind_e {
  LOOM_LOW_FUNC_DECL_IMPORT_KIND_VM = 1,
  LOOM_LOW_FUNC_DECL_IMPORT_KIND_NATIVE = 2,
  LOOM_LOW_FUNC_DECL_IMPORT_KIND_ROCASM = 3,
  LOOM_LOW_FUNC_DECL_IMPORT_KIND_OBJECT = 4,
  LOOM_LOW_FUNC_DECL_IMPORT_KIND_COUNT_ = 5,
} loom_low_func_decl_import_kind_t;

// Storage space represented by a low slot record.
typedef enum loom_low_slot_space_e {
  LOOM_LOW_SLOT_SPACE_STACK = 1,
  LOOM_LOW_SLOT_SPACE_SCRATCH = 2,
  LOOM_LOW_SLOT_SPACE_PRIVATE = 3,
  LOOM_LOW_SLOT_SPACE_LDS = 4,
  LOOM_LOW_SLOT_SPACE_COUNT_ = 5,
} loom_low_slot_space_t;

// Target-provided ABI resource imported into a low function body.
typedef enum loom_low_resource_import_kind_e {
  LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER = 1,
  LOOM_LOW_RESOURCE_IMPORT_KIND_VM_STATE = 2,
  LOOM_LOW_RESOURCE_IMPORT_KIND_VM_IMPORT = 3,
  LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BUFFER_RESOURCE = 4,
  LOOM_LOW_RESOURCE_IMPORT_KIND_COUNT_ = 5,
} loom_low_resource_import_kind_t;

// LOOM_OP_LOW_FUNC_DEF: Target-bound low function definition with register-typed signature values.
// low.func.def target(@gfx1100) @add(%lhs: reg<amdgpu.vgpr x1>, %rhs: reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>) {
//   %sum = low.op<amdgpu.v_add_u32>(%lhs, %rhs) : (reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> reg<amdgpu.vgpr x1>
//   low.return %sum : reg<amdgpu.vgpr x1>
// }
LOOM_DEFINE_ISA(loom_low_func_def_isa, LOOM_OP_LOW_FUNC_DEF)
LOOM_DEFINE_VARIADIC_RESULTS(loom_low_func_def_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_func_def_callee, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_func_def_target, 1)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_def_abi, 2, loom_target_abi_kind_t)
LOOM_DEFINE_ATTR_DICT(loom_low_func_def_abi_attrs, 3)
LOOM_DEFINE_ATTR_STRING(loom_low_func_def_export_symbol, 4)
LOOM_DEFINE_ATTR_DICT(loom_low_func_def_export_attrs, 5)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_def_visibility, 6, loom_low_visibility_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_def_cc, 7, loom_low_cc_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_def_purity, 8, loom_low_purity_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_def_allocation, 9, loom_low_allocation_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_def_schedule, 10, loom_low_schedule_t)
LOOM_DEFINE_REGION(loom_low_func_def_body, 0)
enum loom_low_func_def_build_flag_bits_e {
  LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY = 1u << 0,
  LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_CC = 1u << 1,
  LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_PURITY = 1u << 2,
  LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_ALLOCATION = 1u << 3,
  LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_SCHEDULE = 1u << 4,
  LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_ABI = 1u << 5,
  LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_EXPORT_SYMBOL = 1u << 6,
};
typedef uint32_t loom_low_func_def_build_flags_t;
iree_status_t loom_low_func_def_build(
    loom_builder_t* builder,
    loom_low_func_def_build_flags_t build_flags,
    loom_optional uint8_t visibility,
    loom_optional uint8_t cc,
    loom_optional uint8_t purity,
    loom_optional uint8_t allocation,
    loom_optional uint8_t schedule,
    loom_symbol_ref_t target,
    loom_optional uint8_t abi,
    loom_optional loom_named_attr_slice_t abi_attrs,
    loom_optional loom_string_id_t export_symbol,
    loom_optional loom_named_attr_slice_t export_attrs,
    loom_symbol_ref_t callee,
    const loom_type_t* arg_types,
    iree_host_size_t arg_types_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_optional const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_func_def_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_KERNEL_DEF: Target-bound low kernel entry with register-typed launch ABI values. Helper calls stay in low.func.def; kernel launch/export contracts live on this entry op.
// low.kernel.def target(@gfx1100) export("matmul") artifact(@gfx_hsaco) workgroup_size(16, 4, 1) @matmul(%lhs: reg<amdgpu.sgpr x4>, %rhs: reg<amdgpu.sgpr x4>, %out: reg<amdgpu.sgpr x4>) {
//   low.return
// }
LOOM_DEFINE_ISA(loom_low_kernel_def_isa, LOOM_OP_LOW_KERNEL_DEF)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_kernel_def_callee, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_kernel_def_target, 1)
LOOM_DEFINE_ATTR_STRING(loom_low_kernel_def_export_symbol, 2)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_kernel_def_artifact, 3)
LOOM_DEFINE_ATTR_I64(loom_low_kernel_def_export_ordinal, 4)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_kernel_def_export_linkage, 5, loom_target_linkage_t)
LOOM_DEFINE_ATTR_I64(loom_low_kernel_def_workgroup_size_x, 6)
LOOM_DEFINE_ATTR_I64(loom_low_kernel_def_workgroup_size_y, 7)
LOOM_DEFINE_ATTR_I64(loom_low_kernel_def_workgroup_size_z, 8)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_kernel_def_allocation, 9, loom_low_allocation_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_kernel_def_schedule, 10, loom_low_schedule_t)
LOOM_DEFINE_REGION(loom_low_kernel_def_body, 0)
enum loom_low_kernel_def_build_flag_bits_e {
  LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_ALLOCATION = 1u << 0,
  LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_SCHEDULE = 1u << 1,
  LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_EXPORT_SYMBOL = 1u << 2,
  LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_ARTIFACT = 1u << 3,
  LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_EXPORT_ORDINAL = 1u << 4,
  LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_EXPORT_LINKAGE = 1u << 5,
  LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_SIZE_X = 1u << 6,
  LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_SIZE_Y = 1u << 7,
  LOOM_LOW_KERNEL_DEF_BUILD_FLAG_HAS_WORKGROUP_SIZE_Z = 1u << 8,
};
typedef uint32_t loom_low_kernel_def_build_flags_t;
iree_status_t loom_low_kernel_def_build(
    loom_builder_t* builder,
    loom_low_kernel_def_build_flags_t build_flags,
    loom_optional uint8_t allocation,
    loom_optional uint8_t schedule,
    loom_symbol_ref_t target,
    loom_optional loom_string_id_t export_symbol,
    loom_optional loom_symbol_ref_t artifact,
    loom_optional int64_t export_ordinal,
    loom_optional uint8_t export_linkage,
    loom_optional int64_t workgroup_size_x,
    loom_optional int64_t workgroup_size_y,
    loom_optional int64_t workgroup_size_z,
    loom_symbol_ref_t callee,
    const loom_type_t* arg_types,
    iree_host_size_t arg_types_count,
    loom_optional const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_kernel_def_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_FUNC_DECL: Target-bound low function declaration with register-typed signature values.
// low.func.decl target(@gfx1100) @extern_add(%lhs: reg<amdgpu.vgpr x1>, %rhs: reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>)
LOOM_DEFINE_ISA(loom_low_func_decl_isa, LOOM_OP_LOW_FUNC_DECL)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_low_func_decl_args, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_low_func_decl_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_func_decl_callee, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_func_decl_target, 1)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_decl_abi, 2, loom_target_abi_kind_t)
LOOM_DEFINE_ATTR_DICT(loom_low_func_decl_abi_attrs, 3)
LOOM_DEFINE_ATTR_STRING(loom_low_func_decl_export_symbol, 4)
LOOM_DEFINE_ATTR_DICT(loom_low_func_decl_export_attrs, 5)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_decl_visibility, 6, loom_low_visibility_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_decl_cc, 7, loom_low_cc_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_decl_purity, 8, loom_low_purity_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_decl_allocation, 9, loom_low_allocation_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_decl_schedule, 10, loom_low_schedule_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_decl_import_kind, 12, loom_low_func_decl_import_kind_t)
LOOM_DEFINE_ATTR_STRING(loom_low_func_decl_code_symbol, 13)
enum loom_low_func_decl_build_flag_bits_e {
  LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_VISIBILITY = 1u << 0,
  LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_CC = 1u << 1,
  LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_PURITY = 1u << 2,
  LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_ALLOCATION = 1u << 3,
  LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_SCHEDULE = 1u << 4,
  LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_IMPORT_KIND = 1u << 5,
  LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_CODE_SYMBOL = 1u << 6,
  LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_ABI = 1u << 7,
  LOOM_LOW_FUNC_DECL_BUILD_FLAG_HAS_EXPORT_SYMBOL = 1u << 8,
};
typedef uint32_t loom_low_func_decl_build_flags_t;
iree_status_t loom_low_func_decl_build(
    loom_builder_t* builder,
    loom_low_func_decl_build_flags_t build_flags,
    loom_optional uint8_t visibility,
    loom_optional uint8_t cc,
    loom_optional uint8_t purity,
    loom_optional uint8_t allocation,
    loom_optional uint8_t schedule,
    loom_optional uint8_t import_kind,
    loom_optional loom_string_id_t code_symbol,
    loom_symbol_ref_t target,
    loom_optional uint8_t abi,
    loom_optional loom_named_attr_slice_t abi_attrs,
    loom_optional loom_string_id_t export_symbol,
    loom_optional loom_named_attr_slice_t export_attrs,
    loom_symbol_ref_t callee,
    const loom_type_t* arg_types,
    iree_host_size_t arg_types_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_optional const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_func_decl_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_RETURN: Return register values from a low function.
// low.return
LOOM_DEFINE_ISA(loom_low_return_isa, LOOM_OP_LOW_RETURN)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_low_return_values, 0)
iree_status_t loom_low_return_build(
    loom_builder_t* builder,
    const loom_value_id_t* values,
    iree_host_size_t values_count,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_LOW_FUNC_CALL: Direct call from one low function body to another same-target low function.
// %result = low.func.call @extern_add(%lhs, %rhs) : (reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>)
LOOM_DEFINE_ISA(loom_low_func_call_isa, LOOM_OP_LOW_FUNC_CALL)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_low_func_call_operands, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_low_func_call_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_func_call_callee, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_func_call_purity, 1, loom_low_purity_t)
enum loom_low_func_call_build_flag_bits_e {
  LOOM_LOW_FUNC_CALL_BUILD_FLAG_HAS_PURITY = 1u << 0,
};
typedef uint32_t loom_low_func_call_build_flags_t;
iree_status_t loom_low_func_call_build(
    loom_builder_t* builder,
    loom_low_func_call_build_flags_t build_flags,
    loom_optional uint8_t purity,
    loom_symbol_ref_t callee,
    loom_may_consume const loom_value_id_t* operands,
    iree_host_size_t operands_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
loom_trait_flags_t loom_low_func_call_effective_traits(const loom_op_t* op);
iree_status_t loom_low_func_call_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_OP: Descriptor-backed target instruction over virtual registers.
// %sum = low.op<amdgpu.v_add_u32>(%lhs, %rhs) : (reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> reg<amdgpu.vgpr x1>
LOOM_DEFINE_ISA(loom_low_op_isa, LOOM_OP_LOW_OP)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_low_op_operands, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_low_op_results, 0)
LOOM_DEFINE_ATTR_STRING(loom_low_op_opcode, 0)
LOOM_DEFINE_ATTR_I64(loom_low_op_descriptor_id, 1)
LOOM_DEFINE_ATTR_DICT(loom_low_op_attrs, 2)
iree_status_t loom_low_op_build(
    loom_builder_t* builder,
    loom_string_id_t opcode,
    loom_may_consume const loom_value_id_t* operands,
    iree_host_size_t operands_count,
    loom_optional loom_named_attr_slice_t attrs,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_op_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_CONST: Descriptor-backed constant or immediate materialization into a register.
// %c0 = low.const<amdgpu.s_mov_b32> {imm = 0} : reg<amdgpu.sgpr x1>
LOOM_DEFINE_ISA(loom_low_const_isa, LOOM_OP_LOW_CONST)
LOOM_DEFINE_RESULT(loom_low_const_result, 0)
LOOM_DEFINE_ATTR_STRING(loom_low_const_opcode, 0)
LOOM_DEFINE_ATTR_I64(loom_low_const_descriptor_id, 1)
LOOM_DEFINE_ATTR_DICT(loom_low_const_attrs, 2)
iree_status_t loom_low_const_build(
    loom_builder_t* builder,
    loom_string_id_t opcode,
    loom_optional loom_named_attr_slice_t attrs,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_const_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_COPY: Explicit virtual-register copy used by lowering and allocation.
// %copy = low.copy %value : reg<amdgpu.vgpr x1> -> reg<amdgpu.vgpr x1>
LOOM_DEFINE_ISA(loom_low_copy_isa, LOOM_OP_LOW_COPY)
LOOM_DEFINE_OPERAND(loom_low_copy_source, 0)
LOOM_DEFINE_RESULT(loom_low_copy_result, 0)
iree_status_t loom_low_copy_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_copy_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_SLICE: Project a contiguous subrange from a register-range value.
// %lane = low.slice %quad[2] : reg<amdgpu.vgpr x4> -> reg<amdgpu.vgpr>
LOOM_DEFINE_ISA(loom_low_slice_isa, LOOM_OP_LOW_SLICE)
LOOM_DEFINE_OPERAND(loom_low_slice_source, 0)
LOOM_DEFINE_RESULT(loom_low_slice_result, 0)
LOOM_DEFINE_ATTR_I64(loom_low_slice_offset, 0)
iree_status_t loom_low_slice_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    int64_t offset,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_slice_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_CONCAT: Compose one register-range value from ordered register subranges.
// %pair = low.concat(%lo, %hi) : (reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>
LOOM_DEFINE_ISA(loom_low_concat_isa, LOOM_OP_LOW_CONCAT)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_low_concat_sources, 0)
LOOM_DEFINE_RESULT(loom_low_concat_result, 0)
iree_status_t loom_low_concat_build(
    loom_builder_t* builder,
    loom_may_consume const loom_value_id_t* sources,
    iree_host_size_t sources_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_LOW_INVOKE: Invoke an explicitly selected translated low function from non-low IR.
// %result = low.invoke @extern_add(%lhs, %rhs) : (i32, i32) -> (i32)
LOOM_DEFINE_ISA(loom_low_invoke_isa, LOOM_OP_LOW_INVOKE)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_low_invoke_operands, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_low_invoke_results, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_invoke_callee, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_invoke_purity, 1, loom_low_purity_t)
enum loom_low_invoke_build_flag_bits_e {
  LOOM_LOW_INVOKE_BUILD_FLAG_HAS_PURITY = 1u << 0,
};
typedef uint32_t loom_low_invoke_build_flags_t;
iree_status_t loom_low_invoke_build(
    loom_builder_t* builder,
    loom_low_invoke_build_flags_t build_flags,
    loom_optional uint8_t purity,
    loom_symbol_ref_t callee,
    loom_may_consume const loom_value_id_t* operands,
    iree_host_size_t operands_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
loom_trait_flags_t loom_low_invoke_effective_traits(const loom_op_t* op);
iree_status_t loom_low_invoke_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_SLOT: Explicit function-owned stack, scratch, private, or LDS storage slot.
// low.slot @spill0 {function = @kernel, space = scratch, size = 16, align = 4}
LOOM_DEFINE_ISA(loom_low_slot_isa, LOOM_OP_LOW_SLOT)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_slot_symbol, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_slot_function, 1)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_slot_space, 2, loom_low_slot_space_t)
LOOM_DEFINE_ATTR_I64(loom_low_slot_size, 3)
LOOM_DEFINE_ATTR_I64(loom_low_slot_align, 4)
iree_status_t loom_low_slot_build(
    loom_builder_t* builder,
    loom_symbol_ref_t symbol,
    loom_symbol_ref_t function,
    loom_low_slot_space_t space,
    int64_t size,
    int64_t align,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_slot_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_SPILL: Explicit spill store from a register value into a low slot.
// low.spill %value, @spill0 {offset = 0} : reg<amdgpu.vgpr x4>
LOOM_DEFINE_ISA(loom_low_spill_isa, LOOM_OP_LOW_SPILL)
LOOM_DEFINE_OPERAND(loom_low_spill_value, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_spill_slot, 0)
LOOM_DEFINE_ATTR_I64(loom_low_spill_offset, 1)
iree_status_t loom_low_spill_build(
    loom_builder_t* builder,
    loom_value_id_t value,
    loom_symbol_ref_t slot,
    int64_t offset,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_spill_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_RELOAD: Explicit reload from a low slot into a register value.
// %reload = low.reload @spill0 {offset = 0} : reg<amdgpu.vgpr x4>
LOOM_DEFINE_ISA(loom_low_reload_isa, LOOM_OP_LOW_RELOAD)
LOOM_DEFINE_RESULT(loom_low_reload_result, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_reload_slot, 0)
LOOM_DEFINE_ATTR_I64(loom_low_reload_offset, 1)
iree_status_t loom_low_reload_build(
    loom_builder_t* builder,
    loom_symbol_ref_t slot,
    int64_t offset,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_reload_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_FRAME_INDEX: Symbolic address calculation for a low slot before target frame layout.
// %addr = low.frame_index @spill0 {offset = 0} : reg<x86.gpr>
LOOM_DEFINE_ISA(loom_low_frame_index_isa, LOOM_OP_LOW_FRAME_INDEX)
LOOM_DEFINE_RESULT(loom_low_frame_index_result, 0)
LOOM_DEFINE_ATTR_SYMBOL(loom_low_frame_index_slot, 0)
LOOM_DEFINE_ATTR_I64(loom_low_frame_index_offset, 1)
iree_status_t loom_low_frame_index_build(
    loom_builder_t* builder,
    loom_symbol_ref_t slot,
    int64_t offset,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_frame_index_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_BR: Unconditional branch to a low successor block, forwarding register values.
// low.br ^done
LOOM_DEFINE_ISA(loom_low_br_isa, LOOM_OP_LOW_BR)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_low_br_args, 0)
LOOM_DEFINE_SUCCESSOR(loom_low_br_dest, 0)
iree_status_t loom_low_br_build(
    loom_builder_t* builder,
    loom_block_t* dest,
    const loom_value_id_t* args,
    iree_host_size_t args_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_br_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_COND_BR: Conditional branch to one of two low successor blocks based on a register predicate.
// low.cond_br %condition, ^then, ^else : reg<vm.i32>
LOOM_DEFINE_ISA(loom_low_cond_br_isa, LOOM_OP_LOW_COND_BR)
LOOM_DEFINE_OPERAND(loom_low_cond_br_condition, 0)
LOOM_DEFINE_SUCCESSOR(loom_low_cond_br_true_dest, 0)
LOOM_DEFINE_SUCCESSOR(loom_low_cond_br_false_dest, 1)
iree_status_t loom_low_cond_br_build(
    loom_builder_t* builder,
    loom_value_id_t condition,
    loom_block_t* true_dest,
    loom_block_t* false_dest,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_cond_br_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_RESOURCE: Import a function-local target resource into a low register value.
// %state = low.resource<vm_state> {index = 0, semantic_type = i64} : reg<vm.i64>
LOOM_DEFINE_ISA(loom_low_resource_isa, LOOM_OP_LOW_RESOURCE)
LOOM_DEFINE_RESULT(loom_low_resource_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_low_resource_import_kind, 0, loom_low_resource_import_kind_t)
LOOM_DEFINE_ATTR_I64(loom_low_resource_index, 1)
LOOM_DEFINE_ATTR_TYPE(loom_low_resource_semantic_type, 2)
LOOM_DEFINE_ATTR_I64(loom_low_resource_valid_byte_count, 3)
LOOM_DEFINE_ATTR_I64(loom_low_resource_cache_swizzle_stride, 4)
enum loom_low_resource_build_flag_bits_e {
  LOOM_LOW_RESOURCE_BUILD_FLAG_HAS_VALID_BYTE_COUNT = 1u << 0,
  LOOM_LOW_RESOURCE_BUILD_FLAG_HAS_CACHE_SWIZZLE_STRIDE = 1u << 1,
};
typedef uint32_t loom_low_resource_build_flags_t;
iree_status_t loom_low_resource_build(
    loom_builder_t* builder,
    loom_low_resource_build_flags_t build_flags,
    loom_low_resource_import_kind_t import_kind,
    int64_t index,
    uint32_t semantic_type,
    loom_optional int64_t valid_byte_count,
    loom_optional int64_t cache_swizzle_stride,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_resource_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_LOW_LIVE_IN: Import a target-provided ABI live-in register value at low-function entry.
// %kernarg = low.live_in<amdgpu.kernarg_segment_ptr> : reg<amdgpu.sgpr x2>
LOOM_DEFINE_ISA(loom_low_live_in_isa, LOOM_OP_LOW_LIVE_IN)
LOOM_DEFINE_RESULT(loom_low_live_in_result, 0)
LOOM_DEFINE_ATTR_STRING(loom_low_live_in_source, 0)
LOOM_DEFINE_ATTR_DICT(loom_low_live_in_attrs, 1)
iree_status_t loom_low_live_in_build(
    loom_builder_t* builder,
    loom_string_id_t source,
    loom_optional loom_named_attr_slice_t attrs,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_low_live_in_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the low dialect.
const loom_op_vtable_t* const* loom_low_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the low dialect.
const loom_op_semantics_t* loom_low_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a low op kind, or empty metadata.
loom_op_semantics_t loom_low_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_LOW_OPS_H_
