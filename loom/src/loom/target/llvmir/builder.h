// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured LLVM IR instruction builder.

#ifndef LOOM_TARGET_LLVMIR_BUILDER_H_
#define LOOM_TARGET_LLVMIR_BUILDER_H_

#include "loom/target/llvmir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_llvmir_binop_e {
  LOOM_LLVMIR_BINOP_ADD = 0,
  LOOM_LLVMIR_BINOP_SUB = 1,
  LOOM_LLVMIR_BINOP_MUL = 2,
  LOOM_LLVMIR_BINOP_AND = 3,
  LOOM_LLVMIR_BINOP_OR = 4,
  LOOM_LLVMIR_BINOP_XOR = 5,
  LOOM_LLVMIR_BINOP_SHL = 6,
  LOOM_LLVMIR_BINOP_LSHR = 7,
  LOOM_LLVMIR_BINOP_ASHR = 8,
  LOOM_LLVMIR_BINOP_FADD = 9,
  LOOM_LLVMIR_BINOP_FSUB = 10,
  LOOM_LLVMIR_BINOP_FMUL = 11,
} loom_llvmir_binop_t;

typedef enum loom_llvmir_memory_flag_bits_e {
  LOOM_LLVMIR_MEMORY_VOLATILE = 1u << 0,
} loom_llvmir_memory_flags_t;

typedef enum loom_llvmir_inline_asm_flag_bits_e {
  LOOM_LLVMIR_INLINE_ASM_SIDE_EFFECT = 1u << 0,
  LOOM_LLVMIR_INLINE_ASM_ALIGN_STACK = 1u << 1,
  LOOM_LLVMIR_INLINE_ASM_INTEL_DIALECT = 1u << 2,
} loom_llvmir_inline_asm_flags_t;

typedef struct loom_llvmir_phi_incoming_t {
  // Incoming SSA value from the predecessor block.
  loom_llvmir_value_id_t value;
  // Predecessor block that branches to the phi's containing block.
  loom_llvmir_block_id_t predecessor;
} loom_llvmir_phi_incoming_t;

typedef struct loom_llvmir_binop_desc_t {
  // Optional result name without the leading percent sign.
  iree_string_view_t result_name;
  // Result type and expected operand type.
  loom_llvmir_type_id_t result_type;
  // Binary operation opcode.
  loom_llvmir_binop_t op;
  // Left operand value.
  loom_llvmir_value_id_t lhs;
  // Right operand value.
  loom_llvmir_value_id_t rhs;
} loom_llvmir_binop_desc_t;

typedef struct loom_llvmir_phi_desc_t {
  // Optional result name without the leading percent sign.
  iree_string_view_t result_name;
  // Result type and expected incoming value type.
  loom_llvmir_type_id_t result_type;
  // Incoming block/value pairs in printed order.
  const loom_llvmir_phi_incoming_t* incoming;
  // Number of entries in |incoming|.
  iree_host_size_t incoming_count;
} loom_llvmir_phi_desc_t;

typedef struct loom_llvmir_gep_desc_t {
  // Optional result name without the leading percent sign.
  iree_string_view_t result_name;
  // Pointer result type.
  loom_llvmir_type_id_t result_type;
  // Pointee element type used by LLVM's opaque-pointer GEP syntax.
  loom_llvmir_type_id_t element_type;
  // Base pointer value.
  loom_llvmir_value_id_t base;
  // Index values in GEP order.
  const loom_llvmir_value_id_t* indices;
  // Number of entries in |indices|.
  iree_host_size_t index_count;
} loom_llvmir_gep_desc_t;

typedef struct loom_llvmir_load_desc_t {
  // Optional result name without the leading percent sign.
  iree_string_view_t result_name;
  // Loaded value type.
  loom_llvmir_type_id_t result_type;
  // Pointer value to load from.
  loom_llvmir_value_id_t pointer;
  // Optional byte alignment, or zero to omit the attribute.
  uint32_t alignment;
  // Memory operation flags.
  loom_llvmir_memory_flags_t flags;
} loom_llvmir_load_desc_t;

typedef struct loom_llvmir_store_desc_t {
  // Value to store.
  loom_llvmir_value_id_t value;
  // Pointer value to store into.
  loom_llvmir_value_id_t pointer;
  // Optional byte alignment, or zero to omit the attribute.
  uint32_t alignment;
  // Memory operation flags.
  loom_llvmir_memory_flags_t flags;
} loom_llvmir_store_desc_t;

typedef struct loom_llvmir_call_desc_t {
  // Optional result name without the leading percent sign.
  iree_string_view_t result_name;
  // Function declaration or definition to call.
  loom_llvmir_function_id_t callee;
  // Call arguments in callee parameter order.
  const loom_llvmir_value_id_t* args;
  // Number of entries in |args|.
  iree_host_size_t arg_count;
  // Attributes attached to the call result.
  const loom_llvmir_attr_t* result_attrs;
  // Number of entries in |result_attrs|.
  iree_host_size_t result_attr_count;
} loom_llvmir_call_desc_t;

typedef struct loom_llvmir_inline_asm_desc_t {
  // Optional result name without the leading percent sign.
  iree_string_view_t result_name;
  // Inline asm call result type.
  loom_llvmir_type_id_t result_type;
  // Inline asm flags.
  loom_llvmir_inline_asm_flags_t flags;
  // Asm template string.
  iree_string_view_t asm_template;
  // Constraint string.
  iree_string_view_t constraints;
  // Inline asm arguments in call order.
  const loom_llvmir_value_id_t* args;
  // Number of entries in |args|.
  iree_host_size_t arg_count;
} loom_llvmir_inline_asm_desc_t;

iree_status_t loom_llvmir_build_phi(loom_llvmir_block_t* block,
                                    const loom_llvmir_phi_desc_t* desc,
                                    loom_llvmir_value_id_t* out_value_id);

iree_status_t loom_llvmir_build_binop(loom_llvmir_block_t* block,
                                      const loom_llvmir_binop_desc_t* desc,
                                      loom_llvmir_value_id_t* out_value_id);

iree_status_t loom_llvmir_build_gep(loom_llvmir_block_t* block,
                                    const loom_llvmir_gep_desc_t* desc,
                                    loom_llvmir_value_id_t* out_value_id);

iree_status_t loom_llvmir_build_load(loom_llvmir_block_t* block,
                                     const loom_llvmir_load_desc_t* desc,
                                     loom_llvmir_value_id_t* out_value_id);

iree_status_t loom_llvmir_build_store(loom_llvmir_block_t* block,
                                      const loom_llvmir_store_desc_t* desc);

iree_status_t loom_llvmir_build_call(loom_llvmir_block_t* block,
                                     const loom_llvmir_call_desc_t* desc,
                                     loom_llvmir_value_id_t* out_value_id);

iree_status_t loom_llvmir_build_inline_asm(
    loom_llvmir_block_t* block, const loom_llvmir_inline_asm_desc_t* desc,
    loom_llvmir_value_id_t* out_value_id);

iree_status_t loom_llvmir_build_ret_void(loom_llvmir_block_t* block);

iree_status_t loom_llvmir_build_ret(loom_llvmir_block_t* block,
                                    loom_llvmir_value_id_t value);

iree_status_t loom_llvmir_build_br(loom_llvmir_block_t* block,
                                   loom_llvmir_block_id_t target);

iree_status_t loom_llvmir_build_cond_br(loom_llvmir_block_t* block,
                                        loom_llvmir_value_id_t condition,
                                        loom_llvmir_block_id_t true_block,
                                        loom_llvmir_block_id_t false_block);

iree_status_t loom_llvmir_build_unreachable(loom_llvmir_block_t* block);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_BUILDER_H_
