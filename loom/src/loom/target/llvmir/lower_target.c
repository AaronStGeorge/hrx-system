// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lowering for target punch-through Loom ops into structured LLVMIR.

#include "loom/ir/context.h"
#include "loom/ops/llvmir/ops.h"
#include "loom/target/llvmir/lower_internal.h"

static iree_status_t loom_llvmir_lowering_string_attr(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    iree_string_view_t attr_name, loom_string_id_t string_id,
    iree_string_view_t* out_string) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= state->source_module->strings.count) {
    iree_string_view_t op_name = loom_op_name(state->source_module, op);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "%.*s attribute %.*s references invalid string id %u",
        (int)op_name.size, op_name.data, (int)attr_name.size, attr_name.data,
        (unsigned)string_id);
  }
  *out_string = state->source_module->strings.entries[string_id];
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_inline_asm_flags(
    loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_llvmir_inline_asm_flags_t* out_flags) {
  uint8_t source_flags = loom_llvmir_inline_asm_flags(op);
  uint8_t supported_flags = LOOM_LLVMIR_ASMFLAGS_SIDEEFFECT |
                            LOOM_LLVMIR_ASMFLAGS_ALIGNSTACK |
                            LOOM_LLVMIR_ASMFLAGS_INTELDIALECT;
  if ((source_flags & ~supported_flags) != 0) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "inline asm has unknown instance flags");
  }

  loom_llvmir_inline_asm_flags_t target_flags = 0;
  if (source_flags & LOOM_LLVMIR_ASMFLAGS_SIDEEFFECT) {
    target_flags |= LOOM_LLVMIR_INLINE_ASM_SIDE_EFFECT;
  }
  if (source_flags & LOOM_LLVMIR_ASMFLAGS_ALIGNSTACK) {
    target_flags |= LOOM_LLVMIR_INLINE_ASM_ALIGN_STACK;
  }
  if (source_flags & LOOM_LLVMIR_ASMFLAGS_INTELDIALECT) {
    target_flags |= LOOM_LLVMIR_INLINE_ASM_INTEL_DIALECT;
  }
  *out_flags = target_flags;
  return iree_ok_status();
}

iree_status_t loom_llvmir_lowering_lower_inline_asm(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  if (op->result_count > 1) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "inline asm lowering supports at most one direct result");
  }
  if (op->tied_result_count > 0) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "inline asm lowering does not support tied results");
  }

  loom_llvmir_inline_asm_flags_t flags = 0;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_inline_asm_flags(state, op, &flags));

  iree_string_view_t asm_template = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_string_attr(
      state, op, IREE_SV("asm_template"),
      loom_llvmir_inline_asm_asm_template(op), &asm_template));
  iree_string_view_t constraints = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_string_attr(
      state, op, IREE_SV("constraints"), loom_llvmir_inline_asm_constraints(op),
      &constraints));

  loom_llvmir_value_id_t* args = NULL;
  iree_status_t status = iree_ok_status();
  if (op->operand_count > 0) {
    status = iree_allocator_malloc(
        state->allocator, op->operand_count * sizeof(loom_llvmir_value_id_t),
        (void**)&args);
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_lowering_lookup_operands(state, op, args);
  }

  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  if (iree_status_is_ok(status) && op->result_count == 0) {
    status =
        loom_llvmir_module_get_void_type(state->target_module, &result_type);
  } else if (iree_status_is_ok(status)) {
    loom_value_id_t result_id = loom_op_const_results(op)[0];
    status = loom_llvmir_lowering_lower_type(
        state, loom_module_value_type(state->source_module, result_id),
        &result_type);
  }

  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  if (iree_status_is_ok(status)) {
    loom_llvmir_inline_asm_desc_t desc = {0};
    desc.result_type = result_type;
    desc.flags = flags;
    desc.asm_template = asm_template;
    desc.constraints = constraints;
    desc.args = args;
    desc.arg_count = op->operand_count;
    if (op->result_count == 1) {
      desc.result_name =
          loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
    }
    status = loom_llvmir_build_inline_asm(target_block, &desc, &result);
  }
  if (iree_status_is_ok(status) && op->result_count == 1) {
    status = loom_llvmir_lowering_map_single_result(state, op, result);
  }
  iree_allocator_free(state->allocator, args);
  return status;
}
