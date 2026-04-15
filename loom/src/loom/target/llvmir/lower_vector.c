// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lowering for target-neutral Loom vector ops into structured LLVMIR ops.

#include "loom/ops/vector/ops.h"
#include "loom/target/llvmir/lower_internal.h"

static iree_status_t loom_llvmir_lowering_vector_integer_flags(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    uint8_t instance_flags, loom_llvmir_integer_arithmetic_flags_t* out_flags) {
  const uint8_t known_flags =
      LOOM_VECTOR_INTOVERFLOWFLAGS_NSW | LOOM_VECTOR_INTOVERFLOWFLAGS_NUW;
  if (iree_any_bit_set(instance_flags, (uint8_t)~known_flags)) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "unknown vector integer overflow flags");
  }
  loom_llvmir_integer_arithmetic_flags_t flags =
      LOOM_LLVMIR_INTEGER_ARITHMETIC_NONE;
  if (iree_any_bit_set(instance_flags, LOOM_VECTOR_INTOVERFLOWFLAGS_NUW)) {
    flags |= LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_UNSIGNED_WRAP;
  }
  if (iree_any_bit_set(instance_flags, LOOM_VECTOR_INTOVERFLOWFLAGS_NSW)) {
    flags |= LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_SIGNED_WRAP;
  }
  *out_flags = flags;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_vector_fast_math_flags(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    uint8_t instance_flags, loom_llvmir_fast_math_flags_t* out_flags) {
  const uint8_t known_flags = LOOM_VECTOR_FLOATASSUMPTIONFLAGS_NNAN |
                              LOOM_VECTOR_FLOATASSUMPTIONFLAGS_NINF |
                              LOOM_VECTOR_FLOATASSUMPTIONFLAGS_NSZ;
  if (iree_any_bit_set(instance_flags, (uint8_t)~known_flags)) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "unknown vector floating-point assumption flags");
  }
  loom_llvmir_fast_math_flags_t flags = LOOM_LLVMIR_FAST_MATH_NONE;
  if (iree_any_bit_set(instance_flags, LOOM_VECTOR_FLOATASSUMPTIONFLAGS_NNAN)) {
    flags |= LOOM_LLVMIR_FAST_MATH_NO_NANS;
  }
  if (iree_any_bit_set(instance_flags, LOOM_VECTOR_FLOATASSUMPTIONFLAGS_NINF)) {
    flags |= LOOM_LLVMIR_FAST_MATH_NO_INFS;
  }
  if (iree_any_bit_set(instance_flags, LOOM_VECTOR_FLOATASSUMPTIONFLAGS_NSZ)) {
    flags |= LOOM_LLVMIR_FAST_MATH_NO_SIGNED_ZEROS;
  }
  *out_flags = flags;
  return iree_ok_status();
}

static bool loom_llvmir_lowering_vector_op_is_float_binop(const loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_VECTOR_ADDF:
    case LOOM_OP_VECTOR_SUBF:
    case LOOM_OP_VECTOR_MULF:
    case LOOM_OP_VECTOR_DIVF:
    case LOOM_OP_VECTOR_REMF:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_llvmir_lowering_vector_binop_from_op(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_llvmir_binop_t* out_binop) {
  switch (op->kind) {
    case LOOM_OP_VECTOR_ADDF:
      *out_binop = LOOM_LLVMIR_BINOP_FADD;
      return iree_ok_status();
    case LOOM_OP_VECTOR_SUBF:
      *out_binop = LOOM_LLVMIR_BINOP_FSUB;
      return iree_ok_status();
    case LOOM_OP_VECTOR_MULF:
      *out_binop = LOOM_LLVMIR_BINOP_FMUL;
      return iree_ok_status();
    case LOOM_OP_VECTOR_DIVF:
      *out_binop = LOOM_LLVMIR_BINOP_FDIV;
      return iree_ok_status();
    case LOOM_OP_VECTOR_REMF:
      *out_binop = LOOM_LLVMIR_BINOP_FREM;
      return iree_ok_status();
    case LOOM_OP_VECTOR_ADDI:
      *out_binop = LOOM_LLVMIR_BINOP_ADD;
      return iree_ok_status();
    case LOOM_OP_VECTOR_SUBI:
      *out_binop = LOOM_LLVMIR_BINOP_SUB;
      return iree_ok_status();
    case LOOM_OP_VECTOR_MULI:
      *out_binop = LOOM_LLVMIR_BINOP_MUL;
      return iree_ok_status();
    case LOOM_OP_VECTOR_DIVSI:
      *out_binop = LOOM_LLVMIR_BINOP_SDIV;
      return iree_ok_status();
    case LOOM_OP_VECTOR_DIVUI:
      *out_binop = LOOM_LLVMIR_BINOP_UDIV;
      return iree_ok_status();
    case LOOM_OP_VECTOR_REMSI:
      *out_binop = LOOM_LLVMIR_BINOP_SREM;
      return iree_ok_status();
    case LOOM_OP_VECTOR_REMUI:
      *out_binop = LOOM_LLVMIR_BINOP_UREM;
      return iree_ok_status();
    case LOOM_OP_VECTOR_ANDI:
      *out_binop = LOOM_LLVMIR_BINOP_AND;
      return iree_ok_status();
    case LOOM_OP_VECTOR_ORI:
      *out_binop = LOOM_LLVMIR_BINOP_OR;
      return iree_ok_status();
    case LOOM_OP_VECTOR_XORI:
      *out_binop = LOOM_LLVMIR_BINOP_XOR;
      return iree_ok_status();
    case LOOM_OP_VECTOR_SHLI:
      *out_binop = LOOM_LLVMIR_BINOP_SHL;
      return iree_ok_status();
    case LOOM_OP_VECTOR_SHRSI:
      *out_binop = LOOM_LLVMIR_BINOP_ASHR;
      return iree_ok_status();
    case LOOM_OP_VECTOR_SHRUI:
      *out_binop = LOOM_LLVMIR_BINOP_LSHR;
      return iree_ok_status();
    default:
      return loom_llvmir_lowering_unsupported_op(
          state, op, "vector op has no direct LLVM binary opcode mapping");
  }
}

iree_status_t loom_llvmir_lowering_lower_vector_binop(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t operands[2] = {LOOM_LLVMIR_VALUE_ID_INVALID,
                                        LOOM_LLVMIR_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_operands(state, op, operands));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module,
                             loom_op_const_results(op)[0]),
      &result_type));
  loom_llvmir_binop_t binop = LOOM_LLVMIR_BINOP_ADD;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_vector_binop_from_op(state, op, &binop));

  loom_llvmir_binop_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
  desc.result_type = result_type;
  desc.op = binop;
  desc.lhs = operands[0];
  desc.rhs = operands[1];
  if (loom_llvmir_lowering_vector_op_is_float_binop(op)) {
    IREE_RETURN_IF_ERROR(loom_llvmir_lowering_vector_fast_math_flags(
        state, op, op->instance_flags, &desc.fast_math_flags));
  } else {
    IREE_RETURN_IF_ERROR(loom_llvmir_lowering_vector_integer_flags(
        state, op, op->instance_flags, &desc.integer_flags));
  }
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_binop(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

iree_status_t loom_llvmir_lowering_lower_vector_negf(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_value(
      state, loom_vector_negf_input(op), &value));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module, loom_vector_negf_result(op)),
      &result_type));

  loom_llvmir_unop_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_vector_negf_result(op));
  desc.result_type = result_type;
  desc.op = LOOM_LLVMIR_UNOP_FNEG;
  desc.value = value;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_vector_fast_math_flags(
      state, op, op->instance_flags, &desc.fast_math_flags));
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_unop(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

static iree_status_t loom_llvmir_lowering_vector_icmp_predicate(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    uint8_t source_predicate, loom_llvmir_icmp_predicate_t* out_predicate) {
  switch (source_predicate) {
    case LOOM_VECTOR_CMPI_PREDICATE_EQ:
      *out_predicate = LOOM_LLVMIR_ICMP_EQ;
      return iree_ok_status();
    case LOOM_VECTOR_CMPI_PREDICATE_NE:
      *out_predicate = LOOM_LLVMIR_ICMP_NE;
      return iree_ok_status();
    case LOOM_VECTOR_CMPI_PREDICATE_SLT:
      *out_predicate = LOOM_LLVMIR_ICMP_SLT;
      return iree_ok_status();
    case LOOM_VECTOR_CMPI_PREDICATE_SLE:
      *out_predicate = LOOM_LLVMIR_ICMP_SLE;
      return iree_ok_status();
    case LOOM_VECTOR_CMPI_PREDICATE_SGT:
      *out_predicate = LOOM_LLVMIR_ICMP_SGT;
      return iree_ok_status();
    case LOOM_VECTOR_CMPI_PREDICATE_SGE:
      *out_predicate = LOOM_LLVMIR_ICMP_SGE;
      return iree_ok_status();
    case LOOM_VECTOR_CMPI_PREDICATE_ULT:
      *out_predicate = LOOM_LLVMIR_ICMP_ULT;
      return iree_ok_status();
    case LOOM_VECTOR_CMPI_PREDICATE_ULE:
      *out_predicate = LOOM_LLVMIR_ICMP_ULE;
      return iree_ok_status();
    case LOOM_VECTOR_CMPI_PREDICATE_UGT:
      *out_predicate = LOOM_LLVMIR_ICMP_UGT;
      return iree_ok_status();
    case LOOM_VECTOR_CMPI_PREDICATE_UGE:
      *out_predicate = LOOM_LLVMIR_ICMP_UGE;
      return iree_ok_status();
    default:
      return loom_llvmir_lowering_unsupported_op(
          state, op, "unknown vector integer comparison predicate");
  }
}

iree_status_t loom_llvmir_lowering_lower_vector_icmp(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t operands[2] = {LOOM_LLVMIR_VALUE_ID_INVALID,
                                        LOOM_LLVMIR_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_operands(state, op, operands));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module,
                             loom_op_const_results(op)[0]),
      &result_type));
  loom_llvmir_icmp_predicate_t predicate = LOOM_LLVMIR_ICMP_EQ;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_vector_icmp_predicate(
      state, op, loom_vector_cmpi_predicate(op), &predicate));

  loom_llvmir_icmp_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
  desc.result_type = result_type;
  desc.predicate = predicate;
  desc.lhs = operands[0];
  desc.rhs = operands[1];
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_icmp(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

static iree_status_t loom_llvmir_lowering_vector_fcmp_predicate(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    uint8_t source_predicate, loom_llvmir_fcmp_predicate_t* out_predicate) {
  switch (source_predicate) {
    case LOOM_VECTOR_CMPF_PREDICATE_OEQ:
      *out_predicate = LOOM_LLVMIR_FCMP_OEQ;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_OGT:
      *out_predicate = LOOM_LLVMIR_FCMP_OGT;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_OGE:
      *out_predicate = LOOM_LLVMIR_FCMP_OGE;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_OLT:
      *out_predicate = LOOM_LLVMIR_FCMP_OLT;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_OLE:
      *out_predicate = LOOM_LLVMIR_FCMP_OLE;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_ONE:
      *out_predicate = LOOM_LLVMIR_FCMP_ONE;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_ORD:
      *out_predicate = LOOM_LLVMIR_FCMP_ORD;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_UEQ:
      *out_predicate = LOOM_LLVMIR_FCMP_UEQ;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_UGT:
      *out_predicate = LOOM_LLVMIR_FCMP_UGT;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_UGE:
      *out_predicate = LOOM_LLVMIR_FCMP_UGE;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_ULT:
      *out_predicate = LOOM_LLVMIR_FCMP_ULT;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_ULE:
      *out_predicate = LOOM_LLVMIR_FCMP_ULE;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_UNE:
      *out_predicate = LOOM_LLVMIR_FCMP_UNE;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_UNO:
      *out_predicate = LOOM_LLVMIR_FCMP_UNO;
      return iree_ok_status();
    default:
      return loom_llvmir_lowering_unsupported_op(
          state, op, "unknown vector floating-point comparison predicate");
  }
}

iree_status_t loom_llvmir_lowering_lower_vector_fcmp(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t operands[2] = {LOOM_LLVMIR_VALUE_ID_INVALID,
                                        LOOM_LLVMIR_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_operands(state, op, operands));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module,
                             loom_op_const_results(op)[0]),
      &result_type));
  loom_llvmir_fcmp_predicate_t predicate = LOOM_LLVMIR_FCMP_FALSE;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_vector_fcmp_predicate(
      state, op, loom_vector_cmpf_predicate(op), &predicate));

  loom_llvmir_fcmp_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
  desc.result_type = result_type;
  desc.predicate = predicate;
  desc.lhs = operands[0];
  desc.rhs = operands[1];
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_fcmp(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

iree_status_t loom_llvmir_lowering_lower_vector_select(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t operands[3] = {LOOM_LLVMIR_VALUE_ID_INVALID,
                                        LOOM_LLVMIR_VALUE_ID_INVALID,
                                        LOOM_LLVMIR_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_operands(state, op, operands));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module,
                             loom_vector_select_result(op)),
      &result_type));

  loom_llvmir_select_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_vector_select_result(op));
  desc.result_type = result_type;
  desc.condition = operands[0];
  desc.true_value = operands[1];
  desc.false_value = operands[2];
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_select(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

static iree_status_t loom_llvmir_lowering_vector_cast_op_from_kind(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_llvmir_cast_op_t* out_cast_op) {
  switch (op->kind) {
    case LOOM_OP_VECTOR_SITOFP:
      *out_cast_op = LOOM_LLVMIR_CAST_SIGNED_INT_TO_FP;
      return iree_ok_status();
    case LOOM_OP_VECTOR_UITOFP:
      *out_cast_op = LOOM_LLVMIR_CAST_UNSIGNED_INT_TO_FP;
      return iree_ok_status();
    case LOOM_OP_VECTOR_FPTOSI:
      *out_cast_op = LOOM_LLVMIR_CAST_FP_TO_SIGNED_INT;
      return iree_ok_status();
    case LOOM_OP_VECTOR_FPTOUI:
      *out_cast_op = LOOM_LLVMIR_CAST_FP_TO_UNSIGNED_INT;
      return iree_ok_status();
    case LOOM_OP_VECTOR_EXTF:
      *out_cast_op = LOOM_LLVMIR_CAST_FP_EXTEND;
      return iree_ok_status();
    case LOOM_OP_VECTOR_FPTRUNC:
      *out_cast_op = LOOM_LLVMIR_CAST_FP_TRUNCATE;
      return iree_ok_status();
    case LOOM_OP_VECTOR_EXTSI:
      *out_cast_op = LOOM_LLVMIR_CAST_SIGN_EXTEND;
      return iree_ok_status();
    case LOOM_OP_VECTOR_EXTUI:
      *out_cast_op = LOOM_LLVMIR_CAST_ZERO_EXTEND;
      return iree_ok_status();
    case LOOM_OP_VECTOR_TRUNCI:
      *out_cast_op = LOOM_LLVMIR_CAST_TRUNCATE;
      return iree_ok_status();
    case LOOM_OP_VECTOR_BITCAST:
      *out_cast_op = LOOM_LLVMIR_CAST_BITCAST;
      return iree_ok_status();
    default:
      return loom_llvmir_lowering_unsupported_op(
          state, op, "vector op has no direct LLVM cast opcode mapping");
  }
}

iree_status_t loom_llvmir_lowering_lower_vector_cast(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t operand = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_value(
      state, loom_op_const_operands(op)[0], &operand));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module,
                             loom_op_const_results(op)[0]),
      &result_type));
  loom_llvmir_cast_op_t cast_op = LOOM_LLVMIR_CAST_TRUNCATE;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_vector_cast_op_from_kind(state, op, &cast_op));

  loom_llvmir_cast_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
  desc.result_type = result_type;
  desc.op = cast_op;
  desc.value = operand;
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_cast(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}
