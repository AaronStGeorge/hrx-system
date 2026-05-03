// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/emit/llvmir/lower/internal.h"

static_assert((int)LOOM_INDEX_CMP_PREDICATE_EQ ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_EQ,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_NE ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_NE,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_SLT ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_SLT,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_SLE ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_SLE,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_SGT ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_SGT,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_SGE ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_SGE,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_ULT ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_ULT,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_ULE ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_ULE,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_UGT ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_UGT,
              "index and scalar integer predicate enums must align");
static_assert((int)LOOM_INDEX_CMP_PREDICATE_UGE ==
                  (int)LOOM_SCALAR_CMPI_PREDICATE_UGE,
              "index and scalar integer predicate enums must align");

static loom_llvmir_integer_arithmetic_flags_t
loom_llvmir_lowering_integer_flags(uint8_t instance_flags) {
  loom_llvmir_integer_arithmetic_flags_t flags =
      LOOM_LLVMIR_INTEGER_ARITHMETIC_NONE;
  if (iree_any_bit_set(instance_flags, LOOM_SCALAR_INTOVERFLOWFLAGS_NUW)) {
    flags |= LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_UNSIGNED_WRAP;
  }
  if (iree_any_bit_set(instance_flags, LOOM_SCALAR_INTOVERFLOWFLAGS_NSW)) {
    flags |= LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_SIGNED_WRAP;
  }
  return flags;
}

static loom_llvmir_fast_math_flags_t loom_llvmir_lowering_fast_math_flags(
    uint8_t instance_flags) {
  loom_llvmir_fast_math_flags_t flags = LOOM_LLVMIR_FAST_MATH_NONE;
  if (iree_any_bit_set(instance_flags, LOOM_SCALAR_FASTMATHFLAGS_REASSOC)) {
    flags |= LOOM_LLVMIR_FAST_MATH_ALLOW_REASSOC;
  }
  if (iree_any_bit_set(instance_flags, LOOM_SCALAR_FASTMATHFLAGS_NNAN)) {
    flags |= LOOM_LLVMIR_FAST_MATH_NO_NANS;
  }
  if (iree_any_bit_set(instance_flags, LOOM_SCALAR_FASTMATHFLAGS_NINF)) {
    flags |= LOOM_LLVMIR_FAST_MATH_NO_INFS;
  }
  if (iree_any_bit_set(instance_flags, LOOM_SCALAR_FASTMATHFLAGS_NSZ)) {
    flags |= LOOM_LLVMIR_FAST_MATH_NO_SIGNED_ZEROS;
  }
  if (iree_any_bit_set(instance_flags, LOOM_SCALAR_FASTMATHFLAGS_ARCP)) {
    flags |= LOOM_LLVMIR_FAST_MATH_ALLOW_RECIPROCAL;
  }
  if (iree_any_bit_set(instance_flags, LOOM_SCALAR_FASTMATHFLAGS_CONTRACT)) {
    flags |= LOOM_LLVMIR_FAST_MATH_ALLOW_CONTRACT;
  }
  if (iree_any_bit_set(instance_flags, LOOM_SCALAR_FASTMATHFLAGS_AFN)) {
    flags |= LOOM_LLVMIR_FAST_MATH_APPROX_FUNC;
  }
  return flags;
}

static bool loom_llvmir_lowering_op_is_float_binop(const loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_SCALAR_ADDF:
    case LOOM_OP_SCALAR_SUBF:
    case LOOM_OP_SCALAR_MULF:
    case LOOM_OP_SCALAR_DIVF:
    case LOOM_OP_SCALAR_REMF:
      return true;
    default:
      return false;
  }
}

static bool loom_llvmir_lowering_index_type_is_unsigned(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET;
}

static iree_status_t loom_llvmir_lowering_binop_from_op(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_llvmir_binop_t* out_binop) {
  switch (op->kind) {
    case LOOM_OP_SCALAR_ADDI:
    case LOOM_OP_INDEX_ADD:
      *out_binop = LOOM_LLVMIR_BINOP_ADD;
      return iree_ok_status();
    case LOOM_OP_SCALAR_SUBI:
    case LOOM_OP_INDEX_SUB:
      *out_binop = LOOM_LLVMIR_BINOP_SUB;
      return iree_ok_status();
    case LOOM_OP_SCALAR_MULI:
    case LOOM_OP_INDEX_MUL:
      *out_binop = LOOM_LLVMIR_BINOP_MUL;
      return iree_ok_status();
    case LOOM_OP_SCALAR_DIVSI:
      *out_binop = LOOM_LLVMIR_BINOP_SDIV;
      return iree_ok_status();
    case LOOM_OP_SCALAR_DIVUI:
      *out_binop = LOOM_LLVMIR_BINOP_UDIV;
      return iree_ok_status();
    case LOOM_OP_INDEX_DIV: {
      loom_type_t result_type =
          loom_module_value_type(state->source_module, loom_op_results(op)[0]);
      *out_binop = loom_llvmir_lowering_index_type_is_unsigned(result_type)
                       ? LOOM_LLVMIR_BINOP_UDIV
                       : LOOM_LLVMIR_BINOP_SDIV;
      return iree_ok_status();
    }
    case LOOM_OP_SCALAR_REMSI:
      *out_binop = LOOM_LLVMIR_BINOP_SREM;
      return iree_ok_status();
    case LOOM_OP_SCALAR_REMUI:
      *out_binop = LOOM_LLVMIR_BINOP_UREM;
      return iree_ok_status();
    case LOOM_OP_INDEX_REM: {
      loom_type_t result_type =
          loom_module_value_type(state->source_module, loom_op_results(op)[0]);
      *out_binop = loom_llvmir_lowering_index_type_is_unsigned(result_type)
                       ? LOOM_LLVMIR_BINOP_UREM
                       : LOOM_LLVMIR_BINOP_SREM;
      return iree_ok_status();
    }
    case LOOM_OP_SCALAR_ADDF:
      *out_binop = LOOM_LLVMIR_BINOP_FADD;
      return iree_ok_status();
    case LOOM_OP_SCALAR_SUBF:
      *out_binop = LOOM_LLVMIR_BINOP_FSUB;
      return iree_ok_status();
    case LOOM_OP_SCALAR_MULF:
      *out_binop = LOOM_LLVMIR_BINOP_FMUL;
      return iree_ok_status();
    case LOOM_OP_SCALAR_DIVF:
      *out_binop = LOOM_LLVMIR_BINOP_FDIV;
      return iree_ok_status();
    case LOOM_OP_SCALAR_REMF:
      *out_binop = LOOM_LLVMIR_BINOP_FREM;
      return iree_ok_status();
    case LOOM_OP_SCALAR_ANDI:
      *out_binop = LOOM_LLVMIR_BINOP_AND;
      return iree_ok_status();
    case LOOM_OP_SCALAR_ORI:
      *out_binop = LOOM_LLVMIR_BINOP_OR;
      return iree_ok_status();
    case LOOM_OP_SCALAR_XORI:
      *out_binop = LOOM_LLVMIR_BINOP_XOR;
      return iree_ok_status();
    case LOOM_OP_SCALAR_SHLI:
      *out_binop = LOOM_LLVMIR_BINOP_SHL;
      return iree_ok_status();
    case LOOM_OP_SCALAR_SHRSI:
      *out_binop = LOOM_LLVMIR_BINOP_ASHR;
      return iree_ok_status();
    case LOOM_OP_SCALAR_SHRUI:
      *out_binop = LOOM_LLVMIR_BINOP_LSHR;
      return iree_ok_status();
    default:
      return loom_llvmir_lowering_unsupported_op(state, op,
                                                 "no binary opcode mapping");
  }
}

iree_status_t loom_llvmir_lowering_lower_binop(
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
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_binop_from_op(state, op, &binop));

  loom_llvmir_binop_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
  desc.result_type = result_type;
  desc.op = binop;
  desc.lhs = operands[0];
  desc.rhs = operands[1];
  if (loom_llvmir_lowering_op_is_float_binop(op)) {
    desc.fast_math_flags =
        loom_llvmir_lowering_fast_math_flags(op->instance_flags);
  } else {
    desc.integer_flags = loom_llvmir_lowering_integer_flags(op->instance_flags);
  }
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_binop(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

iree_status_t loom_llvmir_lowering_lower_negf(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_value(
      state, loom_op_const_operands(op)[0], &value));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module,
                             loom_op_const_results(op)[0]),
      &result_type));

  loom_llvmir_unop_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
  desc.result_type = result_type;
  desc.op = LOOM_LLVMIR_UNOP_FNEG;
  desc.value = value;
  desc.fast_math_flags =
      loom_llvmir_lowering_fast_math_flags(op->instance_flags);
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_unop(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

iree_status_t loom_llvmir_lowering_lower_constant(
    loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_attribute_t value_attr) {
  loom_type_t result_source_type = loom_module_value_type(
      state->source_module, loom_op_const_results(op)[0]);
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lower_type(state, result_source_type, &result_type));

  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  if (loom_type_is_scalar(result_source_type) &&
      loom_scalar_type_is_integer(loom_type_element_type(result_source_type))) {
    if (value_attr.kind != LOOM_ATTR_I64 && value_attr.kind != LOOM_ATTR_BOOL) {
      return loom_llvmir_lowering_unsupported_op(
          state, op, "integer constants require an i64 or bool attribute");
    }
    uint64_t value = value_attr.kind == LOOM_ATTR_BOOL
                         ? value_attr.raw
                         : (uint64_t)value_attr.i64;
    IREE_RETURN_IF_ERROR(loom_llvmir_module_add_integer_constant(
        state->target_module, result_type, value, &result));
  } else if (loom_type_is_scalar(result_source_type) &&
             (loom_type_element_type(result_source_type) ==
                  LOOM_SCALAR_TYPE_INDEX ||
              loom_type_element_type(result_source_type) ==
                  LOOM_SCALAR_TYPE_OFFSET)) {
    if (value_attr.kind != LOOM_ATTR_I64) {
      return loom_llvmir_lowering_unsupported_op(
          state, op, "index/offset constants require an i64 attribute");
    }
    IREE_RETURN_IF_ERROR(loom_llvmir_module_add_integer_constant(
        state->target_module, result_type, (uint64_t)value_attr.i64, &result));
  } else if (loom_type_is_scalar(result_source_type) &&
             loom_type_element_type(result_source_type) ==
                 LOOM_SCALAR_TYPE_F64) {
    if (value_attr.kind != LOOM_ATTR_F64) {
      return loom_llvmir_lowering_unsupported_op(
          state, op, "f64 constants require an f64 attribute");
    }
    uint64_t bits = 0;
    memcpy(&bits, &value_attr.f64, sizeof(bits));
    IREE_RETURN_IF_ERROR(loom_llvmir_module_add_float_bits_constant(
        state->target_module, result_type, bits, &result));
  } else {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "constant attribute/type combination is not supported yet");
  }
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

iree_status_t loom_llvmir_lowering_lower_index_madd(
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
                             loom_op_const_results(op)[0]),
      &result_type));

  loom_llvmir_binop_desc_t multiply_desc = {0};
  multiply_desc.result_type = result_type;
  multiply_desc.op = LOOM_LLVMIR_BINOP_MUL;
  multiply_desc.lhs = operands[0];
  multiply_desc.rhs = operands[1];
  loom_llvmir_value_id_t product = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(target_block, &multiply_desc, &product));

  loom_llvmir_binop_desc_t add_desc = {0};
  add_desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
  add_desc.result_type = result_type;
  add_desc.op = LOOM_LLVMIR_BINOP_ADD;
  add_desc.lhs = product;
  add_desc.rhs = operands[2];
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_binop(target_block, &add_desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

typedef enum loom_llvmir_index_minmax_kind_e {
  LOOM_LLVMIR_INDEX_MINMAX_MIN,
  LOOM_LLVMIR_INDEX_MINMAX_MAX,
} loom_llvmir_index_minmax_kind_t;

static iree_status_t loom_llvmir_lowering_lower_index_minmax(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op, loom_llvmir_index_minmax_kind_t kind) {
  loom_llvmir_value_id_t operands[2] = {LOOM_LLVMIR_VALUE_ID_INVALID,
                                        LOOM_LLVMIR_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_operands(state, op, operands));

  loom_llvmir_type_id_t condition_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_scalar_type(
      state, LOOM_SCALAR_TYPE_I1, &condition_type));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module,
                             loom_op_const_results(op)[0]),
      &result_type));

  loom_llvmir_icmp_desc_t cmp_desc = {0};
  cmp_desc.result_name = iree_string_view_empty();
  cmp_desc.result_type = condition_type;
  cmp_desc.predicate = LOOM_LLVMIR_ICMP_SLE;
  cmp_desc.lhs = operands[0];
  cmp_desc.rhs = operands[1];
  loom_llvmir_value_id_t condition = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_icmp(target_block, &cmp_desc, &condition));

  loom_llvmir_select_desc_t select_desc = {0};
  select_desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
  select_desc.result_type = result_type;
  select_desc.condition = condition;
  const bool select_lhs_on_true = kind == LOOM_LLVMIR_INDEX_MINMAX_MIN;
  select_desc.true_value = select_lhs_on_true ? operands[0] : operands[1];
  select_desc.false_value = select_lhs_on_true ? operands[1] : operands[0];
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_select(target_block, &select_desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

iree_status_t loom_llvmir_lowering_lower_index_min(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  return loom_llvmir_lowering_lower_index_minmax(state, target_block, op,
                                                 LOOM_LLVMIR_INDEX_MINMAX_MIN);
}

iree_status_t loom_llvmir_lowering_lower_index_max(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  return loom_llvmir_lowering_lower_index_minmax(state, target_block, op,
                                                 LOOM_LLVMIR_INDEX_MINMAX_MAX);
}

static iree_status_t loom_llvmir_lowering_icmp_predicate(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    uint8_t source_predicate, loom_llvmir_icmp_predicate_t* out_predicate) {
  switch (source_predicate) {
    case LOOM_SCALAR_CMPI_PREDICATE_EQ:
      *out_predicate = LOOM_LLVMIR_ICMP_EQ;
      return iree_ok_status();
    case LOOM_SCALAR_CMPI_PREDICATE_NE:
      *out_predicate = LOOM_LLVMIR_ICMP_NE;
      return iree_ok_status();
    case LOOM_SCALAR_CMPI_PREDICATE_SLT:
      *out_predicate = LOOM_LLVMIR_ICMP_SLT;
      return iree_ok_status();
    case LOOM_SCALAR_CMPI_PREDICATE_SLE:
      *out_predicate = LOOM_LLVMIR_ICMP_SLE;
      return iree_ok_status();
    case LOOM_SCALAR_CMPI_PREDICATE_SGT:
      *out_predicate = LOOM_LLVMIR_ICMP_SGT;
      return iree_ok_status();
    case LOOM_SCALAR_CMPI_PREDICATE_SGE:
      *out_predicate = LOOM_LLVMIR_ICMP_SGE;
      return iree_ok_status();
    case LOOM_SCALAR_CMPI_PREDICATE_ULT:
      *out_predicate = LOOM_LLVMIR_ICMP_ULT;
      return iree_ok_status();
    case LOOM_SCALAR_CMPI_PREDICATE_ULE:
      *out_predicate = LOOM_LLVMIR_ICMP_ULE;
      return iree_ok_status();
    case LOOM_SCALAR_CMPI_PREDICATE_UGT:
      *out_predicate = LOOM_LLVMIR_ICMP_UGT;
      return iree_ok_status();
    case LOOM_SCALAR_CMPI_PREDICATE_UGE:
      *out_predicate = LOOM_LLVMIR_ICMP_UGE;
      return iree_ok_status();
    default:
      return loom_llvmir_lowering_unsupported_op(
          state, op, "unknown integer comparison predicate");
  }
}

static iree_status_t loom_llvmir_lowering_fcmp_predicate(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    uint8_t source_predicate, loom_llvmir_fcmp_predicate_t* out_predicate) {
  switch (source_predicate) {
    case LOOM_SCALAR_CMPF_PREDICATE_OEQ:
      *out_predicate = LOOM_LLVMIR_FCMP_OEQ;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_OGT:
      *out_predicate = LOOM_LLVMIR_FCMP_OGT;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_OGE:
      *out_predicate = LOOM_LLVMIR_FCMP_OGE;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_OLT:
      *out_predicate = LOOM_LLVMIR_FCMP_OLT;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_OLE:
      *out_predicate = LOOM_LLVMIR_FCMP_OLE;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_ONE:
      *out_predicate = LOOM_LLVMIR_FCMP_ONE;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_ORD:
      *out_predicate = LOOM_LLVMIR_FCMP_ORD;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_UEQ:
      *out_predicate = LOOM_LLVMIR_FCMP_UEQ;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_UGT:
      *out_predicate = LOOM_LLVMIR_FCMP_UGT;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_UGE:
      *out_predicate = LOOM_LLVMIR_FCMP_UGE;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_ULT:
      *out_predicate = LOOM_LLVMIR_FCMP_ULT;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_ULE:
      *out_predicate = LOOM_LLVMIR_FCMP_ULE;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_UNE:
      *out_predicate = LOOM_LLVMIR_FCMP_UNE;
      return iree_ok_status();
    case LOOM_SCALAR_CMPF_PREDICATE_UNO:
      *out_predicate = LOOM_LLVMIR_FCMP_UNO;
      return iree_ok_status();
    default:
      return loom_llvmir_lowering_unsupported_op(
          state, op, "unknown floating-point comparison predicate");
  }
}

iree_status_t loom_llvmir_lowering_lower_icmp(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op, uint8_t source_predicate) {
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
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_icmp_predicate(
      state, op, source_predicate, &predicate));

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

iree_status_t loom_llvmir_lowering_lower_fcmp(
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
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_fcmp_predicate(
      state, op, loom_scalar_cmpf_predicate(op), &predicate));

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

static iree_status_t loom_llvmir_lowering_cast_op_from_kind(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_llvmir_cast_op_t* out_cast_op) {
  switch (op->kind) {
    case LOOM_OP_SCALAR_SITOFP:
      *out_cast_op = LOOM_LLVMIR_CAST_SIGNED_INT_TO_FP;
      return iree_ok_status();
    case LOOM_OP_SCALAR_UITOFP:
      *out_cast_op = LOOM_LLVMIR_CAST_UNSIGNED_INT_TO_FP;
      return iree_ok_status();
    case LOOM_OP_SCALAR_FPTOSI:
      *out_cast_op = LOOM_LLVMIR_CAST_FP_TO_SIGNED_INT;
      return iree_ok_status();
    case LOOM_OP_SCALAR_FPTOUI:
      *out_cast_op = LOOM_LLVMIR_CAST_FP_TO_UNSIGNED_INT;
      return iree_ok_status();
    case LOOM_OP_SCALAR_EXTF:
      *out_cast_op = LOOM_LLVMIR_CAST_FP_EXTEND;
      return iree_ok_status();
    case LOOM_OP_SCALAR_FPTRUNC:
      *out_cast_op = LOOM_LLVMIR_CAST_FP_TRUNCATE;
      return iree_ok_status();
    case LOOM_OP_SCALAR_EXTSI:
      *out_cast_op = LOOM_LLVMIR_CAST_SIGN_EXTEND;
      return iree_ok_status();
    case LOOM_OP_SCALAR_EXTUI:
      *out_cast_op = LOOM_LLVMIR_CAST_ZERO_EXTEND;
      return iree_ok_status();
    case LOOM_OP_SCALAR_TRUNCI:
      *out_cast_op = LOOM_LLVMIR_CAST_TRUNCATE;
      return iree_ok_status();
    case LOOM_OP_SCALAR_BITCAST:
      *out_cast_op = LOOM_LLVMIR_CAST_BITCAST;
      return iree_ok_status();
    default:
      return loom_llvmir_lowering_unsupported_op(state, op,
                                                 "no cast opcode mapping");
  }
}

iree_status_t loom_llvmir_lowering_lower_cast(
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
      loom_llvmir_lowering_cast_op_from_kind(state, op, &cast_op));

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

static iree_status_t loom_llvmir_lowering_integer_bitwidth(
    loom_llvmir_lowering_state_t* state, loom_type_t type,
    uint32_t* out_bitwidth) {
  if (!loom_type_is_scalar(type)) {
    return loom_llvmir_lowering_unsupported_type(
        state, type, "index.cast only supports scalar integer-like types");
  }
  switch (loom_type_element_type(type)) {
    case LOOM_SCALAR_TYPE_INDEX:
      *out_bitwidth = state->target_profile->target_env->index_bitwidth;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_OFFSET:
      *out_bitwidth = state->target_profile->target_env->offset_bitwidth;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I1:
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_I64:
      *out_bitwidth =
          (uint32_t)loom_scalar_type_bitwidth(loom_type_element_type(type));
      return iree_ok_status();
    default:
      return loom_llvmir_lowering_unsupported_type(
          state, type, "index.cast only supports integer-like scalar types");
  }
}

iree_status_t loom_llvmir_lowering_lower_index_cast(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_value_id_t source_value_id = loom_op_const_operands(op)[0];
  loom_value_id_t result_value_id = loom_op_const_results(op)[0];
  loom_llvmir_value_id_t operand = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_value(state, source_value_id, &operand));

  loom_type_t source_type =
      loom_module_value_type(state->source_module, source_value_id);
  loom_type_t result_source_type =
      loom_module_value_type(state->source_module, result_value_id);
  uint32_t source_bitwidth = 0;
  uint32_t result_bitwidth = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_integer_bitwidth(state, source_type,
                                                             &source_bitwidth));
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_integer_bitwidth(
      state, result_source_type, &result_bitwidth));
  if (source_bitwidth == result_bitwidth) {
    return loom_llvmir_lowering_map_value(state, result_value_id, operand);
  }

  loom_llvmir_cast_op_t cast_op = LOOM_LLVMIR_CAST_TRUNCATE;
  if (source_bitwidth < result_bitwidth) {
    cast_op = loom_llvmir_lowering_index_type_is_unsigned(source_type)
                  ? LOOM_LLVMIR_CAST_ZERO_EXTEND
                  : LOOM_LLVMIR_CAST_SIGN_EXTEND;
  }
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lower_type(state, result_source_type, &result_type));
  loom_llvmir_cast_desc_t desc = {0};
  desc.result_name = loom_llvmir_lowering_value_name(state, result_value_id);
  desc.result_type = result_type;
  desc.op = cast_op;
  desc.value = operand;
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_cast(target_block, &desc, &result));
  return loom_llvmir_lowering_map_value(state, result_value_id, result);
}

iree_status_t loom_llvmir_lowering_lower_select(
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
                             loom_op_const_results(op)[0]),
      &result_type));

  loom_llvmir_select_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
  desc.result_type = result_type;
  desc.condition = operands[0];
  desc.true_value = operands[1];
  desc.false_value = operands[2];
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_select(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}
