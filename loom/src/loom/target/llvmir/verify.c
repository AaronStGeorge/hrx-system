// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/llvmir/verify.h"

#include "loom/target/llvmir/types.h"

static bool loom_llvmir_is_terminator(const loom_llvmir_instruction_t* inst) {
  return inst->kind == LOOM_LLVMIR_INST_RET ||
         inst->kind == LOOM_LLVMIR_INST_BR ||
         inst->kind == LOOM_LLVMIR_INST_COND_BR ||
         inst->kind == LOOM_LLVMIR_INST_UNREACHABLE;
}

static const loom_llvmir_type_t* loom_llvmir_verify_type(
    const loom_llvmir_module_t* module, loom_llvmir_type_id_t type_id) {
  return type_id < module->type_count ? &module->types[type_id] : NULL;
}

static const loom_llvmir_value_t* loom_llvmir_verify_value(
    const loom_llvmir_module_t* module, loom_llvmir_value_id_t value_id) {
  return value_id < module->value_count ? &module->values[value_id] : NULL;
}

static loom_llvmir_type_id_t loom_llvmir_verify_value_type(
    const loom_llvmir_module_t* module, loom_llvmir_value_id_t value_id) {
  const loom_llvmir_value_t* value = loom_llvmir_verify_value(module, value_id);
  return value ? value->type_id : LOOM_LLVMIR_TYPE_ID_INVALID;
}

static loom_llvmir_function_t* loom_llvmir_verify_function_ref(
    const loom_llvmir_module_t* module, loom_llvmir_function_id_t function_id) {
  return function_id < module->function_count ? module->functions[function_id]
                                              : NULL;
}

static iree_status_t loom_llvmir_verify_expected_value_type(
    const loom_llvmir_module_t* module, loom_llvmir_value_id_t value_id,
    loom_llvmir_type_id_t expected_type) {
  loom_llvmir_type_id_t actual_type =
      loom_llvmir_verify_value_type(module, value_id);
  if (actual_type == LOOM_LLVMIR_TYPE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM instruction references unknown value");
  }
  if (actual_type != expected_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM instruction operand type mismatch");
  }
  return iree_ok_status();
}

static bool loom_llvmir_verify_type_is_i1(const loom_llvmir_module_t* module,
                                          loom_llvmir_type_id_t type_id) {
  const loom_llvmir_type_t* type = loom_llvmir_verify_type(module, type_id);
  return type && type->kind == LOOM_LLVMIR_TYPE_INTEGER && type->bit_width == 1;
}

static iree_status_t loom_llvmir_verify_mask_result_type(
    const loom_llvmir_module_t* module, loom_llvmir_type_id_t operand_type_id,
    loom_llvmir_type_id_t result_type_id) {
  const loom_llvmir_type_t* operand_type =
      loom_llvmir_verify_type(module, operand_type_id);
  const loom_llvmir_type_t* result_type =
      loom_llvmir_verify_type(module, result_type_id);
  if (!operand_type || !result_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM compare references unknown type");
  }
  if (operand_type->kind != LOOM_LLVMIR_TYPE_VECTOR) {
    if (!loom_llvmir_verify_type_is_i1(module, result_type_id)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM scalar compare result must be i1");
    }
    return iree_ok_status();
  }
  if (result_type->kind != LOOM_LLVMIR_TYPE_VECTOR ||
      result_type->element_count != operand_type->element_count ||
      !loom_llvmir_verify_type_is_i1(module, result_type->element_type)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM vector compare result must be a same-lane i1 vector");
  }
  return iree_ok_status();
}

static bool loom_llvmir_verify_type_is_integer_or_pointer_like(
    const loom_llvmir_module_t* module, const loom_llvmir_type_t* type) {
  if (!type) return false;
  if (type->kind == LOOM_LLVMIR_TYPE_INTEGER ||
      type->kind == LOOM_LLVMIR_TYPE_POINTER) {
    return true;
  }
  if (type->kind != LOOM_LLVMIR_TYPE_VECTOR) return false;
  const loom_llvmir_type_t* element_type =
      loom_llvmir_verify_type(module, type->element_type);
  return element_type && (element_type->kind == LOOM_LLVMIR_TYPE_INTEGER ||
                          element_type->kind == LOOM_LLVMIR_TYPE_POINTER);
}

static bool loom_llvmir_verify_type_is_float_like(
    const loom_llvmir_module_t* module, const loom_llvmir_type_t* type) {
  if (!type) return false;
  if (type->kind == LOOM_LLVMIR_TYPE_FLOAT) return true;
  if (type->kind != LOOM_LLVMIR_TYPE_VECTOR) return false;
  const loom_llvmir_type_t* element_type =
      loom_llvmir_verify_type(module, type->element_type);
  return element_type && element_type->kind == LOOM_LLVMIR_TYPE_FLOAT;
}

static bool loom_llvmir_verify_icmp_predicate(
    loom_llvmir_icmp_predicate_t predicate) {
  switch (predicate) {
    case LOOM_LLVMIR_ICMP_EQ:
    case LOOM_LLVMIR_ICMP_NE:
    case LOOM_LLVMIR_ICMP_UGT:
    case LOOM_LLVMIR_ICMP_UGE:
    case LOOM_LLVMIR_ICMP_ULT:
    case LOOM_LLVMIR_ICMP_ULE:
    case LOOM_LLVMIR_ICMP_SGT:
    case LOOM_LLVMIR_ICMP_SGE:
    case LOOM_LLVMIR_ICMP_SLT:
    case LOOM_LLVMIR_ICMP_SLE:
      return true;
    default:
      return false;
  }
}

static bool loom_llvmir_verify_fcmp_predicate(
    loom_llvmir_fcmp_predicate_t predicate) {
  switch (predicate) {
    case LOOM_LLVMIR_FCMP_FALSE:
    case LOOM_LLVMIR_FCMP_OEQ:
    case LOOM_LLVMIR_FCMP_OGT:
    case LOOM_LLVMIR_FCMP_OGE:
    case LOOM_LLVMIR_FCMP_OLT:
    case LOOM_LLVMIR_FCMP_OLE:
    case LOOM_LLVMIR_FCMP_ONE:
    case LOOM_LLVMIR_FCMP_ORD:
    case LOOM_LLVMIR_FCMP_UNO:
    case LOOM_LLVMIR_FCMP_UEQ:
    case LOOM_LLVMIR_FCMP_UGT:
    case LOOM_LLVMIR_FCMP_UGE:
    case LOOM_LLVMIR_FCMP_ULT:
    case LOOM_LLVMIR_FCMP_ULE:
    case LOOM_LLVMIR_FCMP_UNE:
    case LOOM_LLVMIR_FCMP_TRUE:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_llvmir_verify_compare(
    const loom_llvmir_module_t* module,
    const loom_llvmir_instruction_t* instruction, loom_llvmir_value_id_t lhs,
    loom_llvmir_value_id_t rhs,
    bool (*is_valid_operand_type)(const loom_llvmir_module_t*,
                                  const loom_llvmir_type_t*)) {
  if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM compare has no result value");
  }
  loom_llvmir_type_id_t lhs_type_id =
      loom_llvmir_verify_value_type(module, lhs);
  loom_llvmir_type_id_t rhs_type_id =
      loom_llvmir_verify_value_type(module, rhs);
  if (lhs_type_id == LOOM_LLVMIR_TYPE_ID_INVALID ||
      rhs_type_id == LOOM_LLVMIR_TYPE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM compare references unknown value");
  }
  if (lhs_type_id != rhs_type_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM compare operand type mismatch");
  }
  if (!is_valid_operand_type(module,
                             loom_llvmir_verify_type(module, lhs_type_id))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM compare has invalid operand type");
  }
  return loom_llvmir_verify_mask_result_type(
      module, lhs_type_id,
      loom_llvmir_verify_value_type(module, instruction->result_value_id));
}

static iree_status_t loom_llvmir_verify_select(
    const loom_llvmir_module_t* module,
    const loom_llvmir_instruction_t* instruction) {
  if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM select has no result value");
  }
  loom_llvmir_type_id_t result_type_id =
      loom_llvmir_verify_value_type(module, instruction->result_value_id);
  IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
      module, instruction->select.true_value, result_type_id));
  IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
      module, instruction->select.false_value, result_type_id));

  loom_llvmir_type_id_t condition_type_id =
      loom_llvmir_verify_value_type(module, instruction->select.condition);
  if (condition_type_id == LOOM_LLVMIR_TYPE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM select references unknown condition");
  }
  if (loom_llvmir_verify_type_is_i1(module, condition_type_id)) {
    return iree_ok_status();
  }

  const loom_llvmir_type_t* condition_type =
      loom_llvmir_verify_type(module, condition_type_id);
  const loom_llvmir_type_t* result_type =
      loom_llvmir_verify_type(module, result_type_id);
  if (!condition_type || condition_type->kind != LOOM_LLVMIR_TYPE_VECTOR ||
      !result_type || result_type->kind != LOOM_LLVMIR_TYPE_VECTOR ||
      condition_type->element_count != result_type->element_count ||
      !loom_llvmir_verify_type_is_i1(module, condition_type->element_type)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM vector select condition must be a same-lane i1 vector");
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_verify_instruction(
    const loom_llvmir_module_t* module, const loom_llvmir_function_t* function,
    const loom_llvmir_block_t* block,
    const loom_llvmir_instruction_t* instruction) {
  (void)block;
  switch (instruction->kind) {
    case LOOM_LLVMIR_INST_PHI:
      if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM phi has no result value");
      }
      for (iree_host_size_t i = 0; i < instruction->phi.incoming_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
            module, instruction->phi.incoming[i].value,
            loom_llvmir_verify_value_type(module,
                                          instruction->result_value_id)));
        if (instruction->phi.incoming[i].predecessor >= function->block_count) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "LLVM phi references unknown predecessor");
        }
      }
      return iree_ok_status();
    case LOOM_LLVMIR_INST_BINOP:
      if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM binop has no result value");
      }
      IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
          module, instruction->binop.lhs,
          loom_llvmir_verify_value_type(module, instruction->result_value_id)));
      return loom_llvmir_verify_expected_value_type(
          module, instruction->binop.rhs,
          loom_llvmir_verify_value_type(module, instruction->result_value_id));
    case LOOM_LLVMIR_INST_ICMP:
      if (!loom_llvmir_verify_icmp_predicate(instruction->icmp.predicate)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown LLVM icmp predicate");
      }
      return loom_llvmir_verify_compare(
          module, instruction, instruction->icmp.lhs, instruction->icmp.rhs,
          loom_llvmir_verify_type_is_integer_or_pointer_like);
    case LOOM_LLVMIR_INST_FCMP:
      if (!loom_llvmir_verify_fcmp_predicate(instruction->fcmp.predicate)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unknown LLVM fcmp predicate");
      }
      return loom_llvmir_verify_compare(
          module, instruction, instruction->fcmp.lhs, instruction->fcmp.rhs,
          loom_llvmir_verify_type_is_float_like);
    case LOOM_LLVMIR_INST_SELECT:
      return loom_llvmir_verify_select(module, instruction);
    case LOOM_LLVMIR_INST_GEP:
      if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM gep has no result value");
      }
      if (!loom_llvmir_verify_value(module, instruction->gep.base)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM gep references unknown base");
      }
      for (iree_host_size_t i = 0; i < instruction->gep.index_count; ++i) {
        if (!loom_llvmir_verify_value(module, instruction->gep.indices[i])) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "LLVM gep references unknown index");
        }
      }
      return iree_ok_status();
    case LOOM_LLVMIR_INST_LOAD:
      if (instruction->result_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM load has no result value");
      }
      if (!loom_llvmir_verify_value(module, instruction->load.pointer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM load references unknown pointer");
      }
      return iree_ok_status();
    case LOOM_LLVMIR_INST_STORE:
      if (!loom_llvmir_verify_value(module, instruction->store.value) ||
          !loom_llvmir_verify_value(module, instruction->store.pointer)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM store references unknown value");
      }
      return iree_ok_status();
    case LOOM_LLVMIR_INST_CALL: {
      loom_llvmir_function_t* callee =
          loom_llvmir_verify_function_ref(module, instruction->call.callee);
      if (!callee) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM call references unknown callee");
      }
      if (callee->parameter_count != instruction->call.arg_count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM call arg count mismatch");
      }
      for (iree_host_size_t i = 0; i < instruction->call.arg_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_llvmir_verify_expected_value_type(
            module, instruction->call.args[i], callee->parameters[i].type_id));
      }
      return iree_ok_status();
    }
    case LOOM_LLVMIR_INST_INLINE_ASM:
      for (iree_host_size_t i = 0; i < instruction->inline_asm.arg_count; ++i) {
        if (!loom_llvmir_verify_value(module,
                                      instruction->inline_asm.args[i])) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "LLVM inline asm references unknown arg");
        }
      }
      return iree_ok_status();
    case LOOM_LLVMIR_INST_RET:
      if (!instruction->ret.has_value) {
        const loom_llvmir_type_t* return_type =
            loom_llvmir_verify_type(module, function->return_type);
        if (!return_type || return_type->kind != LOOM_LLVMIR_TYPE_VOID) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "LLVM ret void in non-void function");
        }
        return iree_ok_status();
      }
      return loom_llvmir_verify_expected_value_type(
          module, instruction->ret.value, function->return_type);
    case LOOM_LLVMIR_INST_BR:
      if (instruction->br.target >= function->block_count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM branch references unknown block");
      }
      return iree_ok_status();
    case LOOM_LLVMIR_INST_COND_BR:
      if (!loom_llvmir_verify_value(module, instruction->cond_br.condition)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM cond_br references unknown condition");
      }
      const loom_llvmir_type_t* condition_type = loom_llvmir_verify_type(
          module, loom_llvmir_verify_value_type(
                      module, instruction->cond_br.condition));
      if (!condition_type || condition_type->kind != LOOM_LLVMIR_TYPE_INTEGER ||
          condition_type->bit_width != 1) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM cond_br condition must be i1");
      }
      if (instruction->cond_br.true_block >= function->block_count ||
          instruction->cond_br.false_block >= function->block_count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM cond_br references unknown block");
      }
      return iree_ok_status();
    case LOOM_LLVMIR_INST_UNREACHABLE:
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown LLVM instruction kind");
  }
}

static iree_status_t loom_llvmir_verify_function(
    const loom_llvmir_module_t* module,
    const loom_llvmir_function_t* function) {
  if (function->kind == LOOM_LLVMIR_FUNCTION_DECLARATION) {
    return iree_ok_status();
  }
  if (function->block_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM function definition has no blocks");
  }
  for (iree_host_size_t i = 0; i < function->block_count; ++i) {
    const loom_llvmir_block_t* block = function->blocks[i];
    if (block->instruction_count == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM block has no terminator");
    }
    bool saw_non_phi = false;
    for (iree_host_size_t j = 0; j < block->instruction_count; ++j) {
      const loom_llvmir_instruction_t* instruction = &block->instructions[j];
      if (instruction->kind == LOOM_LLVMIR_INST_PHI && saw_non_phi) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM phi appears after non-phi instruction");
      }
      if (instruction->kind != LOOM_LLVMIR_INST_PHI) saw_non_phi = true;
      if (loom_llvmir_is_terminator(instruction) &&
          j + 1 != block->instruction_count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM terminator is not last in block");
      }
      IREE_RETURN_IF_ERROR(
          loom_llvmir_verify_instruction(module, function, block, instruction));
    }
    if (!loom_llvmir_is_terminator(
            &block->instructions[block->instruction_count - 1])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM block does not end with terminator");
    }
  }
  return iree_ok_status();
}

iree_status_t loom_llvmir_verify_module(const loom_llvmir_module_t* module) {
  IREE_ASSERT_ARGUMENT(module);
  for (iree_host_size_t i = 0; i < module->function_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_verify_function(module, module->functions[i]));
  }
  return iree_ok_status();
}
