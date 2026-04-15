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
