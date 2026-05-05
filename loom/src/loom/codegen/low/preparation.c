// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/preparation.h"

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/pass_environment.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/builder.h"
#include "loom/pass/interpreter.h"
#include "loom/pass/program.h"

static iree_status_t loom_low_preparation_build_pipeline(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  loom_op_t* run_op = NULL;
  iree_status_t status = loom_pass_ir_build_run(
      builder, IREE_SV("cse"), loom_make_named_attr_slice(NULL, 0), &run_op);
  if (iree_status_is_ok(status)) {
    status =
        loom_pass_ir_build_run(builder, IREE_SV("low-select-operand-forms"),
                               loom_make_named_attr_slice(NULL, 0), &run_op);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_pass_ir_build_run(builder, IREE_SV("low-dce"),
                               loom_make_named_attr_slice(NULL, 0), &run_op);
  }
  return status;
}

static iree_status_t loom_low_preparation_validate_function(
    loom_module_t* module, loom_op_t* low_func_op) {
  if (!low_func_op || !loom_low_function_def_isa(low_func_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low preparation expected a low function");
  }
  if (!loom_func_like_isa(loom_func_like_cast(module, low_func_op))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low preparation expected a func-like low op");
  }
  return iree_ok_status();
}

iree_status_t loom_low_prepare_functions_for_packetization(
    loom_module_t* module, loom_op_t* const* low_func_ops,
    iree_host_size_t low_func_count,
    const loom_low_preparation_options_t* options,
    iree_arena_block_pool_t* block_pool) {
  if (low_func_count == 0) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < low_func_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_preparation_validate_function(module, low_func_ops[i]));
  }

  loom_module_t* pipeline_module = NULL;
  iree_status_t status = loom_module_allocate(
      module->context, IREE_SV("__low_prepare_packetization"), block_pool, NULL,
      module->allocator, &pipeline_module);
  const loom_op_t* pipeline_op = NULL;
  if (iree_status_is_ok(status)) {
    loom_op_t* mutable_pipeline_op = NULL;
    status = loom_pass_ir_build_pipeline(
        pipeline_module, IREE_SV("__low_prepare_packetization"),
        LOOM_PASS_ANCHOR_FUNC, loom_low_preparation_build_pipeline, NULL,
        &mutable_pipeline_op);
    pipeline_op = mutable_pipeline_op;
  }

  loom_low_pass_environment_storage_t environment_storage = {0};
  loom_pass_environment_t environment =
      loom_low_pass_environment_storage_initialize(
          options->descriptor_registry, /*lower_policy_registry=*/NULL,
          /*legality_provider_list=*/NULL, &environment_storage);
  loom_pass_program_t program = {0};
  if (iree_status_is_ok(status)) {
    const loom_pass_program_compile_options_t compile_options = {
        .registry = options->pass_registry,
        .environment = environment,
    };
    status = loom_pass_program_compile_pipeline(
        pipeline_module, pipeline_op, &compile_options, block_pool, &program);
  }
  if (iree_status_is_ok(status)) {
    const loom_pass_interpreter_options_t interpreter_options = {
        .block_pool = block_pool,
        .diagnostic_emitter = options->diagnostic_emitter,
        .environment = environment,
    };
    for (iree_host_size_t i = 0;
         i < low_func_count && iree_status_is_ok(status); ++i) {
      loom_pass_run_result_t run_result = {0};
      status = loom_pass_interpreter_run_function(
          &program, module, loom_func_like_cast(module, low_func_ops[i]),
          &interpreter_options, &run_result);
      if (iree_status_is_ok(status) && run_result.error_count != 0) {
        break;
      }
    }
  }

  loom_pass_program_deinitialize(&program);
  if (pipeline_module != NULL) {
    loom_module_free(pipeline_module);
  }
  return status;
}
