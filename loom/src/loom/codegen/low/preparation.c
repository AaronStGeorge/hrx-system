// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/preparation.h"

#include <inttypes.h>

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/pass_environment.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/pass/ops.h"
#include "loom/pass/interpreter.h"
#include "loom/pass/program.h"

static iree_status_t loom_low_preparation_register_pass_dialect(
    loom_context_t* context) {
  iree_host_size_t vtable_count = 0;
  const loom_op_vtable_t* const* vtables =
      loom_pass_dialect_vtables(&vtable_count);
  iree_host_size_t semantics_count = 0;
  const loom_op_semantics_t* semantics =
      loom_pass_dialect_op_semantics(&semantics_count);
  if (semantics_count != vtable_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "pass dialect semantics count %" PRIhsz
                            " does not match vtable count %" PRIhsz,
                            semantics_count, vtable_count);
  }
  IREE_RETURN_IF_ERROR(loom_context_register_dialect(
      context, LOOM_DIALECT_PASS, vtables, (uint16_t)vtable_count));
  return loom_context_register_dialect_semantics(
      context, LOOM_DIALECT_PASS, semantics, (uint16_t)semantics_count);
}

static iree_status_t loom_low_preparation_build_run(
    loom_builder_t* builder, iree_string_view_t pass_key) {
  loom_string_id_t key_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(builder->module, pass_key, &key_id));
  loom_op_t* run_op = NULL;
  return loom_pass_run_build(builder, key_id,
                             loom_make_named_attr_slice(NULL, 0),
                             LOOM_LOCATION_UNKNOWN, &run_op);
}

static iree_status_t loom_low_preparation_build_pipeline(
    loom_module_t* pipeline_module, const loom_op_t** out_pipeline_op) {
  IREE_ASSERT_ARGUMENT(pipeline_module);
  IREE_ASSERT_ARGUMENT(out_pipeline_op);
  *out_pipeline_op = NULL;

  loom_builder_t builder;
  loom_builder_initialize(pipeline_module, &pipeline_module->arena,
                          loom_module_block(pipeline_module), &builder);

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      pipeline_module, IREE_SV("__low_prepare_packetization"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_add_symbol(pipeline_module, name_id, &symbol_id));
  loom_op_t* pipeline_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_pipeline_build(
      &builder, LOOM_PASS_ANCHOR_FUNC,
      (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id},
      LOOM_LOCATION_UNKNOWN, &pipeline_op));

  loom_builder_ip_t pipeline_ip = loom_builder_enter_region(
      &builder, pipeline_op, loom_pass_pipeline_body(pipeline_op));
  iree_status_t status =
      loom_low_preparation_build_run(&builder, IREE_SV("cse"));
  if (iree_status_is_ok(status)) {
    status = loom_low_preparation_build_run(
        &builder, IREE_SV("low-select-operand-forms"));
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_preparation_build_run(&builder, IREE_SV("low-dce"));
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* yield_op = NULL;
    status = loom_pass_yield_build(&builder, LOOM_LOCATION_UNKNOWN, &yield_op);
  }
  loom_builder_restore(&builder, pipeline_ip);
  if (iree_status_is_ok(status)) {
    *out_pipeline_op = pipeline_op;
  }
  return status;
}

static iree_status_t loom_low_preparation_validate_function(
    loom_module_t* module, loom_op_t* low_func_op) {
  IREE_ASSERT_ARGUMENT(module);
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
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(options && options->pass_registry &&
                       options->descriptor_registry);
  IREE_ASSERT_ARGUMENT(block_pool);
  if (low_func_count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_ARGUMENT(low_func_ops);

  for (iree_host_size_t i = 0; i < low_func_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_preparation_validate_function(module, low_func_ops[i]));
  }

  loom_context_t pipeline_context;
  loom_context_initialize(module->allocator, &pipeline_context);
  iree_status_t status =
      loom_low_preparation_register_pass_dialect(&pipeline_context);
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(&pipeline_context);
  }

  loom_module_t* pipeline_module = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_module_allocate(
        &pipeline_context, IREE_SV("__low_prepare_packetization"), block_pool,
        NULL, module->allocator, &pipeline_module);
  }
  const loom_op_t* pipeline_op = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_preparation_build_pipeline(pipeline_module, &pipeline_op);
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
      status = loom_pass_interpreter_run_function(
          &program, module, loom_func_like_cast(module, low_func_ops[i]),
          &interpreter_options);
    }
  }

  loom_pass_program_deinitialize(&program);
  if (pipeline_module != NULL) {
    loom_module_free(pipeline_module);
  }
  loom_context_deinitialize(&pipeline_context);
  return status;
}
