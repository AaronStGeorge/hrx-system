// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "pass_program.h"

#include <string.h>

#include "context.h"
#include "diagnostic.h"
#include "iree/base/internal/atomics.h"
#include "loom/ir/module.h"
#include "loom/link/linker.h"
#include "loom/ops/pass/ops.h"
#include "loom/pass/builtin_registry.h"
#include "loom/pass/environment.h"
#include "loom/pass/tooling.h"
#include "loomc/iree.h"
#include "module.h"
#include "result.h"

enum {
  LOOMC_PASS_PROGRAM_DEFAULT_BLOCK_SIZE = 32 * 1024,
};

struct loomc_pass_program_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used for pass-program-owned storage.
  loomc_allocator_t allocator;

  // Context retained by the pass program.
  loomc_context_t* context;

  // Stable arena block pool backing the pipeline module and compiled program.
  iree_arena_block_pool_t block_pool;

  // Owned scratch module containing the selected pass.pipeline IR.
  loom_module_t* pipeline_module;

  // Compiled immutable pass instruction program.
  loom_pass_program_t program;

  // True when program has been initialized and must be deinitialized.
  bool program_initialized;
};

static loomc_status_t loomc_pass_program_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_pass_program_validate_options(
    const loomc_pass_program_options_t* options) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_PASS_PROGRAM_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "pass program options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "pass program options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "pass program option extensions are not supported");
  }
  return loomc_pass_program_validate_string_view(options->identifier);
}

static iree_string_view_t loomc_pass_program_identifier(
    const loomc_pass_program_options_t* options) {
  if (options == NULL || loomc_string_view_is_empty(options->identifier)) {
    return IREE_SV("__loomc_pass_program");
  }
  return iree_string_view_from_loomc(options->identifier);
}

static bool loomc_pass_program_status_is_result_diagnostic(
    loomc_status_t status) {
  switch (loomc_status_code(status)) {
    case LOOMC_STATUS_INVALID_ARGUMENT:
    case LOOMC_STATUS_NOT_FOUND:
    case LOOMC_STATUS_FAILED_PRECONDITION:
    case LOOMC_STATUS_OUT_OF_RANGE:
    case LOOMC_STATUS_UNIMPLEMENTED:
    case LOOMC_STATUS_INCOMPATIBLE:
      return true;
    default:
      return false;
  }
}

static loomc_status_t loomc_pass_program_allocate_storage(
    loomc_context_t* context, loomc_allocator_t allocator,
    loomc_pass_program_t** out_pass_program) {
  *out_pass_program = NULL;
  loomc_pass_program_t* pass_program = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc(allocator, sizeof(*pass_program),
                                               (void**)&pass_program));
  memset(pass_program, 0, sizeof(*pass_program));
  iree_atomic_ref_count_init(&pass_program->ref_count);
  pass_program->allocator = allocator;
  pass_program->context = context;
  loomc_context_retain(context);
  iree_arena_block_pool_initialize(LOOMC_PASS_PROGRAM_DEFAULT_BLOCK_SIZE,
                                   iree_allocator_from_loomc(allocator),
                                   &pass_program->block_pool);

  *out_pass_program = pass_program;
  return loomc_ok_status();
}

static loomc_status_t loomc_pass_program_allocate_pipeline_module(
    loomc_pass_program_t* pass_program,
    const loomc_pass_program_options_t* options) {
  iree_status_t status = loom_module_allocate(
      loomc_context_loom_context(pass_program->context),
      loomc_pass_program_identifier(options), &pass_program->block_pool, NULL,
      iree_allocator_from_loomc(pass_program->allocator),
      &pass_program->pipeline_module);
  return loomc_status_from_iree(status);
}

static loomc_status_t loomc_pass_program_compile_pipeline_op(
    loomc_pass_program_t* pass_program, const loom_op_t* pipeline_op) {
  loom_pass_program_compile_options_t compile_options = {
      .registry = loom_pass_builtin_registry(),
      .environment = loom_pass_environment_empty(),
  };
  iree_status_t status = loom_pass_program_compile_pipeline(
      pass_program->pipeline_module, pipeline_op, &compile_options,
      &pass_program->block_pool, &pass_program->program);
  pass_program->program_initialized = iree_status_is_ok(status);
  return loomc_status_from_iree(status);
}

static loomc_status_t loomc_pass_program_compile_flat_pipeline(
    loomc_pass_program_t* pass_program, loomc_string_view_t pipeline_text) {
  const loom_op_t* pipeline_op = NULL;
  iree_status_t status = loom_pass_tool_build_flat_pipeline(
      pass_program->pipeline_module, iree_string_view_from_loomc(pipeline_text),
      loom_pass_builtin_registry(), &pipeline_op);
  if (iree_status_is_ok(status)) {
    return loomc_pass_program_compile_pipeline_op(pass_program, pipeline_op);
  }
  return loomc_status_from_iree(status);
}

static iree_string_view_t loomc_pass_program_normalize_symbol_name(
    loomc_string_view_t symbol) {
  iree_string_view_t name =
      iree_string_view_trim(iree_string_view_from_loomc(symbol));
  while (iree_string_view_starts_with_char(name, '@')) {
    name = iree_string_view_remove_prefix(name, 1);
  }
  return name;
}

static loomc_status_t loomc_pass_program_snapshot_pipeline_module(
    loomc_pass_program_t* pass_program, const loom_module_t* source_module,
    iree_string_view_t pipeline_symbol_name,
    const loomc_pass_program_options_t* options) {
  const loom_module_t* source_modules[] = {source_module};
  iree_string_view_t root_symbols[] = {pipeline_symbol_name};
  loom_link_options_t link_options = {
      .module_name = loomc_pass_program_identifier(options),
      .root_symbols =
          {
              .count = IREE_ARRAYSIZE(root_symbols),
              .values = root_symbols,
          },
  };
  loom_module_t* pipeline_module = NULL;
  iree_status_t status = loom_link_materialized_modules(
      source_modules, IREE_ARRAYSIZE(source_modules), &link_options,
      &pass_program->block_pool,
      iree_allocator_from_loomc(pass_program->allocator), &pipeline_module);
  if (iree_status_is_ok(status)) {
    pass_program->pipeline_module = pipeline_module;
  }
  return loomc_status_from_iree(status);
}

static loomc_status_t loomc_pass_program_find_pipeline_symbol(
    const loom_module_t* pipeline_module,
    iree_string_view_t pipeline_symbol_name,
    const loom_op_t** out_pipeline_op) {
  *out_pipeline_op = NULL;
  if (iree_string_view_is_empty(pipeline_symbol_name)) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "pass pipeline symbol name is required"));
  }
  loom_string_id_t name_id =
      loom_module_lookup_string(pipeline_module, pipeline_symbol_name);
  if (name_id == LOOM_STRING_ID_INVALID) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_NOT_FOUND, "pass pipeline @%.*s was not found",
        (int)pipeline_symbol_name.size, pipeline_symbol_name.data));
  }
  uint16_t symbol_id = loom_module_find_symbol(pipeline_module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_NOT_FOUND, "pass pipeline @%.*s was not found",
        (int)pipeline_symbol_name.size, pipeline_symbol_name.data));
  }
  const loom_symbol_t* symbol = &pipeline_module->symbols.entries[symbol_id];
  if (!symbol->defining_op || !loom_pass_pipeline_isa(symbol->defining_op)) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "symbol @%.*s does not define a pass.pipeline",
        (int)pipeline_symbol_name.size, pipeline_symbol_name.data));
  }
  *out_pipeline_op = symbol->defining_op;
  return loomc_ok_status();
}

static loomc_status_t loomc_pass_program_fail_result_from_status(
    loomc_result_t* result, loomc_status_t status) {
  loomc_status_t add_status = loomc_result_fail_status_diagnostic(
      result, NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
      loomc_make_cstring_view("PASS_PROGRAM/INVALID"), status);
  loomc_status_free(status);
  return add_status;
}

loomc_status_t loomc_pass_program_create_empty(
    loomc_context_t* context, const loomc_pass_program_options_t* options,
    loomc_allocator_t allocator, loomc_pass_program_t** out_pass_program) {
  if (out_pass_program == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_pass_program must not be NULL");
  }
  *out_pass_program = NULL;
  if (context == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "context must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_pass_program_validate_options(options));
  allocator = loomc_allocator_or_system(allocator);

  loomc_pass_program_t* pass_program = NULL;
  loomc_status_t status =
      loomc_pass_program_allocate_storage(context, allocator, &pass_program);
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_allocate_pipeline_module(pass_program, options);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_compile_flat_pipeline(
        pass_program, loomc_string_view_empty());
  }
  if (loomc_status_is_ok(status)) {
    *out_pass_program = pass_program;
    pass_program = NULL;
  }
  loomc_pass_program_release(pass_program);
  return status;
}

loomc_status_t loomc_pass_program_create_from_pipeline_text(
    loomc_context_t* context, loomc_string_view_t pipeline_text,
    const loomc_pass_program_options_t* options, loomc_allocator_t allocator,
    loomc_pass_program_t** out_pass_program, loomc_result_t** out_result) {
  if (out_pass_program == NULL || out_result == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "out_pass_program and out_result must not be NULL");
  }
  *out_pass_program = NULL;
  *out_result = NULL;
  if (context == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "context must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_pass_program_validate_string_view(pipeline_text));
  LOOMC_RETURN_IF_ERROR(loomc_pass_program_validate_options(options));

  allocator = loomc_allocator_or_system(allocator);
  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result));

  loomc_pass_program_t* pass_program = NULL;
  loomc_status_t status =
      loomc_pass_program_allocate_storage(context, allocator, &pass_program);
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_allocate_pipeline_module(pass_program, options);
  }
  if (loomc_status_is_ok(status)) {
    status =
        loomc_pass_program_compile_flat_pipeline(pass_program, pipeline_text);
  }
  if (!loomc_status_is_ok(status) &&
      loomc_pass_program_status_is_result_diagnostic(status)) {
    status = loomc_pass_program_fail_result_from_status(result, status);
    loomc_pass_program_release(pass_program);
    pass_program = NULL;
  }

  if (loomc_status_is_ok(status)) {
    *out_pass_program = pass_program;
    *out_result = result;
    pass_program = NULL;
    result = NULL;
  }
  loomc_pass_program_release(pass_program);
  loomc_result_release(result);
  return status;
}

loomc_status_t loomc_pass_program_create_from_module_symbol(
    const loomc_module_t* module, loomc_string_view_t pipeline_symbol,
    const loomc_pass_program_options_t* options, loomc_allocator_t allocator,
    loomc_pass_program_t** out_pass_program, loomc_result_t** out_result) {
  if (out_pass_program == NULL || out_result == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "out_pass_program and out_result must not be NULL");
  }
  *out_pass_program = NULL;
  *out_result = NULL;
  if (module == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_pass_program_validate_string_view(pipeline_symbol));
  LOOMC_RETURN_IF_ERROR(loomc_pass_program_validate_options(options));

  allocator = loomc_allocator_or_system(allocator);
  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result));

  loomc_pass_program_t* pass_program = NULL;
  iree_string_view_t pipeline_symbol_name =
      loomc_pass_program_normalize_symbol_name(pipeline_symbol);
  loomc_context_t* context = loomc_module_context(module);
  const loom_module_t* source_module = loomc_module_const_loom_module(module);
  const loom_op_t* pipeline_op = NULL;
  loomc_status_t status =
      loomc_pass_program_allocate_storage(context, allocator, &pass_program);
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_snapshot_pipeline_module(
        pass_program, source_module, pipeline_symbol_name, options);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_find_pipeline_symbol(
        pass_program->pipeline_module, pipeline_symbol_name, &pipeline_op);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_compile_pipeline_op(pass_program, pipeline_op);
  }
  if (!loomc_status_is_ok(status) &&
      loomc_pass_program_status_is_result_diagnostic(status)) {
    status = loomc_pass_program_fail_result_from_status(result, status);
    loomc_pass_program_release(pass_program);
    pass_program = NULL;
  }

  if (loomc_status_is_ok(status)) {
    *out_pass_program = pass_program;
    *out_result = result;
    pass_program = NULL;
    result = NULL;
  }
  loomc_pass_program_release(pass_program);
  loomc_result_release(result);
  return status;
}

void loomc_pass_program_retain(loomc_pass_program_t* pass_program) {
  if (pass_program == NULL) {
    return;
  }
  iree_atomic_ref_count_inc(&pass_program->ref_count);
}

void loomc_pass_program_release(loomc_pass_program_t* pass_program) {
  if (pass_program == NULL) {
    return;
  }
  if (iree_atomic_ref_count_dec(&pass_program->ref_count) != 1) {
    return;
  }
  loomc_allocator_t allocator = pass_program->allocator;
  if (pass_program->program_initialized) {
    loom_pass_program_deinitialize(&pass_program->program);
  }
  loom_module_free(pass_program->pipeline_module);
  iree_arena_block_pool_deinitialize(&pass_program->block_pool);
  loomc_context_release(pass_program->context);
  loomc_allocator_free(allocator, pass_program);
}

loomc_context_t* loomc_pass_program_context(
    const loomc_pass_program_t* pass_program) {
  return pass_program ? pass_program->context : NULL;
}

const loom_pass_program_t* loomc_pass_program_loom_pass_program(
    const loomc_pass_program_t* pass_program) {
  return pass_program ? &pass_program->program : NULL;
}
