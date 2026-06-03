// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/session.h"

#include "loom/codegen/low/text_asm.h"
#include "loom/error/diagnostic.h"
#include "loom/format/text/parser.h"
#include "loom/ir/module.h"

enum {
  LOOM_RUN_DEFAULT_BLOCK_POOL_BLOCK_SIZE = 32 * 1024,
  LOOM_RUN_DEFAULT_MAX_PARSE_ERRORS = 20,
};

void loom_run_session_options_initialize(
    loom_run_session_options_t* out_options) {
  *out_options = (loom_run_session_options_t){
      .host_allocator = iree_allocator_system(),
      .block_pool_block_size = LOOM_RUN_DEFAULT_BLOCK_POOL_BLOCK_SIZE,
  };
}

iree_status_t loom_run_session_initialize(
    const loom_run_session_options_t* options,
    loom_run_session_t* out_session) {
  *out_session = (loom_run_session_t){
      .host_allocator = options->host_allocator,
  };

  const iree_host_size_t block_pool_block_size =
      options->block_pool_block_size == 0
          ? LOOM_RUN_DEFAULT_BLOCK_POOL_BLOCK_SIZE
          : options->block_pool_block_size;
  iree_arena_block_pool_initialize(
      block_pool_block_size, options->host_allocator, &out_session->block_pool);
  out_session->block_pool_initialized = true;

  iree_status_t status = options->initialize_low_descriptor_registry.fn(
      options->initialize_low_descriptor_registry.user_data,
      &out_session->low_descriptor_registry);
  if (iree_status_is_ok(status)) {
    loom_context_initialize(options->host_allocator, &out_session->context);
    out_session->context_initialized = true;
    status = options->register_context.fn(options->register_context.user_data,
                                          &out_session->context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(&out_session->context);
  }
  if (!iree_status_is_ok(status)) {
    loom_run_session_deinitialize(out_session);
  }
  return status;
}

void loom_run_session_deinitialize(loom_run_session_t* session) {
  if (session == NULL) {
    return;
  }
  if (session->context_initialized) {
    loom_context_deinitialize(&session->context);
  }
  if (session->block_pool_initialized) {
    iree_arena_block_pool_deinitialize(&session->block_pool);
  }
  *session = (loom_run_session_t){0};
}

loom_context_t* loom_run_session_context(loom_run_session_t* session) {
  return &session->context;
}

iree_arena_block_pool_t* loom_run_session_block_pool(
    loom_run_session_t* session) {
  return &session->block_pool;
}

const loom_target_low_descriptor_registry_t*
loom_run_session_low_descriptor_registry(const loom_run_session_t* session) {
  return &session->low_descriptor_registry;
}

void loom_run_module_parse_options_initialize(
    loom_run_module_parse_options_t* out_options) {
  *out_options = (loom_run_module_parse_options_t){
      .diagnostic_sink = {.fn = loom_diagnostic_stderr_sink},
      .max_errors = LOOM_RUN_DEFAULT_MAX_PARSE_ERRORS,
  };
}

iree_status_t loom_run_module_parse(
    loom_run_session_t* session, const loom_run_module_parse_options_t* options,
    loom_run_module_t* out_module) {
  *out_module = (loom_run_module_t){
      .filename = options->filename,
      .source = options->source,
  };

  loom_text_parse_options_t parse_options = {
      .diagnostic_sink = options->diagnostic_sink,
      .max_errors = options->max_errors,
  };
  loom_low_descriptor_text_asm_environment_initialize(
      &session->low_descriptor_registry.registry,
      &parse_options.low_asm_environment);

  iree_status_t status = loom_text_parse(
      options->source, options->filename, &session->context,
      &session->block_pool, &parse_options, &out_module->module);
  if (iree_status_is_ok(status) && out_module->module == NULL) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "input module has parse errors");
  }
  if (iree_status_is_ok(status)) {
    loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
    status = loom_module_register_source(out_module->module, options->filename,
                                         &source_id);
    if (iree_status_is_ok(status)) {
      out_module->source_entry = (loom_source_entry_t){
          .source_id = source_id,
          .source = options->source,
          .filename = options->filename,
      };
      out_module->source_table_resolver = (loom_source_table_resolver_t){
          .entries = &out_module->source_entry,
          .count = 1,
      };
    }
  }
  if (!iree_status_is_ok(status)) {
    loom_run_module_deinitialize(out_module);
  }
  return status;
}

void loom_run_module_deinitialize(loom_run_module_t* run_module) {
  if (run_module == NULL) {
    return;
  }
  loom_module_free(run_module->module);
  *run_module = (loom_run_module_t){0};
}

loom_source_resolver_t loom_run_module_source_resolver(
    loom_run_module_t* run_module) {
  return (loom_source_resolver_t){
      .fn = loom_source_table_resolve,
      .user_data = &run_module->source_table_resolver,
  };
}
