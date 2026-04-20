// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-run-loom: compiles a Loom module and executes the selected export.

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/run_module.h"
#include "iree/vm/api.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/diagnostic.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/target/emit/ireevm/low_registry.h"
#include "loom/target/emit/ireevm/module_compiler.h"
#include "loom/verify/verify.h"

IREE_FLAG(string, loom_target, "",
          "Optional target.bundle symbol to compile, such as '@vm_target'. "
          "When omitted the module must contain exactly one IREE VM target.");
IREE_FLAG(string, loom_module_name, "loom",
          "Module name to store in the compiled VM bytecode archive.");

static iree_status_t iree_run_loom_read_input(
    iree_string_view_t path, iree_allocator_t allocator,
    iree_io_file_contents_t** out_contents) {
  const bool is_stdin = iree_string_view_is_empty(path) ||
                        iree_string_view_equal(path, IREE_SV("-"));
  if (is_stdin) {
    return iree_io_file_contents_read_stdin(allocator, out_contents);
  }
  return iree_io_file_contents_read(path, allocator, out_contents);
}

static iree_string_view_t iree_run_loom_file_contents_string_view(
    const iree_io_file_contents_t* contents) {
  return iree_make_string_view((const char*)contents->const_buffer.data,
                               contents->const_buffer.data_length);
}

static iree_status_t iree_run_loom_source_resolver_for_input(
    loom_context_t* context, iree_string_view_t filename,
    iree_string_view_t source, loom_source_entry_t* out_source_entry,
    loom_source_table_resolver_t* out_source_resolver) {
  IREE_ASSERT_ARGUMENT(out_source_entry);
  IREE_ASSERT_ARGUMENT(out_source_resolver);
  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_context_register_source(context, filename, &source_id));
  *out_source_entry = (loom_source_entry_t){
      .source_id = source_id,
      .source = source,
      .filename = filename,
  };
  *out_source_resolver = (loom_source_table_resolver_t){
      .entries = out_source_entry,
      .count = 1,
  };
  return iree_ok_status();
}

static iree_status_t iree_run_loom_parse_module(
    iree_string_view_t filename, iree_string_view_t source,
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    loom_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);
  loom_text_parse_options_t parse_options = {
      .diagnostic_sink = {.fn = loom_diagnostic_stderr_sink},
      .max_errors = 20,
  };
  loom_low_descriptor_text_asm_environment_initialize(
      &low_registry->registry, &parse_options.low_asm_environment);
  IREE_RETURN_IF_ERROR(loom_text_parse(source, filename, context, block_pool,
                                       &parse_options, out_module));
  if (!*out_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "input module has parse errors");
  }
  return iree_ok_status();
}

static iree_status_t iree_run_loom_compile_to_archive(
    iree_string_view_t filename, iree_string_view_t source,
    loom_module_t* module, iree_allocator_t allocator,
    loom_ireevm_module_archive_t* out_archive) {
  loom_source_entry_t source_entry = {0};
  loom_source_table_resolver_t source_resolver = {0};
  IREE_RETURN_IF_ERROR(iree_run_loom_source_resolver_for_input(
      module->context, filename, source, &source_entry, &source_resolver));
  const loom_ireevm_module_compile_options_t compile_options = {
      .module_name = iree_make_cstring_view(FLAG_loom_module_name),
      .target_symbol = iree_make_cstring_view(FLAG_loom_target),
      .diagnostic_sink = {.fn = loom_diagnostic_stderr_sink},
      .source_resolver = {.fn = loom_source_table_resolve,
                          .user_data = &source_resolver},
      .max_errors = 20,
  };
  return loom_ireevm_compile_module_archive(module, &compile_options, allocator,
                                            out_archive);
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_flags_set_usage(
      "iree-run-loom",
      "Compiles a Loom module to a runtime artifact and executes the selected "
      "export.\n"
      "\n"
      "Usage:\n"
      "  iree-run-loom [file.loom] --function=name --input=... "
      "--expected_output=...\n"
      "  cat module.loom | iree-run-loom - --function=name --input=...\n"
      "\n"
      "The initial path compiles target.preset key \"iree-vm\" into a real "
      "IREE "
      "VM bytecode archive and runs it with the same input/output flags as "
      "iree-run-module.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t allocator = iree_allocator_system();
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, allocator, &block_pool);

  loom_context_t context = {0};
  bool context_initialized = false;
  iree_io_file_contents_t* contents = NULL;
  loom_module_t* module = NULL;
  loom_target_low_descriptor_registry_t low_registry = {0};
  loom_ireevm_module_archive_t archive = {0};
  iree_vm_instance_t* instance = NULL;
  int exit_code = 0;

  iree_status_t status = iree_ok_status();
  if (argc > 2) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "iree-run-loom accepts at most one input file or '-' for stdin; got %d "
        "inputs",
        argc - 1);
  }

  loom_ireevm_low_descriptor_registry_initialize(&low_registry);
  if (iree_status_is_ok(status)) {
    status = loom_op_registry_initialize_context(allocator, &context);
    context_initialized = iree_status_is_ok(status);
  }

  const iree_string_view_t input_path =
      argc < 2 ? iree_string_view_empty() : iree_make_cstring_view(argv[1]);
  const iree_string_view_t filename =
      (argc < 2 || iree_string_view_equal(input_path, IREE_SV("-")))
          ? IREE_SV("<stdin>")
          : input_path;
  iree_string_view_t source = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    status = iree_run_loom_read_input(input_path, allocator, &contents);
    if (iree_status_is_ok(status)) {
      source = iree_run_loom_file_contents_string_view(contents);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_run_loom_parse_module(filename, source, &low_registry,
                                        &context, &block_pool, &module);
  }
  if (iree_status_is_ok(status)) {
    status = iree_run_loom_compile_to_archive(filename, source, module,
                                              allocator, &archive);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tooling_create_instance(allocator, &instance);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tooling_run_module_with_data(
        instance, iree_string_view_empty(),
        iree_make_const_byte_span(archive.data, archive.data_length), allocator,
        &exit_code);
  }

  const bool had_error = !iree_status_is_ok(status);
  if (had_error) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = 1;
  }

  iree_vm_instance_release(instance);
  loom_ireevm_module_archive_deinitialize(&archive, allocator);
  loom_module_free(module);
  iree_io_file_contents_free(contents);
  if (context_initialized) {
    loom_context_deinitialize(&context);
  }
  iree_arena_block_pool_deinitialize(&block_pool);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
