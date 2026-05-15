// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-link: links an explicit set of materialized Loom text modules.

#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/tooling/flags.h"
#include "loom/error/diagnostic.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/linker.h"
#include "loom/ops/op_registry.h"
#include "loom/tooling/config/config.h"
#include "loom/tooling/io/file.h"
#include "loom/util/stream.h"
#include "loom/verify/verify.h"

IREE_FLAG(string, output, "-",
          "Output path. Use '-' or the empty string for stdout.");
IREE_FLAG_LIST(string, root,
               "Root symbol to materialize. Repeat for multiple roots. Empty "
               "links every materialized symbol.");
IREE_FLAG_LIST(string, config,
               "Compile/link-time config binding. Repeat as "
               "--config=key=value. Bindings are materialized after linked "
               "config declarations/defaults merge; unused bindings are "
               "ignored.");
IREE_FLAG(bool, require_resolved_config, false,
          "Require all config.decl symbols to be materialized before output.");
IREE_FLAG(bool, print_config_schema, false,
          "Print config schema JSON instead of linked Loom IR.");
IREE_FLAG(bool, verify, true,
          "Verify the linked output module before printing.");

typedef struct loom_link_input_t {
  // Parsed source module owned by the caller.
  loom_module_t* module;
  // Source file contents kept alive for diagnostic source resolution.
  iree_io_file_contents_t* contents;
  // Source table entry for diagnostics that point into this input.
  loom_source_entry_t source_entry;
} loom_link_input_t;

static iree_status_t loom_link_parse_input(iree_string_view_t path,
                                           iree_string_view_t filename,
                                           loom_context_t* context,
                                           iree_arena_block_pool_t* block_pool,
                                           iree_allocator_t allocator,
                                           loom_link_input_t* out_input) {
  IREE_RETURN_IF_ERROR(
      loom_tooling_read_input_file(path, allocator, &out_input->contents));
  iree_string_view_t source =
      loom_tooling_file_contents_string_view(out_input->contents);

  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_context_register_source(context, filename, &source_id));
  out_input->source_entry = (loom_source_entry_t){
      .source_id = source_id,
      .source = source,
      .filename = filename,
  };

  loom_text_parse_options_t parse_options = {
      .diagnostic_sink = {.fn = loom_diagnostic_stderr_sink},
      .max_errors = 20,
  };
  IREE_RETURN_IF_ERROR(loom_text_parse(source, filename, context, block_pool,
                                       &parse_options, &out_input->module));
  if (!out_input->module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "failed to parse '%.*s'", (int)filename.size,
                            filename.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_link_verify_output(
    const loom_source_entry_t* source_entries, iree_host_size_t source_count,
    loom_module_t* module) {
  loom_source_table_resolver_t source_resolver = {
      .entries = source_entries,
      .count = source_count,
  };
  loom_verify_options_t verify_options = {
      .sink = {.fn = loom_diagnostic_stderr_sink},
      .max_errors = 100,
      .source_resolver = {.fn = loom_source_table_resolve,
                          .user_data = &source_resolver},
  };
  loom_verify_result_t verify_result = {0};
  IREE_RETURN_IF_ERROR(
      loom_verify_module(module, &verify_options, &verify_result));
  if (verify_result.error_count > 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "linked module verification failed with %u error%s",
                            verify_result.error_count,
                            verify_result.error_count == 1 ? "" : "s");
  }
  return iree_ok_status();
}

static iree_status_t loom_link_print_output(iree_string_view_t path,
                                            const loom_module_t* module) {
  bool is_stdout = iree_string_view_is_empty(path) ||
                   iree_string_view_equal(path, iree_make_cstring_view("-"));
  FILE* file = stdout;
  if (!is_stdout) {
    file = fopen(path.data, "wb");
    if (!file) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "failed to open output path '%.*s'",
                              (int)path.size, path.data);
    }
  }

  loom_output_stream_t stream;
  loom_output_stream_for_file(file, &stream);
  iree_status_t status =
      loom_text_print_module(module, &stream, LOOM_TEXT_PRINT_DEFAULT);
  if (iree_status_is_ok(status) && fflush(file) != 0) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "failed to flush linked output");
  }
  if (!is_stdout && fclose(file) != 0) {
    status = iree_status_join(
        status, iree_make_status(IREE_STATUS_DATA_LOSS,
                                 "failed to close output path '%.*s'",
                                 (int)path.size, path.data));
  }
  return status;
}

static iree_status_t loom_link_print_config_schema(iree_string_view_t path,
                                                   const loom_module_t* module,
                                                   iree_allocator_t allocator) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  iree_status_t status =
      loom_tooling_config_format_schema_json(module, &stream);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(&builder, IREE_SV("\n"));
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        path, iree_string_builder_view(&builder), allocator);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static void loom_link_release_inputs(loom_link_input_t* inputs,
                                     iree_host_size_t input_count) {
  if (!inputs) {
    return;
  }
  for (iree_host_size_t i = 0; i < input_count; ++i) {
    loom_module_free(inputs[i].module);
    iree_io_file_contents_free(inputs[i].contents);
  }
}

static iree_status_t loom_link_append_config_flags(
    loom_tooling_config_set_t* config_set) {
  iree_flag_string_list_t assignments = FLAG_config_list();
  for (iree_host_size_t i = 0; i < assignments.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_set_append_assignment(
        config_set, assignments.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_materialize_config_set(
    loom_module_t* module, const loom_tooling_config_set_t* config_set,
    iree_arena_block_pool_t* block_pool,
    loom_tooling_config_materialize_result_t* out_result) {
  *out_result = (loom_tooling_config_materialize_result_t){0};
  loom_tooling_config_materialize_options_t options;
  loom_tooling_config_materialize_options_initialize(&options);
  options.config_set = config_set;
  return loom_tooling_config_materialize_module(module, &options, block_pool,
                                                out_result);
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "loom-link",
      "Links an explicit set of already-materialized Loom text modules.\n"
      "\n"
      "Usage:\n"
      "  loom-link [--output=file] [--root=@symbol] [file...]\n"
      "  loom-link harness.loom corpus.loom --root=@entry "
      "--output=linked.loom\n"
      "\n"
      "Input defaults to stdin when no files are provided. A '-' input reads "
      "stdin and must be the only input path.\n"
      "Repeat --config=key=value to override linked config symbols after "
      "config declarations/defaults merge.\n"
      "Use --require-resolved-config for final outputs that must not retain "
      "config.decl symbols.\n"
      "Use --print-config-schema to print config schema JSON instead of Loom "
      "IR.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t allocator = iree_allocator_system();
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, allocator, &block_pool);

  loom_context_t context = {0};
  bool context_initialized = false;
  loom_link_input_t* inputs = NULL;
  const loom_module_t** source_modules = NULL;
  loom_source_entry_t* source_entries = NULL;
  loom_module_t* linked_module = NULL;
  loom_tooling_config_set_t config_set;
  loom_tooling_config_set_initialize(allocator, &config_set);
  loom_tooling_config_materialize_result_t config_materialize_result = {0};

  iree_host_size_t input_count = argc < 2 ? 1 : (iree_host_size_t)(argc - 1);
  iree_status_t status = iree_allocator_malloc(
      allocator, input_count * sizeof(*inputs), (void**)&inputs);
  if (iree_status_is_ok(status)) {
    memset(inputs, 0, input_count * sizeof(*inputs));
    status =
        iree_allocator_malloc(allocator, input_count * sizeof(*source_modules),
                              (void**)&source_modules);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(allocator, input_count * sizeof(*source_entries),
                              (void**)&source_entries);
  }

  if (iree_status_is_ok(status)) {
    status = loom_op_registry_initialize_context(allocator, &context);
    context_initialized = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = loom_link_append_config_flags(&config_set);
  }
  for (iree_host_size_t i = 0; i < input_count && iree_status_is_ok(status);
       ++i) {
    iree_string_view_t path = argc < 2 ? iree_string_view_empty()
                                       : iree_make_cstring_view(argv[i + 1]);
    if (input_count > 1 &&
        iree_string_view_equal(path, iree_make_cstring_view("-"))) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "'-' stdin input must be the only input path");
      break;
    }
    iree_string_view_t filename =
        loom_tooling_file_path_is_stdio(path) ? IREE_SV("<stdin>") : path;
    status = loom_link_parse_input(path, filename, &context, &block_pool,
                                   allocator, &inputs[i]);
    if (iree_status_is_ok(status)) {
      source_modules[i] = inputs[i].module;
      source_entries[i] = inputs[i].source_entry;
    }
  }

  if (iree_status_is_ok(status)) {
    const iree_flag_string_list_t roots = FLAG_root_list();
    loom_link_options_t link_options = {
        .module_name = IREE_SV("linked"),
        .root_symbols = roots,
    };
    status = loom_link_materialized_modules(source_modules, input_count,
                                            &link_options, &block_pool,
                                            allocator, &linked_module);
  }
  if (iree_status_is_ok(status)) {
    status = loom_link_materialize_config_set(
        linked_module, &config_set, &block_pool, &config_materialize_result);
  }
  if (iree_status_is_ok(status) && FLAG_require_resolved_config) {
    status = loom_tooling_config_require_resolved_module(linked_module, NULL);
  }
  if (iree_status_is_ok(status) && FLAG_verify) {
    status =
        loom_link_verify_output(source_entries, input_count, linked_module);
  }
  if (iree_status_is_ok(status)) {
    if (FLAG_print_config_schema) {
      status = loom_link_print_config_schema(
          iree_make_cstring_view(FLAG_output), linked_module, allocator);
    } else {
      status = loom_link_print_output(iree_make_cstring_view(FLAG_output),
                                      linked_module);
    }
  }

  bool had_error = !iree_status_is_ok(status);
  if (had_error) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
  }

  loom_module_free(linked_module);
  loom_tooling_config_set_deinitialize(&config_set);
  loom_link_release_inputs(inputs, input_count);
  iree_allocator_free(allocator, source_entries);
  iree_allocator_free(allocator, source_modules);
  iree_allocator_free(allocator, inputs);
  if (context_initialized) {
    loom_context_deinitialize(&context);
  }
  iree_arena_block_pool_deinitialize(&block_pool);
  return had_error ? 1 : 0;
}
