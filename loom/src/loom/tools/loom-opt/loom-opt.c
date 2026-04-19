// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-opt: executes Loom pass pipelines and prints the transformed module.

#include <inttypes.h>
#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "loom/codegen/low/allocation_pass.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/diagnostic.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/pass/builtin_registry.h"
#include "loom/pass/registry.h"
#include "loom/pass/tooling.h"
#include "loom/target/all/low_registry.h"
#include "loom/target/presets.h"
#include "loom/util/stream.h"
#include "loom/verify/verify.h"

IREE_FLAG(string, output, "-",
          "Output path. Use '-' or the empty string for stdout.");
IREE_FLAG(string, pipeline, "",
          "Named pass.pipeline symbol to execute from the input module.");
IREE_FLAG_LIST(string, pass,
               "Pass pipeline entry to append. Repeat for multiple passes.");
IREE_FLAG(bool, verify, true,
          "Verify the module before and after executing passes.");
IREE_FLAG(bool, list_passes, false, "Print registered passes and exit.");
IREE_FLAG(string, pass_help, "", "Print detailed help for one pass and exit.");
IREE_FLAG(string, low_asm_descriptor_set, "",
          "Descriptor-set key used when printing low asm regions.");

typedef struct loom_opt_pass_pipeline_config_t {
  // Low allocation pass configuration borrowed by matching pipeline entries.
  loom_low_materialize_allocation_pass_config_t* low_allocation_config;
} loom_opt_pass_pipeline_config_t;

typedef struct loom_opt_diagnostic_emitter_t {
  // Module containing the op referenced by emitted diagnostics.
  const loom_module_t* module;
  // Source resolver for source-backed operation locations.
  loom_source_resolver_t source_resolver;
  // Subsystem identity to store in materialized diagnostics.
  loom_emitter_t emitter;
} loom_opt_diagnostic_emitter_t;

static const char* loom_opt_pass_kind_name(loom_pass_kind_t kind) {
  switch (kind) {
    case LOOM_PASS_MODULE:
      return "module";
    case LOOM_PASS_FUNCTION:
      return "func";
    default:
      return "unknown";
  }
}

static iree_status_t loom_opt_read_input(
    iree_string_view_t path, iree_allocator_t allocator,
    iree_io_file_contents_t** out_contents) {
  bool is_stdin = iree_string_view_is_empty(path) ||
                  iree_string_view_equal(path, iree_make_cstring_view("-"));
  if (is_stdin) {
    return iree_io_file_contents_read_stdin(allocator, out_contents);
  }
  return iree_io_file_contents_read(path, allocator, out_contents);
}

static iree_string_view_t loom_opt_file_contents_string_view(
    const iree_io_file_contents_t* contents) {
  return iree_make_string_view((const char*)contents->const_buffer.data,
                               contents->const_buffer.data_length);
}

static iree_status_t loom_opt_write_output(iree_string_view_t path,
                                           iree_string_view_t output,
                                           iree_allocator_t allocator) {
  bool is_stdout = iree_string_view_is_empty(path) ||
                   iree_string_view_equal(path, iree_make_cstring_view("-"));
  iree_const_byte_span_t bytes =
      iree_make_const_byte_span(output.data, output.size);
  if (!is_stdout) {
    return iree_io_file_contents_write(path, bytes, allocator);
  }

  if (bytes.data_length > 0) {
    size_t write_count = fwrite(bytes.data, bytes.data_length, 1, stdout);
    if (write_count != 1) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "failed to write %" PRIhsz " bytes to stdout",
                              bytes.data_length);
    }
  }
  if (fflush(stdout) != 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "failed to flush stdout");
  }
  return iree_ok_status();
}

static iree_status_t loom_opt_source_resolver_for_input(
    loom_context_t* context, iree_string_view_t filename,
    iree_string_view_t source, loom_source_entry_t* out_source_entry,
    loom_source_table_resolver_t* out_source_resolver) {
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

static bool loom_opt_resolve_emission_location(
    const loom_opt_diagnostic_emitter_t* emitter, const loom_op_t* op,
    loom_source_range_t* out_source_location) {
  if (!emitter || !emitter->module || !op) {
    return false;
  }
  if (!loom_source_resolve(emitter->source_resolver, emitter->module,
                           op->location, out_source_location)) {
    return false;
  }
  if (out_source_location->provenance ==
          LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE &&
      out_source_location->source.size > 0) {
    out_source_location->provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE;
  }
  return true;
}

static iree_host_size_t loom_opt_collect_related_locations(
    const loom_opt_diagnostic_emitter_t* emitter,
    const loom_diagnostic_related_op_t* related_ops,
    iree_host_size_t related_op_count,
    loom_diagnostic_related_location_t* out_related_locations) {
  if (!related_ops || related_op_count == 0) {
    return 0;
  }
  iree_host_size_t related_location_count = 0;
  for (iree_host_size_t i = 0;
       i < related_op_count &&
       related_location_count < LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS;
       ++i) {
    loom_source_range_t source_location = {
        .provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE,
    };
    if (!loom_opt_resolve_emission_location(emitter, related_ops[i].op,
                                            &source_location)) {
      continue;
    }
    out_related_locations[related_location_count++] =
        (loom_diagnostic_related_location_t){
            .label = related_ops[i].label,
            .source_location = source_location,
        };
  }
  return related_location_count;
}

static iree_status_t loom_opt_diagnostic_emitter_emit(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loom_opt_diagnostic_emitter_t* emitter =
      (loom_opt_diagnostic_emitter_t*)user_data;
  if (!emitter || !emission || !emission->error) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "diagnostic emitter requires an emission");
  }

  loom_diagnostic_t diagnostic = {
      .severity = emission->error->severity,
      .error = emission->error,
      .params = emission->params,
      .param_count = emission->param_count,
      .emitter = emitter->emitter,
      .origin = {.provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE},
      .source_location = {.provenance =
                              LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE},
  };

  loom_diagnostic_related_location_t
      related_locations[LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS];
  diagnostic.related_location_count = loom_opt_collect_related_locations(
      emitter, emission->related_ops, emission->related_op_count,
      related_locations);
  if (diagnostic.related_location_count > 0) {
    diagnostic.related_locations = related_locations;
  }

  if (loom_opt_resolve_emission_location(emitter, emission->op,
                                         &diagnostic.source_location)) {
    diagnostic.origin = diagnostic.source_location;
  }
  return loom_diagnostic_stderr_sink(NULL, &diagnostic);
}

static iree_status_t loom_opt_verify_module(
    iree_string_view_t filename, iree_string_view_t source,
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_module_t* module) {
  loom_source_entry_t source_entry = {0};
  loom_source_table_resolver_t source_resolver = {0};
  IREE_RETURN_IF_ERROR(loom_opt_source_resolver_for_input(
      module->context, filename, source, &source_entry, &source_resolver));

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
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module verification failed with %" PRIu32 " error%s",
        verify_result.error_count, verify_result.error_count == 1 ? "" : "s");
  }

  loom_opt_diagnostic_emitter_t low_emitter = {
      .module = module,
      .source_resolver = {.fn = loom_source_table_resolve,
                          .user_data = &source_resolver},
      .emitter = LOOM_EMITTER_VERIFIER,
  };
  loom_low_verify_options_t low_verify_options = {
      .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
      .descriptor_registry = &low_registry->registry,
      .emitter = {.fn = loom_opt_diagnostic_emitter_emit,
                  .user_data = &low_emitter},
      .max_errors = 100,
  };
  loom_low_verify_result_t low_verify_result = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_verify_module(module, &low_verify_options, &low_verify_result));
  if (low_verify_result.error_count > 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low verification failed with %" PRIu32 " error%s",
                            low_verify_result.error_count,
                            low_verify_result.error_count == 1 ? "" : "s");
  }
  return iree_ok_status();
}

static iree_status_t loom_opt_expand_target_presets(
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_module_t* module) {
  const loom_target_preset_registry_t preset_registry =
      loom_target_low_descriptor_registry_presets(low_registry);
  iree_host_size_t expanded_preset_count = 0;
  return loom_target_expand_presets(module, &preset_registry,
                                    &expanded_preset_count);
}

static iree_status_t loom_opt_configure_pass_instruction(
    void* user_data, const loom_pass_program_instruction_t* instruction,
    void** out_pass_user_data) {
  IREE_ASSERT_ARGUMENT(user_data);
  IREE_ASSERT_ARGUMENT(instruction);
  IREE_ASSERT_ARGUMENT(out_pass_user_data);

  loom_opt_pass_pipeline_config_t* config =
      (loom_opt_pass_pipeline_config_t*)user_data;
  *out_pass_user_data = NULL;
  if (instruction->kind == LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE &&
      iree_string_view_equal(instruction->invoke.descriptor->key,
                             IREE_SV("low-materialize-allocation"))) {
    *out_pass_user_data = config->low_allocation_config;
  }
  return iree_ok_status();
}

static iree_status_t loom_opt_join_pass_list(iree_flag_string_list_t passes,
                                             iree_string_builder_t* builder) {
  for (iree_host_size_t i = 0; i < passes.count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_string(builder, IREE_SV(",")));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, passes.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_opt_run_passes(
    iree_string_view_t filename, iree_string_view_t source,
    const loom_target_low_descriptor_registry_t* low_registry,
    iree_arena_block_pool_t* block_pool, loom_module_t* module,
    iree_allocator_t allocator) {
  iree_flag_string_list_t passes = FLAG_pass_list();
  iree_string_view_t pipeline_symbol =
      iree_string_view_trim(iree_make_cstring_view(FLAG_pipeline));
  bool has_pipeline_symbol = !iree_string_view_is_empty(pipeline_symbol);
  bool has_pass_list = passes.count > 0;
  if (has_pipeline_symbol && has_pass_list) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--pipeline and --pass cannot be combined");
  }
  if (!has_pipeline_symbol && !has_pass_list) {
    return iree_ok_status();
  }

  loom_source_entry_t source_entry = {0};
  loom_source_table_resolver_t source_resolver = {0};
  IREE_RETURN_IF_ERROR(loom_opt_source_resolver_for_input(
      module->context, filename, source, &source_entry, &source_resolver));
  loom_opt_diagnostic_emitter_t pass_emitter = {
      .module = module,
      .source_resolver = {.fn = loom_source_table_resolve,
                          .user_data = &source_resolver},
      .emitter = LOOM_EMITTER_PASS,
  };

  loom_low_materialize_allocation_pass_config_t low_allocation_config = {
      .descriptor_registry = &low_registry->registry,
  };
  loom_opt_pass_pipeline_config_t pipeline_config = {
      .low_allocation_config = &low_allocation_config,
  };
  loom_pass_tool_run_options_t run_options = {
      .registry = loom_pass_builtin_registry(),
      .block_pool = block_pool,
      .diagnostic_emitter = {.fn = loom_opt_diagnostic_emitter_emit,
                             .user_data = &pass_emitter},
      .configure =
          {
              .fn = loom_opt_configure_pass_instruction,
              .user_data = &pipeline_config,
          },
  };

  if (has_pipeline_symbol) {
    return loom_pass_tool_run_pipeline_symbol(module, pipeline_symbol,
                                              &run_options);
  }

  iree_string_builder_t pipeline_builder;
  iree_string_builder_initialize(allocator, &pipeline_builder);
  iree_status_t status = loom_opt_join_pass_list(passes, &pipeline_builder);
  if (iree_status_is_ok(status)) {
    status = loom_pass_tool_run_flat_pipeline(
        module, iree_string_builder_view(&pipeline_builder), &run_options);
  }
  iree_string_builder_deinitialize(&pipeline_builder);
  return status;
}

static iree_status_t loom_opt_print_module(
    iree_string_view_t output_path,
    const loom_target_low_descriptor_registry_t* low_registry,
    const loom_module_t* module, iree_allocator_t allocator) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);

  loom_text_print_options_t print_options = {
      .flags = LOOM_TEXT_PRINT_DEFAULT,
      .low_asm_descriptor_set_key =
          iree_make_cstring_view(FLAG_low_asm_descriptor_set),
  };
  loom_low_descriptor_text_asm_environment_initialize(
      &low_registry->registry, &print_options.low_asm_environment);
  iree_status_t status = loom_text_print_module_to_builder_with_options(
      module, &builder, &print_options);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(&builder, IREE_SV("\n"));
  }
  if (iree_status_is_ok(status)) {
    status = loom_opt_write_output(
        output_path, iree_string_builder_view(&builder), allocator);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_opt_print_pass_list(
    const loom_pass_registry_t* registry) {
  IREE_RETURN_IF_ERROR(loom_pass_registry_verify(registry));
  for (iree_host_size_t i = 0; i < registry->descriptor_count; ++i) {
    const loom_pass_descriptor_t* descriptor = &registry->descriptors[i];
    const loom_pass_info_t* info = descriptor->info();
    if (!info) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pass descriptor '%.*s' returned no info",
                              (int)descriptor->key.size, descriptor->key.data);
    }
    fprintf(stdout, "%.*s\t%s\t%.*s\n", (int)descriptor->key.size,
            descriptor->key.data, loom_opt_pass_kind_name(info->kind),
            (int)info->description.size, info->description.data);
  }
  return fflush(stdout) == 0 ? iree_ok_status()
                             : iree_make_status(IREE_STATUS_DATA_LOSS,
                                                "failed to flush stdout");
}

static void loom_opt_print_pass_option_schema(
    const loom_pass_option_schema_t* schema) {
  fprintf(stdout, "    %.*s: ", (int)schema->name.size, schema->name.data);
  switch (schema->kind) {
    case LOOM_PASS_OPTION_SCHEMA_STRING:
      fprintf(stdout, "string");
      break;
    case LOOM_PASS_OPTION_SCHEMA_UINT32:
      fprintf(stdout, "uint32 [%" PRIu32 "..%" PRIu32 "]",
              schema->minimum_uint32, schema->maximum_uint32);
      break;
    case LOOM_PASS_OPTION_SCHEMA_ENUM:
      fprintf(stdout, "enum");
      if (schema->enum_value_count > 0) {
        fprintf(stdout, " {");
        for (uint16_t i = 0; i < schema->enum_value_count; ++i) {
          if (i > 0) {
            fprintf(stdout, ", ");
          }
          fprintf(stdout, "%.*s", (int)schema->enum_values[i].value.size,
                  schema->enum_values[i].value.data);
        }
        fprintf(stdout, "}");
      }
      break;
    default:
      fprintf(stdout, "unknown");
      break;
  }
  fprintf(stdout, "\n");
}

static iree_status_t loom_opt_print_pass_help(
    const loom_pass_registry_t* registry, iree_string_view_t key) {
  IREE_RETURN_IF_ERROR(loom_pass_registry_verify(registry));
  const loom_pass_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_registry_lookup(registry, key, &descriptor));
  if (!descriptor) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "unknown pass '%.*s'",
                            (int)key.size, key.data);
  }
  const loom_pass_info_t* info = descriptor->info();
  if (!info) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass descriptor '%.*s' returned no info",
                            (int)descriptor->key.size, descriptor->key.data);
  }

  fprintf(stdout, "%.*s\n", (int)descriptor->key.size, descriptor->key.data);
  fprintf(stdout, "  kind: %s\n", loom_opt_pass_kind_name(info->kind));
  fprintf(stdout, "  description: %.*s\n", (int)info->description.size,
          info->description.data);
  if (!loom_pass_descriptor_is_available(descriptor)) {
    fprintf(stdout, "  unavailable: %.*s\n",
            (int)descriptor->unavailable_reason.size,
            descriptor->unavailable_reason.data);
  }
  if (descriptor->option_schema_count > 0) {
    fprintf(stdout, "  options:\n");
    for (uint16_t i = 0; i < descriptor->option_schema_count; ++i) {
      loom_opt_print_pass_option_schema(&descriptor->option_schema[i]);
    }
  }
  return fflush(stdout) == 0 ? iree_ok_status()
                             : iree_make_status(IREE_STATUS_DATA_LOSS,
                                                "failed to flush stdout");
}

static iree_status_t loom_opt_parse_module(
    iree_string_view_t filename, iree_string_view_t source,
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    loom_module_t** out_module) {
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

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "loom-opt",
      "Executes Loom pass pipelines and prints the transformed module.\n"
      "\n"
      "Usage:\n"
      "  loom-opt [--pipeline=@name] [--output=file] [file]\n"
      "  loom-opt --pass=canonicalize --pass=cse --pass=dce [file]\n"
      "  cat module.loom | loom-opt --pass=symbol-dce\n"
      "  loom-opt --list-passes\n"
      "  loom-opt --pass-help=canonicalize\n"
      "\n"
      "Input defaults to stdin when no file is provided. Output defaults to "
      "stdout.\n"
      "Use --pipeline to execute a named pass.pipeline symbol from the input "
      "module, or\n"
      "repeat --pass for a shallow command-line pipeline backed by the C pass "
      "registry.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t allocator = iree_allocator_system();
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, allocator, &block_pool);

  loom_context_t context = {0};
  bool context_initialized = false;
  iree_io_file_contents_t* contents = NULL;
  loom_module_t* module = NULL;
  loom_target_low_descriptor_registry_t low_registry = {0};
  const loom_pass_registry_t* pass_registry = loom_pass_builtin_registry();
  iree_string_view_t source = iree_string_view_empty();

  iree_status_t status = iree_ok_status();
  if (FLAG_list_passes) {
    status = loom_opt_print_pass_list(pass_registry);
  }
  iree_string_view_t pass_help =
      iree_string_view_trim(iree_make_cstring_view(FLAG_pass_help));
  if (iree_status_is_ok(status) && !iree_string_view_is_empty(pass_help)) {
    status = loom_opt_print_pass_help(pass_registry, pass_help);
  }

  bool metadata_only =
      FLAG_list_passes || !iree_string_view_is_empty(pass_help);
  if (iree_status_is_ok(status) && !metadata_only && argc > 2) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "loom-opt accepts at most one input file or '-' for stdin; got %d "
        "inputs",
        argc - 1);
  }

  if (iree_status_is_ok(status) && !metadata_only) {
    loom_all_low_descriptor_registry_initialize(&low_registry);
    status = loom_op_registry_initialize_context(allocator, &context);
    context_initialized = iree_status_is_ok(status);
  }

  iree_string_view_t input_path =
      argc < 2 ? iree_string_view_empty() : iree_make_cstring_view(argv[1]);
  iree_string_view_t filename =
      (argc < 2 ||
       iree_string_view_equal(input_path, iree_make_cstring_view("-")))
          ? iree_make_cstring_view("<stdin>")
          : input_path;

  if (iree_status_is_ok(status) && !metadata_only) {
    status = loom_opt_read_input(input_path, allocator, &contents);
    if (iree_status_is_ok(status)) {
      source = loom_opt_file_contents_string_view(contents);
    }
  }
  if (iree_status_is_ok(status) && !metadata_only) {
    status = loom_opt_parse_module(filename, source, &low_registry, &context,
                                   &block_pool, &module);
  }
  if (iree_status_is_ok(status) && !metadata_only) {
    status = loom_opt_expand_target_presets(&low_registry, module);
  }
  if (iree_status_is_ok(status) && !metadata_only && FLAG_verify) {
    status = loom_opt_verify_module(filename, source, &low_registry, module);
  }
  if (iree_status_is_ok(status) && !metadata_only) {
    status = loom_opt_run_passes(filename, source, &low_registry, &block_pool,
                                 module, allocator);
  }
  if (iree_status_is_ok(status) && !metadata_only && FLAG_verify) {
    status = loom_opt_verify_module(filename, source, &low_registry, module);
  }
  if (iree_status_is_ok(status) && !metadata_only) {
    status = loom_opt_print_module(iree_make_cstring_view(FLAG_output),
                                   &low_registry, module, allocator);
  }

  bool had_error = !iree_status_is_ok(status);
  if (had_error) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
  }

  loom_module_free(module);
  iree_io_file_contents_free(contents);
  if (context_initialized) {
    loom_context_deinitialize(&context);
  }
  iree_arena_block_pool_deinitialize(&block_pool);
  return had_error ? 1 : 0;
}
