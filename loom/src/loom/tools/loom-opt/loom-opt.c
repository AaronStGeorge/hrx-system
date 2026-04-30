// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-opt: executes Loom pass pipelines and prints the transformed module.

#include <inttypes.h>
#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "loom/codegen/low/pass_environment.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/diagnostic.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/pass/builtin_registry.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/pass/report.h"
#include "loom/pass/tooling.h"
#include "loom/target/all/low_registry.h"
#include "loom/tooling/execution/session.h"
#include "loom/tooling/io/file.h"
#include "loom/util/json.h"
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
IREE_FLAG(string, pass_report, "",
          "Pass execution report format. Use 'json' or empty/'none'.");
IREE_FLAG(string, pass_reproducer, "",
          "Path to a one-file pass failure reproducer to write on failure.");
IREE_FLAG(string, low_asm_descriptor_set, "",
          "Descriptor-set key used when printing low asm regions.");

typedef enum loom_opt_pass_report_mode_e {
  LOOM_OPT_PASS_REPORT_NONE = 0,
  LOOM_OPT_PASS_REPORT_JSON = 1,
} loom_opt_pass_report_mode_t;

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

static iree_status_t loom_opt_parse_pass_report_mode(
    iree_string_view_t value, loom_opt_pass_report_mode_t* out_mode) {
  value = iree_string_view_trim(value);
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("none"))) {
    *out_mode = LOOM_OPT_PASS_REPORT_NONE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("json"))) {
    *out_mode = LOOM_OPT_PASS_REPORT_JSON;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported --pass-report mode '%.*s'",
                          (int)value.size, value.data);
}

static iree_status_t loom_opt_register_context(void* user_data,
                                               loom_context_t* context) {
  (void)user_data;
  return loom_op_registry_register_all_dialects(context);
}

static iree_status_t loom_opt_initialize_low_descriptor_registry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  (void)user_data;
  loom_all_low_descriptor_registry_initialize(out_registry);
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
    loom_diagnostic_related_location_t* out_related_locations,
    iree_host_size_t* out_omitted_count) {
  *out_omitted_count = 0;
  if (!related_ops || related_op_count == 0) {
    return 0;
  }
  iree_host_size_t related_location_count = 0;
  for (iree_host_size_t i = 0; i < related_op_count; ++i) {
    loom_source_range_t source_location = {
        .provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE,
    };
    if (!loom_opt_resolve_emission_location(emitter, related_ops[i].op,
                                            &source_location)) {
      continue;
    }
    if (related_location_count >= LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS) {
      ++*out_omitted_count;
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
      related_locations, &diagnostic.related_location_omitted_count);
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
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_run_module_t* run_module) {
  loom_module_t* module = run_module->module;
  loom_source_resolver_t source_resolver =
      loom_run_module_source_resolver(run_module);

  loom_verify_options_t verify_options = {
      .sink = {.fn = loom_diagnostic_stderr_sink},
      .max_errors = 100,
      .source_resolver = source_resolver,
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
      .source_resolver = source_resolver,
      .emitter = LOOM_EMITTER_VERIFIER,
  };
  loom_low_verify_options_t low_verify_options = {
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

static iree_status_t loom_opt_append_shell_quoted(
    iree_string_builder_t* builder, iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "'"));
  iree_host_size_t segment_start = 0;
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    if (value.data[i] != '\'') {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder,
        iree_make_string_view(value.data + segment_start, i - segment_start)));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, "'\"'\"'"));
    segment_start = i + 1;
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      builder, iree_make_string_view(value.data + segment_start,
                                     value.size - segment_start)));
  return iree_string_builder_append_cstring(builder, "'");
}

static iree_status_t loom_opt_append_reproducer_run_line(
    iree_string_builder_t* builder, bool use_synthetic_pipeline) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "// RUN: loom-opt --verify=%s", FLAG_verify ? "true" : "false"));
  iree_string_view_t low_asm_descriptor_set = iree_string_view_trim(
      iree_make_cstring_view(FLAG_low_asm_descriptor_set));
  if (!iree_string_view_is_empty(low_asm_descriptor_set)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        builder, " --low-asm-descriptor-set="));
    IREE_RETURN_IF_ERROR(
        loom_opt_append_shell_quoted(builder, low_asm_descriptor_set));
  }
  iree_string_view_t pass_report =
      iree_string_view_trim(iree_make_cstring_view(FLAG_pass_report));
  if (!iree_string_view_is_empty(pass_report)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, " --pass-report="));
    IREE_RETURN_IF_ERROR(loom_opt_append_shell_quoted(builder, pass_report));
  }

  iree_string_view_t pipeline_symbol =
      iree_string_view_trim(iree_make_cstring_view(FLAG_pipeline));
  if (use_synthetic_pipeline) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        builder, " --pipeline='@__command_line'"));
  } else if (!iree_string_view_is_empty(pipeline_symbol)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, " --pipeline="));
    IREE_RETURN_IF_ERROR(
        loom_opt_append_shell_quoted(builder, pipeline_symbol));
  } else {
    iree_flag_string_list_t passes = FLAG_pass_list();
    for (iree_host_size_t i = 0; i < passes.count; ++i) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(builder, " --pass="));
      IREE_RETURN_IF_ERROR(
          loom_opt_append_shell_quoted(builder, passes.values[i]));
    }
  }
  return iree_string_builder_append_cstring(builder, " %s\n");
}

static bool loom_opt_pass_key_is_printable(iree_string_view_t key) {
  if (key.size == 0) return false;
  char first = key.data[0];
  bool valid_start = (first >= 'a' && first <= 'z') ||
                     (first >= 'A' && first <= 'Z') || first == '_' ||
                     first == '$';
  if (!valid_start) return false;
  for (iree_host_size_t i = 1; i < key.size; ++i) {
    char c = key.data[i];
    bool valid = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                 (c >= '0' && c <= '9') || c == '_' || c == '$' || c == '-' ||
                 c == '.';
    if (!valid) return false;
  }
  return true;
}

static iree_status_t loom_opt_append_reproducer_pass_run(
    iree_string_builder_t* builder, iree_string_view_t indent,
    const loom_pass_pipeline_entry_spec_t* spec) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, indent));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "pass.run<"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, spec->name));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ">"));
  if (!iree_string_view_is_empty(spec->options)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " {"));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(builder, spec->options));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}"));
  }
  return iree_string_builder_append_cstring(builder, "\n");
}

static iree_status_t loom_opt_reproducer_function_group_open(
    iree_string_builder_t* builder, bool* in_function_group) {
  if (*in_function_group) {
    return iree_ok_status();
  }
  *in_function_group = true;
  return iree_string_builder_append_cstring(builder,
                                            "  pass.for<func> pipeline {\n");
}

static iree_status_t loom_opt_reproducer_function_group_close(
    iree_string_builder_t* builder, bool* in_function_group) {
  if (!*in_function_group) {
    return iree_ok_status();
  }
  *in_function_group = false;
  return iree_string_builder_append_cstring(builder, "    pass.yield\n  }\n");
}

static iree_status_t loom_opt_build_reproducer_synthetic_pipeline(
    iree_string_builder_t* builder, const loom_pass_registry_t* registry,
    bool* out_use_synthetic_pipeline, iree_allocator_t allocator) {
  *out_use_synthetic_pipeline = false;
  iree_string_view_t pipeline_symbol =
      iree_string_view_trim(iree_make_cstring_view(FLAG_pipeline));
  iree_flag_string_list_t passes = FLAG_pass_list();
  if (!iree_string_view_is_empty(pipeline_symbol) || passes.count == 0) {
    return iree_ok_status();
  }

  iree_string_builder_t pipeline_builder;
  iree_string_builder_initialize(allocator, &pipeline_builder);
  iree_status_t status = iree_string_builder_append_cstring(
      &pipeline_builder, "pass.pipeline<module> @__command_line {\n");

  iree_string_builder_t pass_list_builder;
  iree_string_builder_initialize(allocator, &pass_list_builder);
  if (iree_status_is_ok(status)) {
    status = loom_opt_join_pass_list(passes, &pass_list_builder);
  }

  iree_string_view_t remaining = iree_string_builder_view(&pass_list_builder);
  bool in_function_group = false;
  while (iree_status_is_ok(status)) {
    loom_pass_pipeline_entry_spec_t spec = {0};
    bool has_entry = false;
    status = loom_pass_pipeline_consume_entry(&remaining, &spec, &has_entry);
    if (!iree_status_is_ok(status) || !has_entry) break;
    if (!loom_opt_pass_key_is_printable(spec.name)) {
      iree_string_builder_reset(&pipeline_builder);
      status = iree_ok_status();
      break;
    }

    const loom_pass_descriptor_t* descriptor = NULL;
    status = loom_pass_registry_lookup(registry, spec.name, &descriptor);
    if (!iree_status_is_ok(status)) break;
    const loom_pass_info_t* info = descriptor ? descriptor->info() : NULL;
    if (info && info->kind == LOOM_PASS_FUNCTION) {
      status = loom_opt_reproducer_function_group_open(&pipeline_builder,
                                                       &in_function_group);
      if (iree_status_is_ok(status)) {
        status = loom_opt_append_reproducer_pass_run(&pipeline_builder,
                                                     IREE_SV("    "), &spec);
      }
    } else {
      status = loom_opt_reproducer_function_group_close(&pipeline_builder,
                                                        &in_function_group);
      if (iree_status_is_ok(status)) {
        status = loom_opt_append_reproducer_pass_run(&pipeline_builder,
                                                     IREE_SV("  "), &spec);
      }
    }
  }

  status = iree_status_join(status, loom_opt_reproducer_function_group_close(
                                        &pipeline_builder, &in_function_group));
  if (iree_status_is_ok(status) &&
      iree_string_builder_size(&pipeline_builder)) {
    status = iree_string_builder_append_cstring(&pipeline_builder,
                                                "  pass.yield\n}\n");
  }
  if (iree_status_is_ok(status) &&
      iree_string_builder_size(&pipeline_builder)) {
    status = iree_string_builder_append_string(
        builder, iree_string_builder_view(&pipeline_builder));
  }
  if (iree_status_is_ok(status) &&
      iree_string_builder_size(&pipeline_builder)) {
    *out_use_synthetic_pipeline = true;
  }

  iree_string_builder_deinitialize(&pass_list_builder);
  iree_string_builder_deinitialize(&pipeline_builder);
  return status;
}

static iree_status_t loom_opt_format_pass_selection_json(
    loom_output_stream_t* stream) {
  iree_string_view_t pipeline_symbol =
      iree_string_view_trim(iree_make_cstring_view(FLAG_pipeline));
  if (!iree_string_view_is_empty(pipeline_symbol)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "{\"kind\":\"pipeline\","));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "\"pipeline\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(stream, pipeline_symbol));
    return loom_output_stream_write_cstring(stream, "}");
  }

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "{\"kind\":\"pass-list\","));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "\"passes\":["));
  iree_flag_string_list_t passes = FLAG_pass_list();
  for (iree_host_size_t i = 0; i < passes.count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, i == 0 ? "" : ","));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(stream, passes.values[i]));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]}"));
  return iree_ok_status();
}

static iree_status_t loom_opt_append_commented_block(
    iree_string_builder_t* builder, iree_string_view_t text) {
  bool at_line_start = true;
  iree_host_size_t segment_start = 0;
  for (iree_host_size_t i = 0; i < text.size; ++i) {
    if (at_line_start) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "// "));
      at_line_start = false;
      segment_start = i;
    }
    if (text.data[i] != '\n') {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, iree_make_string_view(text.data + segment_start,
                                       i - segment_start + 1)));
    at_line_start = true;
    segment_start = i + 1;
  }
  if (segment_start < text.size) {
    if (at_line_start) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "// "));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, iree_make_string_view(text.data + segment_start,
                                       text.size - segment_start)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  }
  return iree_ok_status();
}

static iree_status_t loom_opt_append_reproducer_metadata(
    iree_string_builder_t* builder, const loom_pass_registry_t* registry,
    iree_status_code_t failure_status_code, iree_allocator_t allocator) {
  iree_string_builder_t metadata_builder;
  iree_string_builder_initialize(allocator, &metadata_builder);
  loom_output_stream_t metadata_stream;
  loom_output_stream_for_builder(&metadata_builder, &metadata_stream);

  iree_status_t status = loom_output_stream_write_cstring(
      &metadata_stream, "{\n  \"selection\": ");
  if (iree_status_is_ok(status)) {
    status = loom_opt_format_pass_selection_json(&metadata_stream);
  }
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_cstring(&metadata_stream,
                                              ",\n  \"failure_status\": ");
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_write_escaped_cstring(
        &metadata_stream, iree_status_code_string(failure_status_code));
  }
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_cstring(&metadata_stream,
                                              ",\n  \"registry\": ");
  }
  if (iree_status_is_ok(status)) {
    status = loom_pass_report_format_registry_json(registry, &metadata_stream);
  }
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_cstring(&metadata_stream, "}\n");
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(builder,
                                                "// Pass failure metadata:\n");
  }
  if (iree_status_is_ok(status)) {
    status = loom_opt_append_commented_block(
        builder, iree_string_builder_view(&metadata_builder));
  }

  iree_string_builder_deinitialize(&metadata_builder);
  return status;
}

static iree_status_t loom_opt_write_pass_reproducer(
    iree_string_view_t path, iree_string_view_t filename,
    iree_string_view_t source, const loom_pass_registry_t* registry,
    iree_status_code_t failure_status_code, iree_allocator_t allocator) {
  path = iree_string_view_trim(path);
  if (iree_string_view_is_empty(path)) {
    return iree_ok_status();
  }

  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);
  iree_status_t status = iree_string_builder_append_cstring(
      &builder,
      "// Reproducer generated by loom-opt after pass pipeline failure.\n");
  if (iree_status_is_ok(status)) {
    status =
        iree_string_builder_append_format(&builder, "// Original input: %.*s\n",
                                          (int)filename.size, filename.data);
  }
  if (iree_status_is_ok(status)) {
    status = loom_opt_append_reproducer_metadata(
        &builder, registry, failure_status_code, allocator);
  }

  iree_string_builder_t synthetic_pipeline_builder;
  iree_string_builder_initialize(allocator, &synthetic_pipeline_builder);
  bool use_synthetic_pipeline = false;
  if (iree_status_is_ok(status)) {
    status = loom_opt_build_reproducer_synthetic_pipeline(
        &synthetic_pipeline_builder, registry, &use_synthetic_pipeline,
        allocator);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_opt_append_reproducer_run_line(&builder, use_synthetic_pipeline);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&builder, "\n");
  }
  if (iree_status_is_ok(status) && use_synthetic_pipeline) {
    status = iree_string_builder_append_string(
        &builder, iree_string_builder_view(&synthetic_pipeline_builder));
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_cstring(&builder, "\n");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(&builder, source);
  }
  if (iree_status_is_ok(status) &&
      (source.size == 0 || source.data[source.size - 1] != '\n')) {
    status = iree_string_builder_append_cstring(&builder, "\n");
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        path, iree_string_builder_view(&builder), allocator);
  }
  iree_string_builder_deinitialize(&synthetic_pipeline_builder);
  iree_string_builder_deinitialize(&builder);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  if (fprintf(stderr, "pass pipeline reproducer written to %.*s\n",
              (int)path.size, path.data) < 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "failed to write pass reproducer path");
  }
  return fflush(stderr) == 0
             ? iree_ok_status()
             : iree_make_status(IREE_STATUS_DATA_LOSS,
                                "failed to flush pass reproducer path");
}

static iree_status_t loom_opt_run_passes(
    const loom_target_low_descriptor_registry_t* low_registry,
    iree_arena_block_pool_t* block_pool, loom_run_module_t* run_module,
    loom_pass_report_t* report, bool* out_execution_started,
    iree_allocator_t allocator) {
  loom_module_t* module = run_module->module;
  *out_execution_started = false;
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

  loom_source_resolver_t source_resolver =
      loom_run_module_source_resolver(run_module);
  loom_opt_diagnostic_emitter_t pass_emitter = {
      .module = module,
      .source_resolver = source_resolver,
      .emitter = LOOM_EMITTER_PASS,
  };

  loom_low_lower_policy_registry_t low_lower_policy_registry = {0};
  loom_all_low_lower_policy_registry_initialize(&low_lower_policy_registry);
  const loom_target_low_legality_provider_list_t low_legality_provider_list =
      loom_all_low_legality_provider_list();
  loom_low_pass_environment_storage_t low_pass_environment_storage;
  loom_pass_tool_run_options_t run_options = {
      .registry = loom_pass_builtin_registry(),
      .environment = loom_low_pass_environment_storage_initialize(
          &low_registry->registry, &low_lower_policy_registry,
          &low_legality_provider_list, &low_pass_environment_storage),
      .block_pool = block_pool,
      .diagnostic_emitter = {.fn = loom_opt_diagnostic_emitter_emit,
                             .user_data = &pass_emitter},
      .report = report,
  };

  if (has_pipeline_symbol) {
    *out_execution_started = true;
    return loom_pass_tool_run_pipeline_symbol(module, pipeline_symbol,
                                              &run_options);
  }

  iree_string_builder_t pipeline_builder;
  iree_string_builder_initialize(allocator, &pipeline_builder);
  iree_status_t status = loom_opt_join_pass_list(passes, &pipeline_builder);
  if (iree_status_is_ok(status)) {
    *out_execution_started = true;
    status = loom_pass_tool_run_flat_pipeline(
        module, iree_string_builder_view(&pipeline_builder), &run_options);
  }
  iree_string_builder_deinitialize(&pipeline_builder);
  return status;
}

static iree_status_t loom_opt_write_pass_report(
    loom_opt_pass_report_mode_t mode, const loom_pass_report_t* report) {
  switch (mode) {
    case LOOM_OPT_PASS_REPORT_NONE:
      return iree_ok_status();
    case LOOM_OPT_PASS_REPORT_JSON: {
      loom_output_stream_t stream;
      loom_output_stream_for_file(stderr, &stream);
      IREE_RETURN_IF_ERROR(loom_pass_report_format_json(report, &stream));
      return fflush(stderr) == 0
                 ? iree_ok_status()
                 : iree_make_status(IREE_STATUS_DATA_LOSS,
                                    "failed to flush pass report");
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported pass report mode");
  }
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
    status = loom_tooling_write_output_file(
        output_path, iree_string_builder_view(&builder), allocator);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_opt_print_pass_list(
    const loom_pass_registry_t* registry) {
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
      "registry.\n"
      "Use --pass-report=json to print a structured execution report to "
      "stderr.\n"
      "Use --pass-reproducer=file to capture a rerunnable failure file.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t allocator = iree_allocator_system();

  iree_io_file_contents_t* contents = NULL;
  loom_run_session_t run_session = {0};
  loom_run_module_t run_module = {0};
  const loom_pass_registry_t* pass_registry = loom_pass_builtin_registry();
  iree_string_view_t source = iree_string_view_empty();
  loom_opt_pass_report_mode_t pass_report_mode = LOOM_OPT_PASS_REPORT_NONE;
  loom_pass_report_t pass_report = {0};
  bool pass_report_initialized = false;
  bool pass_execution_started = false;
  iree_status_t pass_pipeline_status = iree_ok_status();

  iree_status_t status = loom_opt_parse_pass_report_mode(
      iree_make_cstring_view(FLAG_pass_report), &pass_report_mode);
  if (iree_status_is_ok(status) && FLAG_list_passes) {
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
    loom_run_session_options_t session_options = {0};
    loom_run_session_options_initialize(&session_options);
    session_options.host_allocator = allocator;
    session_options.register_context = (loom_run_register_context_callback_t){
        .fn = loom_opt_register_context,
    };
    session_options.initialize_low_descriptor_registry =
        (loom_run_initialize_low_descriptor_registry_callback_t){
            .fn = loom_opt_initialize_low_descriptor_registry,
        };
    status = loom_run_session_initialize(&session_options, &run_session);
  }
  if (iree_status_is_ok(status) && !metadata_only &&
      pass_report_mode != LOOM_OPT_PASS_REPORT_NONE) {
    loom_pass_report_initialize(allocator, &pass_report);
    pass_report_initialized = true;
  }

  iree_string_view_t input_path =
      argc < 2 ? iree_string_view_empty() : iree_make_cstring_view(argv[1]);
  iree_string_view_t filename =
      (argc < 2 ||
       iree_string_view_equal(input_path, iree_make_cstring_view("-")))
          ? iree_make_cstring_view("<stdin>")
          : input_path;

  if (iree_status_is_ok(status) && !metadata_only) {
    status = loom_tooling_read_input_file(input_path, allocator, &contents);
    if (iree_status_is_ok(status)) {
      source = loom_tooling_file_contents_string_view(contents);
    }
  }
  if (iree_status_is_ok(status) && !metadata_only) {
    loom_run_module_parse_options_t parse_options = {0};
    loom_run_module_parse_options_initialize(&parse_options);
    parse_options.filename = filename;
    parse_options.source = source;
    status = loom_run_module_parse(&run_session, &parse_options, &run_module);
  }
  if (iree_status_is_ok(status) && !metadata_only && FLAG_verify) {
    status = loom_opt_verify_module(
        loom_run_session_low_descriptor_registry(&run_session), &run_module);
  }
  if (iree_status_is_ok(status) && !metadata_only) {
    pass_pipeline_status = loom_opt_run_passes(
        loom_run_session_low_descriptor_registry(&run_session),
        loom_run_session_block_pool(&run_session), &run_module,
        pass_report_initialized ? &pass_report : NULL, &pass_execution_started,
        allocator);
    status = pass_pipeline_status;
  }
  if (!metadata_only && pass_execution_started &&
      !iree_status_is_ok(pass_pipeline_status)) {
    status = iree_status_join(
        status,
        loom_opt_write_pass_reproducer(
            iree_make_cstring_view(FLAG_pass_reproducer), filename, source,
            pass_registry, iree_status_code(pass_pipeline_status), allocator));
  }
  if (!metadata_only && pass_report_initialized && pass_execution_started) {
    iree_status_t report_status =
        loom_opt_write_pass_report(pass_report_mode, &pass_report);
    status = iree_status_join(status, report_status);
  }
  if (iree_status_is_ok(status) && !metadata_only && FLAG_verify) {
    status = loom_opt_verify_module(
        loom_run_session_low_descriptor_registry(&run_session), &run_module);
    if (!iree_status_is_ok(status) && pass_execution_started &&
        iree_status_is_ok(pass_pipeline_status)) {
      status = iree_status_join(
          status,
          loom_opt_write_pass_reproducer(
              iree_make_cstring_view(FLAG_pass_reproducer), filename, source,
              pass_registry, iree_status_code(status), allocator));
    }
  }
  if (iree_status_is_ok(status) && !metadata_only) {
    status = loom_opt_print_module(
        iree_make_cstring_view(FLAG_output),
        loom_run_session_low_descriptor_registry(&run_session),
        run_module.module, allocator);
  }

  bool had_error = !iree_status_is_ok(status);
  if (had_error) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
  }

  if (pass_report_initialized) {
    loom_pass_report_deinitialize(&pass_report);
  }
  loom_run_module_deinitialize(&run_module);
  iree_io_file_contents_free(contents);
  loom_run_session_deinitialize(&run_session);
  return had_error ? 1 : 0;
}
