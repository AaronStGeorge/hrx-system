// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-compile: compiles a Loom module to a runtime artifact.

#include <errno.h>
#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "loom/error/diagnostic.h"
#include "loom/ops/op_registry.h"
#include "loom/tooling/compile/pipeline.h"
#include "loom/tooling/config/config.h"
#include "loom/tooling/execution/compile_report_capture.h"
#include "loom/tooling/execution/execution_provider.h"
#include "loom/tooling/execution/hal/artifact.h"
#include "loom/tooling/execution/hal/candidate.h"
#include "loom/tooling/execution/session.h"
#include "loom/tooling/io/file.h"

#ifndef LOOM_COMPILE_HAVE_AMDGPU
#define LOOM_COMPILE_HAVE_AMDGPU 0
#endif  // LOOM_COMPILE_HAVE_AMDGPU
#ifndef LOOM_COMPILE_HAVE_IREEVM
#define LOOM_COMPILE_HAVE_IREEVM 0
#endif  // LOOM_COMPILE_HAVE_IREEVM

#define LOOM_COMPILE_HAVE_ANY_PROVIDER \
  (LOOM_COMPILE_HAVE_AMDGPU || LOOM_COMPILE_HAVE_IREEVM)

#if LOOM_COMPILE_HAVE_AMDGPU
#include "loom/target/arch/amdgpu/provider.h"
#endif  // LOOM_COMPILE_HAVE_AMDGPU
#if LOOM_COMPILE_HAVE_IREEVM
#include "loom/tooling/execution/ireevm/candidate.h"
#include "loom/tooling/execution/ireevm/provider.h"
#endif  // LOOM_COMPILE_HAVE_IREEVM

IREE_FLAG(string, loom_entry, "",
          "Optional function symbol to compile, such as '@main'. When omitted "
          "the module must contain exactly one function entry compatible with "
          "the selected backend.");
IREE_FLAG(string, loom_backend, "vm",
          "Compilation backend to emit, such as 'vm' or a linked native "
          "backend.");
IREE_FLAG(string, loom_module_name, "loom",
          "Module name to store in VM bytecode archives.");
IREE_FLAG(string, pipeline, "default",
          "Pass pipeline to run before artifact emission. Use 'default' or "
          "empty for the shared prepared-low compile pipeline, 'none' to "
          "disable pass execution, '@symbol' to run a module-local "
          "pass.pipeline, or a comma-separated pass list such as "
          "'canonicalize,cse'.");
IREE_FLAG_LIST(
    string, config,
    "Compile-time config binding. Repeat as --config=key=value. Bindings not "
    "referenced by the loaded module are ignored.");
IREE_FLAG_LIST(string, config_file,
               "JSON/JSONC config object file. Repeat for multiple files. "
               "Nested object keys are flattened with '.' separators.");
IREE_FLAG(string, output, "-",
          "Output path for the primary runtime artifact. For VM this is the VM "
          "bytecode archive; for HAL this is the HAL executable package.");
IREE_FLAG(string, emit_target_artifact, "",
          "Optional output path for a target-native artifact produced beside "
          "the primary runtime artifact, such as AMDGPU HSACO.");
IREE_FLAG(string, compile_report, "",
          "Optional compile report output. Use 'summary'/'details' for "
          "structured JSON, 'text-summary'/'text-details' for human-readable "
          "text, or empty/'none'.");
IREE_FLAG(string, compile_report_output, "stderr",
          "Output path for --compile_report. Use 'stderr', '-', or a file "
          "path.");
IREE_FLAG(int32_t, compile_report_row_limit,
          LOOM_RUN_COMPILE_REPORT_DEFAULT_ROW_LIMIT,
          "Maximum rows per report row category to capture for "
          "--compile_report=details.");

#if LOOM_COMPILE_HAVE_AMDGPU
static const loom_run_execution_provider_t kLoomCompileAmdgpuProvider = {
    .name = IREE_SVL("amdgpu"),
    .target_provider = &loom_amdgpu_target_provider,
};
#endif  // LOOM_COMPILE_HAVE_AMDGPU

#if LOOM_COMPILE_HAVE_ANY_PROVIDER
static const loom_run_execution_provider_t* const kLoomCompileProviders[] = {
#if LOOM_COMPILE_HAVE_AMDGPU
    &kLoomCompileAmdgpuProvider,
#endif  // LOOM_COMPILE_HAVE_AMDGPU
#if LOOM_COMPILE_HAVE_IREEVM
    &loom_ireevm_execution_provider,
#endif  // LOOM_COMPILE_HAVE_IREEVM
};
#endif  // LOOM_COMPILE_HAVE_ANY_PROVIDER

static const loom_run_execution_provider_set_t kLoomCompileProviderSet = {
#if LOOM_COMPILE_HAVE_ANY_PROVIDER
    .providers = kLoomCompileProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomCompileProviders),
#else
    .providers = NULL,
    .provider_count = 0,
#endif  // LOOM_COMPILE_HAVE_ANY_PROVIDER
};

static const loom_run_hal_artifact_provider_registry_t
    kLoomCompileHalArtifactProviderRegistry = {
        .providers = NULL,
        .provider_count = 0,
};

static iree_status_t loom_compile_register_context(void* user_data,
                                                   loom_context_t* context) {
  loom_run_execution_environment_t* environment =
      (loom_run_execution_environment_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_op_registry_register_all_dialects(context));
  const loom_run_register_context_callback_t register_context =
      loom_run_execution_environment_register_context_callback(environment);
  return register_context.fn(register_context.user_data, context);
}

static iree_status_t loom_compile_initialize_session(
    loom_run_execution_environment_t* environment, iree_allocator_t allocator,
    loom_run_session_t* out_session) {
  loom_run_session_options_t session_options = {0};
  loom_run_session_options_initialize(&session_options);
  session_options.host_allocator = allocator;
  session_options.register_context = (loom_run_register_context_callback_t){
      .fn = loom_compile_register_context,
      .user_data = environment,
  };
  session_options.initialize_low_descriptor_registry =
      loom_run_execution_environment_low_descriptor_registry_callback(
          environment);
  return loom_run_session_initialize(&session_options, out_session);
}

static iree_status_t loom_compile_parse_input_module(
    int argc, char** argv, loom_run_session_t* session,
    iree_allocator_t allocator, iree_io_file_contents_t** out_contents,
    loom_run_module_t* out_run_module) {
  if (argc > 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "loom-compile accepts at most one input file or '-' for stdin; got %d "
        "inputs",
        argc - 1);
  }

  const iree_string_view_t input_path =
      argc < 2 ? iree_string_view_empty() : iree_make_cstring_view(argv[1]);
  const iree_string_view_t filename =
      (argc < 2 || iree_string_view_equal(input_path, IREE_SV("-")))
          ? IREE_SV("<stdin>")
          : input_path;
  IREE_RETURN_IF_ERROR(
      loom_tooling_read_input_file(input_path, allocator, out_contents));

  loom_run_module_parse_options_t parse_options = {0};
  loom_run_module_parse_options_initialize(&parse_options);
  parse_options.filename = filename;
  parse_options.source = loom_tooling_file_contents_string_view(*out_contents);
  return loom_run_module_parse(session, &parse_options, out_run_module);
}

static iree_status_t loom_compile_append_config_flags(
    loom_tooling_config_set_t* config_set) {
  iree_flag_string_list_t assignments = FLAG_config_list();
  for (iree_host_size_t i = 0; i < assignments.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_set_append_assignment(
        config_set, assignments.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_append_config_files(
    loom_tooling_config_set_t* config_set, iree_allocator_t allocator) {
  iree_flag_string_list_t paths = FLAG_config_file_list();
  for (iree_host_size_t i = 0; i < paths.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_tooling_config_set_append_json_file(
        config_set, paths.values[i], allocator));
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_materialize_config_set(
    loom_run_session_t* session, loom_run_module_t* run_module,
    const loom_tooling_config_set_t* config_set) {
  loom_tooling_config_materialize_options_t options;
  loom_tooling_config_materialize_options_initialize(&options);
  options.config_set = config_set;
  return loom_tooling_config_materialize_module(
      run_module->module, &options, loom_run_session_block_pool(session), NULL);
}

static iree_status_t loom_compile_report_options_initialize(
    loom_run_compile_report_capture_options_t* out_options) {
  loom_run_compile_report_capture_options_initialize(out_options);
  IREE_RETURN_IF_ERROR(loom_run_compile_report_capture_options_parse_request(
      iree_make_cstring_view(FLAG_compile_report), out_options));
  if (FLAG_compile_report_row_limit < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--compile_report_row_limit must be non-negative; got %d",
        (int)FLAG_compile_report_row_limit);
  }
  out_options->row_limit = (iree_host_size_t)FLAG_compile_report_row_limit;
  return iree_ok_status();
}

static iree_status_t loom_compile_run_pass_pipeline(
    const loom_run_execution_environment_t* environment,
    loom_run_session_t* session, loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* compile_options,
    loom_pass_run_result_t* out_run_result) {
  loom_compile_pipeline_options_t pipeline_options = {0};
  loom_compile_pipeline_options_initialize(&pipeline_options);
  pipeline_options.pipeline = iree_make_cstring_view(FLAG_pipeline);
  pipeline_options.entry_symbol = iree_make_cstring_view(FLAG_loom_entry);
  pipeline_options.target_environment =
      loom_run_execution_environment_target_environment(environment);
  pipeline_options.low_descriptor_registry =
      loom_run_session_low_descriptor_registry(session);
  pipeline_options.diagnostic_sink = (loom_diagnostic_sink_t){
      .fn = loom_diagnostic_stderr_sink,
  };
  pipeline_options.source_resolver =
      loom_run_module_source_resolver(run_module);
  pipeline_options.report = compile_options->report;
  pipeline_options.report_row_storage = compile_options->report_row_storage;

  return loom_compile_run_pipeline(run_module->module, &pipeline_options,
                                   loom_run_session_block_pool(session),
                                   out_run_result);
}

static iree_status_t loom_compile_write_bytes(iree_string_view_t path,
                                              iree_const_byte_span_t contents,
                                              iree_allocator_t allocator) {
  return loom_tooling_write_output_file(
      path,
      iree_make_string_view((const char*)contents.data, contents.data_length),
      allocator);
}

static iree_status_t loom_compile_write_optional_target_artifact(
    const loom_run_hal_artifact_t* artifact, iree_allocator_t allocator) {
  const iree_string_view_t path =
      iree_make_cstring_view(FLAG_emit_target_artifact);
  if (iree_string_view_is_empty(path)) {
    return iree_ok_status();
  }
  if (artifact->target_artifact_data.data == NULL ||
      artifact->target_artifact_data.data_length == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "selected HAL artifact provider produced no "
                            "target-native artifact");
  }
  return loom_compile_write_bytes(path, artifact->target_artifact_data,
                                  allocator);
}

static iree_status_t loom_compile_emit_hal(
    const loom_run_hal_artifact_provider_t* artifact_provider,
    loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* compile_options,
    iree_allocator_t allocator, bool* out_emitted) {
  *out_emitted = false;
  const iree_string_view_t output_path = iree_make_cstring_view(FLAG_output);
  const iree_string_view_t target_artifact_path =
      iree_make_cstring_view(FLAG_emit_target_artifact);
  if (!iree_string_view_is_empty(target_artifact_path) &&
      loom_tooling_file_path_is_stdio(output_path) &&
      loom_tooling_file_path_is_stdio(target_artifact_path)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--output and --emit_target_artifact cannot both write to stdout");
  }

  loom_run_hal_candidate_t candidate = {0};
  iree_status_t status = loom_run_hal_candidate_emit_module_target(
      artifact_provider, run_module, compile_options, allocator, &candidate);
  if (iree_status_is_ok(status) && candidate.compiled) {
    status = loom_compile_write_bytes(
        output_path, candidate.artifact.executable_data, allocator);
  }
  if (iree_status_is_ok(status) && candidate.compiled) {
    status = loom_compile_write_optional_target_artifact(&candidate.artifact,
                                                         allocator);
  }
  if (iree_status_is_ok(status)) {
    *out_emitted = candidate.compiled;
  }
  loom_run_hal_candidate_deinitialize(&candidate);
  return status;
}

static iree_status_t loom_compile_emit_vm(
    loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* compile_options,
    iree_allocator_t allocator, bool* out_emitted) {
  *out_emitted = false;
  if (!iree_string_view_is_empty(
          iree_make_cstring_view(FLAG_emit_target_artifact))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--emit_target_artifact is only valid for HAL artifact providers");
  }
#if LOOM_COMPILE_HAVE_IREEVM
  loom_ireevm_run_candidate_t candidate = {0};
  iree_status_t status = loom_ireevm_run_candidate_emit(
      run_module, compile_options, allocator, &candidate);
  if (iree_status_is_ok(status) && candidate.emitted) {
    status = loom_compile_write_bytes(
        iree_make_cstring_view(FLAG_output),
        iree_make_const_byte_span(candidate.archive.data,
                                  candidate.archive.data_length),
        allocator);
  }
  if (iree_status_is_ok(status)) {
    *out_emitted = candidate.emitted;
  }
  loom_ireevm_run_candidate_deinitialize(&candidate);
  return status;
#else
  (void)run_module;
  (void)compile_options;
  (void)allocator;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "loom-compile was built without VM emission");
#endif  // LOOM_COMPILE_HAVE_IREEVM
}

static iree_status_t loom_compile_make_unknown_backend_status(
    iree_string_view_t backend_name,
    const loom_run_hal_artifact_provider_registry_t* artifact_provider_registry,
    iree_allocator_t allocator) {
  iree_string_builder_t backend_names;
  iree_string_builder_initialize(allocator, &backend_names);
  iree_status_t status =
      iree_string_builder_append_cstring(&backend_names, "vm");
  if (iree_status_is_ok(status) &&
      artifact_provider_registry->provider_count != 0) {
    status = iree_string_builder_append_cstring(&backend_names, ", ");
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_artifact_provider_registry_format_names(
        artifact_provider_registry, &backend_names);
  }
  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&backend_names);
    return status;
  }
  status = iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "unknown --loom_backend='%.*s'; expected registered backend in [%.*s]",
      (int)backend_name.size, backend_name.data,
      (int)iree_string_builder_size(&backend_names),
      iree_string_builder_buffer(&backend_names));
  iree_string_builder_deinitialize(&backend_names);
  return status;
}

static iree_status_t loom_compile_write_report(
    const loom_run_compile_report_capture_t* compile_report_capture,
    iree_allocator_t allocator) {
  if (!loom_run_compile_report_capture_is_enabled(compile_report_capture)) {
    return iree_ok_status();
  }
  const iree_string_view_t path =
      iree_make_cstring_view(FLAG_compile_report_output);
  if (iree_string_view_is_empty(path) ||
      iree_string_view_equal(path, IREE_SV("stderr"))) {
    loom_output_stream_t stream;
    loom_output_stream_for_file(stderr, &stream);
    IREE_RETURN_IF_ERROR(loom_run_compile_report_capture_write_output(
        compile_report_capture, &stream, allocator));
    return fflush(stderr) == 0
               ? iree_ok_status()
               : iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "failed to flush compile report stderr");
  }
  if (loom_tooling_file_path_is_stdio(path)) {
    const iree_string_view_t target_artifact_path =
        iree_make_cstring_view(FLAG_emit_target_artifact);
    if (loom_tooling_file_path_is_stdio(iree_make_cstring_view(FLAG_output)) ||
        (!iree_string_view_is_empty(target_artifact_path) &&
         loom_tooling_file_path_is_stdio(target_artifact_path))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--compile_report_output=- cannot share stdout with --output=- or "
          "--emit_target_artifact=-");
    }
    loom_output_stream_t stream;
    loom_output_stream_for_file(stdout, &stream);
    IREE_RETURN_IF_ERROR(loom_run_compile_report_capture_write_output(
        compile_report_capture, &stream, allocator));
    return fflush(stdout) == 0
               ? iree_ok_status()
               : iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "failed to flush compile report stdout");
  }

  FILE* file = fopen(FLAG_compile_report_output, "wb");
  if (file == NULL) {
    const int open_error = errno;
    return iree_make_status(iree_status_code_from_errno(open_error),
                            "failed to open compile report output '%.*s' (%d)",
                            (int)path.size, path.data, open_error);
  }
  loom_output_stream_t stream;
  loom_output_stream_for_file(file, &stream);
  iree_status_t status = loom_run_compile_report_capture_write_output(
      compile_report_capture, &stream, allocator);
  if (fclose(file) != 0 && iree_status_is_ok(status)) {
    const int close_error = errno;
    status = iree_make_status(iree_status_code_from_errno(close_error),
                              "failed to close compile report output '%.*s' "
                              "(%d)",
                              (int)path.size, path.data, close_error);
  }
  return status;
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_flags_set_usage(
      "loom-compile",
      "Compiles a Loom module to a runtime artifact.\n"
      "\n"
      "Usage:\n"
      "  loom-compile [file.loom] --loom_backend=vm --output=module.vmfb\n"
      "\n"
      "Repeat --config=key=value to materialize compile-time config symbols "
      "before the pass pipeline. Use --config-file=path for a JSON/JSONC "
      "object such as {\"model36\":{\"model\":{\"hidden_size\":4096}}}. "
      "Files and direct bindings share one config set and duplicate keys are "
      "rejected.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t allocator = iree_allocator_system();
  loom_run_execution_environment_t environment = {0};
  iree_status_t status = loom_run_execution_environment_initialize(
      &kLoomCompileProviderSet, &environment);
  loom_tooling_config_set_t config_set;
  loom_tooling_config_set_initialize(allocator, &config_set);

  iree_io_file_contents_t* contents = NULL;
  loom_run_session_t session = {0};
  loom_run_module_t run_module = {0};
  loom_run_compile_report_capture_t compile_report_capture = {0};
  bool emitted = false;
  int exit_code = 0;

  if (iree_status_is_ok(status)) {
    status = loom_compile_initialize_session(&environment, allocator, &session);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_append_config_files(&config_set, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_append_config_flags(&config_set);
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_parse_input_module(argc, argv, &session, allocator,
                                             &contents, &run_module);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_compile_materialize_config_set(&session, &run_module, &config_set);
  }

  loom_run_compile_report_capture_options_t compile_report_options = {0};
  if (iree_status_is_ok(status)) {
    status = loom_compile_report_options_initialize(&compile_report_options);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_compile_report_capture_initialize(
        &compile_report_options, allocator, &compile_report_capture);
  }

  loom_run_candidate_compile_options_t compile_options = {0};
  loom_run_candidate_compile_options_initialize(&compile_options);
  compile_options.module_name = iree_make_cstring_view(FLAG_loom_module_name);
  compile_options.entry_symbol = iree_make_cstring_view(FLAG_loom_entry);
  if (iree_status_is_ok(status)) {
    compile_options.source_resolver =
        loom_run_module_source_resolver(&run_module);
    loom_run_compile_report_capture_configure_compile_options(
        &compile_report_capture, &compile_options);
  }
  if (iree_status_is_ok(status)) {
    loom_pass_run_result_t pass_run_result = {0};
    status = loom_compile_run_pass_pipeline(&environment, &session, &run_module,
                                            &compile_options, &pass_run_result);
    if (iree_status_is_ok(status) && pass_run_result.error_count != 0) {
      exit_code = 1;
    }
  }
  if (iree_status_is_ok(status) && exit_code == 0) {
    status =
        loom_tooling_config_require_resolved_module(run_module.module, NULL);
  }

  const iree_string_view_t backend_name =
      iree_make_cstring_view(FLAG_loom_backend);
  if (iree_status_is_ok(status) && exit_code == 0) {
    const loom_run_hal_artifact_provider_t* artifact_provider =
        loom_run_hal_artifact_provider_registry_lookup(
            &kLoomCompileHalArtifactProviderRegistry, backend_name);
    if (artifact_provider != NULL) {
      status = loom_compile_emit_hal(artifact_provider, &run_module,
                                     &compile_options, allocator, &emitted);
    } else if (iree_string_view_equal(backend_name, IREE_SV("vm"))) {
      status = loom_compile_emit_vm(&run_module, &compile_options, allocator,
                                    &emitted);
    } else {
      status = loom_compile_make_unknown_backend_status(
          backend_name, &kLoomCompileHalArtifactProviderRegistry, allocator);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_compile_write_report(&compile_report_capture, allocator);
  }
  if (iree_status_is_ok(status) && exit_code == 0 && !emitted) {
    exit_code = 1;
  }

  const bool had_error = !iree_status_is_ok(status);
  if (had_error) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = 1;
  }

  loom_run_compile_report_capture_deinitialize(&compile_report_capture);
  loom_run_module_deinitialize(&run_module);
  iree_io_file_contents_free(contents);
  loom_tooling_config_set_deinitialize(&config_set);
  loom_run_session_deinitialize(&session);
  loom_run_execution_environment_deinitialize(&environment);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
