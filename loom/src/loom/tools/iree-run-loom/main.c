// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-run-loom: compiles a Loom module and executes the selected export.

#include "loom/tools/iree-run-loom/main.h"

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/tooling/execution/compile_report_capture.h"
#include "loom/tooling/execution/execution_backend.h"
#include "loom/tooling/execution/one_shot.h"
#include "loom/tooling/execution/session.h"

IREE_FLAG(string, loom_entry, "",
          "Optional function symbol to compile, such as '@main'. When omitted "
          "the module must contain exactly one function entry compatible with "
          "the selected backend.");
IREE_FLAG(string, loom_backend, "vm",
          "Compilation backend to run, such as 'vm' or a linked native "
          "backend.");
IREE_FLAG(string, loom_module_name, "loom",
          "Module name to store in the compiled VM bytecode archive.");
IREE_FLAG(string, function, "",
          "VM export name to invoke. Empty selects the single export.");
IREE_FLAG_LIST(string, input,
               "Appends a VM function input in IREE function I/O syntax.");
IREE_FLAG_LIST(string, output,
               "Appends a VM function output handling spec in IREE function "
               "I/O syntax. Empty prints all outputs.");
IREE_FLAG_LIST(string, expected_output,
               "Appends an expected VM function output in IREE function I/O "
               "syntax. Expected outputs take precedence over --output.");
IREE_FLAG(int32_t, output_max_element_count, 1024,
          "Maximum number of VM output elements to format.");
IREE_FLAG(int32_t, entry_point, 0,
          "HAL executable entry point ordinal to dispatch.");
IREE_FLAG(string, workgroup_count, "1,1,1",
          "HAL dispatch workgroup count as `x,y,z`.");
IREE_FLAG(string, compile_report, "",
          "Optional compile report output. Use 'summary', 'details', or "
          "empty/'none'.");
IREE_FLAG(int32_t, compile_report_row_limit,
          LOOM_RUN_COMPILE_REPORT_DEFAULT_ROW_LIMIT,
          "Maximum pressure and spill rows to capture for "
          "--compile_report=details.");
IREE_FLAG(bool, probe_hal, false,
          "Runs the selected backend's target probe, prints the result, and "
          "exits. Not all backends support probing.");

enum {
  IREE_RUN_LOOM_HAL_MAX_OUTPUT_ELEMENT_COUNT = 1024,
};

typedef struct iree_run_loom_hal_flag_state_t {
  // Binding specs in HAL binding ordinal order.
  iree_string_view_t binding_specs[LOOM_RUN_ONE_SHOT_HAL_MAX_BINDING_COUNT];
  // Calling-convention character for each binding spec.
  char binding_cconv[LOOM_RUN_ONE_SHOT_HAL_MAX_BINDING_COUNT];
  // Number of populated entries in |binding_specs|.
  int32_t binding_count;
  // Expected binding specs in HAL binding ordinal order.
  iree_string_view_t
      expected_binding_specs[LOOM_RUN_ONE_SHOT_HAL_MAX_BINDING_COUNT];
  // Calling-convention character for each expected binding spec.
  char expected_binding_cconv[LOOM_RUN_ONE_SHOT_HAL_MAX_BINDING_COUNT];
  // Number of populated entries in |expected_binding_specs|.
  int32_t expected_binding_count;
} iree_run_loom_hal_flag_state_t;

static iree_run_loom_hal_flag_state_t iree_run_loom_hal_flags = {0};

static iree_status_t iree_run_loom_parse_binding_flag(
    iree_string_view_t flag_name, void* storage, iree_string_view_t value) {
  (void)flag_name;
  (void)storage;
  if (iree_run_loom_hal_flags.binding_count >=
      LOOM_RUN_ONE_SHOT_HAL_MAX_BINDING_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many HAL bindings; maximum is %d",
                            LOOM_RUN_ONE_SHOT_HAL_MAX_BINDING_COUNT);
  }
  const int32_t index = iree_run_loom_hal_flags.binding_count++;
  iree_run_loom_hal_flags.binding_specs[index] = value;
  iree_run_loom_hal_flags.binding_cconv[index] = 'r';
  return iree_ok_status();
}

static void iree_run_loom_print_binding_flag(iree_string_view_t flag_name,
                                             void* storage, FILE* file) {
  (void)storage;
  if (iree_run_loom_hal_flags.binding_count == 0) {
    fprintf(file, "# --%.*s=\"shapextype[=values]\"\n", (int)flag_name.size,
            flag_name.data);
    return;
  }
  for (int32_t i = 0; i < iree_run_loom_hal_flags.binding_count; ++i) {
    iree_string_view_t binding_spec = iree_run_loom_hal_flags.binding_specs[i];
    fprintf(file, "--%.*s=\"%.*s\"\n", (int)flag_name.size, flag_name.data,
            (int)binding_spec.size, binding_spec.data);
  }
}
IREE_FLAG_CALLBACK(iree_run_loom_parse_binding_flag,
                   iree_run_loom_print_binding_flag, NULL, binding,
                   "Appends a HAL dispatch binding. Bindings use the same "
                   "shape/type/data syntax as iree-benchmark-executable.");

static iree_status_t iree_run_loom_parse_expected_binding_flag(
    iree_string_view_t flag_name, void* storage, iree_string_view_t value) {
  (void)flag_name;
  (void)storage;
  if (iree_run_loom_hal_flags.expected_binding_count >=
      LOOM_RUN_ONE_SHOT_HAL_MAX_BINDING_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many expected HAL bindings; maximum is %d",
                            LOOM_RUN_ONE_SHOT_HAL_MAX_BINDING_COUNT);
  }
  const int32_t index = iree_run_loom_hal_flags.expected_binding_count++;
  iree_run_loom_hal_flags.expected_binding_specs[index] = value;
  iree_run_loom_hal_flags.expected_binding_cconv[index] = 'r';
  return iree_ok_status();
}

static void iree_run_loom_print_expected_binding_flag(
    iree_string_view_t flag_name, void* storage, FILE* file) {
  (void)storage;
  if (iree_run_loom_hal_flags.expected_binding_count == 0) {
    fprintf(file, "# --%.*s=\"shapextype=values\"\n", (int)flag_name.size,
            flag_name.data);
    return;
  }
  for (int32_t i = 0; i < iree_run_loom_hal_flags.expected_binding_count; ++i) {
    iree_string_view_t binding_spec =
        iree_run_loom_hal_flags.expected_binding_specs[i];
    fprintf(file, "--%.*s=\"%.*s\"\n", (int)flag_name.size, flag_name.data,
            (int)binding_spec.size, binding_spec.data);
  }
}
IREE_FLAG_CALLBACK(iree_run_loom_parse_expected_binding_flag,
                   iree_run_loom_print_expected_binding_flag, NULL,
                   expected_binding,
                   "Appends an expected HAL binding after dispatch. When "
                   "present, one expected binding must be provided for every "
                   "binding.");

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

static iree_status_t iree_run_loom_register_context(void* user_data,
                                                    loom_context_t* context) {
  (void)user_data;
  return loom_op_registry_register_all_dialects(context);
}

static iree_status_t iree_run_loom_parse_workgroup_count(
    iree_string_view_t value, uint32_t* out_workgroup_count) {
  IREE_ASSERT_ARGUMENT(out_workgroup_count);
  iree_string_view_t remaining = value;
  iree_string_view_t x;
  iree_string_view_split(remaining, ',', &x, &remaining);
  iree_string_view_t y;
  iree_string_view_split(remaining, ',', &y, &remaining);
  iree_string_view_t z = remaining;
  if (!iree_string_view_atoi_uint32(x, &out_workgroup_count[0]) ||
      !iree_string_view_atoi_uint32(y, &out_workgroup_count[1]) ||
      !iree_string_view_atoi_uint32(z, &out_workgroup_count[2])) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid --workgroup_count='%.*s'; expected `x,y,z`", (int)value.size,
        value.data);
  }
  return iree_ok_status();
}

static iree_status_t iree_run_loom_write_stdout(iree_string_view_t output) {
  if (iree_string_view_is_empty(output)) {
    return iree_ok_status();
  }
  if (fwrite(output.data, 1, output.size, stdout) != output.size ||
      fflush(stdout) != 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "failed to write invocation output");
  }
  return iree_ok_status();
}

static iree_status_t iree_run_loom_one_shot_options_initialize(
    const loom_run_execution_backend_t* backend,
    loom_run_one_shot_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(backend);
  IREE_ASSERT_ARGUMENT(out_options);
  loom_run_one_shot_options_initialize(out_options);

  if (iree_any_bit_set(backend->flags,
                       LOOM_RUN_EXECUTION_BACKEND_FLAG_VM_OPTIONS)) {
    if (FLAG_output_max_element_count < 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--output_max_element_count must be non-negative; got %d",
          (int)FLAG_output_max_element_count);
    }
    out_options->vm_function_name = iree_make_cstring_view(FLAG_function);
    out_options->vm_inputs = (loom_run_one_shot_value_specs_t){
        .values = FLAG_input_list().values,
        .count = FLAG_input_list().count,
    };
    out_options->vm_outputs = (loom_run_one_shot_value_specs_t){
        .values = FLAG_output_list().values,
        .count = FLAG_output_list().count,
    };
    out_options->vm_expected_outputs = (loom_run_one_shot_value_specs_t){
        .values = FLAG_expected_output_list().values,
        .count = FLAG_expected_output_list().count,
    };
    out_options->vm_max_output_element_count =
        (iree_host_size_t)FLAG_output_max_element_count;
  }

  if (iree_any_bit_set(backend->flags,
                       LOOM_RUN_EXECUTION_BACKEND_FLAG_HAL_OPTIONS)) {
    if (FLAG_entry_point < 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--entry_point must be non-negative; got %d",
                              (int)FLAG_entry_point);
    }
    out_options->hal_entry_point = (uint32_t)FLAG_entry_point;
    IREE_RETURN_IF_ERROR(iree_run_loom_parse_workgroup_count(
        iree_make_cstring_view(FLAG_workgroup_count),
        out_options->hal_workgroup_count));
    out_options->hal_bindings = (loom_run_one_shot_binding_specs_t){
        .values = iree_run_loom_hal_flags.binding_specs,
        .conventions = iree_run_loom_hal_flags.binding_cconv,
        .count = iree_run_loom_hal_flags.binding_count,
    };
    out_options->hal_expected_bindings = (loom_run_one_shot_binding_specs_t){
        .values = iree_run_loom_hal_flags.expected_binding_specs,
        .conventions = iree_run_loom_hal_flags.expected_binding_cconv,
        .count = iree_run_loom_hal_flags.expected_binding_count,
    };
    out_options->hal_max_output_element_count =
        IREE_RUN_LOOM_HAL_MAX_OUTPUT_ELEMENT_COUNT;
  }
  return iree_ok_status();
}

static iree_status_t iree_run_loom_compile_report_options_initialize(
    loom_run_compile_report_capture_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  loom_run_compile_report_capture_options_initialize(out_options);
  IREE_RETURN_IF_ERROR(loom_run_compile_report_capture_options_parse_mode(
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

static iree_status_t iree_run_loom_make_unknown_backend_status(
    iree_string_view_t backend_name,
    const loom_run_execution_backend_registry_t* backend_registry,
    iree_allocator_t allocator) {
  iree_string_builder_t backend_names;
  iree_string_builder_initialize(allocator, &backend_names);
  iree_status_t status = loom_run_execution_backend_registry_format_names(
      backend_registry, &backend_names);
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

int iree_run_loom_main(int argc, char** argv,
                       const iree_run_loom_configuration_t* configuration) {
  IREE_ASSERT_ARGUMENT(configuration);
  IREE_ASSERT_ARGUMENT(configuration->tool_name);
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_flags_set_usage(
      configuration->tool_name,
      "Compiles a Loom module to a runtime artifact and executes the selected "
      "export.\n"
      "\n"
      "Usage:\n"
      "  iree-run-loom [file.loom] --function=name --input=... "
      "--expected_output=...\n"
      "  cat module.loom | iree-run-loom - --function=name --input=...\n"
      "\n"
      "The 'vm' backend compiles target.preset key \"iree-vm\" into a real "
      "IREE VM bytecode archive and runs it with IREE function I/O syntax for "
      "--input, --output, and --expected_output. Native execution backends "
      "compile target-low kernels into runtime artifacts and dispatch them "
      "through their production runtime path.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t allocator = iree_allocator_system();
  const loom_run_execution_backend_registry_t* backend_registry =
      &configuration->execution_backend_registry;
  const iree_string_view_t backend_name =
      iree_make_cstring_view(FLAG_loom_backend);
  const loom_run_execution_backend_t* backend =
      loom_run_execution_backend_registry_lookup(backend_registry,
                                                 backend_name);
  if (FLAG_probe_hal) {
    loom_run_one_shot_result_t probe_result = {0};
    loom_run_one_shot_result_initialize(allocator, &probe_result);
    iree_status_t probe_status = iree_ok_status();
    if (backend == NULL) {
      probe_status = iree_run_loom_make_unknown_backend_status(
          backend_name, backend_registry, allocator);
    } else if (backend->probe == NULL) {
      probe_status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--probe_hal requires --loom_backend to name a probeable backend");
    } else {
      const loom_run_one_shot_probe_request_t probe_request = {
          .host_allocator = allocator,
          .result = &probe_result,
      };
      probe_status = backend->probe(backend, &probe_request);
    }
    if (iree_status_is_ok(probe_status)) {
      probe_status = iree_run_loom_write_stdout(
          iree_string_builder_view(&probe_result.output));
    }
    int probe_exit_code = 0;
    if (!iree_status_is_ok(probe_status)) {
      iree_status_fprint(stderr, probe_status);
      iree_status_free(probe_status);
      probe_exit_code = 1;
    }
    loom_run_one_shot_result_deinitialize(&probe_result);
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(probe_exit_code);
    return probe_exit_code;
  }

  iree_io_file_contents_t* contents = NULL;
  loom_run_session_t session = {0};
  loom_run_module_t run_module = {0};
  loom_run_compile_report_capture_t compile_report_capture = {0};
  loom_run_one_shot_result_t run_result = {0};
  loom_run_one_shot_result_initialize(allocator, &run_result);
  int exit_code = 0;

  iree_status_t status = iree_ok_status();
  if (argc > 2) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "iree-run-loom accepts at most one input file or '-' for stdin; got %d "
        "inputs",
        argc - 1);
  }
  if (iree_status_is_ok(status) && backend == NULL) {
    status = iree_run_loom_make_unknown_backend_status(
        backend_name, backend_registry, allocator);
  }

  if (iree_status_is_ok(status)) {
    loom_run_session_options_t session_options = {0};
    loom_run_session_options_initialize(&session_options);
    session_options.host_allocator = allocator;
    session_options.register_context = (loom_run_register_context_callback_t){
        .fn = iree_run_loom_register_context,
    };
    session_options.initialize_low_descriptor_registry =
        configuration->initialize_low_descriptor_registry;
    status = loom_run_session_initialize(&session_options, &session);
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
  loom_run_candidate_compile_options_t compile_options = {0};
  loom_run_candidate_compile_options_initialize(&compile_options);
  compile_options.module_name = iree_make_cstring_view(FLAG_loom_module_name);
  compile_options.entry_symbol = iree_make_cstring_view(FLAG_loom_entry);
  loom_run_compile_report_capture_options_t compile_report_options = {0};
  if (iree_status_is_ok(status)) {
    status = iree_run_loom_compile_report_options_initialize(
        &compile_report_options);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_compile_report_capture_initialize(
        &compile_report_options, allocator, &compile_report_capture);
  }
  if (iree_status_is_ok(status)) {
    loom_run_compile_report_capture_configure_compile_options(
        &compile_report_capture, &compile_options);
  }
  loom_run_one_shot_options_t one_shot_options = {0};
  if (iree_status_is_ok(status)) {
    status =
        iree_run_loom_one_shot_options_initialize(backend, &one_shot_options);
  }
  if (iree_status_is_ok(status)) {
    loom_run_module_parse_options_t parse_options = {0};
    loom_run_module_parse_options_initialize(&parse_options);
    parse_options.filename = filename;
    parse_options.source = source;
    status = loom_run_module_parse(&session, &parse_options, &run_module);
  }
  if (iree_status_is_ok(status)) {
    compile_options.source_resolver =
        loom_run_module_source_resolver(&run_module);
  }
  if (iree_status_is_ok(status)) {
    const loom_run_one_shot_request_t run_request = {
        .run_module = &run_module,
        .compile_options = &compile_options,
        .options = &one_shot_options,
        .compile_report_capture = &compile_report_capture,
        .host_allocator = allocator,
        .result = &run_result,
    };
    status = backend->run_one_shot(backend, &run_request);
  }
  if (iree_status_is_ok(status)) {
    status = iree_run_loom_write_stdout(
        iree_string_builder_view(&run_result.output));
  }
  if (iree_status_is_ok(status)) {
    exit_code = run_result.exit_code;
  }

  const bool had_error = !iree_status_is_ok(status);
  if (had_error) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = 1;
  }

  loom_run_compile_report_capture_deinitialize(&compile_report_capture);
  loom_run_one_shot_result_deinitialize(&run_result);
  loom_run_module_deinitialize(&run_module);
  iree_io_file_contents_free(contents);
  loom_run_session_deinitialize(&session);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
