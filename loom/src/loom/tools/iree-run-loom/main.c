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
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/io/stdio_stream.h"
#include "iree/tooling/comparison.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/function_io.h"
#include "iree/tooling/function_util.h"
#include "iree/tooling/run_module.h"
#include "iree/vm/api.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/target/emit/ireevm/module_compiler.h"
#include "loom/tooling/execution/hal_invocation.h"
#include "loom/tooling/execution/hal_runtime.h"
#include "loom/tooling/execution/session.h"

IREE_FLAG(string, loom_target, "",
          "Optional target.bundle symbol to compile, such as '@vm_target'. "
          "When omitted the module must contain exactly one target compatible "
          "with the selected backend.");
IREE_FLAG(string, loom_backend, "vm",
          "Compilation backend to run: 'vm' or a registered HAL backend.");
IREE_FLAG(string, loom_module_name, "loom",
          "Module name to store in the compiled VM bytecode archive.");
IREE_FLAG(int32_t, entry_point, 0,
          "HAL executable entry point ordinal to dispatch.");
IREE_FLAG(string, workgroup_count, "1,1,1",
          "HAL dispatch workgroup count as `x,y,z`.");
IREE_FLAG(bool, probe_hal, false,
          "Creates the selected HAL backend device, probes for a "
          "Loom-supported native target, prints the selected target, and "
          "exits.");

enum {
  IREE_RUN_LOOM_HAL_MAX_OUTPUT_ELEMENT_COUNT = 1024,
};

typedef struct iree_run_loom_hal_flag_state_t {
  // Binding specs in HAL binding ordinal order.
  iree_string_view_t binding_specs[LOOM_RUN_HAL_MAX_BINDING_COUNT];
  // Calling-convention character for each binding spec.
  char binding_cconv[LOOM_RUN_HAL_MAX_BINDING_COUNT];
  // Number of populated entries in |binding_specs|.
  int32_t binding_count;
  // Expected binding specs in HAL binding ordinal order.
  iree_string_view_t expected_binding_specs[LOOM_RUN_HAL_MAX_BINDING_COUNT];
  // Calling-convention character for each expected binding spec.
  char expected_binding_cconv[LOOM_RUN_HAL_MAX_BINDING_COUNT];
  // Number of populated entries in |expected_binding_specs|.
  int32_t expected_binding_count;
} iree_run_loom_hal_flag_state_t;

static iree_run_loom_hal_flag_state_t iree_run_loom_hal_flags = {0};

static iree_status_t iree_run_loom_parse_binding_flag(
    iree_string_view_t flag_name, void* storage, iree_string_view_t value) {
  (void)flag_name;
  (void)storage;
  if (iree_run_loom_hal_flags.binding_count >= LOOM_RUN_HAL_MAX_BINDING_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many HAL bindings; maximum is %d",
                            LOOM_RUN_HAL_MAX_BINDING_COUNT);
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
      LOOM_RUN_HAL_MAX_BINDING_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many expected HAL bindings; maximum is %d",
                            LOOM_RUN_HAL_MAX_BINDING_COUNT);
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

static iree_status_t iree_run_loom_probe_hal_backend(
    const loom_run_hal_backend_t* backend, iree_allocator_t allocator) {
  IREE_ASSERT_ARGUMENT(backend);
  loom_run_hal_runtime_t runtime = {0};
  loom_run_hal_selected_target_t target = {0};
  iree_string_builder_t target_id_builder;
  iree_string_builder_initialize(allocator, &target_id_builder);

  iree_status_t status =
      loom_run_hal_runtime_initialize(backend, allocator, &runtime);
  if (iree_status_is_ok(status)) {
    status = backend->select_target(backend, &runtime, allocator, &target);
  }
  if (iree_status_is_ok(status)) {
    status = backend->format_target(backend, &target, &target_id_builder);
  }
  if (iree_status_is_ok(status)) {
    fprintf(stdout, "hal backend: %.*s\n", (int)backend->name.size,
            backend->name.data);
    fprintf(stdout, "hal driver: %.*s\n", (int)backend->hal_driver_name.size,
            backend->hal_driver_name.data);
    fprintf(stdout, "hal target: %.*s\n",
            (int)iree_string_builder_size(&target_id_builder),
            iree_string_builder_buffer(&target_id_builder));
    if (!iree_string_view_is_empty(target.preset_key)) {
      fprintf(stdout, "hal preset: %.*s\n", (int)target.preset_key.size,
              target.preset_key.data);
    }
  }

  iree_string_builder_deinitialize(&target_id_builder);
  loom_run_hal_runtime_deinitialize(&runtime);
  return status;
}

static iree_status_t iree_run_loom_compile_to_archive(
    loom_run_module_t* run_module, iree_allocator_t allocator,
    loom_ireevm_module_archive_t* out_archive) {
  const loom_ireevm_module_compile_options_t compile_options = {
      .module_name = iree_make_cstring_view(FLAG_loom_module_name),
      .target_symbol = iree_make_cstring_view(FLAG_loom_target),
      .diagnostic_sink = {.fn = loom_diagnostic_stderr_sink},
      .source_resolver = loom_run_module_source_resolver(run_module),
      .max_errors = 20,
  };
  return loom_ireevm_compile_module_archive(
      run_module->module, &compile_options, allocator, out_archive);
}

static iree_status_t iree_run_loom_compile_to_hal_executable(
    const loom_run_hal_backend_t* backend, loom_run_module_t* run_module,
    const loom_run_hal_selected_target_t* target, iree_allocator_t allocator,
    loom_run_hal_executable_t* out_executable) {
  IREE_ASSERT_ARGUMENT(backend);
  IREE_ASSERT_ARGUMENT(out_executable);

  return backend->compile(
      backend, run_module->module, target,
      iree_make_cstring_view(FLAG_loom_target),
      (loom_diagnostic_sink_t){.fn = loom_diagnostic_stderr_sink},
      loom_run_module_source_resolver(run_module), 20, allocator,
      out_executable);
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

static iree_status_t iree_run_loom_hal_process_bindings(
    const loom_run_hal_runtime_t* runtime, iree_vm_list_t* binding_list,
    iree_allocator_t allocator, int* out_exit_code) {
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(out_exit_code);
  *out_exit_code = 0;
  IREE_RETURN_IF_ERROR(
      loom_run_hal_transfer_bindings_to_host(runtime, binding_list));

  iree_io_stream_t* stdout_stream = NULL;
  iree_hal_allocator_t* heap_allocator = NULL;
  iree_vm_list_t* expected_list = NULL;

  iree_status_t status = iree_io_stdio_stream_wrap(
      IREE_IO_STREAM_MODE_WRITABLE, stdout, /*owns_handle=*/false, allocator,
      &stdout_stream);
  if (iree_status_is_ok(status) &&
      iree_run_loom_hal_flags.expected_binding_count == 0) {
    status = iree_tooling_print_variants(
        IREE_SV("binding"), binding_list,
        IREE_RUN_LOOM_HAL_MAX_OUTPUT_ELEMENT_COUNT, stdout_stream, allocator);
  }
  if (iree_status_is_ok(status) &&
      iree_run_loom_hal_flags.expected_binding_count != 0) {
    if (iree_run_loom_hal_flags.expected_binding_count !=
        iree_run_loom_hal_flags.binding_count) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--expected_binding count (%d) must match --binding count (%d)",
          iree_run_loom_hal_flags.expected_binding_count,
          iree_run_loom_hal_flags.binding_count);
    }
  }
  if (iree_status_is_ok(status) &&
      iree_run_loom_hal_flags.expected_binding_count != 0) {
    status = iree_hal_allocator_create_heap(IREE_SV("heap"), allocator,
                                            allocator, &heap_allocator);
  }
  if (iree_status_is_ok(status) &&
      iree_run_loom_hal_flags.expected_binding_count != 0) {
    status = iree_tooling_parse_variants(
        iree_make_string_view(iree_run_loom_hal_flags.expected_binding_cconv,
                              iree_run_loom_hal_flags.expected_binding_count),
        (iree_string_view_list_t){
            .count = iree_run_loom_hal_flags.expected_binding_count,
            .values = iree_run_loom_hal_flags.expected_binding_specs,
        },
        runtime->device, heap_allocator, allocator, &expected_list);
  }
  if (iree_status_is_ok(status) &&
      iree_run_loom_hal_flags.expected_binding_count != 0) {
    const bool did_match = iree_tooling_compare_variant_lists(
        expected_list, binding_list, allocator, stdout);
    if (did_match) {
      fprintf(stdout,
              "[SUCCESS] all HAL bindings matched their expected "
              "values.\n");
    }
    *out_exit_code = did_match ? 0 : 1;
  }

  iree_vm_list_release(expected_list);
  iree_hal_allocator_release(heap_allocator);
  iree_io_stream_release(stdout_stream);
  fflush(stdout);
  return status;
}

static iree_status_t iree_run_loom_run_hal_executable(
    const loom_run_hal_executable_t* executable,
    const loom_run_hal_runtime_t* runtime, iree_allocator_t allocator,
    int* out_exit_code) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(out_exit_code);
  *out_exit_code = 0;

  loom_run_hal_invocation_options_t invocation_options = {0};
  loom_run_hal_invocation_options_initialize(&invocation_options);
  iree_status_t status = iree_run_loom_parse_workgroup_count(
      iree_make_cstring_view(FLAG_workgroup_count),
      invocation_options.workgroup_count);
  if (iree_status_is_ok(status) && FLAG_entry_point < 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--entry_point must be non-negative; got %d",
                              (int)FLAG_entry_point);
  }
  if (iree_status_is_ok(status)) {
    invocation_options.entry_point = (uint32_t)FLAG_entry_point;
  }

  iree_vm_list_t* binding_list = NULL;

  if (iree_status_is_ok(status)) {
    status = iree_tooling_parse_variants(
        iree_make_string_view(iree_run_loom_hal_flags.binding_cconv,
                              iree_run_loom_hal_flags.binding_count),
        (iree_string_view_list_t){
            .count = iree_run_loom_hal_flags.binding_count,
            .values = iree_run_loom_hal_flags.binding_specs,
        },
        runtime->device, iree_hal_device_allocator(runtime->device), allocator,
        &binding_list);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_invocation_execute(runtime, executable, binding_list,
                                             &invocation_options);
  }
  if (iree_status_is_ok(status)) {
    status = iree_run_loom_hal_process_bindings(runtime, binding_list,
                                                allocator, out_exit_code);
  }

  iree_vm_list_release(binding_list);
  return status;
}

static iree_status_t iree_run_loom_make_unknown_backend_status(
    iree_string_view_t backend_name,
    const loom_run_hal_backend_registry_t* hal_backend_registry,
    iree_allocator_t allocator) {
  iree_string_builder_t backend_names;
  iree_string_builder_initialize(allocator, &backend_names);
  iree_status_t status = loom_run_hal_backend_registry_format_names(
      hal_backend_registry, &backend_names);
  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&backend_names);
    return status;
  }
  status = iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "unknown --loom_backend='%.*s'; expected 'vm' or registered HAL backend "
      "in [%.*s]",
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
      "IREE VM bytecode archive and runs it with the same input/output flags "
      "as iree-run-module. Registered HAL backends compile HAL-native "
      "target-low kernels into IREE HAL executables and dispatch them through "
      "the production HAL device path.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t allocator = iree_allocator_system();
  const loom_run_hal_backend_registry_t* hal_backend_registry =
      &configuration->hal_backend_registry;
  const iree_string_view_t backend_name =
      iree_make_cstring_view(FLAG_loom_backend);
  const bool run_vm = iree_string_view_equal(backend_name, IREE_SV("vm"));
  const loom_run_hal_backend_t* hal_backend =
      run_vm ? NULL
             : loom_run_hal_backend_registry_lookup(hal_backend_registry,
                                                    backend_name);
  if (FLAG_probe_hal) {
    iree_status_t probe_status = iree_ok_status();
    if (run_vm) {
      probe_status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--probe_hal requires --loom_backend to name a registered HAL "
          "backend");
    } else if (hal_backend == NULL) {
      probe_status = iree_run_loom_make_unknown_backend_status(
          backend_name, hal_backend_registry, allocator);
    } else {
      probe_status = iree_run_loom_probe_hal_backend(hal_backend, allocator);
    }
    int probe_exit_code = 0;
    if (!iree_status_is_ok(probe_status)) {
      iree_status_fprint(stderr, probe_status);
      iree_status_free(probe_status);
      probe_exit_code = 1;
    }
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(probe_exit_code);
    return probe_exit_code;
  }

  iree_io_file_contents_t* contents = NULL;
  loom_run_session_t session = {0};
  loom_run_module_t run_module = {0};
  loom_ireevm_module_archive_t archive = {0};
  loom_run_hal_executable_t hal_executable = {0};
  loom_run_hal_runtime_t hal_runtime = {0};
  loom_run_hal_selected_target_t selected_hal_target = {0};
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
  if (iree_status_is_ok(status) && !run_vm && hal_backend == NULL) {
    status = iree_run_loom_make_unknown_backend_status(
        backend_name, hal_backend_registry, allocator);
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
  if (iree_status_is_ok(status)) {
    loom_run_module_parse_options_t parse_options = {0};
    loom_run_module_parse_options_initialize(&parse_options);
    parse_options.filename = filename;
    parse_options.source = source;
    status = loom_run_module_parse(&session, &parse_options, &run_module);
  }
  if (iree_status_is_ok(status) && run_vm) {
    status = iree_run_loom_compile_to_archive(&run_module, allocator, &archive);
  }
  if (iree_status_is_ok(status) && run_vm) {
    status = iree_tooling_create_instance(allocator, &instance);
  }
  if (iree_status_is_ok(status) && run_vm) {
    status = iree_tooling_run_module_with_data(
        instance, iree_string_view_empty(),
        iree_make_const_byte_span(archive.data, archive.data_length), allocator,
        &exit_code);
  }
  if (iree_status_is_ok(status) && hal_backend != NULL) {
    status =
        loom_run_hal_runtime_initialize(hal_backend, allocator, &hal_runtime);
  }
  if (iree_status_is_ok(status) && hal_backend != NULL) {
    status = hal_backend->select_target(hal_backend, &hal_runtime, allocator,
                                        &selected_hal_target);
  }
  if (iree_status_is_ok(status) && hal_backend != NULL) {
    status = iree_run_loom_compile_to_hal_executable(
        hal_backend, &run_module, &selected_hal_target, allocator,
        &hal_executable);
  }
  if (iree_status_is_ok(status) && hal_backend != NULL) {
    status = iree_run_loom_run_hal_executable(&hal_executable, &hal_runtime,
                                              allocator, &exit_code);
  }

  const bool had_error = !iree_status_is_ok(status);
  if (had_error) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = 1;
  }

  iree_vm_instance_release(instance);
  loom_run_hal_runtime_deinitialize(&hal_runtime);
  if (hal_backend && hal_backend->deinitialize_executable) {
    hal_backend->deinitialize_executable(hal_backend, &hal_executable,
                                         allocator);
  }
  loom_ireevm_module_archive_deinitialize(&archive, allocator);
  loom_run_module_deinitialize(&run_module);
  iree_io_file_contents_free(contents);
  loom_run_session_deinitialize(&session);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
