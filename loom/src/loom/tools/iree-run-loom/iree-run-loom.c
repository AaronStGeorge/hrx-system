// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-run-loom: compiles a Loom module and executes the selected export.

#include <stdio.h>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/threading/numa.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/io/stdio_stream.h"
#include "iree/modules/hal/types.h"
#include "iree/tooling/comparison.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/function_io.h"
#include "iree/tooling/function_util.h"
#include "iree/tooling/run_module.h"
#include "iree/vm/api.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/diagnostic.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/target/all/low_registry.h"
#include "loom/target/emit/ireevm/module_compiler.h"
#include "loom/tools/iree-run-loom/hal_backend.h"

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
  IREE_RUN_LOOM_HAL_MAX_BINDING_COUNT = 64,
  IREE_RUN_LOOM_HAL_MAX_OUTPUT_ELEMENT_COUNT = 1024,
};

typedef struct iree_run_loom_hal_flag_state_t {
  // Binding specs in HAL binding ordinal order.
  iree_string_view_t binding_specs[IREE_RUN_LOOM_HAL_MAX_BINDING_COUNT];
  // Calling-convention character for each binding spec.
  char binding_cconv[IREE_RUN_LOOM_HAL_MAX_BINDING_COUNT];
  // Number of populated entries in |binding_specs|.
  int32_t binding_count;
  // Expected binding specs in HAL binding ordinal order.
  iree_string_view_t
      expected_binding_specs[IREE_RUN_LOOM_HAL_MAX_BINDING_COUNT];
  // Calling-convention character for each expected binding spec.
  char expected_binding_cconv[IREE_RUN_LOOM_HAL_MAX_BINDING_COUNT];
  // Number of populated entries in |expected_binding_specs|.
  int32_t expected_binding_count;
} iree_run_loom_hal_flag_state_t;

static iree_run_loom_hal_flag_state_t iree_run_loom_hal_flags = {0};

static iree_status_t iree_run_loom_parse_binding_flag(
    iree_string_view_t flag_name, void* storage, iree_string_view_t value) {
  (void)flag_name;
  (void)storage;
  if (iree_run_loom_hal_flags.binding_count >=
      IREE_RUN_LOOM_HAL_MAX_BINDING_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many HAL bindings; maximum is %d",
                            IREE_RUN_LOOM_HAL_MAX_BINDING_COUNT);
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
      IREE_RUN_LOOM_HAL_MAX_BINDING_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many expected HAL bindings; maximum is %d",
                            IREE_RUN_LOOM_HAL_MAX_BINDING_COUNT);
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

static void iree_run_loom_hal_runtime_deinitialize(
    iree_run_loom_hal_runtime_t* runtime) {
  if (runtime == NULL) {
    return;
  }
  iree_hal_executable_cache_release(runtime->executable_cache);
  iree_hal_device_release(runtime->device);
  iree_vm_instance_release(runtime->instance);
  *runtime = (iree_run_loom_hal_runtime_t){0};
}

static iree_status_t iree_run_loom_hal_runtime_initialize(
    const iree_amdgpu_hal_backend_t* backend, iree_allocator_t allocator,
    iree_run_loom_hal_runtime_t* out_runtime) {
  IREE_ASSERT_ARGUMENT(backend);
  IREE_ASSERT_ARGUMENT(out_runtime);
  *out_runtime = (iree_run_loom_hal_runtime_t){0};

  iree_async_proactor_pool_t* proactor_pool = NULL;
  iree_status_t status =
      iree_tooling_create_instance(allocator, &out_runtime->instance);
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL,
        iree_async_proactor_pool_options_default(), allocator, &proactor_pool);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    status = iree_hal_create_device_from_flags(
        iree_hal_available_driver_registry(), backend->hal_driver_name,
        &create_params, allocator, &out_runtime->device);
  }
  iree_async_proactor_pool_release(proactor_pool);
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_cache_create(
        out_runtime->device, IREE_SV("loom"), &out_runtime->executable_cache);
  }
  if (!iree_status_is_ok(status)) {
    iree_run_loom_hal_runtime_deinitialize(out_runtime);
  }
  return status;
}

static iree_status_t iree_run_loom_probe_hal_backend(
    const iree_amdgpu_hal_backend_t* backend, iree_allocator_t allocator) {
  IREE_ASSERT_ARGUMENT(backend);
  iree_run_loom_hal_runtime_t runtime = {0};
  iree_run_loom_hal_selected_target_t target = {0};
  iree_string_builder_t target_id_builder;
  iree_string_builder_initialize(allocator, &target_id_builder);

  iree_status_t status =
      iree_run_loom_hal_runtime_initialize(backend, allocator, &runtime);
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
  iree_run_loom_hal_runtime_deinitialize(&runtime);
  return status;
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

static iree_status_t iree_run_loom_compile_to_hal_executable(
    iree_string_view_t filename, iree_string_view_t source,
    const iree_amdgpu_hal_backend_t* backend, loom_module_t* module,
    const iree_run_loom_hal_selected_target_t* target,
    iree_allocator_t allocator,
    iree_run_loom_hal_executable_t* out_executable) {
  IREE_ASSERT_ARGUMENT(backend);
  IREE_ASSERT_ARGUMENT(out_executable);

  loom_source_entry_t source_entry = {0};
  loom_source_table_resolver_t source_resolver = {0};
  IREE_RETURN_IF_ERROR(iree_run_loom_source_resolver_for_input(
      module->context, filename, source, &source_entry, &source_resolver));
  return backend->compile(
      backend, module, target, iree_make_cstring_view(FLAG_loom_target),
      (loom_diagnostic_sink_t){.fn = loom_diagnostic_stderr_sink},
      (loom_source_resolver_t){.fn = loom_source_table_resolve,
                               .user_data = &source_resolver},
      20, allocator, out_executable);
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

static iree_status_t iree_run_loom_hal_binding_refs_from_list(
    iree_vm_list_t* binding_list, iree_hal_buffer_ref_t* binding_refs,
    iree_host_size_t binding_ref_capacity) {
  IREE_ASSERT_ARGUMENT(binding_list);
  IREE_ASSERT_ARGUMENT(binding_refs);
  const iree_host_size_t binding_count = iree_vm_list_size(binding_list);
  if (binding_count > binding_ref_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL binding count %" PRIhsz
                            " exceeds capacity %" PRIhsz,
                            binding_count, binding_ref_capacity);
  }
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    iree_vm_ref_t value = iree_vm_ref_null();
    IREE_RETURN_IF_ERROR(iree_vm_list_get_ref_assign(binding_list, i, &value));
    iree_hal_buffer_t* buffer = NULL;
    if (iree_hal_buffer_isa(value)) {
      buffer = iree_hal_buffer_deref(value);
    } else if (iree_hal_buffer_view_isa(value)) {
      buffer = iree_hal_buffer_view_buffer(iree_hal_buffer_view_deref(value));
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HAL binding %" PRIhsz " is not a buffer or buffer view", i);
    }
    binding_refs[i] =
        iree_hal_make_buffer_ref(buffer, 0, IREE_HAL_WHOLE_BUFFER);
  }
  return iree_ok_status();
}

static iree_status_t iree_run_loom_hal_dispatch(
    iree_hal_device_t* device, iree_hal_executable_t* executable,
    iree_vm_list_t* binding_list, uint32_t* workgroup_count) {
  iree_hal_buffer_ref_t binding_refs[IREE_RUN_LOOM_HAL_MAX_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(iree_run_loom_hal_binding_refs_from_list(
      binding_list, binding_refs, IREE_ARRAYSIZE(binding_refs)));

  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_hal_semaphore_t* semaphore = NULL;
  uint64_t signal_value = 1;

  iree_status_t status = iree_hal_command_buffer_create(
      device,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_ref_list_t bindings = {
        .count = iree_vm_list_size(binding_list),
        .values = binding_refs,
    };
    iree_hal_dispatch_config_t config = iree_hal_make_static_dispatch_config(
        workgroup_count[0], workgroup_count[1], workgroup_count[2]);
    status = iree_hal_command_buffer_dispatch(
        command_buffer, executable, (uint32_t)FLAG_entry_point, config,
        iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_list_t wait_semaphores = iree_hal_semaphore_list_empty();
    iree_hal_semaphore_list_t signal_semaphores = {
        .count = 1,
        .semaphores = &semaphore,
        .payload_values = &signal_value,
    };
    status = iree_hal_device_queue_execute(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphores, signal_semaphores,
        command_buffer, iree_hal_buffer_binding_table_empty(),
        IREE_HAL_EXECUTE_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(semaphore, signal_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }

  iree_hal_semaphore_release(semaphore);
  iree_hal_command_buffer_release(command_buffer);
  return status;
}

static iree_status_t iree_run_loom_hal_process_bindings(
    iree_hal_device_t* device, iree_vm_list_t* binding_list,
    iree_allocator_t allocator, int* out_exit_code) {
  IREE_ASSERT_ARGUMENT(out_exit_code);
  *out_exit_code = 0;
  iree_hal_allocator_t* device_allocator = iree_hal_device_allocator(device);
  iree_hal_buffer_params_t host_params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = 0,
  };
  IREE_RETURN_IF_ERROR(iree_tooling_transfer_variants(
      binding_list, device, device_allocator, host_params,
      /*wait_fence=*/NULL, /*signal_fence=*/NULL));

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
        device, heap_allocator, allocator, &expected_list);
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
    const iree_run_loom_hal_executable_t* executable,
    const iree_run_loom_hal_runtime_t* runtime, iree_allocator_t allocator,
    int* out_exit_code) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(out_exit_code);
  *out_exit_code = 0;

  uint32_t workgroup_count[3] = {1, 1, 1};
  iree_status_t status = iree_run_loom_parse_workgroup_count(
      iree_make_cstring_view(FLAG_workgroup_count), workgroup_count);

  iree_hal_executable_t* hal_executable = NULL;
  iree_vm_list_t* binding_list = NULL;

  if (runtime->device == NULL || runtime->executable_cache == NULL) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HAL runtime is not initialized");
  }
  if (iree_status_is_ok(status)) {
    iree_hal_executable_params_t executable_params;
    iree_hal_executable_params_initialize(&executable_params);
    executable_params.caching_mode =
        IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION |
        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
    executable_params.executable_format = executable->executable_format;
    executable_params.executable_data = executable->executable_data;
    status = iree_hal_executable_cache_prepare_executable(
        runtime->executable_cache, &executable_params, &hal_executable);
  }
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
    status = iree_run_loom_hal_dispatch(runtime->device, hal_executable,
                                        binding_list, workgroup_count);
  }
  if (iree_status_is_ok(status)) {
    status = iree_run_loom_hal_process_bindings(runtime->device, binding_list,
                                                allocator, out_exit_code);
  }

  iree_vm_list_release(binding_list);
  iree_hal_executable_release(hal_executable);
  return status;
}

static iree_status_t iree_run_loom_make_unknown_backend_status(
    iree_string_view_t backend_name,
    const iree_amdgpu_hal_backend_registry_t* hal_backend_registry,
    iree_allocator_t allocator) {
  iree_string_builder_t backend_names;
  iree_string_builder_initialize(allocator, &backend_names);
  iree_status_t status = iree_amdgpu_hal_backend_registry_format_names(
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
      "The 'vm' backend compiles target.preset key \"iree-vm\" into a real "
      "IREE VM bytecode archive and runs it with the same input/output flags "
      "as iree-run-module. Registered HAL backends compile HAL-native "
      "target-low kernels into IREE HAL executables and dispatch them through "
      "the production HAL device path.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t allocator = iree_allocator_system();
  iree_amdgpu_hal_backend_registry_t hal_backend_registry;
  iree_amdgpu_hal_backend_registry_initialize(&hal_backend_registry);
  const iree_string_view_t backend_name =
      iree_make_cstring_view(FLAG_loom_backend);
  const bool run_vm = iree_string_view_equal(backend_name, IREE_SV("vm"));
  const iree_amdgpu_hal_backend_t* hal_backend =
      run_vm ? NULL
             : iree_amdgpu_hal_backend_registry_lookup(&hal_backend_registry,
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
          backend_name, &hal_backend_registry, allocator);
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

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, allocator, &block_pool);

  loom_context_t context = {0};
  bool context_initialized = false;
  iree_io_file_contents_t* contents = NULL;
  loom_module_t* module = NULL;
  loom_target_low_descriptor_registry_t low_registry = {0};
  loom_ireevm_module_archive_t archive = {0};
  iree_run_loom_hal_executable_t hal_executable = {0};
  iree_run_loom_hal_runtime_t hal_runtime = {0};
  iree_run_loom_hal_selected_target_t selected_hal_target = {0};
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
        backend_name, &hal_backend_registry, allocator);
  }

  loom_all_low_descriptor_registry_initialize(&low_registry);
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
  if (iree_status_is_ok(status) && run_vm) {
    status = iree_run_loom_compile_to_archive(filename, source, module,
                                              allocator, &archive);
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
    status = iree_run_loom_hal_runtime_initialize(hal_backend, allocator,
                                                  &hal_runtime);
  }
  if (iree_status_is_ok(status) && hal_backend != NULL) {
    status = hal_backend->select_target(hal_backend, &hal_runtime, allocator,
                                        &selected_hal_target);
  }
  if (iree_status_is_ok(status) && hal_backend != NULL) {
    status = iree_run_loom_compile_to_hal_executable(
        filename, source, hal_backend, module, &selected_hal_target, allocator,
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
  iree_run_loom_hal_runtime_deinitialize(&hal_runtime);
  if (hal_backend && hal_backend->deinitialize_executable) {
    hal_backend->deinitialize_executable(hal_backend, &hal_executable,
                                         allocator);
  }
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
