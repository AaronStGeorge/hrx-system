// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/amdgpu_hal_backend.h"

#include "loom/ops/target/ops.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/hal_executable.h"
#include "loom/target/emit/native/amdgpu/module_compiler.h"
#include "loom/tooling/execution/hal_runtime.h"

static const iree_string_view_t kIreeRunLoomAmdgpuCurrentPreset =
    IREE_SVL("amdgpu-current");

static iree_status_t iree_run_loom_amdgpu_format_target_id(
    const loom_amdgpu_processor_info_t* processor,
    iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(processor);
  iree_string_builder_reset(builder);
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "amdgcn-amd-amdhsa--"));
  return iree_string_builder_append_string(builder, processor->target_cpu);
}

static iree_status_t iree_run_loom_amdgpu_processor_is_compile_supported(
    const loom_amdgpu_processor_info_t* processor, bool* out_supported) {
  IREE_ASSERT_ARGUMENT(processor);
  IREE_ASSERT_ARGUMENT(out_supported);
  *out_supported = false;
  if (iree_string_view_is_empty(processor->target_cpu) ||
      iree_string_view_is_empty(processor->low_preset_key) ||
      iree_string_view_is_empty(processor->descriptor_set_key) ||
      processor->elf_machine_flags == 0 ||
      processor->kernel_descriptor_profile ==
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE) {
    return iree_ok_status();
  }

  const loom_amdgpu_descriptor_set_info_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_info_lookup_descriptor_set_by_id(
      processor->descriptor_set_stable_id, &descriptor_set));
  *out_supported = descriptor_set->supports_descriptor_packet_encoding;
  return iree_ok_status();
}

static iree_status_t iree_run_loom_amdgpu_select_target(
    const loom_run_hal_backend_t* backend,
    const loom_run_hal_runtime_t* runtime, iree_allocator_t allocator,
    loom_run_hal_selected_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(backend);
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = (loom_run_hal_selected_target_t){0};
  if (runtime->executable_cache == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL executable cache is required for %.*s",
                            (int)backend->name.size, backend->name.data);
  }

  iree_string_builder_t target_id_builder;
  iree_string_builder_initialize(allocator, &target_id_builder);

  iree_status_t status = iree_ok_status();
  const iree_host_size_t processor_count =
      loom_amdgpu_target_info_processor_count();
  for (iree_host_size_t i = 0;
       i < processor_count && out_target->data == NULL &&
       iree_status_is_ok(status);
       ++i) {
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_processor_at(i);
    bool compile_supported = false;
    status = iree_run_loom_amdgpu_processor_is_compile_supported(
        processor, &compile_supported);
    if (!iree_status_is_ok(status) || !compile_supported) {
      continue;
    }
    status =
        iree_run_loom_amdgpu_format_target_id(processor, &target_id_builder);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (iree_hal_executable_cache_can_prepare_format(
            runtime->executable_cache,
            IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION,
            iree_string_builder_view(&target_id_builder))) {
      *out_target = (loom_run_hal_selected_target_t){
          .data = processor,
          .preset_key = processor->low_preset_key,
      };
    }
  }

  iree_string_builder_deinitialize(&target_id_builder);
  if (iree_status_is_ok(status) && out_target->data == NULL) {
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "selected %.*s HAL device has no Loom-supported native target",
        (int)backend->target_family_name.size,
        backend->target_family_name.data);
  }
  return status;
}

static iree_status_t iree_run_loom_amdgpu_format_target(
    const loom_run_hal_backend_t* backend,
    const loom_run_hal_selected_target_t* target,
    iree_string_builder_t* output) {
  IREE_ASSERT_ARGUMENT(backend);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(output);
  if (target->data == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s target selection is empty",
                            (int)backend->name.size, backend->name.data);
  }
  const loom_amdgpu_processor_info_t* processor =
      (const loom_amdgpu_processor_info_t*)target->data;
  return iree_run_loom_amdgpu_format_target_id(processor, output);
}

static iree_status_t iree_run_loom_amdgpu_rewrite_current_profiles(
    loom_module_t* module, const loom_amdgpu_processor_info_t* processor) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(processor);
  if (iree_string_view_is_empty(processor->low_preset_key)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "selected AMDGPU processor has no low preset key");
  }

  loom_string_id_t selected_key_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, processor->low_preset_key, &selected_key_id));

  loom_block_t* block = loom_module_block(module);
  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (!loom_target_profile_isa(op)) {
      continue;
    }
    const loom_string_id_t key_id = loom_target_profile_preset(op);
    if (key_id == LOOM_STRING_ID_INVALID || key_id >= module->strings.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "target profile has invalid preset string id");
    }
    iree_string_view_t key = module->strings.entries[key_id];
    if (iree_string_view_equal(key, kIreeRunLoomAmdgpuCurrentPreset)) {
      loom_op_attrs(op)[loom_target_profile_preset_ATTR_INDEX] =
          loom_attr_string(selected_key_id);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_run_loom_amdgpu_compile(
    const loom_run_hal_backend_t* backend, loom_module_t* module,
    const loom_run_hal_selected_target_t* target,
    iree_string_view_t entry_symbol, loom_diagnostic_sink_t diagnostic_sink,
    loom_source_resolver_t source_resolver, uint32_t max_errors,
    loom_target_compile_report_t* report, iree_allocator_t allocator,
    loom_run_hal_executable_t* out_executable) {
  IREE_ASSERT_ARGUMENT(backend);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = (loom_run_hal_executable_t){0};

  IREE_ASSERT_ARGUMENT(target->data);
  const loom_amdgpu_processor_info_t* processor =
      (const loom_amdgpu_processor_info_t*)target->data;
  IREE_RETURN_IF_ERROR(
      iree_run_loom_amdgpu_rewrite_current_profiles(module, processor));

  loom_amdgpu_hal_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, sizeof(*executable),
                                             (void**)&executable));
  *executable = (loom_amdgpu_hal_executable_t){0};

  const loom_amdgpu_module_compile_options_t compile_options = {
      .entry_symbol = entry_symbol,
      .target_cpu = processor->target_cpu,
      .diagnostic_sink = diagnostic_sink,
      .source_resolver = source_resolver,
      .max_errors = max_errors,
      .report = report,
      .report_row_storage =
          {
              .pressure_rows = report ? report->pressure_rows : NULL,
              .pressure_row_capacity =
                  report ? report->pressure_row_capacity : 0,
              .spill_rows = report ? report->spill_rows : NULL,
              .spill_row_capacity = report ? report->spill_row_capacity : 0,
          },
  };
  iree_status_t status = loom_amdgpu_compile_hal_executable(
      module, &compile_options, allocator, executable);
  if (iree_status_is_ok(status)) {
    *out_executable = (loom_run_hal_executable_t){
        .executable_format = executable->executable_format,
        .executable_data = iree_make_const_byte_span(executable->data,
                                                     executable->data_length),
        .storage = executable,
    };
  } else {
    loom_amdgpu_hal_executable_deinitialize(executable, allocator);
    iree_allocator_free(allocator, executable);
  }
  return status;
}

static void iree_run_loom_amdgpu_deinitialize_executable(
    const loom_run_hal_backend_t* backend,
    loom_run_hal_executable_t* executable, iree_allocator_t allocator) {
  (void)backend;
  if (!executable || !executable->storage) {
    return;
  }
  loom_amdgpu_hal_executable_t* amdgpu_executable =
      (loom_amdgpu_hal_executable_t*)executable->storage;
  loom_amdgpu_hal_executable_deinitialize(amdgpu_executable, allocator);
  iree_allocator_free(allocator, amdgpu_executable);
  *executable = (loom_run_hal_executable_t){0};
}

const loom_run_hal_backend_t iree_run_loom_amdgpu_hal_backend = {
    .name = IREE_SVL("amdgpu-hal"),
    .hal_driver_name = IREE_SVL("amdgpu"),
    .target_family_name = IREE_SVL("AMDGPU"),
    .select_target = iree_run_loom_amdgpu_select_target,
    .format_target = iree_run_loom_amdgpu_format_target,
    .compile = iree_run_loom_amdgpu_compile,
    .deinitialize_executable = iree_run_loom_amdgpu_deinitialize_executable,
};
