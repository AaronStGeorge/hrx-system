// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal_invocation.h"

#include "iree/modules/hal/types.h"
#include "iree/tooling/comparison.h"
#include "iree/tooling/function_io.h"
#include "iree/tooling/function_util.h"
#include "loom/target/launch.h"

enum {
  LOOM_RUN_HAL_DEFAULT_MAX_OUTPUT_ELEMENT_COUNT = 1024,
};

void loom_run_hal_invocation_options_initialize(
    loom_run_hal_invocation_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  *out_options = (loom_run_hal_invocation_options_t){
      .entry_point = 0,
      .workgroup_count = {1, 1, 1},
  };
}

void loom_run_hal_invocation_request_initialize(
    loom_run_hal_invocation_request_t* out_request) {
  IREE_ASSERT_ARGUMENT(out_request);
  *out_request = (loom_run_hal_invocation_request_t){0};
  loom_run_hal_invocation_options_initialize(&out_request->options);
}

void loom_run_hal_invocation_plan_initialize(
    loom_run_hal_invocation_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_run_hal_invocation_plan_t){0};
  loom_run_hal_invocation_options_initialize(&out_plan->options);
}

void loom_run_hal_invocation_plan_deinitialize(
    loom_run_hal_invocation_plan_t* plan) {
  if (plan == NULL) {
    return;
  }
  iree_vm_list_release(plan->expected_bindings);
  iree_hal_allocator_release(plan->expected_binding_allocator);
  iree_vm_list_release(plan->bindings);
  *plan = (loom_run_hal_invocation_plan_t){0};
}

void loom_run_hal_prepared_candidate_initialize(
    loom_run_hal_prepared_candidate_t* out_candidate) {
  IREE_ASSERT_ARGUMENT(out_candidate);
  *out_candidate = (loom_run_hal_prepared_candidate_t){0};
}

void loom_run_hal_prepared_candidate_deinitialize(
    loom_run_hal_prepared_candidate_t* candidate) {
  if (candidate == NULL) {
    return;
  }
  iree_hal_executable_release(candidate->executable);
  *candidate = (loom_run_hal_prepared_candidate_t){0};
}

void loom_run_hal_iteration_initialize(
    loom_run_hal_iteration_t* out_iteration) {
  IREE_ASSERT_ARGUMENT(out_iteration);
  *out_iteration = (loom_run_hal_iteration_t){0};
}

void loom_run_hal_iteration_deinitialize(loom_run_hal_iteration_t* iteration) {
  if (iteration == NULL) {
    return;
  }
  iree_vm_list_release(iteration->bindings);
  *iteration = (loom_run_hal_iteration_t){0};
}

void loom_run_hal_invocation_result_initialize(
    iree_allocator_t allocator, loom_run_hal_invocation_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(out_result);
  *out_result = (loom_run_hal_invocation_result_t){0};
  iree_string_builder_initialize(allocator, &out_result->output);
}

void loom_run_hal_invocation_result_deinitialize(
    loom_run_hal_invocation_result_t* result) {
  if (result == NULL) {
    return;
  }
  iree_string_builder_deinitialize(&result->output);
  *result = (loom_run_hal_invocation_result_t){0};
}

iree_status_t loom_run_hal_executable_prepare(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_executable_t* executable,
    iree_hal_executable_t** out_hal_executable) {
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_hal_executable);
  *out_hal_executable = NULL;
  if (runtime->device == NULL || runtime->executable_cache == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL runtime is not initialized");
  }

  iree_hal_executable_params_t executable_params;
  iree_hal_executable_params_initialize(&executable_params);
  executable_params.caching_mode =
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION |
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
  executable_params.executable_format = executable->executable_format;
  executable_params.executable_data = executable->executable_data;
  return iree_hal_executable_cache_prepare_executable(
      runtime->executable_cache, &executable_params, out_hal_executable);
}

iree_status_t loom_run_hal_prepared_candidate_prepare(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_executable_t* executable,
    loom_run_hal_prepared_candidate_t* out_candidate) {
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_candidate);
  loom_run_hal_prepared_candidate_initialize(out_candidate);
  iree_status_t status = loom_run_hal_executable_prepare(
      runtime, executable, &out_candidate->executable);
  if (iree_status_is_ok(status)) {
    out_candidate->target_bundle = executable->target_bundle;
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_prepared_candidate_deinitialize(out_candidate);
  }
  return status;
}

static iree_status_t loom_run_hal_binding_refs_from_list(
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

iree_status_t loom_run_hal_dispatch(
    iree_hal_device_t* device, iree_hal_executable_t* executable,
    iree_vm_list_t* binding_list,
    const loom_run_hal_invocation_options_t* options) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(binding_list);
  IREE_ASSERT_ARGUMENT(options);

  iree_hal_buffer_ref_t binding_refs[LOOM_RUN_HAL_MAX_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(loom_run_hal_binding_refs_from_list(
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
        options->workgroup_count[0], options->workgroup_count[1],
        options->workgroup_count[2]);
    status = iree_hal_command_buffer_dispatch(
        command_buffer, executable, options->entry_point, config,
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

iree_status_t loom_run_hal_invocation_execute(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_executable_t* executable, iree_vm_list_t* binding_list,
    const loom_run_hal_invocation_options_t* options) {
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(binding_list);
  IREE_ASSERT_ARGUMENT(options);

  loom_run_hal_prepared_candidate_t candidate = {0};
  iree_status_t status =
      loom_run_hal_prepared_candidate_prepare(runtime, executable, &candidate);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_dispatch(runtime->device, candidate.executable,
                                   binding_list, options);
  }
  loom_run_hal_prepared_candidate_deinitialize(&candidate);
  return status;
}

iree_status_t loom_run_hal_transfer_bindings_to_host(
    const loom_run_hal_runtime_t* runtime, iree_vm_list_t* binding_list) {
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(binding_list);
  if (runtime->device == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL runtime is not initialized");
  }

  iree_hal_allocator_t* device_allocator =
      iree_hal_device_allocator(runtime->device);
  iree_hal_buffer_params_t host_params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = 0,
  };
  return iree_tooling_transfer_variants(
      binding_list, runtime->device, device_allocator, host_params,
      /*wait_fence=*/NULL, /*signal_fence=*/NULL);
}

static iree_status_t loom_run_hal_binding_specs_validate(
    const loom_run_hal_binding_specs_t* specs,
    iree_string_view_t binding_list_name) {
  IREE_ASSERT_ARGUMENT(specs);
  if (specs->count > LOOM_RUN_HAL_MAX_BINDING_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%.*s binding count %" PRIhsz " exceeds maximum %d",
                            (int)binding_list_name.size, binding_list_name.data,
                            specs->count, LOOM_RUN_HAL_MAX_BINDING_COUNT);
  }
  if (specs->count > 0 &&
      (specs->values == NULL || specs->conventions == NULL)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s binding specs require values and calling conventions",
        (int)binding_list_name.size, binding_list_name.data);
  }
  return iree_ok_status();
}

static iree_string_view_t loom_run_hal_binding_specs_conventions(
    const loom_run_hal_binding_specs_t* specs) {
  IREE_ASSERT_ARGUMENT(specs);
  return iree_make_string_view(specs->conventions, specs->count);
}

static iree_string_view_list_t loom_run_hal_binding_specs_values(
    const loom_run_hal_binding_specs_t* specs) {
  IREE_ASSERT_ARGUMENT(specs);
  return (iree_string_view_list_t){
      .count = specs->count,
      .values = specs->values,
  };
}

static iree_status_t loom_run_hal_parse_binding_specs(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_binding_specs_t* specs,
    iree_hal_allocator_t* device_allocator, iree_allocator_t allocator,
    iree_vm_list_t** out_list) {
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(specs);
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(out_list);
  IREE_RETURN_IF_ERROR(
      loom_run_hal_binding_specs_validate(specs, IREE_SV("HAL")));
  return iree_tooling_parse_variants(
      loom_run_hal_binding_specs_conventions(specs),
      loom_run_hal_binding_specs_values(specs), runtime->device,
      device_allocator, allocator, out_list);
}

static iree_status_t loom_run_hal_process_invocation_bindings(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_invocation_plan_t* plan, iree_vm_list_t* binding_list,
    iree_allocator_t allocator, loom_run_hal_invocation_result_t* result) {
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(binding_list);
  IREE_ASSERT_ARGUMENT(result);
  IREE_RETURN_IF_ERROR(
      loom_run_hal_transfer_bindings_to_host(runtime, binding_list));

  const iree_host_size_t max_output_element_count =
      plan->max_output_element_count == 0
          ? LOOM_RUN_HAL_DEFAULT_MAX_OUTPUT_ELEMENT_COUNT
          : plan->max_output_element_count;
  if (plan->expected_bindings == NULL) {
    return iree_tooling_format_variants(IREE_SV("binding"), binding_list,
                                        max_output_element_count,
                                        &result->output);
  }

  iree_status_t status = iree_ok_status();
  const bool did_match = iree_tooling_compare_variant_lists_and_append(
      plan->expected_bindings, binding_list, allocator, &result->output);
  if (did_match) {
    status = iree_string_builder_append_cstring(
        &result->output,
        "[SUCCESS] all HAL bindings matched their expected values.\n");
  }
  result->exit_code = did_match ? 0 : 1;
  return status;
}

static iree_status_t loom_run_hal_invocation_plan_validate(
    const loom_run_hal_invocation_plan_t* plan) {
  IREE_ASSERT_ARGUMENT(plan);
  if (plan->bindings == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL invocation plan requires bindings");
  }
  if (plan->expected_bindings != NULL &&
      iree_vm_list_size(plan->expected_bindings) !=
          iree_vm_list_size(plan->bindings)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected HAL binding count %" PRIhsz
                            " must match input binding count %" PRIhsz,
                            iree_vm_list_size(plan->expected_bindings),
                            iree_vm_list_size(plan->bindings));
  }
  return iree_ok_status();
}

static iree_status_t loom_run_hal_prepared_candidate_validate_dispatch(
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan) {
  IREE_ASSERT_ARGUMENT(candidate);
  IREE_ASSERT_ARGUMENT(plan);
  const loom_target_bundle_t* target_bundle = candidate->target_bundle;
  if (target_bundle == NULL) {
    return iree_ok_status();
  }
  if (target_bundle->snapshot == NULL || target_bundle->export_plan == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL prepared candidate target bundle is missing "
                            "snapshot or export plan");
  }
  if (target_bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL prepared candidate target bundle must use HAL kernel ABI");
  }
  const loom_target_dispatch_workgroup_count_t workgroup_count = {
      .x = plan->options.workgroup_count[0],
      .y = plan->options.workgroup_count[1],
      .z = plan->options.workgroup_count[2],
  };
  return loom_target_validate_hal_dispatch_workgroup_count(
      target_bundle->snapshot, &target_bundle->export_plan->hal_kernel,
      &workgroup_count);
}

iree_status_t loom_run_hal_invocation_plan_prepare_from_specs(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_invocation_options_t* options,
    const loom_run_hal_binding_specs_t* bindings,
    const loom_run_hal_binding_specs_t* expected_bindings,
    iree_host_size_t max_output_element_count, iree_allocator_t allocator,
    loom_run_hal_invocation_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(bindings);
  IREE_ASSERT_ARGUMENT(expected_bindings);
  IREE_ASSERT_ARGUMENT(out_plan);
  loom_run_hal_invocation_plan_initialize(out_plan);
  IREE_RETURN_IF_ERROR(
      loom_run_hal_binding_specs_validate(bindings, IREE_SV("HAL")));
  IREE_RETURN_IF_ERROR(loom_run_hal_binding_specs_validate(
      expected_bindings, IREE_SV("expected HAL")));
  if (expected_bindings->count != 0 &&
      expected_bindings->count != bindings->count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected HAL binding count %" PRIhsz
                            " must match input binding count %" PRIhsz,
                            expected_bindings->count, bindings->count);
  }
  if (runtime->device == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL runtime is not initialized");
  }

  iree_status_t status = loom_run_hal_parse_binding_specs(
      runtime, bindings, iree_hal_device_allocator(runtime->device), allocator,
      &out_plan->bindings);
  if (iree_status_is_ok(status) && expected_bindings->count != 0) {
    status =
        iree_hal_allocator_create_heap(IREE_SV("heap"), allocator, allocator,
                                       &out_plan->expected_binding_allocator);
  }
  if (iree_status_is_ok(status) && expected_bindings->count != 0) {
    status = loom_run_hal_parse_binding_specs(
        runtime, expected_bindings, out_plan->expected_binding_allocator,
        allocator, &out_plan->expected_bindings);
  }
  if (iree_status_is_ok(status)) {
    out_plan->options = *options;
    out_plan->max_output_element_count = max_output_element_count;
  } else {
    loom_run_hal_invocation_plan_deinitialize(out_plan);
  }
  return status;
}

iree_status_t loom_run_hal_invocation_dispatch_plan(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan, iree_allocator_t allocator,
    loom_run_hal_iteration_t* out_iteration) {
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(candidate);
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(out_iteration);
  loom_run_hal_iteration_initialize(out_iteration);
  IREE_RETURN_IF_ERROR(loom_run_hal_invocation_plan_validate(plan));
  if (candidate->executable == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL prepared candidate requires an executable");
  }
  IREE_RETURN_IF_ERROR(
      loom_run_hal_prepared_candidate_validate_dispatch(candidate, plan));
  if (runtime->device == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL runtime is not initialized");
  }

  iree_status_t status =
      iree_vm_list_clone(plan->bindings, allocator, &out_iteration->bindings);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_dispatch(runtime->device, candidate->executable,
                                   out_iteration->bindings, &plan->options);
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_iteration_deinitialize(out_iteration);
  }
  return status;
}

iree_status_t loom_run_hal_invocation_collect_results(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_invocation_plan_t* plan,
    const loom_run_hal_iteration_t* iteration, iree_allocator_t allocator,
    loom_run_hal_invocation_result_t* result) {
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(iteration);
  IREE_ASSERT_ARGUMENT(result);
  iree_string_builder_reset(&result->output);
  result->exit_code = 0;
  IREE_RETURN_IF_ERROR(loom_run_hal_invocation_plan_validate(plan));
  if (iteration->bindings == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL iteration requires bindings");
  }
  if (iree_vm_list_size(iteration->bindings) !=
      iree_vm_list_size(plan->bindings)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL iteration binding count %" PRIhsz
                            " must match plan binding count %" PRIhsz,
                            iree_vm_list_size(iteration->bindings),
                            iree_vm_list_size(plan->bindings));
  }
  if (runtime->device == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL runtime is not initialized");
  }
  return loom_run_hal_process_invocation_bindings(
      runtime, plan, iteration->bindings, allocator, result);
}

iree_status_t loom_run_hal_invocation_run_prepared(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan, iree_allocator_t allocator,
    loom_run_hal_invocation_result_t* result) {
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(candidate);
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(result);
  iree_string_builder_reset(&result->output);
  result->exit_code = 0;

  loom_run_hal_iteration_t iteration = {0};
  iree_status_t status = loom_run_hal_invocation_dispatch_plan(
      runtime, candidate, plan, allocator, &iteration);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_invocation_collect_results(runtime, plan, &iteration,
                                                     allocator, result);
  }
  loom_run_hal_iteration_deinitialize(&iteration);
  return status;
}

iree_status_t loom_run_hal_invocation_run_plan(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_executable_t* executable,
    const loom_run_hal_invocation_plan_t* plan, iree_allocator_t allocator,
    loom_run_hal_invocation_result_t* result) {
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(result);
  iree_string_builder_reset(&result->output);
  result->exit_code = 0;
  IREE_RETURN_IF_ERROR(loom_run_hal_invocation_plan_validate(plan));
  loom_run_hal_prepared_candidate_t candidate = {0};
  iree_status_t status =
      loom_run_hal_prepared_candidate_prepare(runtime, executable, &candidate);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_invocation_run_prepared(runtime, &candidate, plan,
                                                  allocator, result);
  }
  loom_run_hal_prepared_candidate_deinitialize(&candidate);
  return status;
}

iree_status_t loom_run_hal_invocation_run(
    const loom_run_hal_invocation_request_t* request,
    iree_allocator_t allocator, loom_run_hal_invocation_result_t* result) {
  IREE_ASSERT_ARGUMENT(request && request->runtime && request->executable);
  IREE_ASSERT_ARGUMENT(result);
  iree_string_builder_reset(&result->output);
  result->exit_code = 0;
  loom_run_hal_invocation_plan_t plan = {0};
  iree_status_t status = loom_run_hal_invocation_plan_prepare_from_specs(
      request->runtime, &request->options, &request->bindings,
      &request->expected_bindings, request->max_output_element_count, allocator,
      &plan);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_invocation_run_plan(
        request->runtime, request->executable, &plan, allocator, result);
  }
  loom_run_hal_invocation_plan_deinitialize(&plan);
  return status;
}
