// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/vm_invocation.h"

#include "iree/hal/api.h"
#include "iree/io/vec_stream.h"
#include "iree/tooling/comparison.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/function_io.h"
#include "iree/tooling/function_util.h"
#include "iree/vm/bytecode/module.h"

enum {
  LOOM_RUN_VM_DEFAULT_MAX_OUTPUT_ELEMENT_COUNT = 1024,
  LOOM_RUN_VM_OUTPUT_STREAM_BLOCK_SIZE = 4096,
};

iree_status_t loom_run_vm_runtime_initialize(
    iree_allocator_t allocator, loom_run_vm_runtime_t* out_runtime) {
  IREE_ASSERT_ARGUMENT(out_runtime);
  *out_runtime = (loom_run_vm_runtime_t){0};
  iree_status_t status =
      iree_tooling_create_instance(allocator, &out_runtime->instance);
  if (!iree_status_is_ok(status)) {
    loom_run_vm_runtime_deinitialize(out_runtime);
  }
  return status;
}

void loom_run_vm_runtime_deinitialize(loom_run_vm_runtime_t* runtime) {
  if (runtime == NULL) {
    return;
  }
  iree_vm_instance_release(runtime->instance);
  *runtime = (loom_run_vm_runtime_t){0};
}

void loom_run_vm_invocation_options_initialize(
    loom_run_vm_invocation_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  *out_options = (loom_run_vm_invocation_options_t){
      .max_output_element_count = LOOM_RUN_VM_DEFAULT_MAX_OUTPUT_ELEMENT_COUNT,
  };
}

void loom_run_vm_invocation_plan_initialize(
    loom_run_vm_invocation_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_run_vm_invocation_plan_t){
      .output_mode = LOOM_RUN_VM_OUTPUT_PLAN_MODE_FORMAT_ALL,
      .max_output_element_count = LOOM_RUN_VM_DEFAULT_MAX_OUTPUT_ELEMENT_COUNT,
  };
}

void loom_run_vm_invocation_plan_deinitialize(
    loom_run_vm_invocation_plan_t* plan) {
  if (plan == NULL) {
    return;
  }
  iree_vm_list_release(plan->expected_outputs);
  iree_vm_list_release(plan->inputs);
  *plan = (loom_run_vm_invocation_plan_t){0};
}

void loom_run_vm_invocation_request_initialize(
    loom_run_vm_invocation_request_t* out_request) {
  IREE_ASSERT_ARGUMENT(out_request);
  *out_request = (loom_run_vm_invocation_request_t){0};
  loom_run_vm_invocation_options_initialize(&out_request->options);
}

void loom_run_vm_invocation_result_initialize(
    iree_allocator_t allocator, loom_run_vm_invocation_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(out_result);
  *out_result = (loom_run_vm_invocation_result_t){0};
  iree_string_builder_initialize(allocator, &out_result->output);
}

void loom_run_vm_invocation_result_deinitialize(
    loom_run_vm_invocation_result_t* result) {
  if (result == NULL) {
    return;
  }
  iree_string_builder_deinitialize(&result->output);
  *result = (loom_run_vm_invocation_result_t){0};
}

static iree_status_t loom_run_vm_value_specs_validate(
    const loom_run_vm_value_specs_t* specs, iree_string_view_t list_name) {
  IREE_ASSERT_ARGUMENT(specs);
  if (specs->count > 0 && specs->values == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s specs require values", (int)list_name.size,
                            list_name.data);
  }
  return iree_ok_status();
}

static iree_string_view_list_t loom_run_vm_value_specs_list(
    const loom_run_vm_value_specs_t* specs) {
  IREE_ASSERT_ARGUMENT(specs);
  return (iree_string_view_list_t){
      .count = specs->count,
      .values = specs->values,
  };
}

iree_status_t loom_run_vm_invocation_plan_prepare_from_specs(
    const loom_run_vm_invocation_plan_prepare_request_t* request,
    iree_allocator_t allocator, loom_run_vm_invocation_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(request);
  IREE_ASSERT_ARGUMENT(request->options);
  IREE_ASSERT_ARGUMENT(out_plan);
  loom_run_vm_invocation_plan_initialize(out_plan);
  IREE_RETURN_IF_ERROR(loom_run_vm_value_specs_validate(
      &request->options->inputs, IREE_SV("VM input")));
  IREE_RETURN_IF_ERROR(loom_run_vm_value_specs_validate(
      &request->options->outputs, IREE_SV("VM output")));
  IREE_RETURN_IF_ERROR(loom_run_vm_value_specs_validate(
      &request->options->expected_outputs, IREE_SV("expected VM output")));

  iree_hal_allocator_t* heap_allocator = NULL;
  iree_status_t status = iree_tooling_parse_variants(
      request->arguments_cconv,
      loom_run_vm_value_specs_list(&request->options->inputs), request->device,
      request->device_allocator, allocator, &out_plan->inputs);
  if (iree_status_is_ok(status) &&
      request->options->expected_outputs.count != 0) {
    status = iree_hal_allocator_create_heap(IREE_SV("heap"), allocator,
                                            allocator, &heap_allocator);
  }
  if (iree_status_is_ok(status) &&
      request->options->expected_outputs.count != 0) {
    status = iree_tooling_parse_variants(
        request->results_cconv,
        loom_run_vm_value_specs_list(&request->options->expected_outputs),
        request->device, heap_allocator, allocator,
        &out_plan->expected_outputs);
  }
  if (iree_status_is_ok(status)) {
    out_plan->max_output_element_count =
        request->options->max_output_element_count == 0
            ? LOOM_RUN_VM_DEFAULT_MAX_OUTPUT_ELEMENT_COUNT
            : request->options->max_output_element_count;
    if (request->options->expected_outputs.count != 0) {
      out_plan->output_mode = LOOM_RUN_VM_OUTPUT_PLAN_MODE_COMPARE_EXPECTED;
    } else if (request->options->outputs.count != 0) {
      out_plan->output_mode = LOOM_RUN_VM_OUTPUT_PLAN_MODE_WRITE_SPECS;
      out_plan->output_specs = request->options->outputs;
    } else {
      out_plan->output_mode = LOOM_RUN_VM_OUTPUT_PLAN_MODE_FORMAT_ALL;
    }
  } else {
    loom_run_vm_invocation_plan_deinitialize(out_plan);
  }
  iree_hal_allocator_release(heap_allocator);
  return status;
}

static iree_status_t loom_run_vm_load_archive_module(
    const loom_run_vm_runtime_t* runtime,
    const loom_ireevm_module_archive_t* archive, iree_allocator_t allocator,
    iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(archive);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  if (runtime->instance == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM runtime is not initialized");
  }
  if (archive->data == NULL || archive->data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM archive is empty");
  }
  return iree_vm_bytecode_module_create(
      runtime->instance, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
      iree_make_const_byte_span(archive->data, archive->data_length),
      iree_allocator_null(), allocator, out_module);
}

static iree_status_t loom_run_vm_create_context(
    const loom_run_vm_runtime_t* runtime, iree_vm_module_t* module,
    iree_string_view_t default_device_uri, iree_allocator_t allocator,
    iree_vm_context_t** out_context, iree_hal_device_t** out_device,
    iree_hal_allocator_t** out_device_allocator) {
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(out_context);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_ASSERT_ARGUMENT(out_device_allocator);
  *out_context = NULL;
  *out_device = NULL;
  *out_device_allocator = NULL;

  iree_vm_module_t* modules[] = {module};
  return iree_tooling_create_context_from_flags(
      runtime->instance, IREE_ARRAYSIZE(modules), modules, default_device_uri,
      allocator, out_context, out_device, out_device_allocator);
}

static iree_status_t loom_run_vm_lookup_function(
    iree_vm_module_t* module, iree_string_view_t function_name,
    iree_vm_function_t* out_function) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(out_function);
  if (iree_string_view_is_empty(function_name)) {
    return iree_tooling_find_single_exported_function(module, out_function);
  }
  return iree_vm_module_lookup_function_by_name(
      module, IREE_VM_FUNCTION_LINKAGE_EXPORT, function_name, out_function);
}

static iree_status_t loom_run_vm_annotate_status_with_function_decl(
    iree_status_t status, iree_vm_function_t function) {
  if (iree_status_is_ok(status) || iree_vm_function_is_null(function)) {
    return status;
  }
  iree_string_view_t declaration = iree_vm_function_lookup_attr_by_name(
      &function, IREE_SV("iree.abi.declaration"));
  if (iree_string_view_is_empty(declaration)) {
    return status;
  }
  return iree_status_annotate_f(status, "`%.*s`", (int)declaration.size,
                                declaration.data);
}

static iree_status_t loom_run_vm_append_stream_block(
    void* user_data, iree_const_byte_span_t block) {
  iree_string_builder_t* builder = (iree_string_builder_t*)user_data;
  return iree_string_builder_append_string(
      builder,
      iree_make_string_view((const char*)block.data, block.data_length));
}

static iree_status_t loom_run_vm_write_outputs_to_result(
    iree_vm_list_t* outputs, const loom_run_vm_value_specs_t* output_specs,
    iree_host_size_t max_output_element_count, iree_allocator_t allocator,
    loom_run_vm_invocation_result_t* result) {
  IREE_ASSERT_ARGUMENT(outputs);
  IREE_ASSERT_ARGUMENT(output_specs);
  IREE_ASSERT_ARGUMENT(result);

  iree_io_stream_t* output_stream = NULL;
  iree_status_t status = iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE, LOOM_RUN_VM_OUTPUT_STREAM_BLOCK_SIZE,
      allocator, &output_stream);
  if (iree_status_is_ok(status)) {
    status = iree_tooling_write_variants(
        outputs, loom_run_vm_value_specs_list(output_specs),
        max_output_element_count, output_stream, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_vec_stream_enumerate_blocks(
        output_stream, loom_run_vm_append_stream_block, &result->output);
  }
  iree_io_stream_release(output_stream);
  return status;
}

static iree_status_t loom_run_vm_process_outputs(
    const loom_run_vm_invocation_plan_t* plan, iree_vm_list_t* outputs,
    iree_allocator_t allocator, loom_run_vm_invocation_result_t* result) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(outputs);
  IREE_ASSERT_ARGUMENT(result);

  switch (plan->output_mode) {
    case LOOM_RUN_VM_OUTPUT_PLAN_MODE_FORMAT_ALL:
      return iree_tooling_format_variants(IREE_SV("result"), outputs,
                                          plan->max_output_element_count,
                                          &result->output);
    case LOOM_RUN_VM_OUTPUT_PLAN_MODE_WRITE_SPECS:
      return loom_run_vm_write_outputs_to_result(outputs, &plan->output_specs,
                                                 plan->max_output_element_count,
                                                 allocator, result);
    case LOOM_RUN_VM_OUTPUT_PLAN_MODE_COMPARE_EXPECTED:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown VM output plan mode %d",
                              (int)plan->output_mode);
  }
  if (plan->expected_outputs == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM expected output comparison requires values");
  }

  iree_status_t status = iree_ok_status();
  const bool did_match = iree_tooling_compare_variant_lists_and_append(
      plan->expected_outputs, outputs, allocator, &result->output);
  if (did_match) {
    status = iree_string_builder_append_cstring(
        &result->output,
        "[SUCCESS] all function outputs matched their expected values.\n");
  }
  result->exit_code = did_match ? 0 : 1;
  return status;
}

iree_status_t loom_run_vm_invocation_invoke_plan(
    const loom_run_vm_invocation_plan_t* plan, iree_vm_context_t* context,
    iree_vm_function_t function, iree_hal_device_t* device,
    iree_hal_allocator_t* device_allocator, iree_allocator_t allocator,
    loom_run_vm_invocation_result_t* result) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(result);
  iree_string_builder_reset(&result->output);
  result->exit_code = 0;
  if (plan->inputs == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM invocation plan requires inputs");
  }
  if (plan->output_mode == LOOM_RUN_VM_OUTPUT_PLAN_MODE_WRITE_SPECS) {
    IREE_RETURN_IF_ERROR(loom_run_vm_value_specs_validate(
        &plan->output_specs, IREE_SV("VM output")));
  } else if (plan->output_mode ==
             LOOM_RUN_VM_OUTPUT_PLAN_MODE_COMPARE_EXPECTED) {
    if (plan->expected_outputs == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "VM expected output comparison requires values");
    }
  } else if (plan->output_mode != LOOM_RUN_VM_OUTPUT_PLAN_MODE_FORMAT_ALL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown VM output plan mode %d",
                            (int)plan->output_mode);
  }
  if (device != NULL && device_allocator == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM device invocation requires a HAL allocator");
  }

  iree_vm_list_t* inputs = NULL;
  iree_hal_fence_t* finish_fence = NULL;
  iree_vm_list_t* outputs = NULL;

  iree_status_t status = iree_vm_list_clone(plan->inputs, allocator, &inputs);
  if (iree_status_is_ok(status) && device == NULL) {
    iree_string_view_t model = iree_vm_function_lookup_attr_by_name(
        &function, IREE_SV("iree.abi.model"));
    if (iree_string_view_equal(model, IREE_SV("coarse-fences"))) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "async VM invocation requires a HAL device");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_tooling_append_async_fences(
        inputs, function, device, /*wait_fence=*/NULL, &finish_fence);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 16,
                                 allocator, &outputs);
  }
  if (iree_status_is_ok(status)) {
    iree_string_view_t function_name = iree_vm_function_name(&function);
    status = iree_string_builder_append_format(&result->output, "EXEC @%.*s\n",
                                               (int)function_name.size,
                                               function_name.data);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                            /*policy=*/NULL, inputs, outputs, allocator);
  }
  if (iree_status_is_ok(status) && finish_fence != NULL) {
    status = iree_hal_fence_wait(finish_fence, iree_infinite_timeout(),
                                 IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (iree_status_is_ok(status) && device != NULL) {
    iree_hal_buffer_params_t target_params = {
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
        .min_alignment = 0,
    };
    status = iree_tooling_transfer_variants(outputs, device, device_allocator,
                                            target_params, /*wait_fence=*/NULL,
                                            /*signal_fence=*/NULL);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_vm_process_outputs(plan, outputs, allocator, result);
  }

  iree_vm_list_release(outputs);
  iree_hal_fence_release(finish_fence);
  iree_vm_list_release(inputs);
  return status;
}

static iree_status_t loom_run_vm_invoke_function(
    const loom_run_vm_invocation_options_t* options, iree_vm_context_t* context,
    iree_vm_function_t function, iree_hal_device_t* device,
    iree_hal_allocator_t* device_allocator, iree_allocator_t allocator,
    loom_run_vm_invocation_result_t* result) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(result);

  iree_vm_function_signature_t signature =
      iree_vm_function_signature(&function);
  iree_string_view_t arguments_cconv = iree_string_view_empty();
  iree_string_view_t results_cconv = iree_string_view_empty();
  iree_status_t status = iree_vm_function_call_get_cconv_fragments(
      &signature, &arguments_cconv, &results_cconv);

  loom_run_vm_invocation_plan_t plan = {0};
  if (iree_status_is_ok(status)) {
    const loom_run_vm_invocation_plan_prepare_request_t prepare_request = {
        .options = options,
        .arguments_cconv = arguments_cconv,
        .results_cconv = results_cconv,
        .device = device,
        .device_allocator = device_allocator,
    };
    status = loom_run_vm_invocation_plan_prepare_from_specs(&prepare_request,
                                                            allocator, &plan);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_vm_invocation_invoke_plan(
        &plan, context, function, device, device_allocator, allocator, result);
  }
  loom_run_vm_invocation_plan_deinitialize(&plan);
  return status;
}

iree_status_t loom_run_vm_invocation_run(
    const loom_run_vm_invocation_request_t* request, iree_allocator_t allocator,
    loom_run_vm_invocation_result_t* result) {
  IREE_ASSERT_ARGUMENT(request);
  IREE_ASSERT_ARGUMENT(request->runtime);
  IREE_ASSERT_ARGUMENT(request->archive);
  IREE_ASSERT_ARGUMENT(result);
  iree_string_builder_reset(&result->output);
  result->exit_code = 0;

  iree_vm_module_t* module = NULL;
  iree_vm_context_t* context = NULL;
  iree_hal_device_t* device = NULL;
  iree_hal_allocator_t* device_allocator = NULL;
  iree_vm_function_t function = {0};

  iree_status_t status = loom_run_vm_load_archive_module(
      request->runtime, request->archive, allocator, &module);
  if (iree_status_is_ok(status)) {
    status = loom_run_vm_create_context(
        request->runtime, module, request->options.default_device_uri,
        allocator, &context, &device, &device_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_vm_lookup_function(module, request->options.function_name,
                                         &function);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_vm_invoke_function(&request->options, context, function,
                                         device, device_allocator, allocator,
                                         result);
  }
  status = loom_run_vm_annotate_status_with_function_decl(status, function);

  iree_hal_allocator_release(device_allocator);
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_module_release(module);
  return status;
}
