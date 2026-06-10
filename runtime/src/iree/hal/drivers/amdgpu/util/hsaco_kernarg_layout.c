// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/hsaco_kernarg_layout.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/abi/kernel_args.h"

static bool iree_hal_amdgpu_hsaco_kernarg_layout_arg_kind_is_hidden(
    iree_hal_amdgpu_hsaco_metadata_arg_kind_t kind) {
  return kind == IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_HIDDEN ||
         kind == IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_HIDDEN_NONE;
}

static iree_status_t iree_hal_amdgpu_hsaco_kernarg_layout_check_u16_field(
    iree_string_view_t symbol_name, iree_string_view_t field_name,
    iree_host_size_t value) {
  if (value > IREE_HAL_AMDGPU_HSACO_KERNARG_LAYOUT_MAX_BYTE_LENGTH) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU kernel `%.*s` %.*s %" PRIhsz
        " exceeds runtime kernarg layout limit %u",
        (int)symbol_name.size, symbol_name.data, (int)field_name.size,
        field_name.data, value,
        IREE_HAL_AMDGPU_HSACO_KERNARG_LAYOUT_MAX_BYTE_LENGTH);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_hsaco_kernarg_layout_analyze(
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel,
    iree_host_size_t parameter_capacity,
    iree_hal_amdgpu_hsaco_kernarg_parameter_t* out_parameters,
    iree_hal_amdgpu_hsaco_kernarg_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(kernel);
  IREE_ASSERT_ARGUMENT(out_layout);
  IREE_ASSERT_ARGUMENT(out_parameters || parameter_capacity == 0);
  memset(out_layout, 0, sizeof(*out_layout));

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_kernarg_layout_check_u16_field(
      kernel->symbol_name, IREE_SV("kernarg segment size"),
      kernel->kernarg_segment_size));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_kernarg_layout_check_u16_field(
      kernel->symbol_name, IREE_SV("kernarg segment alignment"),
      kernel->kernarg_segment_alignment));

  iree_host_size_t parameter_count = 0;
  iree_host_size_t binding_count = 0;
  iree_host_size_t constant_byte_length = 0;
  iree_host_size_t explicit_kernarg_size = 0;
  iree_host_size_t implicit_args_offset = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < kernel->arg_count; ++i) {
    const iree_hal_amdgpu_hsaco_metadata_arg_t* arg = &kernel->args[i];
    if (iree_hal_amdgpu_hsaco_kernarg_layout_arg_kind_is_hidden(arg->kind)) {
      implicit_args_offset =
          iree_min(implicit_args_offset, (iree_host_size_t)arg->offset);
      continue;
    }

    iree_host_size_t arg_end = 0;
    if (!iree_host_size_checked_add(arg->offset, arg->size, &arg_end)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU kernel `%.*s` argument %" PRIhsz " range overflows",
          (int)kernel->symbol_name.size, kernel->symbol_name.data, i);
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_kernarg_layout_check_u16_field(
        kernel->symbol_name, IREE_SV("argument end offset"), arg_end));
    if (arg_end > kernel->kernarg_segment_size) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel `%.*s` argument %" PRIhsz " ends at %" PRIhsz
          " beyond kernarg segment size %u",
          (int)kernel->symbol_name.size, kernel->symbol_name.data, i, arg_end,
          kernel->kernarg_segment_size);
    }

    const iree_host_size_t current_parameter = parameter_count++;
    iree_hal_amdgpu_hsaco_kernarg_parameter_t parameter = {0};
    parameter.kernarg_offset = (uint16_t)arg->offset;
    parameter.byte_length = (uint16_t)arg->size;
    switch (arg->kind) {
      case IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER: {
        if (arg->size != sizeof(uint64_t)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "AMDGPU kernel `%.*s` global_buffer argument %" PRIhsz
              " has unsupported size %u",
              (int)kernel->symbol_name.size, kernel->symbol_name.data, i,
              arg->size);
        }
        IREE_RETURN_IF_ERROR(
            iree_hal_amdgpu_hsaco_kernarg_layout_check_u16_field(
                kernel->symbol_name, IREE_SV("binding ordinal"),
                binding_count));
        parameter.kind = IREE_HAL_AMDGPU_HSACO_KERNARG_PARAMETER_KIND_BINDING;
        parameter.source_offset = (uint16_t)binding_count++;
        break;
      }
      case IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE: {
        IREE_RETURN_IF_ERROR(
            iree_hal_amdgpu_hsaco_kernarg_layout_check_u16_field(
                kernel->symbol_name, IREE_SV("constant source offset"),
                constant_byte_length));
        parameter.kind = IREE_HAL_AMDGPU_HSACO_KERNARG_PARAMETER_KIND_CONSTANT;
        parameter.source_offset = (uint16_t)constant_byte_length;
        if (!iree_host_size_checked_add(constant_byte_length, arg->size,
                                        &constant_byte_length)) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "AMDGPU kernel `%.*s` constant byte length overflows",
              (int)kernel->symbol_name.size, kernel->symbol_name.data);
        }
        break;
      }
      default:
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU kernel `%.*s` argument %" PRIhsz
            " uses unsupported dispatchable value_kind `%.*s`",
            (int)kernel->symbol_name.size, kernel->symbol_name.data, i,
            (int)arg->value_kind.size, arg->value_kind.data);
    }

    if (out_parameters) {
      if (current_parameter >= parameter_capacity) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "AMDGPU kernel `%.*s` kernarg parameter output capacity too small",
            (int)kernel->symbol_name.size, kernel->symbol_name.data);
      }
      out_parameters[current_parameter] = parameter;
    }
    explicit_kernarg_size = iree_max(explicit_kernarg_size, arg_end);
  }

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_kernarg_layout_check_u16_field(
      kernel->symbol_name, IREE_SV("parameter count"), parameter_count));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_kernarg_layout_check_u16_field(
      kernel->symbol_name, IREE_SV("binding count"), binding_count));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_kernarg_layout_check_u16_field(
      kernel->symbol_name, IREE_SV("constant byte length"),
      constant_byte_length));

  iree_host_size_t total_kernarg_size = kernel->kernarg_segment_size;
  if (implicit_args_offset != IREE_HOST_SIZE_MAX) {
    if (explicit_kernarg_size > implicit_args_offset) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel `%.*s` has visible arguments interleaved with hidden "
          "implicit kernargs",
          (int)kernel->symbol_name.size, kernel->symbol_name.data);
    }
    if (!iree_host_size_has_alignment(
            implicit_args_offset,
            iree_alignof(iree_amdgpu_kernel_implicit_args_t))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel `%.*s` implicit kernarg suffix is not aligned",
          (int)kernel->symbol_name.size, kernel->symbol_name.data);
    }
    iree_host_size_t implicit_args_end = 0;
    if (!iree_host_size_checked_add(
            implicit_args_offset,
            (iree_host_size_t)IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE,
            &implicit_args_end)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU kernel `%.*s` implicit kernarg suffix overflows",
          (int)kernel->symbol_name.size, kernel->symbol_name.data);
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_kernarg_layout_check_u16_field(
        kernel->symbol_name, IREE_SV("implicit kernarg suffix end"),
        implicit_args_end));
    total_kernarg_size = iree_max(total_kernarg_size, implicit_args_end);
    out_layout->flags |=
        IREE_HAL_AMDGPU_HSACO_KERNARG_LAYOUT_FLAG_IMPLICIT_ARGS;
    out_layout->implicit_args_offset = (uint16_t)implicit_args_offset;
  } else {
    out_layout->implicit_args_offset = UINT16_MAX;
  }

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_hsaco_kernarg_layout_check_u16_field(
      kernel->symbol_name, IREE_SV("total kernarg size"), total_kernarg_size));
  out_layout->parameter_count = (uint16_t)parameter_count;
  out_layout->binding_count = (uint16_t)binding_count;
  out_layout->constant_byte_length = (uint16_t)constant_byte_length;
  out_layout->explicit_kernarg_size = (uint16_t)explicit_kernarg_size;
  out_layout->total_kernarg_size = (uint16_t)total_kernarg_size;
  out_layout->kernarg_alignment = (uint16_t)kernel->kernarg_segment_alignment;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_hsaco_kernarg_layout_calculate(
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel,
    iree_hal_amdgpu_hsaco_kernarg_layout_t* out_layout) {
  return iree_hal_amdgpu_hsaco_kernarg_layout_analyze(
      kernel, /*parameter_capacity=*/0, /*out_parameters=*/NULL, out_layout);
}

iree_status_t iree_hal_amdgpu_hsaco_kernarg_layout_populate_parameters(
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel,
    iree_host_size_t parameter_capacity,
    iree_hal_amdgpu_hsaco_kernarg_parameter_t* out_parameters) {
  IREE_ASSERT_ARGUMENT(out_parameters || parameter_capacity == 0);
  iree_hal_amdgpu_hsaco_kernarg_layout_t layout;
  return iree_hal_amdgpu_hsaco_kernarg_layout_analyze(
      kernel, parameter_capacity, out_parameters, &layout);
}
