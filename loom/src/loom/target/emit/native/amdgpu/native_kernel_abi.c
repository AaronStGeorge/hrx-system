// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/native_kernel_abi.h"

#include <inttypes.h>

static iree_status_t loom_amdgpu_native_kernel_abi_validate(
    const loom_amdgpu_native_kernel_abi_t* abi) {
  if (abi == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU native kernel ABI is required");
  }
  if (abi->kernarg_segment_alignment == 0 ||
      !iree_host_size_is_power_of_two(abi->kernarg_segment_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU native kernel ABI kernarg segment alignment must be a power of "
        "two");
  }
  if (abi->argument_count != 0 && abi->arguments == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU native kernel ABI argument array is required");
  }

  iree_host_size_t previous_argument_end = 0;
  for (iree_host_size_t i = 0; i < abi->argument_count; ++i) {
    const loom_amdgpu_native_kernel_abi_argument_t* argument =
        &abi->arguments[i];
    if (argument->alignment != 0 &&
        !iree_host_size_is_power_of_two(argument->alignment)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU native kernel ABI argument %" PRIhsz
                              " alignment must be a power of two",
                              i);
    }
    if (argument->alignment != 0 &&
        !iree_host_size_has_alignment(argument->offset, argument->alignment)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU native kernel ABI argument %" PRIhsz
                              " offset %" PRIu32 " is not aligned to %" PRIu32,
                              i, argument->offset, argument->alignment);
    }

    iree_host_size_t argument_end = 0;
    if (!iree_host_size_checked_add(argument->offset, argument->size,
                                    &argument_end) ||
        argument_end > abi->kernarg_segment_size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU native kernel ABI argument %" PRIhsz
                              " exceeds kernarg segment size %" PRIu32,
                              i, abi->kernarg_segment_size);
    }
    if (i > 0 && argument->offset < previous_argument_end) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU native kernel ABI argument %" PRIhsz
                              " overlaps a previous argument",
                              i);
    }
    previous_argument_end = argument_end;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_native_kernel_abi_apply_to_hsaco_kernel(
    const loom_amdgpu_native_kernel_abi_t* abi,
    loom_amdgpu_hsaco_kernel_t* kernel, iree_arena_allocator_t* arena) {
  if (kernel == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HSACO kernel is required");
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_native_kernel_abi_validate(abi));

  loom_amdgpu_metadata_argument_t* arguments = NULL;
  if (abi->argument_count != 0) {
    if (arena == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU native kernel ABI argument cloning requires an arena");
    }
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, abi->argument_count, sizeof(arguments[0]), (void**)&arguments));
    for (iree_host_size_t i = 0; i < abi->argument_count; ++i) {
      const loom_amdgpu_native_kernel_abi_argument_t* argument =
          &abi->arguments[i];
      arguments[i] = (loom_amdgpu_metadata_argument_t){
          .name = argument->name,
          .offset = argument->offset,
          .size = argument->size,
          .alignment = argument->alignment,
          .kind = argument->kind,
          .address_space = argument->address_space,
          .access = argument->access,
          .actual_access = argument->actual_access,
      };
    }
  }

  kernel->metadata.kernarg_segment_size = abi->kernarg_segment_size;
  kernel->metadata.kernarg_segment_alignment = abi->kernarg_segment_alignment;
  kernel->metadata.arguments = arguments;
  kernel->metadata.argument_count = abi->argument_count;
  kernel->descriptor_options.flags = abi->descriptor_flags;
  return iree_ok_status();
}
