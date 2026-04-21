// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_hsaco.h"

#include "loom/target/emit/native/amdgpu/encoding.h"
#include "loom/target/emit/native/amdgpu/hsaco.h"
#include "loom/target/emit/native/amdgpu/kernel_record.h"

static iree_status_t loom_amdgpu_emit_kernel_hsaco_internal(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    const loom_amdgpu_kernel_hsaco_options_t* options, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena) {
  if (options != NULL && options->summary != NULL) {
    *options->summary = (loom_amdgpu_kernel_hsaco_summary_t){0};
  }
  if (stream == NULL || scratch_arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel HSACO output stream and scratch arena are required");
  }
  loom_amdgpu_kernel_record_t record = {0};
  const loom_amdgpu_kernel_record_options_t record_options = {
      .abi_layout = options ? options->abi_layout : NULL,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_build(
      schedule, allocation, &record_options, &record, scratch_arena));

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  const loom_amdgpu_wait_packet_plan_t* wait_packets =
      options ? options->wait_packets : NULL;
  if (wait_packets != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_encode_instruction_stream_with_wait_packets(
            schedule, allocation, wait_packets, &text, scratch_arena));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_encode_instruction_stream(
        schedule, allocation, &text, scratch_arena));
  }

  const loom_amdgpu_hsaco_kernel_t kernel = {
      .metadata = record.metadata,
      .descriptor_options = {.flags = record.descriptor_flags},
      .text = text,
  };
  const loom_target_snapshot_t* snapshot =
      &schedule->target.bundle_storage.snapshot;
  const loom_amdgpu_hsaco_file_t file = {
      .target = record.target_id,
      .target_cpu = snapshot->target_cpu,
      .kernels = &kernel,
      .kernel_count = 1,
  };
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hsaco_write_file(&file, stream, scratch_arena));

  if (options != NULL && options->summary != NULL) {
    const uint64_t wait_packet_count =
        wait_packets ? wait_packets->packet_count : 0;
    *options->summary = (loom_amdgpu_kernel_hsaco_summary_t){
        .instruction_count = schedule->scheduled_node_count + wait_packet_count,
        .text_byte_count = text.data_length,
        .text_storage_byte_count = text.data_length,
        .private_segment_fixed_size =
            record.metadata.private_segment_fixed_size,
        .group_segment_fixed_size = record.metadata.group_segment_fixed_size,
    };
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_kernel_hsaco(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    const loom_amdgpu_kernel_hsaco_options_t* options, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena) {
  return loom_amdgpu_emit_kernel_hsaco_internal(schedule, allocation, options,
                                                stream, scratch_arena);
}
