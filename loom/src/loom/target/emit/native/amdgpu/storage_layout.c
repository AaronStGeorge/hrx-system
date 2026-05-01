// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/storage_layout.h"

#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"

typedef struct loom_amdgpu_storage_layout_state_t {
  // Fixed segment sizes accumulated so far.
  loom_amdgpu_storage_layout_segment_sizes_t sizes;
  // Optional destination records to populate.
  loom_amdgpu_storage_layout_record_t* records;
  // Maximum number of records in |records|.
  iree_host_size_t record_capacity;
  // Number of storage records counted or populated so far.
  iree_host_size_t record_count;
  // Optional storage value to resolve while scanning.
  loom_value_id_t resolve_storage_value_id;
  // Optional resolved reservation destination.
  loom_amdgpu_storage_layout_reservation_t* resolved_reservation;
  // Whether |resolved_reservation| has been populated.
  bool resolved;
} loom_amdgpu_storage_layout_state_t;

static bool loom_amdgpu_storage_layout_u64_is_power_of_two(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

static iree_status_t loom_amdgpu_storage_layout_checked_align_u64(
    uint64_t value, uint64_t alignment, uint64_t* out_aligned_value) {
  *out_aligned_value = 0;
  if (!loom_amdgpu_storage_layout_u64_is_power_of_two(alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU storage layout alignment must be a power of two");
  }
  const uint64_t alignment_mask = alignment - 1;
  if (value > UINT64_MAX - alignment_mask) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU storage layout alignment overflows");
  }
  *out_aligned_value = (value + alignment_mask) & ~alignment_mask;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_storage_layout_visit_reserve(
    const loom_module_t* module, const loom_op_t* reserve_op,
    loom_amdgpu_storage_layout_state_t* state) {
  const int64_t signed_byte_size =
      loom_low_storage_reserve_byte_length(reserve_op);
  const int64_t signed_byte_alignment =
      loom_low_storage_reserve_byte_alignment(reserve_op);
  if (signed_byte_size < 0 || signed_byte_alignment <= 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU storage layout requires positive storage "
                            "sizes and alignments");
  }

  const loom_value_id_t storage_value_id =
      loom_low_storage_reserve_storage(reserve_op);
  const loom_type_t storage_type =
      loom_module_value_type(module, storage_value_id);
  if (!loom_type_is_storage(storage_type)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU storage layout found a reserve with non-storage result type");
  }
  const loom_storage_space_t storage_space =
      loom_type_storage_space(storage_type);

  uint64_t* segment_size = NULL;
  switch (storage_space) {
    case LOOM_STORAGE_SPACE_WORKGROUP:
      segment_size = &state->sizes.group_segment_fixed_size;
      break;
    case LOOM_STORAGE_SPACE_PRIVATE:
      segment_size = &state->sizes.private_segment_fixed_size;
      break;
    case LOOM_STORAGE_SPACE_STACK:
    case LOOM_STORAGE_SPACE_SCRATCH:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AMDGPU native emission does not lower stack/scratch storage yet");
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU storage layout found an unknown storage space");
  }

  const uint64_t byte_size = (uint64_t)signed_byte_size;
  const uint64_t byte_alignment = (uint64_t)signed_byte_alignment;
  uint64_t aligned_segment_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_checked_align_u64(
      *segment_size, byte_alignment, &aligned_segment_size));
  if (aligned_segment_size > UINT64_MAX - byte_size) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU storage layout fixed segment size overflows");
  }

  const loom_amdgpu_storage_layout_reservation_t reservation = {
      .space = storage_space,
      .byte_offset = aligned_segment_size,
      .byte_size = byte_size,
      .byte_alignment = byte_alignment,
  };
  if (state->records != NULL) {
    if (state->record_count >= state->record_capacity) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU storage layout record count changed while building");
    }
    state->records[state->record_count] = (loom_amdgpu_storage_layout_record_t){
        .storage_value_id = storage_value_id,
        .reservation = reservation,
    };
  }
  if (state->resolved_reservation != NULL &&
      state->resolve_storage_value_id == storage_value_id) {
    *state->resolved_reservation = reservation;
    state->resolved = true;
  }
  ++state->record_count;
  *segment_size = aligned_segment_size + byte_size;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_storage_layout_scan(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_amdgpu_storage_layout_record_t* records,
    iree_host_size_t record_capacity, loom_value_id_t resolve_storage_value_id,
    loom_amdgpu_storage_layout_reservation_t* resolved_reservation,
    loom_amdgpu_storage_layout_state_t* out_state) {
  *out_state = (loom_amdgpu_storage_layout_state_t){
      .records = records,
      .record_capacity = record_capacity,
      .resolve_storage_value_id = resolve_storage_value_id,
      .resolved_reservation = resolved_reservation,
  };
  const loom_region_t* body = loom_low_function_const_body(function_op);
  if (body == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU storage layout requires a low function "
                            "body");
  }
  const loom_block_t* block = NULL;
  const loom_op_t* op = NULL;
  loom_region_for_each_block(body, block) {
    loom_block_for_each_op(block, op) {
      if (!loom_low_storage_reserve_isa(op)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_storage_layout_visit_reserve(module, op, out_state));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_storage_layout_collect_segment_sizes(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_amdgpu_storage_layout_segment_sizes_t* out_sizes) {
  *out_sizes = (loom_amdgpu_storage_layout_segment_sizes_t){0};
  loom_amdgpu_storage_layout_state_t state;
  IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_scan(
      module, function_op, NULL, 0, LOOM_VALUE_ID_INVALID, NULL, &state));
  *out_sizes = state.sizes;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_storage_layout_build(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_arena_allocator_t* arena, loom_amdgpu_storage_layout_t* out_layout) {
  *out_layout = (loom_amdgpu_storage_layout_t){0};
  loom_amdgpu_storage_layout_state_t count_state;
  IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_scan(
      module, function_op, NULL, 0, LOOM_VALUE_ID_INVALID, NULL, &count_state));
  loom_amdgpu_storage_layout_record_t* records = NULL;
  if (count_state.record_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, count_state.record_count, sizeof(*records), (void**)&records));
  }
  loom_amdgpu_storage_layout_state_t build_state;
  IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_scan(
      module, function_op, records, count_state.record_count,
      LOOM_VALUE_ID_INVALID, NULL, &build_state));
  if (build_state.record_count != count_state.record_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU storage layout record count changed while building");
  }
  *out_layout = (loom_amdgpu_storage_layout_t){
      .segment_sizes = build_state.sizes,
      .records = records,
      .record_count = build_state.record_count,
  };
  return iree_ok_status();
}

iree_status_t loom_amdgpu_storage_layout_lookup(
    const loom_amdgpu_storage_layout_t* layout,
    loom_value_id_t storage_value_id,
    loom_amdgpu_storage_layout_reservation_t* out_reservation) {
  *out_reservation = (loom_amdgpu_storage_layout_reservation_t){0};
  for (iree_host_size_t i = 0; i < layout->record_count; ++i) {
    const loom_amdgpu_storage_layout_record_t* record = &layout->records[i];
    if (record->storage_value_id != storage_value_id) {
      continue;
    }
    *out_reservation = record->reservation;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "AMDGPU storage layout could not resolve referenced storage value");
}

iree_status_t loom_amdgpu_storage_layout_resolve(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_value_id_t storage_value_id,
    loom_amdgpu_storage_layout_reservation_t* out_reservation) {
  *out_reservation = (loom_amdgpu_storage_layout_reservation_t){0};
  loom_amdgpu_storage_layout_state_t state;
  IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_scan(
      module, function_op, NULL, 0, storage_value_id, out_reservation, &state));
  if (!state.resolved) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "AMDGPU storage layout could not resolve referenced storage value");
  }
  return iree_ok_status();
}
