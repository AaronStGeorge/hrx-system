// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/slot_layout.h"

#include "loom/codegen/low/function.h"

typedef struct loom_amdgpu_slot_layout_state_t {
  // Function symbol whose slots are being laid out.
  loom_symbol_ref_t function_ref;
  // Fixed segment sizes accumulated so far.
  loom_amdgpu_slot_layout_segment_sizes_t sizes;
  // Optional destination records to populate.
  loom_amdgpu_slot_layout_record_t* records;
  // Maximum number of records in |records|.
  iree_host_size_t record_capacity;
  // Number of slot records counted or populated so far.
  iree_host_size_t record_count;
} loom_amdgpu_slot_layout_state_t;

static bool loom_amdgpu_slot_layout_symbol_ref_equal(loom_symbol_ref_t lhs,
                                                     loom_symbol_ref_t rhs) {
  return lhs.module_id == rhs.module_id && lhs.symbol_id == rhs.symbol_id;
}

static bool loom_amdgpu_slot_layout_u64_is_power_of_two(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

static iree_status_t loom_amdgpu_slot_layout_checked_align_u64(
    uint64_t value, uint64_t alignment, uint64_t* out_aligned_value) {
  IREE_ASSERT_ARGUMENT(out_aligned_value);
  *out_aligned_value = 0;
  if (!loom_amdgpu_slot_layout_u64_is_power_of_two(alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU low slot layout alignment must be a power of two");
  }
  const uint64_t alignment_mask = alignment - 1;
  if (value > UINT64_MAX - alignment_mask) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU low slot layout alignment overflows");
  }
  *out_aligned_value = (value + alignment_mask) & ~alignment_mask;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_slot_layout_visit_slot(
    const loom_op_t* slot_op, loom_amdgpu_slot_layout_state_t* state) {
  const int64_t signed_slot_size = loom_low_slot_size(slot_op);
  const int64_t signed_slot_alignment = loom_low_slot_align(slot_op);
  if (signed_slot_size < 0 || signed_slot_alignment <= 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU low slot layout requires positive slot sizes and alignments");
  }

  uint64_t* segment_size = NULL;
  switch (loom_low_slot_space(slot_op)) {
    case LOOM_LOW_SLOT_SPACE_LDS:
      segment_size = &state->sizes.group_segment_fixed_size;
      break;
    case LOOM_LOW_SLOT_SPACE_PRIVATE:
      segment_size = &state->sizes.private_segment_fixed_size;
      break;
    case LOOM_LOW_SLOT_SPACE_STACK:
    case LOOM_LOW_SLOT_SPACE_SCRATCH:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AMDGPU native emission does not lower stack/scratch low slots yet");
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU low slot layout found an unknown storage space");
  }

  const uint64_t slot_size = (uint64_t)signed_slot_size;
  const uint64_t slot_alignment = (uint64_t)signed_slot_alignment;
  uint64_t aligned_segment_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_slot_layout_checked_align_u64(
      *segment_size, slot_alignment, &aligned_segment_size));
  if (aligned_segment_size > UINT64_MAX - slot_size) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU low slot layout fixed segment size overflows");
  }

  if (state->records != NULL) {
    if (state->record_count >= state->record_capacity) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU low slot layout record count changed while building");
    }
    state->records[state->record_count] = (loom_amdgpu_slot_layout_record_t){
        .slot_ref = loom_low_slot_symbol(slot_op),
        .slot =
            {
                .space = (loom_low_slot_space_t)loom_low_slot_space(slot_op),
                .byte_offset = aligned_segment_size,
                .byte_size = slot_size,
                .byte_alignment = slot_alignment,
            },
    };
  }
  ++state->record_count;
  *segment_size = aligned_segment_size + slot_size;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_slot_layout_scan(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_amdgpu_slot_layout_record_t* records, iree_host_size_t record_capacity,
    loom_amdgpu_slot_layout_state_t* out_state) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(function_op);
  IREE_ASSERT_ARGUMENT(out_state);
  *out_state = (loom_amdgpu_slot_layout_state_t){
      .function_ref = loom_low_function_callee(function_op),
      .records = records,
      .record_capacity = record_capacity,
  };
  const loom_block_t* module_block =
      loom_region_const_entry_block(module->body);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(module_block, op) {
    if (!loom_low_slot_isa(op) ||
        !loom_amdgpu_slot_layout_symbol_ref_equal(loom_low_slot_function(op),
                                                  out_state->function_ref)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_slot_layout_visit_slot(op, out_state));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_slot_layout_collect_segment_sizes(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_amdgpu_slot_layout_segment_sizes_t* out_sizes) {
  IREE_ASSERT_ARGUMENT(out_sizes);
  *out_sizes = (loom_amdgpu_slot_layout_segment_sizes_t){0};
  loom_amdgpu_slot_layout_state_t state;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_slot_layout_scan(module, function_op, NULL, 0, &state));
  *out_sizes = state.sizes;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_slot_layout_build(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_arena_allocator_t* arena, loom_amdgpu_slot_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_layout);
  *out_layout = (loom_amdgpu_slot_layout_t){0};
  loom_amdgpu_slot_layout_state_t count_state;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_slot_layout_scan(module, function_op, NULL, 0, &count_state));
  loom_amdgpu_slot_layout_record_t* records = NULL;
  if (count_state.record_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, count_state.record_count, sizeof(*records), (void**)&records));
  }
  loom_amdgpu_slot_layout_state_t build_state;
  IREE_RETURN_IF_ERROR(loom_amdgpu_slot_layout_scan(
      module, function_op, records, count_state.record_count, &build_state));
  if (build_state.record_count != count_state.record_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU low slot layout record count changed while building");
  }
  *out_layout = (loom_amdgpu_slot_layout_t){
      .segment_sizes = build_state.sizes,
      .records = records,
      .record_count = build_state.record_count,
  };
  return iree_ok_status();
}

iree_status_t loom_amdgpu_slot_layout_lookup_slot(
    const loom_amdgpu_slot_layout_t* layout, loom_symbol_ref_t slot_ref,
    loom_amdgpu_slot_layout_slot_t* out_slot) {
  IREE_ASSERT_ARGUMENT(layout);
  IREE_ASSERT_ARGUMENT(out_slot);
  *out_slot = (loom_amdgpu_slot_layout_slot_t){0};
  for (iree_host_size_t i = 0; i < layout->record_count; ++i) {
    const loom_amdgpu_slot_layout_record_t* record = &layout->records[i];
    if (!loom_amdgpu_slot_layout_symbol_ref_equal(record->slot_ref, slot_ref)) {
      continue;
    }
    *out_slot = record->slot;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "AMDGPU low slot layout could not resolve referenced slot");
}
