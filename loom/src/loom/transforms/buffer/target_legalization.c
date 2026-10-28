// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/buffer/target_legalization.h"

#include <stdint.h>

#include "iree/base/internal/math.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ir/types.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_table.h"

static iree_status_t loom_buffer_legalize_build_offset_constant(
    loom_builder_t* builder, int64_t value, loom_location_id_t location,
    loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64(value), loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
      location, &op));
  *out_value = loom_index_constant_result(op);
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize_build_scalar_constant(
    loom_builder_t* builder, loom_scalar_type_t scalar_type, int64_t value,
    loom_location_id_t location, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(builder, loom_attr_i64(value),
                                                  loom_type_scalar(scalar_type),
                                                  location, &op));
  *out_value = loom_scalar_constant_result(op);
  return iree_ok_status();
}

// Builds the physical address of one byte in a bulk operation range.
static iree_status_t loom_buffer_legalize_build_byte_offset(
    loom_builder_t* builder, loom_value_id_t base_offset,
    loom_value_id_t relative_offset, loom_location_id_t location,
    loom_value_id_t* out_byte_offset) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_add_build(
      builder, base_offset, relative_offset,
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), location, &op));
  *out_byte_offset = loom_index_add_result(op);
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize_build_byte_load(
    loom_builder_t* builder, loom_value_id_t source,
    loom_value_id_t source_offset, loom_value_id_t relative_offset,
    loom_location_id_t location, loom_value_id_t* out_value) {
  loom_value_id_t byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_byte_offset(
      builder, source_offset, relative_offset, location, &byte_offset));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_buffer_load_i8_u_build(builder, source, byte_offset, location, &op));
  *out_value = loom_buffer_load_i8_u_result(op);
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize_build_byte_store(
    loom_builder_t* builder, loom_value_id_t value, loom_value_id_t target,
    loom_value_id_t target_offset, loom_value_id_t relative_offset,
    loom_location_id_t location) {
  loom_value_id_t byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_byte_offset(
      builder, target_offset, relative_offset, location, &byte_offset));
  loom_op_t* op = NULL;
  return loom_buffer_store_i8_build(builder, value, target, byte_offset,
                                    location, &op);
}

// Selects transfers from producer-retained root and offset alignment facts.
// No stronger alignment requirement is introduced by the generated views.
static uint8_t loom_buffer_legalize_transfer_widths(
    const loom_target_legalization_context_t* context, loom_value_id_t buffer,
    loom_value_id_t offset) {
  const loom_target_bulk_memory_policy_t policy = context->bulk_memory_policy;
  if ((policy.scalar_byte_widths | policy.vector_byte_widths) == 0) {
    return 1;
  }
  uint64_t alignment = 1;
  loom_value_fact_buffer_reference_t reference = {0};
  if (context->fact_table != NULL &&
      loom_value_facts_query_buffer_reference(
          &context->fact_table->context,
          loom_value_fact_table_lookup(context->fact_table, buffer),
          &reference)) {
    const loom_value_facts_t offset_facts =
        loom_value_fact_table_lookup(context->fact_table, offset);
    alignment = reference.minimum_alignment;
    if (!loom_value_facts_is_zero(offset_facts)) {
      alignment = iree_math_gcd_u64(alignment, offset_facts.known_divisor);
    }
  }
  uint8_t widths = 1;
  for (uint32_t width = 2; width <= 64; width *= 2) {
    const uint32_t required_alignment = iree_min(
        width, (policy.scalar_byte_widths & width) ? policy.scalar_alignment
                                                   : policy.vector_alignment);
    if (((policy.scalar_byte_widths | policy.vector_byte_widths) & width) &&
        alignment >= required_alignment) {
      widths |= width;
    }
  }
  return widths;
}

static loom_type_t loom_buffer_legalize_transfer_type(
    const loom_target_bulk_memory_policy_t* policy, uint32_t width) {
  if (policy->vector_byte_widths & width) {
    return loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                               width / 4, 0);
  }
  return loom_type_scalar(width == 8   ? LOOM_SCALAR_TYPE_I64
                          : width == 4 ? LOOM_SCALAR_TYPE_I32
                          : width == 2 ? LOOM_SCALAR_TYPE_I16
                                       : LOOM_SCALAR_TYPE_I8);
}

static iree_status_t loom_buffer_legalize_build_transfer_view(
    loom_builder_t* builder, loom_value_id_t buffer, loom_value_id_t offset,
    loom_value_id_t relative_offset, loom_type_t transfer_type,
    loom_location_id_t location, loom_value_id_t* out_view) {
  loom_value_id_t byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_byte_offset(
      builder, offset, relative_offset, location, &byte_offset));
  const uint32_t lane_count =
      loom_type_is_vector(transfer_type)
          ? loom_type_dim_static_size_at(transfer_type, 0)
          : 1;
  loom_type_t view_type = loom_type_view_with_alignment(
      loom_type_shaped_1d(LOOM_TYPE_VIEW, loom_type_element_type(transfer_type),
                          lane_count, 0),
      1);
  loom_op_t* view_op = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_view_build(builder, buffer, byte_offset,
                                              view_type, location, &view_op));
  *out_view = loom_buffer_view_result(view_op);
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize_build_load(
    loom_builder_t* builder, loom_value_id_t source,
    loom_value_id_t source_offset, loom_value_id_t relative_offset,
    loom_type_t transfer_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  if (loom_type_equal(transfer_type, loom_type_scalar(LOOM_SCALAR_TYPE_I8))) {
    return loom_buffer_legalize_build_byte_load(
        builder, source, source_offset, relative_offset, location, out_value);
  }
  loom_value_id_t view = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_transfer_view(
      builder, source, source_offset, relative_offset, transfer_type, location,
      &view));
  const int64_t origin = 0;
  loom_op_t* load = NULL;
  if (loom_type_is_vector(transfer_type)) {
    IREE_RETURN_IF_ERROR(loom_vector_load_build(builder, 0, 0, view, NULL, 0,
                                                &origin, 1, 0, 0, transfer_type,
                                                location, &load));
    *out_value = loom_vector_load_result(load);
  } else {
    IREE_RETURN_IF_ERROR(loom_view_load_build(builder, 0, 0, view, NULL, 0,
                                              &origin, 1, 0, 0, transfer_type,
                                              location, &load));
    *out_value = loom_view_load_result(load);
  }
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize_build_store(
    loom_builder_t* builder, loom_value_id_t value, loom_value_id_t target,
    loom_value_id_t target_offset, loom_value_id_t relative_offset,
    loom_type_t transfer_type, loom_location_id_t location) {
  if (loom_type_equal(transfer_type, loom_type_scalar(LOOM_SCALAR_TYPE_I8))) {
    return loom_buffer_legalize_build_byte_store(
        builder, value, target, target_offset, relative_offset, location);
  }
  loom_value_id_t view = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_transfer_view(
      builder, target, target_offset, relative_offset, transfer_type, location,
      &view));
  const int64_t origin = 0;
  loom_op_t* store = NULL;
  if (loom_type_is_vector(transfer_type)) {
    return loom_vector_store_build(builder, 0, 0, value, view, NULL, 0, &origin,
                                   1, 0, 0, location, &store);
  }
  return loom_view_store_build(builder, 0, 0, value, view, NULL, 0, &origin, 1,
                               0, 0, location, &store);
}

static iree_status_t loom_buffer_legalize_build_copy_chunk(
    loom_builder_t* builder, loom_value_id_t relative_offset,
    loom_type_t transfer_type, loom_value_id_t source,
    loom_value_id_t source_offset, loom_value_id_t target,
    loom_value_id_t target_offset, loom_location_id_t location) {
  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_load(
      builder, source, source_offset, relative_offset, transfer_type, location,
      &value));
  return loom_buffer_legalize_build_store(builder, value, target, target_offset,
                                          relative_offset, transfer_type,
                                          location);
}

// Converts to an integer payload for bit arithmetic, keeping address-domain
// operations restricted to their canonical coordinate/byte semantics.
static iree_status_t loom_buffer_legalize_build_aligned_end(
    loom_builder_t* builder, loom_value_id_t byte_length, uint32_t width,
    loom_location_id_t location, loom_value_id_t* out_end) {
  if (width == 1) {
    *out_end = byte_length;
    return iree_ok_status();
  }
  const loom_type_t offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  const loom_type_t bits_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  loom_op_t* bits_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(builder, byte_length, offset_type,
                                             bits_type, location, &bits_op));
  loom_value_id_t mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
      builder, LOOM_SCALAR_TYPE_I64, -(int64_t)width, location, &mask));
  loom_op_t* end_bits_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_andi_build(builder, loom_index_cast_result(bits_op), mask,
                             bits_type, location, &end_bits_op));
  loom_op_t* end_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_cast_build(builder, loom_scalar_andi_result(end_bits_op),
                            bits_type, offset_type, location, &end_op));
  *out_end = loom_index_cast_result(end_op);
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize_copy(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);

  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_offset_constant(
      &rewriter->builder, 0, op->location, &zero));
  const uint8_t widths =
      loom_buffer_legalize_transfer_widths(context, loom_buffer_copy_source(op),
                                           loom_buffer_copy_source_offset(op)) &
      loom_buffer_legalize_transfer_widths(context, loom_buffer_copy_target(op),
                                           loom_buffer_copy_target_offset(op));
  const loom_value_id_t byte_length = loom_buffer_copy_byte_length(op);
  int64_t static_length = 0;
  const bool is_static = loom_value_facts_as_exact_i64(
      loom_value_fact_table_lookup(context->fact_table, byte_length),
      &static_length);
  int64_t static_start = 0;
  loom_value_id_t start = zero;
  uint32_t previous_width = 0;
  iree_status_t status = iree_ok_status();
  for (uint32_t width = 64; width != 0 && iree_status_is_ok(status);
       width /= 2) {
    if (!(widths & width)) {
      continue;
    }
    const int64_t static_end = static_length & -(int64_t)width;
    if (is_static && static_end == static_start) {
      continue;
    }
    loom_value_id_t end = LOOM_VALUE_ID_INVALID;
    if (is_static) {
      status = loom_buffer_legalize_build_offset_constant(
          &rewriter->builder, static_end, op->location, &end);
    } else {
      status = loom_buffer_legalize_build_aligned_end(
          &rewriter->builder, byte_length, width, op->location, &end);
    }
    if (!iree_status_is_ok(status)) {
      break;
    }
    const bool single = is_static && static_end - static_start == width;
    loom_builder_ip_t saved_ip = loom_builder_save(&rewriter->builder);
    loom_value_id_t position = start;
    if (!single) {
      loom_op_t* region_op = NULL;
      if (!is_static && previous_width == 2 * width) {
        loom_op_t* condition_op = NULL;
        status = loom_index_cmp_build(&rewriter->builder,
                                      LOOM_INDEX_CMP_PREDICATE_ULT, start, end,
                                      op->location, &condition_op);
        if (iree_status_is_ok(status)) {
          status = loom_scf_if_build(&rewriter->builder, 0,
                                     loom_index_cmp_result(condition_op), NULL,
                                     0, NULL, 0, op->location, &region_op);
        }
        if (iree_status_is_ok(status)) {
          loom_builder_enter_region(&rewriter->builder, region_op,
                                    loom_scf_if_then_region(region_op));
        }
      } else {
        loom_value_id_t step = LOOM_VALUE_ID_INVALID;
        status = loom_buffer_legalize_build_offset_constant(
            &rewriter->builder, width, op->location, &step);
        if (iree_status_is_ok(status)) {
          status = loom_scf_for_build(
              &rewriter->builder, 0, start, end, step, NULL, 0, NULL, NULL, 0,
              LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, 0, 0, op->location,
              &region_op);
        }
        if (iree_status_is_ok(status)) {
          loom_builder_enter_region(&rewriter->builder, region_op,
                                    loom_scf_for_body(region_op));
          position = loom_region_entry_arg_id(loom_scf_for_body(region_op), 0);
        }
      }
    }
    if (iree_status_is_ok(status)) {
      status = loom_buffer_legalize_build_copy_chunk(
          &rewriter->builder, position,
          loom_buffer_legalize_transfer_type(&context->bulk_memory_policy,
                                             width),
          loom_buffer_copy_source(op), loom_buffer_copy_source_offset(op),
          loom_buffer_copy_target(op), loom_buffer_copy_target_offset(op),
          op->location);
    }
    if (!single && iree_status_is_ok(status)) {
      loom_op_t* yield_op = NULL;
      status = loom_scf_yield_build(&rewriter->builder, NULL, 0, op->location,
                                    &yield_op);
    }
    if (!single) {
      loom_builder_restore(&rewriter->builder, saved_ip);
    }
    start = end;
    static_start = static_end;
    previous_width = width;
  }
  IREE_RETURN_IF_ERROR(status);

  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static loom_scalar_type_t loom_buffer_legalize_pattern_integer_type(
    loom_scalar_type_t pattern_type) {
  switch (loom_scalar_type_bitwidth(pattern_type)) {
    case 8:
      return LOOM_SCALAR_TYPE_I8;
    case 16:
      return LOOM_SCALAR_TYPE_I16;
    case 32:
      return LOOM_SCALAR_TYPE_I32;
    case 64:
      return LOOM_SCALAR_TYPE_I64;
    default:
      IREE_ASSERT_UNREACHABLE(
          "verified buffer.fill pattern must have a supported byte width");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_buffer_legalize_build_pattern_bits(
    loom_builder_t* builder, loom_value_id_t pattern,
    loom_scalar_type_t pattern_type, loom_location_id_t location,
    loom_scalar_type_t* out_working_type, loom_value_id_t* out_pattern_bits) {
  const loom_scalar_type_t integer_type =
      loom_buffer_legalize_pattern_integer_type(pattern_type);
  loom_value_id_t pattern_bits = pattern;
  if (loom_scalar_type_is_float(pattern_type)) {
    loom_op_t* bitcast_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(
        builder, pattern, loom_type_scalar(pattern_type),
        loom_type_scalar(integer_type), location, &bitcast_op));
    pattern_bits = loom_scalar_bitcast_result(bitcast_op);
  }

  const uint32_t pattern_bitwidth = loom_scalar_type_bitwidth(integer_type);
  if (pattern_bitwidth < 32) {
    loom_op_t* extend_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_extui_build(
        builder, pattern_bits, loom_type_scalar(integer_type),
        loom_type_scalar(LOOM_SCALAR_TYPE_I32), location, &extend_op));
    pattern_bits = loom_scalar_extui_result(extend_op);

    // Repeat a two-byte pattern across the i32 working value so rotating the
    // working value by one byte preserves the authored pattern period.
    if (pattern_bitwidth == 16) {
      loom_value_id_t halfword_shift = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
          builder, LOOM_SCALAR_TYPE_I32, 16, location, &halfword_shift));
      loom_op_t* high_half_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_shli_build(
          builder, /*instance_flags=*/0, pattern_bits, halfword_shift,
          loom_type_scalar(LOOM_SCALAR_TYPE_I32), location, &high_half_op));
      loom_op_t* repeated_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_ori_build(
          builder, pattern_bits, loom_scalar_shli_result(high_half_op),
          loom_type_scalar(LOOM_SCALAR_TYPE_I32), location, &repeated_op));
      pattern_bits = loom_scalar_ori_result(repeated_op);
    }

    *out_working_type = LOOM_SCALAR_TYPE_I32;
    *out_pattern_bits = pattern_bits;
    return iree_ok_status();
  }

  *out_working_type = integer_type;
  *out_pattern_bits = pattern_bits;
  return iree_ok_status();
}

// Projects the low byte of the rotating pattern into the canonical i32 byte
// carrier used by buffer.store.i8. The store consumes only the low eight bits,
// so an i32 working value needs no separate mask or truncate operation.
static iree_status_t loom_buffer_legalize_build_pattern_byte(
    loom_builder_t* builder, loom_value_id_t pattern_bits,
    loom_scalar_type_t working_type, loom_location_id_t location,
    loom_value_id_t* out_byte) {
  if (working_type == LOOM_SCALAR_TYPE_I32) {
    *out_byte = pattern_bits;
    return iree_ok_status();
  }

  loom_op_t* carrier_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_trunci_build(
      builder, pattern_bits, loom_type_scalar(working_type),
      loom_type_scalar(LOOM_SCALAR_TYPE_I32), location, &carrier_op));
  *out_byte = loom_scalar_trunci_result(carrier_op);
  return iree_ok_status();
}

// Rotates a multi-byte working pattern right by one byte. Both shifts have
// compile-time-constant amounts, which keeps the reference legalization in the
// scalar baseline shared by targets without a general variable-shift form.
static iree_status_t loom_buffer_legalize_build_rotated_pattern(
    loom_builder_t* builder, loom_value_id_t pattern_bits,
    loom_scalar_type_t working_type, loom_value_id_t byte_shift,
    loom_value_id_t wrap_shift, loom_location_id_t location,
    loom_value_id_t* out_rotated_pattern) {
  const loom_type_t working_loom_type = loom_type_scalar(working_type);
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_shrui_build(
      builder, pattern_bits, byte_shift, working_loom_type, location, &low_op));
  loom_op_t* high_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_shli_build(
      builder, /*instance_flags=*/0, pattern_bits, wrap_shift,
      working_loom_type, location, &high_op));
  loom_op_t* rotated_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_ori_build(builder, loom_scalar_shrui_result(low_op),
                            loom_scalar_shli_result(high_op), working_loom_type,
                            location, &rotated_op));
  *out_rotated_pattern = loom_scalar_ori_result(rotated_op);
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize_build_fill_loop_body(
    loom_builder_t* builder, loom_op_t* loop, loom_value_id_t target,
    loom_value_id_t target_offset, loom_value_id_t pattern_bits,
    loom_scalar_type_t working_type, loom_value_id_t byte_shift,
    loom_value_id_t wrap_shift, loom_location_id_t location) {
  const loom_region_t* body = loom_scf_for_body(loop);
  const loom_value_id_t relative_offset = loom_region_entry_arg_id(body, 0);
  const bool rotates_pattern = byte_shift != LOOM_VALUE_ID_INVALID;
  const loom_value_id_t current_pattern =
      rotates_pattern ? loom_region_entry_arg_id(body, 1) : pattern_bits;
  loom_value_id_t byte = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_pattern_byte(
      builder, current_pattern, working_type, location, &byte));
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_byte_store(
      builder, byte, target, target_offset, relative_offset, location));

  if (!rotates_pattern) {
    loom_op_t* yield_op = NULL;
    return loom_scf_yield_build(builder, /*values=*/NULL, /*value_count=*/0,
                                location, &yield_op);
  }

  loom_value_id_t next_pattern = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_rotated_pattern(
      builder, current_pattern, working_type, byte_shift, wrap_shift, location,
      &next_pattern));
  loom_op_t* yield_op = NULL;
  return loom_scf_yield_build(builder, &next_pattern, 1, location, &yield_op);
}

static iree_status_t loom_buffer_legalize_fill(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);

  const loom_scalar_type_t pattern_type = loom_type_element_type(
      loom_module_value_type(context->module, loom_buffer_fill_pattern(op)));
  loom_scalar_type_t working_type = LOOM_SCALAR_TYPE_NONE;
  loom_value_id_t pattern_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_pattern_bits(
      &rewriter->builder, loom_buffer_fill_pattern(op), pattern_type,
      op->location, &working_type, &pattern_bits));

  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_offset_constant(
      &rewriter->builder, 0, op->location, &zero));
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_offset_constant(
      &rewriter->builder, 1, op->location, &one));
  loom_value_id_t byte_shift = LOOM_VALUE_ID_INVALID;
  loom_value_id_t wrap_shift = LOOM_VALUE_ID_INVALID;
  const int64_t pattern_byte_count =
      loom_scalar_type_bitwidth(pattern_type) / 8;
  if (pattern_byte_count > 1) {
    IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
        &rewriter->builder, working_type, 8, op->location, &byte_shift));
    IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
        &rewriter->builder, working_type,
        loom_scalar_type_bitwidth(working_type) - 8, op->location,
        &wrap_shift));
  }

  const loom_value_id_t initial_pattern =
      pattern_byte_count > 1 ? pattern_bits : LOOM_VALUE_ID_INVALID;
  const loom_value_id_t* iter_args =
      initial_pattern == LOOM_VALUE_ID_INVALID ? NULL : &initial_pattern;
  const iree_host_size_t iter_arg_count = iter_args == NULL ? 0 : 1;
  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &rewriter->builder, /*build_flags=*/0, zero,
      loom_buffer_fill_byte_length(op), one, iter_args, iter_arg_count,
      /*result_types=*/NULL,
      /*tied_results=*/NULL, /*tied_result_count=*/0,
      /*pipeline_depth=*/LOOM_VALUE_ID_INVALID,
      /*unroll_factor=*/LOOM_VALUE_ID_INVALID, /*unroll_policy=*/0,
      /*unroll_schedule=*/0, op->location, &loop));

  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &rewriter->builder, loop, loom_scf_for_body(loop));
  iree_status_t status = loom_buffer_legalize_build_fill_loop_body(
      &rewriter->builder, loop, loom_buffer_fill_target(op),
      loom_buffer_fill_target_offset(op), pattern_bits, working_type,
      byte_shift, wrap_shift, op->location);
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

typedef struct loom_buffer_compare_constants_t {
  // Canonical equal comparison result.
  loom_value_id_t equal;
  // Canonical less-than comparison result.
  loom_value_id_t less;
  // Canonical greater-than comparison result.
  loom_value_id_t greater;
} loom_buffer_compare_constants_t;

static iree_status_t loom_buffer_legalize_build_compare_byte_order(
    loom_builder_t* builder, loom_value_id_t lhs, loom_value_id_t rhs,
    const loom_buffer_compare_constants_t* constants,
    loom_location_id_t location, loom_value_id_t* out_order) {
  loom_op_t* equal_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
      builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, lhs, rhs, location, &equal_op));
  loom_op_t* less_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
      builder, LOOM_SCALAR_CMPI_PREDICATE_ULT, lhs, rhs, location, &less_op));
  const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* unequal_order_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(
      builder, loom_scalar_cmpi_result(less_op), constants->less,
      constants->greater, i32_type, location, &unequal_order_op));
  loom_op_t* byte_order_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(
      builder, loom_scalar_cmpi_result(equal_op), constants->equal,
      loom_scf_select_result(unequal_order_op), i32_type, location,
      &byte_order_op));
  *out_order = loom_scf_select_result(byte_order_op);
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize_build_compare_before_region(
    loom_builder_t* builder, loom_op_t* loop, loom_value_id_t byte_length,
    const loom_buffer_compare_constants_t* constants,
    loom_location_id_t location) {
  const loom_region_t* before = loom_scf_while_before(loop);
  const loom_value_id_t relative_offset = loom_region_entry_arg_id(before, 0);
  const loom_value_id_t current_order = loom_region_entry_arg_id(before, 1);

  loom_op_t* in_range_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cmp_build(
      builder, LOOM_INDEX_CMP_PREDICATE_ULT, relative_offset, byte_length,
      location, &in_range_op));
  loom_op_t* unresolved_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
      builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, current_order, constants->equal,
      location, &unresolved_op));
  loom_op_t* continue_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
      builder, loom_index_cmp_result(in_range_op),
      loom_scalar_cmpi_result(unresolved_op),
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), location, &continue_op));

  const loom_value_id_t forwarded[] = {relative_offset, current_order};
  loom_op_t* condition_op = NULL;
  return loom_scf_condition_build(builder, loom_scalar_andi_result(continue_op),
                                  forwarded, IREE_ARRAYSIZE(forwarded),
                                  location, &condition_op);
}

static iree_status_t loom_buffer_legalize_build_compare_after_region(
    loom_builder_t* builder, loom_op_t* loop, loom_value_id_t lhs,
    loom_value_id_t lhs_offset, loom_value_id_t rhs, loom_value_id_t rhs_offset,
    loom_value_id_t one_offset,
    const loom_buffer_compare_constants_t* constants,
    loom_location_id_t location) {
  const loom_region_t* after = loom_scf_while_after(loop);
  const loom_value_id_t relative_offset = loom_region_entry_arg_id(after, 0);
  loom_value_id_t lhs_byte = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_byte_load(
      builder, lhs, lhs_offset, relative_offset, location, &lhs_byte));
  loom_value_id_t rhs_byte = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_byte_load(
      builder, rhs, rhs_offset, relative_offset, location, &rhs_byte));
  loom_value_id_t next_order = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_compare_byte_order(
      builder, lhs_byte, rhs_byte, constants, location, &next_order));
  loom_op_t* next_offset_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_add_build(
      builder, relative_offset, one_offset,
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), location, &next_offset_op));

  const loom_value_id_t yielded[] = {loom_index_add_result(next_offset_op),
                                     next_order};
  loom_op_t* yield_op = NULL;
  return loom_scf_yield_build(builder, yielded, IREE_ARRAYSIZE(yielded),
                              location, &yield_op);
}

static iree_status_t loom_buffer_legalize_compare(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);

  loom_value_id_t zero_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_offset_constant(
      &rewriter->builder, 0, op->location, &zero_offset));
  loom_value_id_t one_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_offset_constant(
      &rewriter->builder, 1, op->location, &one_offset));
  loom_buffer_compare_constants_t constants = {0};
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
      &rewriter->builder, LOOM_SCALAR_TYPE_I32, 0, op->location,
      &constants.equal));
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
      &rewriter->builder, LOOM_SCALAR_TYPE_I32, -1, op->location,
      &constants.less));
  IREE_RETURN_IF_ERROR(loom_buffer_legalize_build_scalar_constant(
      &rewriter->builder, LOOM_SCALAR_TYPE_I32, 1, op->location,
      &constants.greater));

  const loom_value_id_t iter_args[] = {zero_offset, constants.equal};
  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_while_build(
      &rewriter->builder, iter_args, IREE_ARRAYSIZE(iter_args),
      /*iter_args_types=*/NULL,
      /*result_types=*/NULL, /*result_count=*/IREE_ARRAYSIZE(iter_args),
      /*tied_results=*/NULL, /*tied_result_count=*/0, op->location, &loop));

  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &rewriter->builder, loop, loom_scf_while_before(loop));
  iree_status_t status = loom_buffer_legalize_build_compare_before_region(
      &rewriter->builder, loop, loom_buffer_compare_byte_length(op), &constants,
      op->location);
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  saved_ip = loom_builder_enter_region(&rewriter->builder, loop,
                                       loom_scf_while_after(loop));
  status = loom_buffer_legalize_build_compare_after_region(
      &rewriter->builder, loop, loom_buffer_compare_lhs(op),
      loom_buffer_compare_lhs_offset(op), loom_buffer_compare_rhs(op),
      loom_buffer_compare_rhs_offset(op), one_offset, &constants, op->location);
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  const loom_value_id_t order = loom_scf_while_results(loop).values[1];
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &order, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &order, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static iree_status_t loom_buffer_legalize(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  switch (op->kind) {
    case LOOM_OP_BUFFER_COPY:
      return loom_buffer_legalize_copy(context, op, out_result);
    case LOOM_OP_BUFFER_FILL:
      return loom_buffer_legalize_fill(context, op, out_result);
    case LOOM_OP_BUFFER_COMPARE:
      return loom_buffer_legalize_compare(context, op, out_result);
    default:
      return iree_ok_status();
  }
}

static const loom_target_legalizer_rule_t kBufferLegalizerRules[] = {
    {
        .root_kind = LOOM_OP_BUFFER_COPY,
        .legalize = loom_buffer_legalize,
    },
    {
        .root_kind = LOOM_OP_BUFFER_FILL,
        .legalize = loom_buffer_legalize,
    },
    {
        .root_kind = LOOM_OP_BUFFER_COMPARE,
        .legalize = loom_buffer_legalize,
    },
};

static const loom_target_legalizer_provider_t kBufferLegalizerProvider = {
    .name = IREE_SVL("buffer"),
    .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE,
    .rules = kBufferLegalizerRules,
    .rule_count = IREE_ARRAYSIZE(kBufferLegalizerRules),
};

const loom_target_legalizer_provider_t* loom_buffer_target_legalizer_provider(
    void) {
  return &kBufferLegalizerProvider;
}
