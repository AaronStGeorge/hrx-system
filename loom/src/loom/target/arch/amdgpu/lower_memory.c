// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cache.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower_internal.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/util/fact_table.h"

static bool loom_amdgpu_module_value_as_workitem_id(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_kernel_dimension_t* out_dimension) {
  IREE_ASSERT_ARGUMENT(out_dimension);
  *out_dimension = LOOM_KERNEL_DIMENSION_COUNT_;
  if (value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_kernel_workitem_id_isa(defining_op)) {
    return false;
  }
  const loom_kernel_dimension_t dimension =
      loom_kernel_workitem_id_dimension(defining_op);
  if (dimension >= LOOM_KERNEL_DIMENSION_COUNT_) {
    return false;
  }
  *out_dimension = dimension;
  return true;
}

static bool loom_amdgpu_module_value_as_workgroup_id(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_kernel_dimension_t* out_dimension) {
  IREE_ASSERT_ARGUMENT(out_dimension);
  *out_dimension = LOOM_KERNEL_DIMENSION_COUNT_;
  if (value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_kernel_workgroup_id_isa(defining_op)) {
    return false;
  }
  const loom_kernel_dimension_t dimension =
      loom_kernel_workgroup_id_dimension(defining_op);
  if (dimension >= LOOM_KERNEL_DIMENSION_COUNT_) {
    return false;
  }
  *out_dimension = dimension;
  return true;
}

typedef uint32_t loom_amdgpu_memory_access_rejection_flags_t;

#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIBE_FAILED ((uint32_t)1u << 0)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_LAYOUT ((uint32_t)1u << 1)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_ELEMENT_WIDTH ((uint32_t)1u << 2)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_RANK ((uint32_t)1u << 3)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE ((uint32_t)1u << 4)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE \
  ((uint32_t)1u << 5)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_COUNT \
  ((uint32_t)1u << 6)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_AXIS ((uint32_t)1u << 7)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_SOURCE \
  ((uint32_t)1u << 8)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE ((uint32_t)1u << 9)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_DYNAMIC_ALIGNMENT \
  ((uint32_t)1u << 10)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_STATIC_OFFSET ((uint32_t)1u << 11)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET \
  ((uint32_t)1u << 12)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_STATIC_ALIGNMENT \
  ((uint32_t)1u << 13)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING \
  ((uint32_t)1u << 14)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE \
  ((uint32_t)1u << 15)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE \
  ((uint32_t)1u << 16)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VIEW_SOURCE ((uint32_t)1u << 17)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VIEW_BASE ((uint32_t)1u << 18)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VIEW_BASE_OVERFLOW \
  ((uint32_t)1u << 19)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_MEMORY_SPACE ((uint32_t)1u << 20)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_ROOT ((uint32_t)1u << 21)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_SCALAR_DYNAMIC_STRIDE \
  ((uint32_t)1u << 22)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_DYNAMIC_INDEX_SOURCE \
  ((uint32_t)1u << 23)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_UNAVAILABLE \
  ((uint32_t)1u << 24)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_ADDRESS \
  ((uint32_t)1u << 25)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_OFFSET_RANGE \
  ((uint32_t)1u << 26)
#define LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_PACKED_REGISTER_FOOTPRINT \
  ((uint32_t)1u << 27)

typedef struct loom_amdgpu_memory_access_diagnostic_t {
  // Rejection bits explaining why an access is not legal for this target.
  loom_amdgpu_memory_access_rejection_flags_t rejection_bits;
} loom_amdgpu_memory_access_diagnostic_t;

#define LOOM_AMDGPU_LDS_BANK_COUNT 32u
#define LOOM_AMDGPU_LDS_BANK_WIDTH_BYTES 4u

typedef struct loom_amdgpu_memory_access_bank_summary_t {
  // Distance between adjacent workitems in 32-bit LDS bank words.
  uint32_t bank_stride_words;
  // Estimated conflict degree across a 32-lane bank cycle, or zero when the
  // bank pattern is unknown from the current facts.
  uint32_t conflict_degree;
  // Stable diagnostic reason token.
  iree_string_view_t reason;
} loom_amdgpu_memory_access_bank_summary_t;

static uint32_t loom_amdgpu_gcd_u32(uint32_t lhs, uint32_t rhs) {
  while (rhs != 0) {
    const uint32_t remainder = lhs % rhs;
    lhs = rhs;
    rhs = remainder;
  }
  return lhs;
}

static iree_string_view_t loom_amdgpu_memory_space_name(
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
      return IREE_SV("unknown");
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
      return IREE_SV("global");
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
      return IREE_SV("constant");
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      return IREE_SV("workgroup");
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
      return IREE_SV("private");
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
      return IREE_SV("host");
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
      return IREE_SV("descriptor");
  }
  return IREE_SV("invalid");
}

static iree_string_view_t loom_amdgpu_memory_operation_name(
    loom_amdgpu_memory_operation_kind_t kind) {
  switch (kind) {
    case LOOM_AMDGPU_MEMORY_OPERATION_LOAD:
      return IREE_SV("load");
    case LOOM_AMDGPU_MEMORY_OPERATION_STORE:
      return IREE_SV("store");
  }
  return IREE_SV("invalid");
}

static iree_string_view_t loom_amdgpu_cache_scope_name(uint8_t scope) {
  switch (scope) {
    case LOOM_CACHE_SCOPE_CU:
      return IREE_SV("cu");
    case LOOM_CACHE_SCOPE_SE:
      return IREE_SV("se");
    case LOOM_CACHE_SCOPE_DEVICE:
      return IREE_SV("device");
    case LOOM_CACHE_SCOPE_SYSTEM:
      return IREE_SV("system");
  }
  return IREE_SV("invalid");
}

static iree_string_view_t loom_amdgpu_cache_temporal_name(uint8_t temporal) {
  switch (temporal) {
    case LOOM_CACHE_TEMPORAL_REGULAR:
      return IREE_SV("regular");
    case LOOM_CACHE_TEMPORAL_NON_TEMPORAL:
      return IREE_SV("non_temporal");
    case LOOM_CACHE_TEMPORAL_HIGH_TEMPORAL:
      return IREE_SV("high_temporal");
    case LOOM_CACHE_TEMPORAL_LAST_USE:
      return IREE_SV("last_use");
    case LOOM_CACHE_TEMPORAL_WRITEBACK:
      return IREE_SV("writeback");
    case LOOM_CACHE_TEMPORAL_NON_TEMPORAL_REGULAR:
      return IREE_SV("non_temporal_regular");
    case LOOM_CACHE_TEMPORAL_REGULAR_NON_TEMPORAL:
      return IREE_SV("regular_non_temporal");
    case LOOM_CACHE_TEMPORAL_NON_TEMPORAL_HIGH_TEMPORAL:
      return IREE_SV("non_temporal_high_temporal");
    case LOOM_CACHE_TEMPORAL_NON_TEMPORAL_WRITEBACK:
      return IREE_SV("non_temporal_writeback");
    case LOOM_CACHE_TEMPORAL_BYPASS:
      return IREE_SV("bypass");
  }
  return IREE_SV("invalid");
}

static loom_amdgpu_memory_access_bank_summary_t
loom_amdgpu_memory_access_bank_summary(
    const loom_amdgpu_memory_access_plan_t* access) {
  loom_amdgpu_memory_access_bank_summary_t summary = {
      .reason = IREE_SV("bank-pattern-unknown"),
  };
  if (access->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
      access->dynamic_index == LOOM_VALUE_ID_INVALID ||
      access->dynamic_index_kind != LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR ||
      access->dynamic_index_byte_stride == 0 ||
      (access->dynamic_index_byte_stride % LOOM_AMDGPU_LDS_BANK_WIDTH_BYTES) !=
          0) {
    return summary;
  }

  summary.bank_stride_words =
      access->dynamic_index_byte_stride / LOOM_AMDGPU_LDS_BANK_WIDTH_BYTES;
  summary.conflict_degree = loom_amdgpu_gcd_u32(summary.bank_stride_words,
                                                LOOM_AMDGPU_LDS_BANK_COUNT);
  const uint32_t vector_footprint_bytes =
      access->packet_byte_count != 0
          ? access->packet_byte_count
          : access->element_byte_count * iree_max(access->vgpr_count, 1u);
  if (summary.conflict_degree == 1) {
    summary.reason = access->dynamic_index_byte_stride > vector_footprint_bytes
                         ? IREE_SV("padded-bank-conflict-free")
                         : IREE_SV("bank-conflict-free");
  } else {
    summary.reason = IREE_SV("bank-conflict-risk");
  }
  return summary;
}

static iree_status_t loom_amdgpu_memory_access_descriptor_key(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_plan_t* access,
    iree_string_view_t* out_packet_key) {
  IREE_ASSERT_ARGUMENT(out_packet_key);
  *out_packet_key = IREE_SV("<missing>");
  if (access->descriptor_id == LOOM_LOW_DESCRIPTOR_ID_NONE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU memory access has no selected descriptor");
  }
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_lookup_descriptor_by_id(descriptor_set,
                                                      access->descriptor_id);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "selected AMDGPU memory descriptor id is not present");
  }
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  if (descriptor == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "selected AMDGPU memory descriptor is invalid");
  }
  return loom_low_descriptor_set_string(
      descriptor_set, descriptor->key_string_offset, out_packet_key);
}

static iree_status_t loom_amdgpu_record_memory_access_diagnostic(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_operation_kind_t kind) {
  if (!iree_any_bit_set(loom_target_low_legality_diagnostic_flags(context),
                        LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS) ||
      access->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return iree_ok_status();
  }

  iree_string_view_t packet_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_memory_access_descriptor_key(
      descriptor_set, access, &packet_key));
  const loom_amdgpu_memory_access_bank_summary_t bank_summary =
      loom_amdgpu_memory_access_bank_summary(access);
  return loom_target_low_legality_record_memory_access(
      context, provider, op,
      loom_amdgpu_memory_space_name(access->memory_space),
      loom_amdgpu_memory_operation_name(kind), packet_key, IREE_SV("selected"),
      access->element_byte_count, access->vgpr_count,
      access->dynamic_index_byte_stride, access->vector_lane_byte_stride,
      bank_summary.bank_stride_words, bank_summary.conflict_degree,
      bank_summary.reason);
}

static iree_string_view_t loom_amdgpu_memory_access_rejection_detail(
    loom_amdgpu_memory_access_rejection_flags_t rejection_bits) {
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIBE_FAILED)) {
    return IREE_SV(
        "memory access shape is not representable as a vector "
        "footprint in the source view");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_LAYOUT)) {
    return IREE_SV(
        "AMDGPU memory lowering requires a statically-described dense or "
        "strided view layout");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_ELEMENT_WIDTH)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering requires byte-addressable "
        "view elements");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_RANK)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering currently requires "
        "one-dimensional vectors");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering currently supports up to four 32-bit "
        "memory registers");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering requires unit stride along "
        "the vector axis");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_COUNT)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering currently supports at most "
        "one dynamic index");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_AXIS)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering requires exactly one "
        "well-formed dynamic axis for dynamic accesses");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_SOURCE)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering currently requires dynamic "
        "indices to come from kernel.workitem.id, kernel.workgroup.id, or "
        "VGPR address arithmetic");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering requires a non-negative "
        "32-bit dynamic byte stride");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_SCALAR_DYNAMIC_STRIDE)) {
    return IREE_SV(
        "AMDGPU scalar dynamic buffer offsets currently require a "
        "power-of-two dynamic byte stride");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_DYNAMIC_INDEX_SOURCE)) {
    return IREE_SV(
        "AMDGPU workgroup memory lowering currently requires dynamic "
        "indices to come from kernel.workitem.id");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_UNAVAILABLE)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering rejected this access and global pointer "
        "fallback is unavailable in the selected descriptor set");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_ADDRESS)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering rejected this access and global pointer "
        "fallback is blocked by missing vector address facts");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_OFFSET_RANGE)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering rejected this access and global pointer "
        "fallback cannot encode the static byte offset");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_PACKED_REGISTER_FOOTPRINT)) {
    return IREE_SV(
        "AMDGPU packed integer memory lowering requires the vector footprint "
        "to exactly fill complete 32-bit memory registers");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_DYNAMIC_ALIGNMENT)) {
    return IREE_SV(
        "128-bit AMDGPU buffer memory accesses currently require "
        "16-byte aligned dynamic byte strides");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_STATIC_OFFSET)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering requires a statically "
        "representable descriptor offset");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering requires non-negative "
        "static byte offsets");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_STATIC_ALIGNMENT)) {
    return IREE_SV(
        "128-bit AMDGPU buffer memory accesses currently require "
        "16-byte aligned static byte offsets");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING)) {
    return IREE_SV(
        "selected AMDGPU descriptor set has no buffer memory "
        "descriptor for this access width");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE)) {
    return IREE_SV(
        "selected AMDGPU buffer memory descriptor does not expose "
        "one unsigned offset immediate");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE)) {
    return IREE_SV(
        "AMDGPU buffer memory static byte offset is outside the selected "
        "descriptor's immediate plus scalar SOFFSET range");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VIEW_SOURCE)) {
    return IREE_SV(
        "AMDGPU buffer memory lowering currently requires views to come from "
        "buffer.view");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VIEW_BASE)) {
    return IREE_SV(
        "AMDGPU HAL buffer views currently require exact non-negative static "
        "byte offsets");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VIEW_BASE_OVERFLOW)) {
    return IREE_SV(
        "AMDGPU HAL buffer view byte offset overflows the static memory "
        "access offset");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_MEMORY_SPACE)) {
    return IREE_SV(
        "AMDGPU memory lowering currently supports HAL/global buffer resources "
        "and workgroup LDS roots");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_ROOT)) {
    return IREE_SV(
        "AMDGPU workgroup memory lowering currently requires a buffer.alloca "
        "root");
  }
  return IREE_SV("AMDGPU buffer memory access is not target-legal");
}

static bool loom_amdgpu_memory_access_find_dynamic_axis(
    loom_attribute_t static_indices, uint8_t* out_dynamic_axis) {
  IREE_ASSERT_ARGUMENT(out_dynamic_axis);
  *out_dynamic_axis = UINT8_MAX;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY) {
    return false;
  }
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    if (static_indices.i64_array[i] != INT64_MIN) {
      continue;
    }
    if (*out_dynamic_axis != UINT8_MAX || i > UINT8_MAX) {
      return false;
    }
    *out_dynamic_axis = (uint8_t)i;
  }
  return true;
}

static bool loom_amdgpu_memory_access_static_byte_offset(
    const loom_vector_memory_access_t* vector_access,
    loom_attribute_t static_indices, uint8_t dynamic_axis,
    int64_t* out_static_byte_offset) {
  IREE_ASSERT_ARGUMENT(out_static_byte_offset);
  *out_static_byte_offset = 0;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY) {
    return false;
  }
  if (static_indices.count > LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK) {
    return false;
  }

  int64_t static_origin[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    if (i == dynamic_axis) {
      if (static_indices.i64_array[i] != INT64_MIN) {
        return false;
      }
      continue;
    }
    if (static_indices.i64_array[i] == INT64_MIN) {
      return false;
    }
    static_origin[i] = static_indices.i64_array[i];
  }
  loom_attribute_t static_origin_attr =
      loom_attr_i64_array(static_origin, static_indices.count);
  int64_t lane_indices[] = {0};
  return loom_vector_memory_access_static_lane_byte_offset(
      vector_access, static_origin_attr, lane_indices,
      IREE_ARRAYSIZE(lane_indices), out_static_byte_offset);
}

static bool loom_amdgpu_memory_access_static_byte_offset_is_usable(
    int64_t static_byte_offset,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (static_byte_offset < 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET;
    return false;
  }
  return true;
}

static bool loom_amdgpu_memory_access_register_footprint(
    loom_type_t vector_type, loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  uint32_t register_count = loom_amdgpu_vector_32bit_lane_count(vector_type);
  if (register_count != 0) {
    access->vgpr_count = register_count;
    access->packet_byte_count = register_count * 4u;
    return true;
  }

  uint32_t payload_bit_count = 0;
  if (!loom_amdgpu_type_packed_integer_storage(vector_type, &payload_bit_count,
                                               &register_count)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE;
    return false;
  }
  const uint32_t packet_bit_count = register_count * 32u;
  if (payload_bit_count != packet_bit_count) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_PACKED_REGISTER_FOOTPRINT;
    return false;
  }
  access->vgpr_count = register_count;
  access->packet_byte_count = packet_bit_count / 8u;
  return true;
}

static bool loom_amdgpu_memory_access_has_32bit_lanes(
    const loom_amdgpu_memory_access_plan_t* access) {
  return access->packet_byte_count ==
         access->element_byte_count * iree_max(access->vgpr_count, 1u);
}

static bool loom_amdgpu_memory_access_view_reference(
    const loom_value_fact_table_t* fact_table, loom_value_id_t view_value_id,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic,
    loom_value_fact_view_reference_t* out_reference) {
  IREE_ASSERT_ARGUMENT(out_reference);
  *out_reference = (loom_value_fact_view_reference_t){0};
  if (fact_table == NULL ||
      !loom_value_facts_query_view_reference(
          &fact_table->context,
          loom_value_fact_table_lookup(fact_table, view_value_id),
          out_reference)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VIEW_SOURCE;
    return false;
  }
  return true;
}

static bool loom_amdgpu_memory_access_add_view_base_byte_offset(
    const loom_value_fact_table_t* fact_table, loom_value_id_t view_value_id,
    loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic,
    int64_t* inout_static_byte_offset) {
  IREE_ASSERT_ARGUMENT(inout_static_byte_offset);
  loom_value_fact_view_reference_t view_reference = {0};
  if (!loom_amdgpu_memory_access_view_reference(fact_table, view_value_id,
                                                diagnostic, &view_reference)) {
    return false;
  }

  int64_t view_base_byte_offset = 0;
  if (!loom_amdgpu_value_facts_as_exact_non_negative_i64(
          view_reference.base_byte_offset, &view_base_byte_offset)) {
    diagnostic->rejection_bits |= LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VIEW_BASE;
    return false;
  }
  int64_t static_byte_offset = 0;
  if (!iree_checked_add_i64(*inout_static_byte_offset, view_base_byte_offset,
                            &static_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VIEW_BASE_OVERFLOW;
    return false;
  }
  *inout_static_byte_offset = static_byte_offset;
  access->memory_space = view_reference.memory_space;
  access->root_value_id = view_reference.root_value_id;
  return true;
}

typedef struct loom_amdgpu_memory_descriptor_family_t {
  // Number of VGPR lanes moved by the memory packet.
  uint32_t vgpr_count;
  // Direction of the memory packet.
  loom_amdgpu_memory_operation_kind_t kind;
  // Addressing form required by the descriptor family.
  loom_amdgpu_memory_address_form_t address_form;
  // Candidate descriptor stable IDs ordered by preference.
  const uint64_t* descriptor_ids;
  // Number of entries in descriptor_ids.
  iree_host_size_t descriptor_id_count;
} loom_amdgpu_memory_descriptor_family_t;

typedef struct loom_amdgpu_ds2_memory_descriptor_candidate_t {
  // Direction of the memory packet.
  loom_amdgpu_memory_operation_kind_t kind;
  // Candidate descriptor stable ID.
  uint64_t descriptor_id;
} loom_amdgpu_ds2_memory_descriptor_candidate_t;

static bool loom_amdgpu_select_buffer_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_address_form_t address_form,
    loom_amdgpu_memory_operation_kind_t kind, uint64_t* out_descriptor_id,
    uint32_t* out_descriptor_ordinal) {
  IREE_ASSERT_ARGUMENT(out_descriptor_id);
  IREE_ASSERT_ARGUMENT(out_descriptor_ordinal);
  *out_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  *out_descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  static const uint64_t kLoadB32DescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_DWORD,
  };
  static const uint64_t kLoadB32OffZeroDescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_DWORD_OFF_ZERO,
  };
  static const uint64_t kLoadB64DescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_B64,
      LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_DWORDX2,
  };
  static const uint64_t kLoadB128DescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_B128,
      LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_DWORDX4,
  };
  static const uint64_t kStoreB32DescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_STORE_DWORD,
  };
  static const uint64_t kStoreB32OffZeroDescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_STORE_DWORD_OFF_ZERO,
  };
  static const uint64_t kStoreB64DescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_STORE_B64,
      LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_STORE_DWORDX2,
  };
  static const uint64_t kStoreB128DescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_STORE_B128,
      LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_STORE_DWORDX4,
  };
  static const loom_amdgpu_memory_descriptor_family_t kFamilies[] = {
      {
          .vgpr_count = 1,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
          .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
          .descriptor_ids = kLoadB32DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kLoadB32DescriptorIds),
      },
      {
          .vgpr_count = 1,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
          .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO,
          .descriptor_ids = kLoadB32OffZeroDescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kLoadB32OffZeroDescriptorIds),
      },
      {
          .vgpr_count = 2,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
          .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
          .descriptor_ids = kLoadB64DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kLoadB64DescriptorIds),
      },
      {
          .vgpr_count = 4,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
          .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
          .descriptor_ids = kLoadB128DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kLoadB128DescriptorIds),
      },
      {
          .vgpr_count = 1,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
          .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
          .descriptor_ids = kStoreB32DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kStoreB32DescriptorIds),
      },
      {
          .vgpr_count = 1,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
          .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO,
          .descriptor_ids = kStoreB32OffZeroDescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kStoreB32OffZeroDescriptorIds),
      },
      {
          .vgpr_count = 2,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
          .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
          .descriptor_ids = kStoreB64DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kStoreB64DescriptorIds),
      },
      {
          .vgpr_count = 4,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
          .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
          .descriptor_ids = kStoreB128DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kStoreB128DescriptorIds),
      },
  };

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kFamilies); ++i) {
    const loom_amdgpu_memory_descriptor_family_t* family = &kFamilies[i];
    if (family->vgpr_count != access->vgpr_count || family->kind != kind ||
        family->address_form != address_form) {
      continue;
    }
    for (iree_host_size_t j = 0; j < family->descriptor_id_count; ++j) {
      const uint64_t descriptor_id = family->descriptor_ids[j];
      const uint32_t descriptor_ordinal =
          loom_low_descriptor_set_lookup_descriptor_by_id(descriptor_set,
                                                          descriptor_id);
      if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
        continue;
      }
      *out_descriptor_id = descriptor_id;
      *out_descriptor_ordinal = descriptor_ordinal;
      return true;
    }
    return false;
  }
  return false;
}

static bool loom_amdgpu_select_global_saddr_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_operation_kind_t kind, uint64_t* out_descriptor_id,
    uint32_t* out_descriptor_ordinal) {
  IREE_ASSERT_ARGUMENT(out_descriptor_id);
  IREE_ASSERT_ARGUMENT(out_descriptor_ordinal);
  *out_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  *out_descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  static const uint64_t kLoadB32DescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_LOAD_B32_SADDR,
  };
  static const uint64_t kLoadB64DescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_LOAD_B64_SADDR,
  };
  static const uint64_t kLoadB128DescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_LOAD_B128_SADDR,
  };
  static const uint64_t kStoreB32DescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_STORE_B32_SADDR,
  };
  static const uint64_t kStoreB64DescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_STORE_B64_SADDR,
  };
  static const uint64_t kStoreB128DescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_STORE_B128_SADDR,
  };
  static const loom_amdgpu_memory_descriptor_family_t kFamilies[] = {
      {
          .vgpr_count = 1,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
          .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR,
          .descriptor_ids = kLoadB32DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kLoadB32DescriptorIds),
      },
      {
          .vgpr_count = 2,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
          .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR,
          .descriptor_ids = kLoadB64DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kLoadB64DescriptorIds),
      },
      {
          .vgpr_count = 4,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
          .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR,
          .descriptor_ids = kLoadB128DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kLoadB128DescriptorIds),
      },
      {
          .vgpr_count = 1,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
          .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR,
          .descriptor_ids = kStoreB32DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kStoreB32DescriptorIds),
      },
      {
          .vgpr_count = 2,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
          .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR,
          .descriptor_ids = kStoreB64DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kStoreB64DescriptorIds),
      },
      {
          .vgpr_count = 4,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
          .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR,
          .descriptor_ids = kStoreB128DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kStoreB128DescriptorIds),
      },
  };

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kFamilies); ++i) {
    const loom_amdgpu_memory_descriptor_family_t* family = &kFamilies[i];
    if (family->vgpr_count != access->vgpr_count || family->kind != kind) {
      continue;
    }
    const uint64_t descriptor_id = family->descriptor_ids[0];
    const uint32_t descriptor_ordinal =
        loom_low_descriptor_set_lookup_descriptor_by_id(descriptor_set,
                                                        descriptor_id);
    if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
      return false;
    }
    *out_descriptor_id = descriptor_id;
    *out_descriptor_ordinal = descriptor_ordinal;
    return true;
  }
  return false;
}

static bool loom_amdgpu_select_ds_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_operation_kind_t kind, uint64_t* out_descriptor_id,
    uint32_t* out_descriptor_ordinal) {
  IREE_ASSERT_ARGUMENT(out_descriptor_id);
  IREE_ASSERT_ARGUMENT(out_descriptor_ordinal);
  *out_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  *out_descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  static const uint64_t kLoadB32DescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_DS_READ_B32,
  };
  static const uint64_t kLoadB64DescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_DS_READ_B64,
  };
  static const uint64_t kLoadB96DescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_DS_READ_B96,
  };
  static const uint64_t kLoadB128DescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_DS_READ_B128,
  };
  static const uint64_t kStoreB32DescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_DS_WRITE_B32,
  };
  static const uint64_t kStoreB64DescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_DS_WRITE_B64,
  };
  static const uint64_t kStoreB96DescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_DS_WRITE_B96,
  };
  static const uint64_t kStoreB128DescriptorIds[] = {
      LOOM_AMDGPU_DESCRIPTOR_ID_DS_WRITE_B128,
  };
  static const loom_amdgpu_memory_descriptor_family_t kFamilies[] = {
      {
          .vgpr_count = 1,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
          .descriptor_ids = kLoadB32DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kLoadB32DescriptorIds),
      },
      {
          .vgpr_count = 2,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
          .descriptor_ids = kLoadB64DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kLoadB64DescriptorIds),
      },
      {
          .vgpr_count = 3,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
          .descriptor_ids = kLoadB96DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kLoadB96DescriptorIds),
      },
      {
          .vgpr_count = 4,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
          .descriptor_ids = kLoadB128DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kLoadB128DescriptorIds),
      },
      {
          .vgpr_count = 1,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
          .descriptor_ids = kStoreB32DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kStoreB32DescriptorIds),
      },
      {
          .vgpr_count = 2,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
          .descriptor_ids = kStoreB64DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kStoreB64DescriptorIds),
      },
      {
          .vgpr_count = 3,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
          .descriptor_ids = kStoreB96DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kStoreB96DescriptorIds),
      },
      {
          .vgpr_count = 4,
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
          .descriptor_ids = kStoreB128DescriptorIds,
          .descriptor_id_count = IREE_ARRAYSIZE(kStoreB128DescriptorIds),
      },
  };

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kFamilies); ++i) {
    const loom_amdgpu_memory_descriptor_family_t* family = &kFamilies[i];
    if (family->vgpr_count != access->vgpr_count || family->kind != kind) {
      continue;
    }
    const uint64_t descriptor_id = family->descriptor_ids[0];
    const uint32_t descriptor_ordinal =
        loom_low_descriptor_set_lookup_descriptor_by_id(descriptor_set,
                                                        descriptor_id);
    if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
      return false;
    }
    *out_descriptor_id = descriptor_id;
    *out_descriptor_ordinal = descriptor_ordinal;
    return true;
  }
  return false;
}

static bool loom_amdgpu_select_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_operation_kind_t kind, uint64_t* out_descriptor_id,
    uint32_t* out_descriptor_ordinal) {
  switch (access->memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      return loom_amdgpu_select_ds_memory_descriptor(descriptor_set, access,
                                                     kind, out_descriptor_id,
                                                     out_descriptor_ordinal);
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
      return loom_amdgpu_select_buffer_memory_descriptor(
          descriptor_set, access, access->address_form, kind, out_descriptor_id,
          out_descriptor_ordinal);
    default:
      break;
  }
  return false;
}

typedef struct loom_amdgpu_descriptor_offset_immediate_info_t {
  // Low immediate kind required by all offset immediate fields.
  loom_low_immediate_kind_t kind;
  // Minimum encoded value accepted by signed offset immediate fields.
  int64_t signed_min;
  // Maximum encoded value accepted by every offset immediate field.
  uint64_t unsigned_max;
  // Byte count represented by one encoded offset unit.
  uint32_t unit_byte_count;
} loom_amdgpu_descriptor_offset_immediate_info_t;

static bool loom_amdgpu_immediate_encoding_address_unit_byte_count(
    uint16_t encoding_id, uint32_t* out_unit_byte_count) {
  IREE_ASSERT_ARGUMENT(out_unit_byte_count);
  *out_unit_byte_count = 0;
  switch (encoding_id) {
    case LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_BYTE:
      *out_unit_byte_count = 1;
      return true;
    case LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_DWORD:
      *out_unit_byte_count = 4;
      return true;
    case LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_QWORD:
      *out_unit_byte_count = 8;
      return true;
    case LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_DWORD_STRIDE64:
      *out_unit_byte_count = 4 * 64;
      return true;
    case LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_QWORD_STRIDE64:
      *out_unit_byte_count = 8 * 64;
      return true;
    case LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_DS16:
      *out_unit_byte_count = 1;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_descriptor_offset_immediate_info(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal, uint16_t expected_offset_immediate_count,
    loom_low_immediate_kind_t expected_kind,
    loom_amdgpu_descriptor_offset_immediate_info_t* out_info) {
  IREE_ASSERT_ARGUMENT(out_info);
  *out_info = (loom_amdgpu_descriptor_offset_immediate_info_t){
      .kind = expected_kind,
      .signed_min = INT64_MIN,
      .unsigned_max = UINT64_MAX,
  };
  if (descriptor_ordinal >= descriptor_set->descriptor_count) {
    return false;
  }
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_ordinal];
  if (descriptor->immediate_count != 0 &&
      descriptor->immediate_start >= descriptor_set->immediate_count) {
    return false;
  }
  uint16_t offset_immediate_count = 0;
  uint64_t unsigned_max = UINT64_MAX;
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const uint32_t immediate_index = descriptor->immediate_start + i;
    if (immediate_index >= descriptor_set->immediate_count) {
      return false;
    }
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[immediate_index];
    uint32_t unit_byte_count = 0;
    if (!loom_amdgpu_immediate_encoding_address_unit_byte_count(
            immediate->encoding_id, &unit_byte_count)) {
      continue;
    }
    if (immediate->kind != expected_kind) {
      return false;
    }
    if (offset_immediate_count == 0) {
      out_info->unit_byte_count = unit_byte_count;
    } else if (out_info->unit_byte_count != unit_byte_count) {
      return false;
    }
    if (expected_kind == LOOM_LOW_IMMEDIATE_KIND_SIGNED) {
      out_info->signed_min =
          iree_max(out_info->signed_min, immediate->signed_min);
    }
    unsigned_max = iree_min(unsigned_max, immediate->unsigned_max);
    ++offset_immediate_count;
  }
  if (offset_immediate_count != expected_offset_immediate_count) {
    return false;
  }
  out_info->unsigned_max = unsigned_max;
  return true;
}

static bool loom_amdgpu_memory_access_plan_try_select_buffer_off_zero(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_plan_t* access) {
  if (access->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
      access->dynamic_index != LOOM_VALUE_ID_INVALID ||
      access->scalar_byte_offset != 0) {
    return false;
  }
  uint64_t descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_buffer_memory_descriptor(
          descriptor_set, access,
          LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO, kind, &descriptor_id,
          &descriptor_ordinal)) {
    return false;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1,
          LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info) ||
      offset_info.unit_byte_count != 1 ||
      (uint64_t)access->immediate_offset > offset_info.unsigned_max) {
    return false;
  }
  access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO;
  access->descriptor_id = descriptor_id;
  return true;
}

static bool loom_amdgpu_memory_access_split_static_offset(
    loom_amdgpu_memory_access_plan_t* access, uint32_t offset_unit_byte_count,
    uint64_t offset_unsigned_max,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (access->static_byte_offset < 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET;
    return false;
  }
  if (offset_unit_byte_count == 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }

  const uint64_t static_byte_offset = (uint64_t)access->static_byte_offset;
  if (access->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    if ((static_byte_offset % offset_unit_byte_count) != 0) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE;
      return false;
    }
    const uint64_t encoded_offset = static_byte_offset / offset_unit_byte_count;
    if (encoded_offset > offset_unsigned_max) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE;
      return false;
    }
    access->immediate_offset = (uint32_t)encoded_offset;
    access->scalar_byte_offset = 0;
    return true;
  }
  if (offset_unit_byte_count != 1) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }

  uint64_t immediate_offset = iree_min(static_byte_offset, offset_unsigned_max);
  if (access->vgpr_count == 4) {
    immediate_offset &= ~UINT64_C(15);
  }
  const uint64_t scalar_byte_offset = static_byte_offset - immediate_offset;
  if (scalar_byte_offset > UINT32_MAX) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE;
    return false;
  }
  access->immediate_offset = (uint32_t)immediate_offset;
  access->scalar_byte_offset = (uint32_t)scalar_byte_offset;
  return true;
}

static bool loom_amdgpu_memory_access_split_global_saddr_static_offset(
    loom_amdgpu_memory_access_plan_t* access,
    const loom_amdgpu_descriptor_offset_immediate_info_t* offset_info,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (offset_info->unit_byte_count != 1 ||
      offset_info->kind != LOOM_LOW_IMMEDIATE_KIND_SIGNED) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE |
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_UNAVAILABLE;
    return false;
  }
  const int64_t static_byte_offset = access->static_byte_offset;
  const int64_t signed_max = offset_info->unsigned_max > INT64_MAX
                                 ? INT64_MAX
                                 : (int64_t)offset_info->unsigned_max;
  if (static_byte_offset < offset_info->signed_min ||
      static_byte_offset > signed_max) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE |
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_OFFSET_RANGE;
    return false;
  }
  access->immediate_offset = static_byte_offset;
  access->scalar_byte_offset = 0;
  return true;
}

static bool loom_amdgpu_memory_access_split_ds2_static_offset(
    loom_amdgpu_memory_access_plan_t* access, uint32_t offset_unit_byte_count,
    uint64_t offset_unsigned_max) {
  if (access->static_byte_offset < 0) {
    return false;
  }
  int64_t secondary_byte_offset = 0;
  if (!iree_checked_add_i64(access->static_byte_offset,
                            access->vector_lane_byte_stride,
                            &secondary_byte_offset) ||
      secondary_byte_offset < 0) {
    return false;
  }

  const uint64_t byte_offsets[] = {
      (uint64_t)access->static_byte_offset,
      (uint64_t)secondary_byte_offset,
  };
  uint32_t encoded_offsets[IREE_ARRAYSIZE(byte_offsets)] = {0};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(byte_offsets); ++i) {
    if ((byte_offsets[i] % offset_unit_byte_count) != 0) {
      return false;
    }
    const uint64_t encoded_offset = byte_offsets[i] / offset_unit_byte_count;
    if (encoded_offset > offset_unsigned_max) {
      return false;
    }
    encoded_offsets[i] = (uint32_t)encoded_offset;
  }

  access->immediate_offset = encoded_offsets[0];
  access->secondary_immediate_offset = encoded_offsets[1];
  access->scalar_byte_offset = 0;
  return true;
}

static bool loom_amdgpu_select_ds2_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  static const loom_amdgpu_ds2_memory_descriptor_candidate_t kCandidates[] = {
      {
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
          .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_READ2_B32,
      },
      {
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
          .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_WRITE2_B32,
      },
      {
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
          .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_READ2ST64_B32,
      },
      {
          .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
          .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_WRITE2ST64_B32,
      },
  };

  bool found_kind_descriptor = false;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kCandidates); ++i) {
    const loom_amdgpu_ds2_memory_descriptor_candidate_t* candidate =
        &kCandidates[i];
    if (candidate->kind != kind) {
      continue;
    }
    const uint32_t descriptor_ordinal =
        loom_low_descriptor_set_lookup_descriptor_by_id(
            descriptor_set, candidate->descriptor_id);
    if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
      continue;
    }
    found_kind_descriptor = true;
    loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
    if (!loom_amdgpu_descriptor_offset_immediate_info(
            descriptor_set, descriptor_ordinal, 2,
            LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info)) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
      return false;
    }
    if (!loom_amdgpu_memory_access_split_ds2_static_offset(
            access, offset_info.unit_byte_count, offset_info.unsigned_max)) {
      continue;
    }
    access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR;
    access->descriptor_id = candidate->descriptor_id;
    return true;
  }

  diagnostic->rejection_bits |=
      found_kind_descriptor
          ? LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE
          : LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING;
  return false;
}

// The current slot builder may reorder multiple LDS slots around the low
// function anchor, so a single source LDS root is the only source-local proof
// that the selected slot will receive byte offset zero.
static bool loom_amdgpu_source_function_proves_zero_lds_slot_base(
    loom_func_like_t source_function, loom_value_id_t root_value_id) {
  if (!loom_func_like_isa(source_function)) {
    return false;
  }
  loom_region_t* body = loom_func_like_body(source_function);
  if (body == NULL) {
    return false;
  }
  bool found_root = false;
  uint32_t workgroup_alloca_count = 0;
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_buffer_alloca_isa(op) ||
          loom_buffer_alloca_memory_space(op) !=
              LOOM_BUFFER_MEMORY_SPACE_WORKGROUP) {
        continue;
      }
      ++workgroup_alloca_count;
      found_root = found_root || loom_buffer_alloca_result(op) == root_value_id;
      if (workgroup_alloca_count > 1) {
        return false;
      }
    }
  }
  return found_root && workgroup_alloca_count == 1;
}

static bool loom_amdgpu_try_select_ds_addtid_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_func_like_t source_function, loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_operation_kind_t kind) {
  if (access->vgpr_count != 1 ||
      access->dynamic_index == LOOM_VALUE_ID_INVALID ||
      access->dynamic_index_kind != LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR ||
      access->dynamic_index_source !=
          LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID ||
      access->dynamic_index_dimension != LOOM_KERNEL_DIMENSION_X ||
      access->dynamic_index_byte_stride != access->element_byte_count ||
      access->vector_lane_byte_stride != access->element_byte_count ||
      !loom_amdgpu_source_function_proves_zero_lds_slot_base(
          source_function, access->root_value_id)) {
    return false;
  }

  const uint64_t descriptor_id =
      kind == LOOM_AMDGPU_MEMORY_OPERATION_LOAD
          ? LOOM_AMDGPU_DESCRIPTOR_ID_DS_READ_ADDTID_B32
          : LOOM_AMDGPU_DESCRIPTOR_ID_DS_WRITE_ADDTID_B32;
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_lookup_descriptor_by_id(descriptor_set,
                                                      descriptor_id);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return false;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1,
          LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info)) {
    return false;
  }

  loom_amdgpu_memory_access_diagnostic_t ignored_diagnostic = {0};
  if (!loom_amdgpu_memory_access_split_static_offset(
          access, offset_info.unit_byte_count, offset_info.unsigned_max,
          &ignored_diagnostic)) {
    return false;
  }
  access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID;
  access->descriptor_id = descriptor_id;
  return true;
}

static bool loom_amdgpu_descriptor_has_implicit_operand(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal) {
  if (descriptor_ordinal >= descriptor_set->descriptor_count) {
    return false;
  }
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_ordinal];
  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    if (iree_any_bit_set(operands[i].flags, LOOM_LOW_OPERAND_FLAG_IMPLICIT)) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_memory_access_plan_try_select_buffer(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (access->vgpr_count == 4 &&
      access->dynamic_index != LOOM_VALUE_ID_INVALID &&
      (access->dynamic_index_byte_stride & 15) != 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_DYNAMIC_ALIGNMENT;
    return false;
  }
  if (access->vgpr_count == 4 && (access->static_byte_offset & 15) != 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_STATIC_ALIGNMENT;
    return false;
  }

  uint64_t descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_buffer_memory_descriptor(
          descriptor_set, access, LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT, kind,
          &descriptor_id, &descriptor_ordinal)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING;
    return false;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1,
          LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }
  if (!loom_amdgpu_memory_access_split_static_offset(
          access, offset_info.unit_byte_count, offset_info.unsigned_max,
          diagnostic)) {
    return false;
  }
  access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT;
  access->descriptor_id = descriptor_id;
  loom_amdgpu_memory_access_plan_try_select_buffer_off_zero(descriptor_set,
                                                            kind, access);
  return true;
}

static bool loom_amdgpu_memory_access_plan_try_select_global_saddr(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (access->dynamic_index_kind == LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_ADDRESS;
    return false;
  }

  uint64_t descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_global_saddr_memory_descriptor(
          descriptor_set, access, kind, &descriptor_id, &descriptor_ordinal)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING |
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_UNAVAILABLE;
    return false;
  }
  if (loom_amdgpu_descriptor_has_implicit_operand(descriptor_set,
                                                  descriptor_ordinal)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING |
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_UNAVAILABLE;
    return false;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1, LOOM_LOW_IMMEDIATE_KIND_SIGNED,
          &offset_info)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE |
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_UNAVAILABLE;
    return false;
  }
  if (!loom_amdgpu_memory_access_split_global_saddr_static_offset(
          access, &offset_info, diagnostic)) {
    return false;
  }
  access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR;
  access->descriptor_id = descriptor_id;
  return true;
}

static bool loom_amdgpu_memory_access_select(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_func_like_t source_function, loom_value_id_t view_value_id,
    loom_value_slice_t dynamic_indices, loom_attribute_t static_indices,
    loom_type_t view_type, loom_type_t vector_type,
    loom_amdgpu_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_plan_t* out_access,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic) {
  IREE_ASSERT_ARGUMENT(out_access);
  IREE_ASSERT_ARGUMENT(out_diagnostic);
  *out_access = (loom_amdgpu_memory_access_plan_t){
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
      .root_value_id = LOOM_VALUE_ID_INVALID,
      .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
      .dynamic_index = LOOM_VALUE_ID_INVALID,
      .dynamic_index_kind = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE,
      .dynamic_index_source = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOURCE_NONE,
      .dynamic_index_dimension = LOOM_KERNEL_DIMENSION_COUNT_,
      .dynamic_index_byte_shift = LOOM_AMDGPU_MEMORY_ACCESS_BYTE_SHIFT_NONE,
      .descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE,
  };
  *out_diagnostic = (loom_amdgpu_memory_access_diagnostic_t){0};

  loom_vector_memory_access_t vector_access;
  if (!loom_vector_memory_access_describe(module, view_type, vector_type,
                                          &vector_access)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIBE_FAILED;
    return false;
  }
  switch (vector_access.layout_kind) {
    case LOOM_VECTOR_MEMORY_LAYOUT_DENSE:
    case LOOM_VECTOR_MEMORY_LAYOUT_STRIDED:
      break;
    case LOOM_VECTOR_MEMORY_LAYOUT_UNKNOWN:
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_LAYOUT;
      return false;
  }
  if (vector_access.static_element_byte_count <= 0 ||
      vector_access.static_element_byte_count > UINT32_MAX) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_ELEMENT_WIDTH;
    return false;
  }
  out_access->element_byte_count =
      (uint32_t)vector_access.static_element_byte_count;
  if (vector_access.vector_rank != 1) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_RANK;
    return false;
  }

  if (!loom_amdgpu_memory_access_register_footprint(vector_type, out_access,
                                                    out_diagnostic)) {
    return false;
  }
  if (out_access->vgpr_count == 0 ||
      out_access->vgpr_count > LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE;
    return false;
  }

  int64_t vector_axis_stride = 0;
  if (!loom_vector_memory_access_static_axis_stride(
          &vector_access, vector_access.first_vector_axis,
          &vector_axis_stride)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
    return false;
  }
  int64_t vector_lane_byte_stride = 0;
  if (!iree_checked_mul_i64(vector_axis_stride,
                            vector_access.static_element_byte_count,
                            &vector_lane_byte_stride) ||
      vector_lane_byte_stride <= 0 || vector_lane_byte_stride > UINT32_MAX) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
    return false;
  }
  out_access->vector_lane_byte_stride = (uint32_t)vector_lane_byte_stride;

  if (dynamic_indices.count == 0) {
    int64_t lane_indices[] = {0};
    int64_t static_byte_offset = 0;
    if (!loom_vector_memory_access_static_lane_byte_offset(
            &vector_access, static_indices, lane_indices,
            IREE_ARRAYSIZE(lane_indices), &static_byte_offset)) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_STATIC_OFFSET;
      return false;
    }
    if (!loom_amdgpu_memory_access_add_view_base_byte_offset(
            fact_table, view_value_id, out_access, out_diagnostic,
            &static_byte_offset)) {
      return false;
    }
    if (!loom_amdgpu_memory_access_static_byte_offset_is_usable(
            static_byte_offset, out_diagnostic)) {
      return false;
    }
    out_access->static_byte_offset = static_byte_offset;
  } else {
    if (dynamic_indices.count != 1) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_COUNT;
      return false;
    }
    uint8_t dynamic_axis = UINT8_MAX;
    if (!loom_amdgpu_memory_access_find_dynamic_axis(static_indices,
                                                     &dynamic_axis) ||
        dynamic_axis == UINT8_MAX || dynamic_axis >= vector_access.view_rank) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_AXIS;
      return false;
    }
    loom_kernel_dimension_t dynamic_index_dimension =
        LOOM_KERNEL_DIMENSION_COUNT_;
    const bool dynamic_index_is_workitem_id =
        loom_amdgpu_module_value_as_workitem_id(
            module, dynamic_indices.values[0], &dynamic_index_dimension);
    const bool dynamic_index_is_workgroup_id =
        !dynamic_index_is_workitem_id &&
        loom_amdgpu_module_value_as_workgroup_id(
            module, dynamic_indices.values[0], &dynamic_index_dimension);
    const bool dynamic_index_is_computed_vaddr =
        !dynamic_index_is_workitem_id && !dynamic_index_is_workgroup_id &&
        loom_amdgpu_module_value_prefers_vgpr(module,
                                              dynamic_indices.values[0]);
    if (!dynamic_index_is_workitem_id && !dynamic_index_is_workgroup_id &&
        !dynamic_index_is_computed_vaddr) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_SOURCE;
      return false;
    }

    int64_t axis_stride = 0;
    if (!loom_vector_memory_access_static_axis_stride(
            &vector_access, dynamic_axis, &axis_stride)) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE;
      return false;
    }
    int64_t dynamic_index_byte_stride = 0;
    if (!iree_checked_mul_i64(axis_stride,
                              vector_access.static_element_byte_count,
                              &dynamic_index_byte_stride) ||
        dynamic_index_byte_stride < 0 ||
        dynamic_index_byte_stride > UINT32_MAX) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE;
      return false;
    }
    int64_t static_byte_offset = 0;
    if (!loom_amdgpu_memory_access_static_byte_offset(
            &vector_access, static_indices, dynamic_axis,
            &static_byte_offset)) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_STATIC_OFFSET;
      return false;
    }
    if (!loom_amdgpu_memory_access_add_view_base_byte_offset(
            fact_table, view_value_id, out_access, out_diagnostic,
            &static_byte_offset)) {
      return false;
    }
    if (dynamic_index_is_workgroup_id &&
        out_access->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_DYNAMIC_INDEX_SOURCE;
      return false;
    }
    if (!loom_amdgpu_memory_access_static_byte_offset_is_usable(
            static_byte_offset, out_diagnostic)) {
      return false;
    }
    out_access->dynamic_index = dynamic_indices.values[0];
    out_access->dynamic_index_kind =
        dynamic_index_is_workgroup_id ? LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET
                                      : LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
    if (dynamic_index_is_workgroup_id) {
      out_access->dynamic_index_source =
          LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID;
    } else if (dynamic_index_is_workitem_id) {
      out_access->dynamic_index_source =
          LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID;
    } else {
      out_access->dynamic_index_source =
          LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOURCE_COMPUTED_VADDR;
    }
    out_access->dynamic_index_dimension = dynamic_index_dimension;
    out_access->dynamic_index_byte_stride = (uint32_t)dynamic_index_byte_stride;
    if (loom_amdgpu_u32_is_power_of_two((uint32_t)dynamic_index_byte_stride)) {
      uint32_t dynamic_index_byte_shift = 0;
      uint32_t remaining_stride = (uint32_t)dynamic_index_byte_stride;
      while (remaining_stride > 1) {
        remaining_stride >>= 1;
        ++dynamic_index_byte_shift;
      }
      out_access->dynamic_index_byte_shift = dynamic_index_byte_shift;
    }
    if (out_access->dynamic_index_kind ==
            LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET &&
        dynamic_index_byte_stride != 1 &&
        out_access->dynamic_index_byte_shift ==
            LOOM_AMDGPU_MEMORY_ACCESS_BYTE_SHIFT_NONE) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_SCALAR_DYNAMIC_STRIDE;
      return false;
    }
    out_access->static_byte_offset = static_byte_offset;
  }

  if (out_access->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    if (out_access->root_value_id >= module->values.count) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_ROOT;
      return false;
    }
    const loom_value_t* root_value =
        loom_module_value(module, out_access->root_value_id);
    const loom_op_t* root_op = loom_value_is_block_arg(root_value)
                                   ? NULL
                                   : loom_value_def_op(root_value);
    if (root_op == NULL || !loom_buffer_alloca_isa(root_op)) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_ROOT;
      return false;
    }
    if (loom_amdgpu_try_select_ds_addtid_memory_descriptor(
            descriptor_set, source_function, out_access, kind)) {
      return true;
    }
    if (out_access->vector_lane_byte_stride != out_access->element_byte_count) {
      if (out_access->vgpr_count != 2 ||
          !loom_amdgpu_memory_access_has_32bit_lanes(out_access)) {
        out_diagnostic->rejection_bits |=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
        return false;
      }
      return loom_amdgpu_select_ds2_memory_descriptor(
          descriptor_set, out_access, kind, out_diagnostic);
    }
  } else if (out_access->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN &&
             out_access->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL &&
             out_access->memory_space !=
                 LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT &&
             out_access->memory_space !=
                 LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_MEMORY_SPACE;
    return false;
  }
  if (out_access->vector_lane_byte_stride != out_access->element_byte_count) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
    return false;
  }

  if (out_access->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
    if (!loom_amdgpu_select_memory_descriptor(descriptor_set, out_access, kind,
                                              &out_access->descriptor_id,
                                              &descriptor_ordinal)) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING;
      return false;
    }
    loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
    if (!loom_amdgpu_descriptor_offset_immediate_info(
            descriptor_set, descriptor_ordinal, 1,
            LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info)) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
      return false;
    }
    return loom_amdgpu_memory_access_split_static_offset(
        out_access, offset_info.unit_byte_count, offset_info.unsigned_max,
        out_diagnostic);
  }

  if (loom_amdgpu_memory_access_plan_try_select_buffer(
          descriptor_set, kind, out_access, out_diagnostic)) {
    return true;
  }
  return loom_amdgpu_memory_access_plan_try_select_global_saddr(
      descriptor_set, kind, out_access, out_diagnostic);
}

static bool loom_amdgpu_load_memory_access_select_with_source_function(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_func_like_t source_function,
    loom_amdgpu_memory_access_plan_t* out_access) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_amdgpu_memory_access_diagnostic_t diagnostic = {0};
  return loom_amdgpu_memory_access_select(
      module, loom_low_lower_context_fact_table(context),
      loom_low_lower_context_descriptor_set(context), source_function,
      loom_vector_load_view(source_op), loom_vector_load_indices(source_op),
      loom_vector_load_static_indices(source_op),
      loom_module_value_type(module, loom_vector_load_view(source_op)),
      loom_module_value_type(module, loom_vector_load_result(source_op)),
      LOOM_AMDGPU_MEMORY_OPERATION_LOAD, out_access, &diagnostic);
}

static bool loom_amdgpu_load_memory_access_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_plan_t* out_access) {
  return loom_amdgpu_load_memory_access_select_with_source_function(
      context, source_op, loom_low_lower_context_source_function(context),
      out_access);
}

static bool loom_amdgpu_store_memory_access_select_with_source_function(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_func_like_t source_function,
    loom_amdgpu_memory_access_plan_t* out_access) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_amdgpu_memory_access_diagnostic_t diagnostic = {0};
  return loom_amdgpu_memory_access_select(
      module, loom_low_lower_context_fact_table(context),
      loom_low_lower_context_descriptor_set(context), source_function,
      loom_vector_store_view(source_op), loom_vector_store_indices(source_op),
      loom_vector_store_static_indices(source_op),
      loom_module_value_type(module, loom_vector_store_view(source_op)),
      loom_module_value_type(module, loom_vector_store_value(source_op)),
      LOOM_AMDGPU_MEMORY_OPERATION_STORE, out_access, &diagnostic);
}

static bool loom_amdgpu_store_memory_access_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_plan_t* out_access) {
  return loom_amdgpu_store_memory_access_select_with_source_function(
      context, source_op, loom_low_lower_context_source_function(context),
      out_access);
}

static bool loom_amdgpu_memory_cache_policy_is_present(
    const loom_vector_memory_cache_policy_t* policy) {
  return iree_any_bit_set(
      policy->build_flags,
      LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_SCOPE |
          LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_TEMPORAL);
}

static bool loom_amdgpu_memory_cache_policy_is_complete(
    const loom_vector_memory_cache_policy_t* policy) {
  const uint32_t required_flags =
      LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_SCOPE |
      LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_TEMPORAL;
  return policy->build_flags == 0 || policy->build_flags == required_flags;
}

static bool loom_amdgpu_memory_cache_policy_is_regular_device(
    const loom_vector_memory_cache_policy_t* policy) {
  return policy->cache_scope == LOOM_CACHE_SCOPE_DEVICE &&
         policy->cache_temporal == LOOM_CACHE_TEMPORAL_REGULAR;
}

static bool loom_amdgpu_memory_cache_policy_gfx12_th(uint8_t temporal,
                                                     int64_t* out_th) {
  *out_th = 0;
  switch (temporal) {
    case LOOM_CACHE_TEMPORAL_REGULAR:
      *out_th = 0;
      return true;
    case LOOM_CACHE_TEMPORAL_NON_TEMPORAL:
      *out_th = 1;
      return true;
    case LOOM_CACHE_TEMPORAL_HIGH_TEMPORAL:
      *out_th = 2;
      return true;
    case LOOM_CACHE_TEMPORAL_LAST_USE:
    case LOOM_CACHE_TEMPORAL_WRITEBACK:
    case LOOM_CACHE_TEMPORAL_BYPASS:
      *out_th = 3;
      return true;
    case LOOM_CACHE_TEMPORAL_NON_TEMPORAL_REGULAR:
      *out_th = 4;
      return true;
    case LOOM_CACHE_TEMPORAL_REGULAR_NON_TEMPORAL:
      *out_th = 5;
      return true;
    case LOOM_CACHE_TEMPORAL_NON_TEMPORAL_HIGH_TEMPORAL:
      *out_th = 6;
      return true;
    case LOOM_CACHE_TEMPORAL_NON_TEMPORAL_WRITEBACK:
      *out_th = 7;
      return true;
  }
  return false;
}

static bool loom_amdgpu_memory_cache_policy_can_lower(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_plan_t* access,
    const loom_op_t* source_op) {
  loom_vector_memory_cache_policy_t policy = {0};
  if (!loom_vector_memory_cache_policy_from_op(source_op, &policy)) {
    return false;
  }
  if (!loom_amdgpu_memory_cache_policy_is_present(&policy)) {
    return true;
  }
  if (!loom_amdgpu_memory_cache_policy_is_complete(&policy) ||
      access->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return false;
  }

  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_by_id(descriptor_set->stable_id);
  if (descriptor_set_info == NULL) {
    return false;
  }
  switch (descriptor_set_info->vector_memory_cache_policy_encoding) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH: {
      int64_t th = 0;
      return loom_cache_scope_is_valid(policy.cache_scope) &&
             loom_amdgpu_memory_cache_policy_gfx12_th(policy.cache_temporal,
                                                      &th);
    }
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1:
      return policy.cache_scope == LOOM_CACHE_SCOPE_DEVICE &&
             (policy.cache_temporal == LOOM_CACHE_TEMPORAL_REGULAR ||
              policy.cache_temporal == LOOM_CACHE_TEMPORAL_NON_TEMPORAL);
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC:
      return loom_amdgpu_memory_cache_policy_is_regular_device(&policy);
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE:
      return false;
  }
  return false;
}

bool loom_amdgpu_select_vector_load_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_plan_t* out_plan) {
  if (!loom_amdgpu_load_memory_access_select(context, source_op, out_plan)) {
    return false;
  }
  return loom_amdgpu_memory_cache_policy_can_lower(
      loom_low_lower_context_descriptor_set(context), out_plan, source_op);
}

bool loom_amdgpu_select_vector_store_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_plan_t* out_plan) {
  if (!loom_amdgpu_store_memory_access_select(context, source_op, out_plan)) {
    return false;
  }
  return loom_amdgpu_memory_cache_policy_can_lower(
      loom_low_lower_context_descriptor_set(context), out_plan, source_op);
}

bool loom_amdgpu_source_op_selects_m0_descriptor(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t* out_descriptor_id) {
  IREE_ASSERT_ARGUMENT(out_descriptor_id);
  *out_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  loom_amdgpu_memory_access_plan_t plan;
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_LOAD:
      if (!loom_amdgpu_select_vector_load_plan(context, source_op, &plan)) {
        return false;
      }
      break;
    case LOOM_OP_VECTOR_STORE:
      if (!loom_amdgpu_select_vector_store_plan(context, source_op, &plan)) {
        return false;
      }
      break;
    default:
      return false;
  }
  if (plan.address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID) {
    return false;
  }
  *out_descriptor_id = plan.descriptor_id;
  return true;
}

static iree_status_t loom_amdgpu_emit_memory_vaddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* access,
    loom_value_id_t low_base_addr, loom_value_id_t* out_low_vaddr) {
  IREE_ASSERT_ARGUMENT(out_low_vaddr);
  *out_low_vaddr = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  if (access->dynamic_index == LOOM_VALUE_ID_INVALID) {
    if (low_base_addr != LOOM_VALUE_ID_INVALID) {
      *out_low_vaddr = low_base_addr;
      return iree_ok_status();
    }
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32, 0,
                                      vgpr_type, out_low_vaddr);
  }
  if (access->dynamic_index_kind == LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET) {
    if (low_base_addr != LOOM_VALUE_ID_INVALID) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU scalar dynamic offsets are only supported for buffer memory");
    }
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32, 0,
                                      vgpr_type, out_low_vaddr);
  }

  loom_value_id_t low_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, access->dynamic_index, &low_index));
  if (access->dynamic_index_byte_stride == 1) {
    if (low_base_addr == LOOM_VALUE_ID_INVALID) {
      *out_low_vaddr = low_index;
      return iree_ok_status();
    }
    loom_value_id_t operands[] = {low_base_addr, low_index};
    loom_op_t* low_add_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0),
        &vgpr_type, 1, &low_add_op));
    *out_low_vaddr = loom_value_slice_get(loom_low_op_results(low_add_op), 0);
    return iree_ok_status();
  }

  const bool use_shift = access->dynamic_index_byte_shift !=
                         LOOM_AMDGPU_MEMORY_ACCESS_BYTE_SHIFT_NONE;
  loom_value_id_t low_scale = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32,
      use_shift ? access->dynamic_index_byte_shift
                : access->dynamic_index_byte_stride,
      vgpr_type, &low_scale));
  loom_value_id_t operands[] = {
      use_shift ? low_scale : low_index,
      use_shift ? low_index : low_scale,
  };
  loom_op_t* low_offset_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op,
      use_shift ? LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHLREV_B32
                : LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32,
      operands, IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0),
      &vgpr_type, 1, &low_offset_op));
  loom_value_id_t low_offset =
      loom_value_slice_get(loom_low_op_results(low_offset_op), 0);
  if (low_base_addr == LOOM_VALUE_ID_INVALID) {
    *out_low_vaddr = low_offset;
    return iree_ok_status();
  }

  loom_value_id_t add_operands[] = {low_base_addr, low_offset};
  loom_op_t* low_add_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32, add_operands,
      IREE_ARRAYSIZE(add_operands), loom_make_named_attr_slice(NULL, 0),
      &vgpr_type, 1, &low_add_op));
  *out_low_vaddr = loom_value_slice_get(loom_low_op_results(low_add_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_memory_soffset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* access,
    loom_value_id_t* out_low_soffset) {
  IREE_ASSERT_ARGUMENT(out_low_soffset);
  *out_low_soffset = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  if (access->dynamic_index_kind != LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET) {
    return loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B32,
        access->scalar_byte_offset, sgpr_type, out_low_soffset);
  }

  loom_value_id_t low_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, access->dynamic_index, &low_index));
  loom_value_id_t low_dynamic_offset = low_index;
  if (access->dynamic_index_byte_stride != 1) {
    IREE_ASSERT(access->dynamic_index_byte_shift !=
                LOOM_AMDGPU_MEMORY_ACCESS_BYTE_SHIFT_NONE);
    loom_value_id_t low_shift = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B32,
        access->dynamic_index_byte_shift, sgpr_type, &low_shift));
    loom_value_id_t shift_operands[] = {low_index, low_shift};
    loom_op_t* low_shift_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_LSHL_B32,
        shift_operands, IREE_ARRAYSIZE(shift_operands),
        loom_make_named_attr_slice(NULL, 0), &sgpr_type, 1, &low_shift_op));
    low_dynamic_offset =
        loom_value_slice_get(loom_low_op_results(low_shift_op), 0);
  }

  if (access->scalar_byte_offset == 0) {
    *out_low_soffset = low_dynamic_offset;
    return iree_ok_status();
  }

  loom_value_id_t low_static_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B32,
      access->scalar_byte_offset, sgpr_type, &low_static_offset));
  loom_value_id_t add_operands[] = {low_dynamic_offset, low_static_offset};
  loom_op_t* low_add_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_ADD_U32, add_operands,
      IREE_ARRAYSIZE(add_operands), loom_make_named_attr_slice(NULL, 0),
      &sgpr_type, 1, &low_add_op));
  *out_low_soffset = loom_value_slice_get(loom_low_op_results(low_add_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_memory_saddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_resource, loom_value_id_t* out_low_saddr) {
  IREE_ASSERT_ARGUMENT(out_low_saddr);
  *out_low_saddr = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &sgpr_x2_type));
  return loom_amdgpu_emit_low_slice(context, source_op, low_resource,
                                    /*offset=*/0, sgpr_x2_type, out_low_saddr);
}

static iree_status_t loom_amdgpu_memory_cache_policy_rejected_status(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_plan_t* access,
    const loom_vector_memory_cache_policy_t* policy) {
  if (!loom_amdgpu_memory_cache_policy_is_complete(policy)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU vector memory cache policy requires both cache_scope and "
        "cache_temporal");
  }
  const iree_string_view_t scope_name =
      loom_amdgpu_cache_scope_name(policy->cache_scope);
  const iree_string_view_t temporal_name =
      loom_amdgpu_cache_temporal_name(policy->cache_temporal);
  if (access->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU workgroup memory packets cannot encode cache policy "
        "%.*s/%.*s",
        (int)scope_name.size, scope_name.data, (int)temporal_name.size,
        temporal_name.data);
  }
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_by_id(descriptor_set->stable_id);
  if (descriptor_set_info == NULL) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AMDGPU descriptor set stable ID 0x%016" PRIx64
                            " has no cache-policy target-info row",
                            descriptor_set->stable_id);
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDGPU descriptor set '%.*s' cannot faithfully encode vector memory "
      "cache policy %.*s/%.*s",
      (int)descriptor_set_info->descriptor_set_key.size,
      descriptor_set_info->descriptor_set_key.data, (int)scope_name.size,
      scope_name.data, (int)temporal_name.size, temporal_name.data);
}

static iree_status_t loom_amdgpu_append_memory_attr(
    loom_low_lower_context_t* context, iree_string_view_t name, int64_t value,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count) {
  if (*inout_attr_count >= attr_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU memory attr capacity exceeded");
  }
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, name, &name_id));
  attrs[*inout_attr_count] = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
  *inout_attr_count += 1;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_memory_cache_attrs(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* access, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  loom_vector_memory_cache_policy_t policy = {0};
  if (!loom_vector_memory_cache_policy_from_op(source_op, &policy)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU vector memory op has malformed cache "
                            "policy attributes");
  }
  if (!loom_amdgpu_memory_cache_policy_is_present(&policy)) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  if (!loom_amdgpu_memory_cache_policy_can_lower(descriptor_set, access,
                                                 source_op)) {
    return loom_amdgpu_memory_cache_policy_rejected_status(descriptor_set,
                                                           access, &policy);
  }

  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_by_id(descriptor_set->stable_id);
  switch (descriptor_set_info->vector_memory_cache_policy_encoding) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH: {
      int64_t th = 0;
      if (!loom_amdgpu_memory_cache_policy_gfx12_th(policy.cache_temporal,
                                                    &th)) {
        return loom_amdgpu_memory_cache_policy_rejected_status(descriptor_set,
                                                               access, &policy);
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_append_memory_attr(
          context, IREE_SV("scope"), policy.cache_scope, attrs, attr_capacity,
          inout_attr_count));
      return loom_amdgpu_append_memory_attr(context, IREE_SV("th"), th, attrs,
                                            attr_capacity, inout_attr_count);
    }
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1:
      if (policy.cache_temporal == LOOM_CACHE_TEMPORAL_NON_TEMPORAL) {
        return loom_amdgpu_append_memory_attr(context, IREE_SV("nt"), 1, attrs,
                                              attr_capacity, inout_attr_count);
      }
      return iree_ok_status();
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC:
      return iree_ok_status();
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE:
      break;
  }
  return loom_amdgpu_memory_cache_policy_rejected_status(descriptor_set, access,
                                                         &policy);
}

static iree_status_t loom_amdgpu_make_memory_attrs(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* access, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* out_attr_count) {
  IREE_ASSERT_ARGUMENT(access);
  IREE_ASSERT_ARGUMENT(attrs);
  IREE_ASSERT_ARGUMENT(out_attr_count);
  *out_attr_count = 0;
  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_memory_attr(
        context, IREE_SV("offset0"), access->immediate_offset, attrs,
        attr_capacity, out_attr_count));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_memory_attr(
        context, IREE_SV("offset1"), access->secondary_immediate_offset, attrs,
        attr_capacity, out_attr_count));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_memory_attr(
        context, IREE_SV("offset"), access->immediate_offset, attrs,
        attr_capacity, out_attr_count));
  }
  return loom_amdgpu_append_memory_cache_attrs(
      context, source_op, access, attrs, attr_capacity, out_attr_count);
}

iree_status_t loom_amdgpu_lower_vector_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* access) {
  IREE_ASSERT_ARGUMENT(access);
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_load_view(source_op), &low_resource));

  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  if (access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO &&
      access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID) {
    const loom_value_id_t low_base_addr =
        access->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP
            ? low_resource
            : LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_vaddr(
        context, source_op, access, low_base_addr, &low_vaddr));
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, loom_vector_load_result(source_op), &result_type));

  loom_named_attr_t attrs[5] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_memory_attrs(
      context, source_op, access, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  if (access->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID) {
      loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_m0_live_in(context, &low_m0));
      loom_value_id_t operands[] = {low_m0};
      loom_op_t* low_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_low_op(context, source_op, access->descriptor_id,
                                  operands, IREE_ARRAYSIZE(operands),
                                  loom_make_named_attr_slice(attrs, attr_count),
                                  &result_type, 1, &low_op));
      return loom_low_lower_bind_value(
          context, loom_vector_load_result(source_op),
          loom_value_slice_get(loom_low_op_results(low_op), 0));
    }
    loom_value_id_t operands[] = {low_vaddr};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, access->descriptor_id, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
        &result_type, 1, &low_op));
    return loom_low_lower_bind_value(
        context, loom_vector_load_result(source_op),
        loom_value_slice_get(loom_low_op_results(low_op), 0));
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
    loom_value_id_t low_saddr = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr(
        context, source_op, low_resource, &low_saddr));
    loom_value_id_t operands[] = {
        low_vaddr,
        low_saddr,
    };
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, access->descriptor_id, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
        &result_type, 1, &low_op));
    return loom_low_lower_bind_value(
        context, loom_vector_load_result(source_op),
        loom_value_slice_get(loom_low_op_results(low_op), 0));
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO) {
    loom_value_id_t operands[] = {low_resource};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, access->descriptor_id, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
        &result_type, 1, &low_op));
    return loom_low_lower_bind_value(
        context, loom_vector_load_result(source_op),
        loom_value_slice_get(loom_low_op_results(low_op), 0));
  }

  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_soffset(context, source_op,
                                                       access, &low_soffset));
  loom_value_id_t operands[] = {
      low_resource,
      low_vaddr,
      low_soffset,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, access->descriptor_id, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
      &result_type, 1, &low_op));
  return loom_low_lower_bind_value(
      context, loom_vector_load_result(source_op),
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

iree_status_t loom_amdgpu_lower_vector_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_plan_t* access) {
  IREE_ASSERT_ARGUMENT(access);
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_store_value(source_op), &low_value));
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_store_view(source_op), &low_resource));

  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  if (access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO &&
      access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID) {
    const loom_value_id_t low_base_addr =
        access->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP
            ? low_resource
            : LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_vaddr(
        context, source_op, access, low_base_addr, &low_vaddr));
  }

  loom_named_attr_t attrs[5] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_memory_attrs(
      context, source_op, access, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  if (access->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID) {
      loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_m0_live_in(context, &low_m0));
      loom_value_id_t operands[] = {
          low_value,
          low_m0,
      };
      loom_op_t* low_op = NULL;
      return loom_amdgpu_emit_low_op(
          context, source_op, access->descriptor_id, operands,
          IREE_ARRAYSIZE(operands),
          loom_make_named_attr_slice(attrs, attr_count),
          /*result_types=*/NULL, /*result_count=*/0, &low_op);
    }
    if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR) {
      loom_type_t lane_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      loom_value_id_t low_value0 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_value, 0, lane_type, &low_value0));
      loom_value_id_t low_value1 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_value, 1, lane_type, &low_value1));
      loom_value_id_t operands[] = {
          low_vaddr,
          low_value0,
          low_value1,
      };
      loom_op_t* low_op = NULL;
      return loom_amdgpu_emit_low_op(
          context, source_op, access->descriptor_id, operands,
          IREE_ARRAYSIZE(operands),
          loom_make_named_attr_slice(attrs, attr_count),
          /*result_types=*/NULL, /*result_count=*/0, &low_op);
    }
    loom_value_id_t operands[] = {
        low_vaddr,
        low_value,
    };
    loom_op_t* low_op = NULL;
    return loom_amdgpu_emit_low_op(
        context, source_op, access->descriptor_id, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
        /*result_types=*/NULL, /*result_count=*/0, &low_op);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO) {
    loom_value_id_t operands[] = {
        low_value,
        low_resource,
    };
    loom_op_t* low_op = NULL;
    return loom_amdgpu_emit_low_op(
        context, source_op, access->descriptor_id, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
        /*result_types=*/NULL, /*result_count=*/0, &low_op);
  }

  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR) {
    loom_value_id_t low_saddr = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_saddr(
        context, source_op, low_resource, &low_saddr));
    loom_value_id_t operands[] = {
        low_vaddr,
        low_value,
        low_saddr,
    };
    loom_op_t* low_op = NULL;
    return loom_amdgpu_emit_low_op(
        context, source_op, access->descriptor_id, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
        /*result_types=*/NULL, /*result_count=*/0, &low_op);
  }

  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_soffset(context, source_op,
                                                       access, &low_soffset));
  loom_value_id_t operands[] = {
      low_value,
      low_resource,
      low_vaddr,
      low_soffset,
  };
  loom_op_t* low_op = NULL;
  return loom_amdgpu_emit_low_op(
      context, source_op, access->descriptor_id, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
      /*result_types=*/NULL, /*result_count=*/0, &low_op);
}

iree_status_t loom_amdgpu_low_legality_verify_vector_memory(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_target_low_legality_descriptor_set(context);
  loom_value_slice_t dynamic_indices = {0};
  loom_attribute_t static_indices = {0};
  loom_value_id_t view_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t view_type = loom_type_none();
  loom_type_t vector_type = loom_type_none();
  loom_amdgpu_memory_operation_kind_t kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD;
  switch (op->kind) {
    case LOOM_OP_VECTOR_LOAD:
      dynamic_indices = loom_vector_load_indices(op);
      static_indices = loom_vector_load_static_indices(op);
      view_value_id = loom_vector_load_view(op);
      view_type = loom_module_value_type(module, loom_vector_load_view(op));
      vector_type = loom_module_value_type(module, loom_vector_load_result(op));
      kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD;
      break;
    case LOOM_OP_VECTOR_STORE:
      dynamic_indices = loom_vector_store_indices(op);
      static_indices = loom_vector_store_static_indices(op);
      view_value_id = loom_vector_store_view(op);
      view_type = loom_module_value_type(module, loom_vector_store_view(op));
      vector_type = loom_module_value_type(module, loom_vector_store_value(op));
      kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE;
      break;
    default:
      *out_handled = false;
      return iree_ok_status();
  }

  loom_amdgpu_memory_access_plan_t access = {0};
  loom_amdgpu_memory_access_diagnostic_t diagnostic = {0};
  const loom_func_like_t source_function =
      iree_any_bit_set(loom_target_low_legality_diagnostic_flags(context),
                       LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS)
          ? loom_target_low_legality_function(context)
          : (loom_func_like_t){0};
  if (!loom_amdgpu_memory_access_select(
          module, loom_target_low_legality_fact_table(context), descriptor_set,
          source_function, view_value_id, dynamic_indices, static_indices,
          view_type, vector_type, kind, &access, &diagnostic)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("memory"), loom_op_name(module, op),
        loom_amdgpu_memory_access_rejection_detail(diagnostic.rejection_bits));
  }
  if (!loom_amdgpu_memory_cache_policy_can_lower(descriptor_set, &access, op)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("cache"), loom_op_name(module, op),
        IREE_SV("AMDGPU vector memory cache policy is not representable by "
                "the selected descriptor set"));
  }
  return loom_amdgpu_record_memory_access_diagnostic(
      provider, context, op, descriptor_set, &access, kind);
}
