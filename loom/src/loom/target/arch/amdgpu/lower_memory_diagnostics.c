// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/target/arch/amdgpu/lower_internal.h"
#include "loom/target/arch/amdgpu/lower_memory_internal.h"

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

static loom_amdgpu_memory_access_bank_summary_t
loom_amdgpu_memory_access_bank_summary(
    const loom_amdgpu_memory_access_plan_t* access) {
  loom_amdgpu_memory_access_bank_summary_t summary = {
      .reason = IREE_SV("bank-pattern-unknown"),
  };
  if (access->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
      access->source.dynamic_index == LOOM_VALUE_ID_INVALID ||
      access->dynamic_index_kind != LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR ||
      access->source.dynamic_index_byte_stride == 0 ||
      (access->source.dynamic_index_byte_stride %
       LOOM_AMDGPU_LDS_BANK_WIDTH_BYTES) != 0) {
    return summary;
  }

  summary.bank_stride_words = access->source.dynamic_index_byte_stride /
                              LOOM_AMDGPU_LDS_BANK_WIDTH_BYTES;
  summary.conflict_degree = loom_amdgpu_gcd_u32(summary.bank_stride_words,
                                                LOOM_AMDGPU_LDS_BANK_COUNT);
  const uint32_t vector_footprint_bytes =
      access->packet_byte_count != 0 ? access->packet_byte_count
                                     : access->source.element_byte_count *
                                           iree_max(access->vgpr_count, 1u);
  if (summary.conflict_degree == 1) {
    summary.reason =
        access->source.dynamic_index_byte_stride > vector_footprint_bytes
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

iree_status_t loom_amdgpu_record_memory_access_diagnostic(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_operation_kind_t kind) {
  if (!iree_any_bit_set(loom_target_low_legality_diagnostic_flags(context),
                        LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS) ||
      access->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return iree_ok_status();
  }

  iree_string_view_t packet_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_memory_access_descriptor_key(
      descriptor_set, access, &packet_key));
  const loom_amdgpu_memory_access_bank_summary_t bank_summary =
      loom_amdgpu_memory_access_bank_summary(access);
  return loom_target_low_legality_record_memory_access(
      context, provider, op,
      loom_amdgpu_memory_space_name(access->source.memory_space),
      loom_amdgpu_memory_operation_name(kind), packet_key, IREE_SV("selected"),
      access->source.element_byte_count, access->vgpr_count,
      access->source.dynamic_index_byte_stride,
      access->source.vector_lane_byte_stride, bank_summary.bank_stride_words,
      bank_summary.conflict_degree, bank_summary.reason);
}

iree_string_view_t loom_amdgpu_memory_access_rejection_detail(
    loom_amdgpu_memory_access_rejection_flags_t rejection_bits) {
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
