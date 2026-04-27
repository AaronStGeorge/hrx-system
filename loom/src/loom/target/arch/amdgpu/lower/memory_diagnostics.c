// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/target/arch/amdgpu/lower/internal.h"
#include "loom/target/arch/amdgpu/lower/memory_internal.h"

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
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
      return IREE_SV("generic");
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

typedef struct loom_amdgpu_memory_access_rejection_detail_t {
  // Rejection flag tested by this row.
  loom_amdgpu_memory_access_rejection_flags_t rejection_bit;
  // Stable diagnostic detail returned when rejection_bit is present.
  iree_string_view_t detail;
} loom_amdgpu_memory_access_rejection_detail_t;

static const loom_amdgpu_memory_access_rejection_detail_t
    kAmdgpuMemoryAccessRejectionDetails[] = {
        {
            .rejection_bit = LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE,
            .detail = IREE_SVL(
                "AMDGPU buffer memory lowering currently supports up to four "
                "32-bit memory registers"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE,
            .detail = IREE_SVL(
                "AMDGPU buffer memory lowering requires unit stride along "
                "the vector axis"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_SOURCE,
            .detail = IREE_SVL(
                "AMDGPU buffer memory lowering currently requires dynamic "
                "indices to come from kernel.workitem.id, kernel.workgroup.id, "
                "or VGPR address arithmetic"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE,
            .detail = IREE_SVL(
                "AMDGPU buffer memory lowering requires a non-negative "
                "32-bit dynamic byte stride"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_SCALAR_DYNAMIC_STRIDE,
            .detail = IREE_SVL(
                "AMDGPU scalar dynamic buffer offsets currently require a "
                "power-of-two dynamic byte stride"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_DYNAMIC_INDEX_SOURCE,
            .detail = IREE_SVL(
                "AMDGPU workgroup memory lowering currently requires dynamic "
                "indices to come from kernel.workitem.id"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_UNAVAILABLE,
            .detail = IREE_SVL(
                "AMDGPU buffer memory lowering rejected this access and global "
                "pointer fallback is unavailable in the selected descriptor "
                "set"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_ADDRESS,
            .detail = IREE_SVL(
                "AMDGPU buffer memory lowering rejected this access and global "
                "pointer fallback is blocked by missing vector address facts"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_OFFSET_RANGE,
            .detail = IREE_SVL(
                "AMDGPU buffer memory lowering rejected this access and global "
                "pointer fallback cannot encode the static byte offset"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_PACKED_REGISTER_FOOTPRINT,
            .detail = IREE_SVL(
                "AMDGPU packed integer memory lowering requires the vector "
                "footprint to exactly fill complete 32-bit memory registers"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_FLAT_DYNAMIC_ADDRESS,
            .detail = IREE_SVL(
                "AMDGPU flat memory lowering currently requires a static "
                "source address"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE,
            .detail = IREE_SVL(
                "AMDGPU memory lowering requires dynamic byte offsets to be "
                "proven non-negative and 32-bit"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_DYNAMIC_ALIGNMENT,
            .detail = IREE_SVL(
                "128-bit AMDGPU buffer memory accesses currently require "
                "16-byte aligned dynamic byte strides"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET,
            .detail = IREE_SVL(
                "AMDGPU buffer memory lowering requires non-negative static "
                "byte offsets"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_STATIC_ALIGNMENT,
            .detail = IREE_SVL(
                "128-bit AMDGPU buffer memory accesses currently require "
                "16-byte aligned static byte offsets"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING,
            .detail =
                IREE_SVL("selected AMDGPU descriptor set has no buffer memory "
                         "descriptor for this access width"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE,
            .detail = IREE_SVL(
                "selected AMDGPU buffer memory descriptor does not expose one "
                "unsigned offset immediate"),
        },
        {
            .rejection_bit =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE,
            .detail = IREE_SVL(
                "AMDGPU memory static byte offset is outside the "
                "selected descriptor's immediate/address operand range"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_MEMORY_SPACE,
            .detail = IREE_SVL(
                "AMDGPU memory lowering currently supports HAL/global buffer "
                "resources and workgroup LDS roots"),
        },
        {
            .rejection_bit = LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_ROOT,
            .detail = IREE_SVL(
                "AMDGPU workgroup memory lowering currently requires a "
                "buffer.alloca root"),
        },
};

static loom_amdgpu_memory_access_bank_summary_t
loom_amdgpu_memory_access_bank_summary(
    const loom_amdgpu_memory_access_plan_t* access) {
  loom_amdgpu_memory_access_bank_summary_t summary = {
      .reason = IREE_SV("bank-pattern-unknown"),
  };
  const loom_low_source_memory_dynamic_term_t* term =
      loom_low_source_memory_access_single_dynamic_term(&access->source);
  if (access->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
      !term ||
      access->dynamic_term_kinds[0] != LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR ||
      term->byte_stride == 0 ||
      (term->byte_stride % LOOM_AMDGPU_LDS_BANK_WIDTH_BYTES) != 0) {
    return summary;
  }

  summary.bank_stride_words =
      term->byte_stride / LOOM_AMDGPU_LDS_BANK_WIDTH_BYTES;
  summary.conflict_degree = loom_amdgpu_gcd_u32(summary.bank_stride_words,
                                                LOOM_AMDGPU_LDS_BANK_COUNT);
  const uint32_t vector_footprint_bytes =
      access->packet_byte_count != 0 ? access->packet_byte_count
                                     : access->source.element_byte_count *
                                           iree_max(access->vgpr_count, 1u);
  if (summary.conflict_degree == 1) {
    summary.reason = term->byte_stride > vector_footprint_bytes
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
      access->source.dynamic_term_count == 1
          ? access->source.dynamic_terms[0].byte_stride
          : 0,
      access->source.vector_lane_byte_stride, bank_summary.bank_stride_words,
      bank_summary.conflict_degree, bank_summary.reason);
}

iree_string_view_t loom_amdgpu_memory_access_rejection_detail(
    loom_amdgpu_memory_access_rejection_flags_t rejection_bits) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kAmdgpuMemoryAccessRejectionDetails); ++i) {
    const loom_amdgpu_memory_access_rejection_detail_t* row =
        &kAmdgpuMemoryAccessRejectionDetails[i];
    if (iree_any_bit_set(rejection_bits, row->rejection_bit)) {
      return row->detail;
    }
  }
  return IREE_SV("AMDGPU buffer memory access is not target-legal");
}
