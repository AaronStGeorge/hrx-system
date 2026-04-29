// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/vector/memory.h"

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower/internal.h"
#include "loom/target/arch/amdgpu/lower/memory_internal.h"
#include "loom/util/fact_table.h"

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
                                               &register_count) &&
      !loom_amdgpu_type_packed_16bit_float_storage(
          vector_type, &payload_bit_count, &register_count)) {
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
         access->source.element_byte_count * iree_max(access->vgpr_count, 1u);
}

typedef uint32_t loom_amdgpu_dynamic_index_source_rule_flags_t;

#define LOOM_AMDGPU_DYNAMIC_INDEX_SOURCE_RULE_REJECT_WORKGROUP_MEMORY \
  ((uint32_t)1u << 0)
#define LOOM_AMDGPU_DYNAMIC_INDEX_SOURCE_RULE_REQUIRE_VGPR_VALUE \
  ((uint32_t)1u << 1)

typedef struct loom_amdgpu_dynamic_index_source_rule_t {
  // Source dynamic-index producer matched by this rule.
  loom_low_source_memory_dynamic_index_source_t source;
  // Target operand path selected by this rule.
  loom_amdgpu_memory_dynamic_index_kind_t dynamic_index_kind;
  // Extra semantic predicates required by this rule.
  loom_amdgpu_dynamic_index_source_rule_flags_t flags;
  // Rejection bits reported when the matched rule fails its predicates.
  loom_amdgpu_memory_access_rejection_flags_t rejection_bits;
} loom_amdgpu_dynamic_index_source_rule_t;

static const loom_amdgpu_dynamic_index_source_rule_t
    kAmdgpuDynamicIndexSourceRules[] = {
        {
            .source = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID,
            .dynamic_index_kind = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET,
            .flags =
                LOOM_AMDGPU_DYNAMIC_INDEX_SOURCE_RULE_REJECT_WORKGROUP_MEMORY,
            .rejection_bits =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_DYNAMIC_INDEX_SOURCE,
        },
        {
            .source = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID,
            .dynamic_index_kind = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR,
        },
        {
            .source = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE,
            .dynamic_index_kind = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR,
            .flags = LOOM_AMDGPU_DYNAMIC_INDEX_SOURCE_RULE_REQUIRE_VGPR_VALUE,
            .rejection_bits =
                LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_SOURCE,
        },
};

bool loom_amdgpu_memory_access_select_dynamic_term_kinds(
    const loom_module_t* module, loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  for (uint8_t term_index = 0; term_index < access->source.dynamic_term_count;
       ++term_index) {
    const loom_low_source_memory_dynamic_term_t* term =
        &access->source.dynamic_terms[term_index];
    if (term->byte_stride < 0 || term->byte_stride > UINT32_MAX) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE;
      return false;
    }
    if (!loom_low_source_memory_dynamic_term_fits_unsigned_bit_count(term,
                                                                     32)) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
      return false;
    }

    bool selected = false;
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(kAmdgpuDynamicIndexSourceRules); ++i) {
      const loom_amdgpu_dynamic_index_source_rule_t* rule =
          &kAmdgpuDynamicIndexSourceRules[i];
      if (rule->source != term->source) {
        continue;
      }
      if (iree_any_bit_set(
              rule->flags,
              LOOM_AMDGPU_DYNAMIC_INDEX_SOURCE_RULE_REJECT_WORKGROUP_MEMORY) &&
          access->source.memory_space ==
              LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
        diagnostic->rejection_bits |= rule->rejection_bits;
        return false;
      }
      if (iree_any_bit_set(
              rule->flags,
              LOOM_AMDGPU_DYNAMIC_INDEX_SOURCE_RULE_REQUIRE_VGPR_VALUE) &&
          !loom_amdgpu_module_value_prefers_vgpr(module, term->index)) {
        diagnostic->rejection_bits |= rule->rejection_bits;
        return false;
      }

      access->dynamic_term_kinds[term_index] = rule->dynamic_index_kind;
      if (rule->dynamic_index_kind ==
              LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET &&
          term->byte_stride != 1 &&
          term->byte_shift == LOOM_AMDGPU_MEMORY_ACCESS_BYTE_SHIFT_NONE) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_SCALAR_DYNAMIC_STRIDE;
        return false;
      }
      selected = true;
      break;
    }
    if (!selected) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_SOURCE;
      return false;
    }
  }
  return true;
}

bool loom_amdgpu_source_memory_offset_fits_u32(
    const loom_low_source_memory_access_plan_t* source,
    int64_t static_byte_offset) {
  return loom_low_source_memory_dynamic_offset_fits_unsigned_bit_count(
      source, static_byte_offset, 32);
}

typedef enum loom_amdgpu_memory_descriptor_domain_e {
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE = 0,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GLOBAL_SADDR = 1,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS = 2,
} loom_amdgpu_memory_descriptor_domain_t;

typedef enum loom_amdgpu_memory_address_attempt_kind_e {
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_DS_ADDTID = 0,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_DS_2ADDR = 1,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_LDS_DEFAULT = 2,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_BUFFER_RESOURCE = 3,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_GLOBAL_SADDR = 4,
} loom_amdgpu_memory_address_attempt_kind_t;

typedef enum loom_amdgpu_memory_address_attempt_result_e {
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE = 0,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED = 1,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED = 2,
} loom_amdgpu_memory_address_attempt_result_t;

typedef struct loom_amdgpu_memory_address_attempt_t {
  // Address-form selector attempted by this row.
  loom_amdgpu_memory_address_attempt_kind_t kind;
} loom_amdgpu_memory_address_attempt_t;

static const loom_amdgpu_memory_address_attempt_t kAmdgpuLdsAddressAttempts[] =
    {
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_DS_ADDTID,
        },
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_DS_2ADDR,
        },
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_LDS_DEFAULT,
        },
};

static const loom_amdgpu_memory_address_attempt_t
    kAmdgpuGlobalAddressAttempts[] = {
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_GLOBAL_SADDR,
        },
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_BUFFER_RESOURCE,
        },
};

typedef struct loom_amdgpu_memory_space_descriptor_domain_t {
  // Source memory-space fact matched by this row.
  loom_value_fact_memory_space_t memory_space;
  // Default descriptor domain selected for this memory space.
  loom_amdgpu_memory_descriptor_domain_t descriptor_domain;
} loom_amdgpu_memory_space_descriptor_domain_t;

static const loom_amdgpu_memory_space_descriptor_domain_t
    kAmdgpuMemorySpaceDescriptorDomains[] = {
        {
            .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
            .descriptor_domain =
                LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
        },
        {
            .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL,
            .descriptor_domain =
                LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
        },
        {
            .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT,
            .descriptor_domain =
                LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
        },
        {
            .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR,
            .descriptor_domain =
                LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
        },
        {
            .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
            .descriptor_domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
        },
};

static bool loom_amdgpu_memory_descriptor_domain_from_memory_space(
    loom_value_fact_memory_space_t memory_space,
    loom_amdgpu_memory_descriptor_domain_t* out_descriptor_domain) {
  IREE_ASSERT_ARGUMENT(out_descriptor_domain);
  *out_descriptor_domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kAmdgpuMemorySpaceDescriptorDomains); ++i) {
    const loom_amdgpu_memory_space_descriptor_domain_t* row =
        &kAmdgpuMemorySpaceDescriptorDomains[i];
    if (row->memory_space == memory_space) {
      *out_descriptor_domain = row->descriptor_domain;
      return true;
    }
  }
  return false;
}

typedef struct loom_amdgpu_memory_descriptor_candidate_t {
  // Memory resource domain accepted by this row.
  loom_amdgpu_memory_descriptor_domain_t domain;
  // Addressing form selected by this row.
  loom_amdgpu_memory_address_form_t address_form;
  // Number of VGPR lanes moved by the memory packet.
  uint32_t vgpr_count;
  // Direction of the memory packet.
  loom_amdgpu_memory_operation_kind_t kind;
  // Stable descriptor ID selected when present in the descriptor set.
  uint64_t descriptor_id;
} loom_amdgpu_memory_descriptor_candidate_t;

static const loom_amdgpu_memory_descriptor_candidate_t
    kAmdgpuMemoryDescriptorCandidates[] = {
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
            .vgpr_count = 1,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_DWORD,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO,
            .vgpr_count = 1,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
            .descriptor_id =
                LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_DWORD_OFF_ZERO,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
            .vgpr_count = 2,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_B64,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
            .vgpr_count = 2,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_DWORDX2,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
            .vgpr_count = 4,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_B128,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
            .vgpr_count = 4,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_DWORDX4,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
            .vgpr_count = 1,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_STORE_DWORD,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO,
            .vgpr_count = 1,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
            .descriptor_id =
                LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_STORE_DWORD_OFF_ZERO,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
            .vgpr_count = 2,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_STORE_B64,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
            .vgpr_count = 2,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_STORE_DWORDX2,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
            .vgpr_count = 4,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_STORE_B128,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
            .vgpr_count = 4,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_STORE_DWORDX4,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GLOBAL_SADDR,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR,
            .vgpr_count = 1,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_LOAD_B32_SADDR,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GLOBAL_SADDR,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR,
            .vgpr_count = 2,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_LOAD_B64_SADDR,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GLOBAL_SADDR,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR,
            .vgpr_count = 4,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_LOAD_B128_SADDR,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GLOBAL_SADDR,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR,
            .vgpr_count = 1,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_STORE_B32_SADDR,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GLOBAL_SADDR,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR,
            .vgpr_count = 2,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_STORE_B64_SADDR,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GLOBAL_SADDR,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR,
            .vgpr_count = 4,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_STORE_B128_SADDR,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
            .vgpr_count = 1,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_READ_B32,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
            .vgpr_count = 2,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_READ_B64,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
            .vgpr_count = 3,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_READ_B96,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
            .vgpr_count = 4,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_READ_B128,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
            .vgpr_count = 1,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_WRITE_B32,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
            .vgpr_count = 2,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_WRITE_B64,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
            .vgpr_count = 3,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_WRITE_B96,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
            .vgpr_count = 4,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_WRITE_B128,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR,
            .vgpr_count = 2,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_READ2_B32,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR,
            .vgpr_count = 2,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_WRITE2_B32,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR,
            .vgpr_count = 2,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_READ2ST64_B32,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR,
            .vgpr_count = 2,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_WRITE2ST64_B32,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID,
            .vgpr_count = 1,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_READ_ADDTID_B32,
        },
        {
            .domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
            .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID,
            .vgpr_count = 1,
            .kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_DS_WRITE_ADDTID_B32,
        },
};

static bool loom_amdgpu_memory_descriptor_candidate_matches(
    const loom_amdgpu_memory_descriptor_candidate_t* candidate,
    loom_amdgpu_memory_descriptor_domain_t domain,
    loom_amdgpu_memory_address_form_t address_form,
    loom_amdgpu_memory_operation_kind_t kind, uint32_t vgpr_count) {
  return candidate->domain == domain &&
         candidate->address_form == address_form && candidate->kind == kind &&
         candidate->vgpr_count == vgpr_count;
}

static bool loom_amdgpu_select_memory_descriptor_candidate(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_descriptor_domain_t domain,
    loom_amdgpu_memory_address_form_t address_form,
    loom_amdgpu_memory_operation_kind_t kind, uint32_t vgpr_count,
    uint64_t* out_descriptor_id, uint32_t* out_descriptor_ordinal) {
  IREE_ASSERT_ARGUMENT(out_descriptor_id);
  IREE_ASSERT_ARGUMENT(out_descriptor_ordinal);
  *out_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  *out_descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kAmdgpuMemoryDescriptorCandidates); ++i) {
    const loom_amdgpu_memory_descriptor_candidate_t* candidate =
        &kAmdgpuMemoryDescriptorCandidates[i];
    if (!loom_amdgpu_memory_descriptor_candidate_matches(
            candidate, domain, address_form, kind, vgpr_count)) {
      continue;
    }
    const uint32_t descriptor_ordinal =
        loom_low_descriptor_set_lookup_descriptor_by_id(
            descriptor_set, candidate->descriptor_id);
    if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
      continue;
    }
    *out_descriptor_id = candidate->descriptor_id;
    *out_descriptor_ordinal = descriptor_ordinal;
    return true;
  }
  return false;
}

static bool loom_amdgpu_select_buffer_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_address_form_t address_form,
    loom_amdgpu_memory_operation_kind_t kind, uint64_t* out_descriptor_id,
    uint32_t* out_descriptor_ordinal) {
  return loom_amdgpu_select_memory_descriptor_candidate(
      descriptor_set, LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
      address_form, kind, access->vgpr_count, out_descriptor_id,
      out_descriptor_ordinal);
}

static bool loom_amdgpu_select_global_saddr_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_operation_kind_t kind, uint64_t* out_descriptor_id,
    uint32_t* out_descriptor_ordinal) {
  return loom_amdgpu_select_memory_descriptor_candidate(
      descriptor_set, LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GLOBAL_SADDR,
      LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR, kind, access->vgpr_count,
      out_descriptor_id, out_descriptor_ordinal);
}

static bool loom_amdgpu_select_ds_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_operation_kind_t kind, uint64_t* out_descriptor_id,
    uint32_t* out_descriptor_ordinal) {
  return loom_amdgpu_select_memory_descriptor_candidate(
      descriptor_set, LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
      LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT, kind, access->vgpr_count,
      out_descriptor_id, out_descriptor_ordinal);
}

static bool loom_amdgpu_select_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_operation_kind_t kind, uint64_t* out_descriptor_id,
    uint32_t* out_descriptor_ordinal) {
  loom_amdgpu_memory_descriptor_domain_t descriptor_domain;
  if (!loom_amdgpu_memory_descriptor_domain_from_memory_space(
          access->source.memory_space, &descriptor_domain)) {
    return false;
  }
  if (descriptor_domain == LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS) {
    return loom_amdgpu_select_ds_memory_descriptor(descriptor_set, access, kind,
                                                   out_descriptor_id,
                                                   out_descriptor_ordinal);
  }
  return loom_amdgpu_select_buffer_memory_descriptor(
      descriptor_set, access, access->address_form, kind, out_descriptor_id,
      out_descriptor_ordinal);
}

typedef struct loom_amdgpu_offset_immediate_encoding_t {
  // Target immediate encoding ID.
  uint16_t encoding_id;
  // Byte count represented by one encoded offset unit.
  uint32_t unit_byte_count;
} loom_amdgpu_offset_immediate_encoding_t;

static const loom_amdgpu_offset_immediate_encoding_t
    kAmdgpuOffsetImmediateEncodings[] = {
        {
            .encoding_id =
                LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_BYTE,
            .unit_byte_count = 1,
        },
        {
            .encoding_id =
                LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_DWORD,
            .unit_byte_count = 4,
        },
        {
            .encoding_id =
                LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_QWORD,
            .unit_byte_count = 8,
        },
        {
            .encoding_id =
                LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_DWORD_STRIDE64,
            .unit_byte_count = 4 * 64,
        },
        {
            .encoding_id =
                LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_QWORD_STRIDE64,
            .unit_byte_count = 8 * 64,
        },
        {
            .encoding_id =
                LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_DS16,
            .unit_byte_count = 1,
        },
};

static bool loom_amdgpu_immediate_encoding_address_unit_byte_count(
    uint16_t encoding_id, uint32_t* out_unit_byte_count) {
  IREE_ASSERT_ARGUMENT(out_unit_byte_count);
  *out_unit_byte_count = 0;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kAmdgpuOffsetImmediateEncodings); ++i) {
    const loom_amdgpu_offset_immediate_encoding_t* encoding =
        &kAmdgpuOffsetImmediateEncodings[i];
    if (encoding->encoding_id == encoding_id) {
      *out_unit_byte_count = encoding->unit_byte_count;
      return true;
    }
  }
  return false;
}

bool loom_amdgpu_descriptor_offset_immediate_info(
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
  if (access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
      loom_low_source_memory_access_is_dynamic(&access->source) ||
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
  if (access->source.static_byte_offset < 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET;
    return false;
  }
  if (offset_unit_byte_count == 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }

  const uint64_t static_byte_offset =
      (uint64_t)access->source.static_byte_offset;
  if (access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
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
    if (!loom_amdgpu_source_memory_offset_fits_u32(
            &access->source, access->source.static_byte_offset)) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
      return false;
    }
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
  if (!loom_amdgpu_source_memory_offset_fits_u32(
          &access->source, access->source.static_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
    return false;
  }
  return true;
}

static bool loom_amdgpu_memory_access_split_lds_default_static_offset(
    loom_amdgpu_memory_access_plan_t* access, uint32_t offset_unit_byte_count,
    uint64_t offset_unsigned_max,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (access->source.static_byte_offset < 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET;
    return false;
  }
  if (offset_unit_byte_count == 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }
  if (!loom_amdgpu_source_memory_offset_fits_u32(
          &access->source, access->source.static_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
    return false;
  }

  const uint64_t static_byte_offset =
      (uint64_t)access->source.static_byte_offset;
  const bool static_fits_immediate =
      (static_byte_offset % offset_unit_byte_count) == 0 &&
      (static_byte_offset / offset_unit_byte_count) <= offset_unsigned_max;
  access->immediate_offset =
      static_fits_immediate
          ? (int64_t)(static_byte_offset / offset_unit_byte_count)
          : 0;
  access->secondary_immediate_offset = 0;
  access->vaddr_static_byte_offset =
      static_fits_immediate ? 0 : (uint32_t)static_byte_offset;
  access->scalar_byte_offset = 0;
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
  const int64_t static_byte_offset = access->source.static_byte_offset;
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
  if (!loom_amdgpu_source_memory_offset_fits_u32(&access->source,
                                                 /*static_byte_offset=*/0)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
    return false;
  }
  access->immediate_offset = static_byte_offset;
  access->scalar_byte_offset = 0;
  return true;
}

static bool loom_amdgpu_memory_access_split_ds2_static_offset(
    loom_amdgpu_memory_access_plan_t* access, uint32_t offset_unit_byte_count,
    uint64_t offset_unsigned_max,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (access->source.static_byte_offset < 0) {
    return false;
  }
  int64_t secondary_byte_offset = 0;
  if (!iree_checked_add_i64(access->source.static_byte_offset,
                            access->source.vector_lane_byte_stride,
                            &secondary_byte_offset) ||
      secondary_byte_offset < 0) {
    return false;
  }

  const uint64_t byte_offsets[] = {
      (uint64_t)access->source.static_byte_offset,
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
  if (!loom_amdgpu_source_memory_offset_fits_u32(&access->source,
                                                 secondary_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
    return false;
  }
  return true;
}

static bool loom_amdgpu_select_ds2_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  bool found_kind_descriptor = false;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kAmdgpuMemoryDescriptorCandidates); ++i) {
    const loom_amdgpu_memory_descriptor_candidate_t* candidate =
        &kAmdgpuMemoryDescriptorCandidates[i];
    if (!loom_amdgpu_memory_descriptor_candidate_matches(
            candidate, LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
            LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR, kind,
            access->vgpr_count)) {
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
            access, offset_info.unit_byte_count, offset_info.unsigned_max,
            diagnostic)) {
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

static bool loom_amdgpu_try_select_ds_addtid_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_value_fact_table_t* fact_table, loom_func_like_t source_function,
    loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_operation_kind_t kind) {
  uint64_t root_byte_offset = 0;
  if (access->vgpr_count != 1 || access->source.dynamic_term_count != 1 ||
      access->dynamic_term_kinds[0] != LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR ||
      access->source.dynamic_terms[0].source !=
          LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID ||
      access->source.dynamic_terms[0].dimension != LOOM_KERNEL_DIMENSION_X ||
      access->source.dynamic_terms[0].byte_stride !=
          access->source.element_byte_count ||
      access->source.vector_lane_byte_stride !=
          access->source.element_byte_count ||
      !loom_amdgpu_source_lds_layout_lookup_root(fact_table, source_function,
                                                 access->source.root_value_id,
                                                 &root_byte_offset) ||
      root_byte_offset != 0) {
    return false;
  }

  uint64_t descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_memory_descriptor_candidate(
          descriptor_set, LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
          LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID, kind, access->vgpr_count,
          &descriptor_id, &descriptor_ordinal)) {
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
      loom_low_source_memory_access_is_dynamic(&access->source)) {
    for (uint8_t i = 0; i < access->source.dynamic_term_count; ++i) {
      if ((access->source.dynamic_terms[i].byte_stride & 15) != 0) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_DYNAMIC_ALIGNMENT;
        return false;
      }
    }
  }
  if (access->vgpr_count == 4 &&
      (access->source.static_byte_offset & 15) != 0) {
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
  for (uint8_t i = 0; i < access->source.dynamic_term_count; ++i) {
    if (access->dynamic_term_kinds[i] !=
        LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET) {
      continue;
    }
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

static bool loom_amdgpu_memory_access_plan_try_select_lds_default(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (access->source.vector_lane_byte_stride !=
      access->source.element_byte_count) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
    return false;
  }

  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_memory_descriptor(descriptor_set, access, kind,
                                            &access->descriptor_id,
                                            &descriptor_ordinal)) {
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
  return loom_amdgpu_memory_access_split_lds_default_static_offset(
      access, offset_info.unit_byte_count, offset_info.unsigned_max,
      diagnostic);
}

static loom_amdgpu_memory_address_attempt_result_t
loom_amdgpu_memory_address_attempt_apply(
    const loom_amdgpu_memory_address_attempt_t* attempt,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_value_fact_table_t* fact_table, loom_func_like_t source_function,
    loom_amdgpu_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  switch (attempt->kind) {
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_DS_ADDTID:
      return loom_amdgpu_try_select_ds_addtid_memory_descriptor(
                 descriptor_set, fact_table, source_function, access, kind)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_DS_2ADDR:
      if (access->source.vector_lane_byte_stride ==
          access->source.element_byte_count) {
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
      }
      if (access->vgpr_count != 2 ||
          !loom_amdgpu_memory_access_has_32bit_lanes(access)) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
      }
      return loom_amdgpu_select_ds2_memory_descriptor(descriptor_set, access,
                                                      kind, diagnostic)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_LDS_DEFAULT:
      return loom_amdgpu_memory_access_plan_try_select_lds_default(
                 descriptor_set, kind, access, diagnostic)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_BUFFER_RESOURCE:
      if (access->source.vector_lane_byte_stride !=
          access->source.element_byte_count) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
      }
      return loom_amdgpu_memory_access_plan_try_select_buffer(
                 descriptor_set, kind, access, diagnostic)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_GLOBAL_SADDR:
      if (access->source.vector_lane_byte_stride !=
          access->source.element_byte_count) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
      }
      return loom_amdgpu_memory_access_plan_try_select_global_saddr(
                 descriptor_set, kind, access, diagnostic)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
  }
  return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
}

static bool loom_amdgpu_memory_access_select_address_form(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_value_fact_table_t* fact_table, loom_func_like_t source_function,
    loom_amdgpu_memory_descriptor_domain_t descriptor_domain,
    loom_amdgpu_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_plan_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  const loom_amdgpu_memory_address_attempt_t* attempts =
      kAmdgpuGlobalAddressAttempts;
  iree_host_size_t attempt_count = IREE_ARRAYSIZE(kAmdgpuGlobalAddressAttempts);
  if (descriptor_domain == LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS) {
    attempts = kAmdgpuLdsAddressAttempts;
    attempt_count = IREE_ARRAYSIZE(kAmdgpuLdsAddressAttempts);
  }
  for (iree_host_size_t i = 0; i < attempt_count; ++i) {
    const loom_amdgpu_memory_address_attempt_result_t result =
        loom_amdgpu_memory_address_attempt_apply(&attempts[i], descriptor_set,
                                                 fact_table, source_function,
                                                 kind, access, diagnostic);
    if (result == LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED) {
      return true;
    }
    if (result == LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED) {
      return false;
    }
  }
  return false;
}

loom_amdgpu_memory_operation_kind_t
loom_amdgpu_memory_operation_kind_from_source(
    const loom_low_source_memory_access_plan_t* source) {
  switch (source->operation_kind) {
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD:
      return LOOM_AMDGPU_MEMORY_OPERATION_LOAD;
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE:
      return LOOM_AMDGPU_MEMORY_OPERATION_STORE;
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_REDUCE:
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_RMW:
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_CMPXCHG:
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_PREFETCH:
      IREE_CHECK_UNREACHABLE();
  }
  IREE_CHECK_UNREACHABLE();
}

static loom_type_t loom_amdgpu_memory_access_source_vector_type(
    const loom_module_t* module, const loom_op_t* source_op) {
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_LOAD:
      return loom_module_value_type(module, loom_vector_load_result(source_op));
    case LOOM_OP_VECTOR_STORE:
      return loom_module_value_type(module, loom_vector_store_value(source_op));
    default:
      return loom_type_none();
  }
}

bool loom_amdgpu_memory_access_select(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_func_like_t source_function, const loom_op_t* source_op,
    loom_amdgpu_memory_access_plan_t* out_access,
    loom_low_source_memory_access_diagnostic_t* out_source_diagnostic,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic) {
  IREE_ASSERT_ARGUMENT(out_access);
  IREE_ASSERT_ARGUMENT(out_source_diagnostic);
  IREE_ASSERT_ARGUMENT(out_diagnostic);
  *out_access = (loom_amdgpu_memory_access_plan_t){
      .address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT,
      .descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE,
  };
  *out_source_diagnostic = (loom_low_source_memory_access_diagnostic_t){0};
  *out_diagnostic = (loom_amdgpu_memory_access_diagnostic_t){0};

  if (!loom_low_source_memory_access_plan_build(module, fact_table, source_op,
                                                &out_access->source,
                                                out_source_diagnostic)) {
    return false;
  }

  const loom_amdgpu_memory_operation_kind_t kind =
      loom_amdgpu_memory_operation_kind_from_source(&out_access->source);

  loom_type_t vector_type =
      loom_amdgpu_memory_access_source_vector_type(module, source_op);
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

  if (out_access->source.vector_lane_byte_stride <= 0 ||
      out_access->source.vector_lane_byte_stride > UINT32_MAX) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
    return false;
  }
  if (!loom_amdgpu_memory_access_static_byte_offset_is_usable(
          out_access->source.static_byte_offset, out_diagnostic)) {
    return false;
  }

  if (!loom_amdgpu_memory_access_select_dynamic_term_kinds(module, out_access,
                                                           out_diagnostic)) {
    return false;
  }

  loom_amdgpu_memory_descriptor_domain_t descriptor_domain;
  if (!loom_amdgpu_memory_descriptor_domain_from_memory_space(
          out_access->source.memory_space, &descriptor_domain)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_MEMORY_SPACE;
    return false;
  }

  if (descriptor_domain == LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS) {
    if (out_access->source.root_value_id >= module->values.count) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_ROOT;
      return false;
    }
    const loom_value_t* root_value =
        loom_module_value(module, out_access->source.root_value_id);
    const loom_op_t* root_op = loom_value_is_block_arg(root_value)
                                   ? NULL
                                   : loom_value_def_op(root_value);
    if (root_op == NULL || !loom_buffer_alloca_isa(root_op)) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_ROOT;
      return false;
    }
  }

  return loom_amdgpu_memory_access_select_address_form(
      descriptor_set, fact_table, source_function, descriptor_domain, kind,
      out_access, out_diagnostic);
}

static bool loom_amdgpu_memory_access_select_with_source_function(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_func_like_t source_function,
    loom_amdgpu_memory_access_plan_t* out_access) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_amdgpu_memory_access_diagnostic_t diagnostic = {0};
  return loom_amdgpu_memory_access_select(
      module, loom_low_lower_context_fact_table(context),
      loom_low_lower_context_descriptor_set(context), source_function,
      source_op, out_access, &source_diagnostic, &diagnostic);
}

static bool loom_amdgpu_memory_access_select_from_context(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_plan_t* out_access) {
  return loom_amdgpu_memory_access_select_with_source_function(
      context, source_op, loom_low_lower_context_source_function(context),
      out_access);
}

iree_status_t loom_amdgpu_select_vector_load_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_plan_t* out_plan, bool* out_selected) {
  IREE_ASSERT_ARGUMENT(out_selected);
  *out_selected = false;
  if (!loom_amdgpu_memory_access_select_from_context(context, source_op,
                                                     out_plan)) {
    return iree_ok_status();
  }
  *out_selected = loom_amdgpu_memory_cache_policy_can_lower(
      loom_low_lower_context_descriptor_set(context), out_plan);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_vector_store_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_plan_t* out_plan, bool* out_selected) {
  IREE_ASSERT_ARGUMENT(out_selected);
  *out_selected = false;
  if (!loom_amdgpu_memory_access_select_from_context(context, source_op,
                                                     out_plan)) {
    return iree_ok_status();
  }
  *out_selected = loom_amdgpu_memory_cache_policy_can_lower(
      loom_low_lower_context_descriptor_set(context), out_plan);
  return iree_ok_status();
}
