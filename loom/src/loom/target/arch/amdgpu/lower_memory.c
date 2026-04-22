// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower_internal.h"
#include "loom/target/arch/amdgpu/lower_memory_internal.h"
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
         access->source.element_byte_count * iree_max(access->vgpr_count, 1u);
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
  switch (access->source.memory_space) {
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
  if (access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
      access->source.dynamic_index != LOOM_VALUE_ID_INVALID ||
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
  access->immediate_offset = static_byte_offset;
  access->scalar_byte_offset = 0;
  return true;
}

static bool loom_amdgpu_memory_access_split_ds2_static_offset(
    loom_amdgpu_memory_access_plan_t* access, uint32_t offset_unit_byte_count,
    uint64_t offset_unsigned_max) {
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
      access->source.dynamic_index == LOOM_VALUE_ID_INVALID ||
      access->dynamic_index_kind != LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR ||
      access->source.dynamic_index_source !=
          LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID ||
      access->source.dynamic_index_dimension != LOOM_KERNEL_DIMENSION_X ||
      access->source.dynamic_index_byte_stride !=
          access->source.element_byte_count ||
      access->source.vector_lane_byte_stride !=
          access->source.element_byte_count ||
      !loom_amdgpu_source_function_proves_zero_lds_slot_base(
          source_function, access->source.root_value_id)) {
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
      access->source.dynamic_index != LOOM_VALUE_ID_INVALID &&
      (access->source.dynamic_index_byte_stride & 15) != 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_DYNAMIC_ALIGNMENT;
    return false;
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

loom_amdgpu_memory_operation_kind_t
loom_amdgpu_memory_operation_kind_from_source(
    const loom_low_source_memory_access_plan_t* source) {
  switch (source->operation_kind) {
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD:
      return LOOM_AMDGPU_MEMORY_OPERATION_LOAD;
    case LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE:
      return LOOM_AMDGPU_MEMORY_OPERATION_STORE;
  }
  return LOOM_AMDGPU_MEMORY_OPERATION_LOAD;
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
      .dynamic_index_kind = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE,
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

  if (out_access->source.dynamic_index != LOOM_VALUE_ID_INVALID) {
    if (out_access->source.dynamic_index_byte_stride < 0 ||
        out_access->source.dynamic_index_byte_stride > UINT32_MAX) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE;
      return false;
    }
    switch (out_access->source.dynamic_index_source) {
      case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID:
        if (out_access->source.memory_space ==
            LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
          out_diagnostic->rejection_bits |=
              LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_WORKGROUP_DYNAMIC_INDEX_SOURCE;
          return false;
        }
        out_access->dynamic_index_kind =
            LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET;
        break;
      case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID:
        out_access->dynamic_index_kind = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
        break;
      case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE:
        if (!loom_amdgpu_module_value_prefers_vgpr(
                module, out_access->source.dynamic_index)) {
          out_diagnostic->rejection_bits |=
              LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_SOURCE;
          return false;
        }
        out_access->dynamic_index_kind = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
        break;
      case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_NONE:
        out_diagnostic->rejection_bits |=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_SOURCE;
        return false;
    }
    if (out_access->dynamic_index_kind ==
            LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET &&
        out_access->source.dynamic_index_byte_stride != 1 &&
        out_access->source.dynamic_index_byte_shift ==
            LOOM_AMDGPU_MEMORY_ACCESS_BYTE_SHIFT_NONE) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_SCALAR_DYNAMIC_STRIDE;
      return false;
    }
  }

  if (out_access->source.memory_space ==
      LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
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
    if (loom_amdgpu_try_select_ds_addtid_memory_descriptor(
            descriptor_set, source_function, out_access, kind)) {
      return true;
    }
    if (out_access->source.vector_lane_byte_stride !=
        out_access->source.element_byte_count) {
      if (out_access->vgpr_count != 2 ||
          !loom_amdgpu_memory_access_has_32bit_lanes(out_access)) {
        out_diagnostic->rejection_bits |=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
        return false;
      }
      return loom_amdgpu_select_ds2_memory_descriptor(
          descriptor_set, out_access, kind, out_diagnostic);
    }
  } else if (out_access->source.memory_space !=
                 LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN &&
             out_access->source.memory_space !=
                 LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL &&
             out_access->source.memory_space !=
                 LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT &&
             out_access->source.memory_space !=
                 LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_MEMORY_SPACE;
    return false;
  }
  if (out_access->source.vector_lane_byte_stride !=
      out_access->source.element_byte_count) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
    return false;
  }

  if (out_access->source.memory_space ==
      LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
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
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_amdgpu_memory_access_diagnostic_t diagnostic = {0};
  return loom_amdgpu_memory_access_select(
      module, loom_low_lower_context_fact_table(context),
      loom_low_lower_context_descriptor_set(context), source_function,
      source_op, out_access, &source_diagnostic, &diagnostic);
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
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_amdgpu_memory_access_diagnostic_t diagnostic = {0};
  return loom_amdgpu_memory_access_select(
      module, loom_low_lower_context_fact_table(context),
      loom_low_lower_context_descriptor_set(context), source_function,
      source_op, out_access, &source_diagnostic, &diagnostic);
}

static bool loom_amdgpu_store_memory_access_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_plan_t* out_access) {
  return loom_amdgpu_store_memory_access_select_with_source_function(
      context, source_op, loom_low_lower_context_source_function(context),
      out_access);
}

bool loom_amdgpu_select_vector_load_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_plan_t* out_plan) {
  if (!loom_amdgpu_load_memory_access_select(context, source_op, out_plan)) {
    return false;
  }
  return loom_amdgpu_memory_cache_policy_can_lower(
      loom_low_lower_context_descriptor_set(context), out_plan);
}

bool loom_amdgpu_select_vector_store_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_plan_t* out_plan) {
  if (!loom_amdgpu_store_memory_access_select(context, source_op, out_plan)) {
    return false;
  }
  return loom_amdgpu_memory_cache_policy_can_lower(
      loom_low_lower_context_descriptor_set(context), out_plan);
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
