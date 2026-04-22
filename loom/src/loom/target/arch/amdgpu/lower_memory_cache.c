// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdint.h>

#include "loom/ops/cache.h"
#include "loom/target/arch/amdgpu/lower_memory_internal.h"
#include "loom/target/arch/amdgpu/target_info.h"

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

bool loom_amdgpu_memory_cache_policy_is_present(
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

bool loom_amdgpu_memory_cache_policy_gfx12_th(uint8_t temporal,
                                              int64_t* out_th) {
  IREE_ASSERT_ARGUMENT(out_th);
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

bool loom_amdgpu_memory_cache_policy_can_lower(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_plan_t* access) {
  const loom_vector_memory_cache_policy_t* policy =
      &access->source.cache_policy;
  if (!loom_amdgpu_memory_cache_policy_is_present(policy)) {
    return true;
  }
  if (!loom_amdgpu_memory_cache_policy_is_complete(policy) ||
      access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
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
      return loom_cache_scope_is_valid(policy->cache_scope) &&
             loom_amdgpu_memory_cache_policy_gfx12_th(policy->cache_temporal,
                                                      &th);
    }
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1:
      return policy->cache_scope == LOOM_CACHE_SCOPE_DEVICE &&
             (policy->cache_temporal == LOOM_CACHE_TEMPORAL_REGULAR ||
              policy->cache_temporal == LOOM_CACHE_TEMPORAL_NON_TEMPORAL);
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC:
      return loom_amdgpu_memory_cache_policy_is_regular_device(policy);
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE:
      return false;
  }
  return false;
}

iree_status_t loom_amdgpu_memory_cache_policy_rejected_status(
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
  if (access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
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
