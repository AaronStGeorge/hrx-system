// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/system_memory.h"

#include <stdint.h>

#include "loom/codegen/low/builder.h"
#include "loom/ops/cache.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

loom_amdgpu_vector_memory_cache_policy_encoding_t
loom_amdgpu_system_memory_cache_policy_encoding(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_at(
          descriptor_set->descriptor_set_ordinal);
  return descriptor_set_info == NULL
             ? LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE
             : descriptor_set_info->vector_memory.cache_policy_encoding;
}

iree_status_t loom_amdgpu_system_memory_build_u32_attr(
    loom_builder_t* builder, iree_string_view_t name, uint32_t value,
    loom_named_attr_t* out_attr) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, name, &name_id));
  *out_attr = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
  return iree_ok_status();
}

iree_status_t loom_amdgpu_system_memory_build_offset_attr(
    loom_builder_t* builder, uint32_t byte_offset,
    loom_named_attr_t* out_attr) {
  return loom_amdgpu_system_memory_build_u32_attr(builder, IREE_SV("offset"),
                                                  byte_offset, out_attr);
}

static iree_status_t loom_amdgpu_system_memory_append_u32_attr(
    loom_builder_t* builder, iree_string_view_t name, uint32_t value,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count) {
  if (*inout_attr_count >= attr_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU system-memory attr capacity exceeded");
  }
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_system_memory_build_u32_attr(builder, name, value, &attr));
  attrs[(*inout_attr_count)++] = attr;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_system_memory_append_scope_attr(
    loom_builder_t* builder, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  return loom_amdgpu_system_memory_append_u32_attr(
      builder, IREE_SV("scope"), LOOM_CACHE_SCOPE_SYSTEM, attrs, attr_capacity,
      inout_attr_count);
}

static iree_status_t loom_amdgpu_system_memory_append_sc0_attr(
    loom_builder_t* builder, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  return loom_amdgpu_system_memory_append_u32_attr(
      builder, IREE_SV("sc0"), 1, attrs, attr_capacity, inout_attr_count);
}

static iree_status_t loom_amdgpu_system_memory_append_sc1_attr(
    loom_builder_t* builder, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  return loom_amdgpu_system_memory_append_u32_attr(
      builder, IREE_SV("sc1"), 1, attrs, attr_capacity, inout_attr_count);
}

static iree_status_t loom_amdgpu_system_memory_append_sc0_sc1_attrs(
    loom_builder_t* builder, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* inout_attr_count) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_sc0_attr(
      builder, attrs, attr_capacity, inout_attr_count));
  return loom_amdgpu_system_memory_append_sc1_attr(
      builder, attrs, attr_capacity, inout_attr_count);
}

iree_status_t loom_amdgpu_system_memory_append_load_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count) {
  switch (loom_amdgpu_system_memory_cache_policy_encoding(descriptor_set)) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC:
      return loom_amdgpu_system_memory_append_u32_attr(
          builder, IREE_SV("glc"), 1, attrs, attr_capacity, inout_attr_count);
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1:
      return loom_amdgpu_system_memory_append_sc0_sc1_attrs(
          builder, attrs, attr_capacity, inout_attr_count);
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH:
      return loom_amdgpu_system_memory_append_scope_attr(
          builder, attrs, attr_capacity, inout_attr_count);
    default:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU descriptor set has no system-memory load policy");
  }
}

iree_status_t loom_amdgpu_system_memory_append_release_store_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count) {
  switch (loom_amdgpu_system_memory_cache_policy_encoding(descriptor_set)) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC:
      return iree_ok_status();
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1:
      return loom_amdgpu_system_memory_append_sc0_sc1_attrs(
          builder, attrs, attr_capacity, inout_attr_count);
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH:
      return loom_amdgpu_system_memory_append_scope_attr(
          builder, attrs, attr_capacity, inout_attr_count);
    default:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU descriptor set has no system-memory release-store policy");
  }
}

iree_status_t loom_amdgpu_system_memory_append_no_return_atomic_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count) {
  switch (loom_amdgpu_system_memory_cache_policy_encoding(descriptor_set)) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC:
      return iree_ok_status();
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1:
      return loom_amdgpu_system_memory_append_sc1_attr(
          builder, attrs, attr_capacity, inout_attr_count);
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH:
      return loom_amdgpu_system_memory_append_scope_attr(
          builder, attrs, attr_capacity, inout_attr_count);
    default:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU descriptor set has no system-memory no-return atomic policy");
  }
}

iree_status_t loom_amdgpu_system_memory_append_return_atomic_attrs(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count) {
  switch (loom_amdgpu_system_memory_cache_policy_encoding(descriptor_set)) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC:
      return iree_ok_status();
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1:
      return loom_amdgpu_system_memory_append_sc0_sc1_attrs(
          builder, attrs, attr_capacity, inout_attr_count);
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH:
      return loom_amdgpu_system_memory_append_scope_attr(
          builder, attrs, attr_capacity, inout_attr_count);
    default:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU descriptor set has no system-memory returning atomic policy");
  }
}

static iree_status_t loom_amdgpu_system_memory_build_explicit_packet(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_named_attr_slice_t attrs,
    loom_location_id_t location) {
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set, descriptor_ref, &descriptor, &opcode_id));
  loom_op_t* op = NULL;
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, /*operands=*/NULL,
      /*operand_count=*/0, attrs, /*result_types=*/NULL, /*result_count=*/0,
      /*tied_results=*/NULL, /*tied_result_count=*/0, location, &op);
}

static iree_status_t loom_amdgpu_system_memory_build_waitcnt_vmem_load(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location) {
  loom_named_attr_t attrs[2] = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_u32_attr(
      builder, IREE_SV("vmcnt"), 0, &attrs[0]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_u32_attr(
      builder, IREE_SV("lgkmcnt"), 15, &attrs[1]));
  return loom_amdgpu_system_memory_build_explicit_packet(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT,
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)), location);
}

static iree_status_t loom_amdgpu_system_memory_build_waitcnt_vmem_store(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location) {
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_u32_attr(
      builder, IREE_SV("vscnt"), 0, &attr));
  return loom_amdgpu_system_memory_build_explicit_packet(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT_VSCNT,
      loom_make_named_attr_slice(&attr, 1), location);
}

static iree_status_t loom_amdgpu_system_memory_build_wait_loadcnt(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location) {
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_u32_attr(
      builder, IREE_SV("loadcnt"), 0, &attr));
  return loom_amdgpu_system_memory_build_explicit_packet(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_LOADCNT,
      loom_make_named_attr_slice(&attr, 1), location);
}

static iree_status_t loom_amdgpu_system_memory_build_wait_storecnt(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location) {
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_u32_attr(
      builder, IREE_SV("storecnt"), 0, &attr));
  return loom_amdgpu_system_memory_build_explicit_packet(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_STORECNT,
      loom_make_named_attr_slice(&attr, 1), location);
}

static iree_status_t loom_amdgpu_system_memory_build_scoped_buffer_wbl2(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location) {
  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_sc0_sc1_attrs(
      builder, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  return loom_amdgpu_system_memory_build_explicit_packet(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_WBL2,
      loom_make_named_attr_slice(attrs, attr_count), location);
}

static iree_status_t loom_amdgpu_system_memory_build_scoped_buffer_inv(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location) {
  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_sc0_sc1_attrs(
      builder, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  return loom_amdgpu_system_memory_build_explicit_packet(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_INV,
      loom_make_named_attr_slice(attrs, attr_count), location);
}

static iree_status_t
loom_amdgpu_system_memory_build_scoped_global_cache_control(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_location_id_t location) {
  loom_named_attr_t attr = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_scope_attr(
      builder, &attr, 1, &attr_count));
  return loom_amdgpu_system_memory_build_explicit_packet(
      builder, descriptor_set, descriptor_ref,
      loom_make_named_attr_slice(&attr, attr_count), location);
}

iree_status_t loom_amdgpu_system_memory_build_release_ordering(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location) {
  switch (loom_amdgpu_system_memory_cache_policy_encoding(descriptor_set)) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_waitcnt_vmem_load(
          builder, descriptor_set, location));
      return loom_amdgpu_system_memory_build_waitcnt_vmem_store(
          builder, descriptor_set, location);
    }
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1:
      return loom_amdgpu_system_memory_build_scoped_buffer_wbl2(
          builder, descriptor_set, location);
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH: {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_system_memory_build_scoped_global_cache_control(
              builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_WB,
              location));
      return loom_amdgpu_system_memory_build_wait_storecnt(
          builder, descriptor_set, location);
    }
    default:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU descriptor set has no system-memory release ordering");
  }
}

iree_status_t loom_amdgpu_system_memory_build_acquire_ordering(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_location_id_t location) {
  switch (loom_amdgpu_system_memory_cache_policy_encoding(descriptor_set)) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_waitcnt_vmem_load(
          builder, descriptor_set, location));
      IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_explicit_packet(
          builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL1_INV,
          loom_named_attr_slice_empty(), location));
      return loom_amdgpu_system_memory_build_explicit_packet(
          builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL0_INV,
          loom_named_attr_slice_empty(), location);
    }
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_waitcnt_vmem_load(
          builder, descriptor_set, location));
      return loom_amdgpu_system_memory_build_scoped_buffer_inv(
          builder, descriptor_set, location);
    }
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_wait_loadcnt(
          builder, descriptor_set, location));
      return loom_amdgpu_system_memory_build_scoped_global_cache_control(
          builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_INV,
          location);
    }
    default:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU descriptor set has no system-memory acquire ordering");
  }
}
