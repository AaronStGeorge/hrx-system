// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU processor and descriptor-set facts.
//
// The declarations here are the stable C contract for target-owned AMDGPU
// facts. Generated target-info tables provide the row data and lookup
// implementation; this header owns the public types so C API design does not
// live inside the table generator. Target-info generators may include this
// header, but must not emit public struct or enum definitions for these facts.

#ifndef LOOM_TARGET_ARCH_AMDGPU_TARGET_INFO_DEFS_H_
#define LOOM_TARGET_ARCH_AMDGPU_TARGET_INFO_DEFS_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Stable target-family identity for AMDGPU low descriptor sets.
#define LOOM_AMDGPU_TARGET_STABLE_ID UINT64_C(0x6c46df5542915cc5)

typedef enum loom_amdgpu_kernel_descriptor_profile_e {
  // No kernel descriptor writer is implemented for this processor yet.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE = 0,
  // GFX11 AMDHSA code-object v5 kernel descriptor packing.
  LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11 = 1,
} loom_amdgpu_kernel_descriptor_profile_t;

typedef enum loom_amdgpu_matrix_feature_profile_e {
  // No matrix instruction feature profile is defined.
  LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_NONE = 0,
  // GFX908 MFMA feature baseline.
  LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908 = 1,
  // GFX90A MFMA feature baseline.
  LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A = 2,
  // GFX940 MFMA/SMFMAC feature baseline.
  LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940 = 3,
  // GFX950 MFMA/SMFMAC feature baseline.
  LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950 = 4,
  // GFX11 WMMA feature baseline.
  LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11 = 5,
  // GFX12 WMMA/SWMMAC feature baseline.
  LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12 = 6,
  // GFX1250 WMMA/SWMMAC feature baseline.
  LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250 = 7,
} loom_amdgpu_matrix_feature_profile_t;

typedef enum loom_amdgpu_buffer_resource_cache_swizzle_e {
  // Buffer resource descriptors do not support cache swizzle.
  LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE = 0,
  // Descriptor word 1 carries a 14-bit byte stride and one enable bit.
  LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT = 1,
} loom_amdgpu_buffer_resource_cache_swizzle_t;

typedef enum loom_amdgpu_vector_memory_cache_policy_encoding_e {
  // Vector memory descriptors do not expose cache-policy immediates.
  LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE = 0,
  // Vector memory descriptors expose GLC/SLC/DLC cache controls.
  LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC = 1,
  // Vector memory descriptors expose NV/SCOPE/TH cache controls.
  LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH = 2,
  // Vector memory descriptors expose NT/SC0/SC1 cache controls.
  LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1 = 3,
} loom_amdgpu_vector_memory_cache_policy_encoding_t;

typedef struct loom_amdgpu_descriptor_set_info_t {
  // Durable descriptor-set identity derived from the descriptor-set key.
  uint64_t descriptor_set_stable_id;
  // Target-low descriptor set key such as `amdgpu.gfx11.core`.
  iree_string_view_t descriptor_set_key;
  // Production target preset key that expands to this descriptor set.
  iree_string_view_t low_preset_key;
  // SOPP opcode used when lowering structural `low.return` to `s_endpgm`.
  uint16_t s_endpgm_opcode;
  // True when descriptor packets have implemented native binary encoding.
  bool supports_descriptor_packet_encoding;
  // Buffer resource descriptor cache-swizzle encoding shape.
  loom_amdgpu_buffer_resource_cache_swizzle_t buffer_resource_cache_swizzle;
  // Vector memory packet cache-policy immediate encoding shape.
  loom_amdgpu_vector_memory_cache_policy_encoding_t
      vector_memory_cache_policy_encoding;
} loom_amdgpu_descriptor_set_info_t;

typedef struct loom_amdgpu_processor_info_t {
  // Processor name used in AMDHSA target IDs, such as `gfx1100`.
  iree_string_view_t target_cpu;
  // Target-low descriptor set key selected for this processor.
  iree_string_view_t descriptor_set_key;
  // Durable descriptor-set identity selected for this processor.
  uint64_t descriptor_set_stable_id;
  // Production target preset key selected for this processor.
  iree_string_view_t low_preset_key;
  // ELF EF_AMDGPU_MACH bits for this processor, or 0 when unknown.
  uint32_t elf_machine_flags;
  // ELF EF_AMDGPU_FEATURE_* bits implied by the selected target-id policy.
  uint32_t elf_feature_flags;
  // Default metadata wavefront size in lanes.
  uint32_t default_wavefront_size;
  // Kernel descriptor packing profile implemented for this processor.
  loom_amdgpu_kernel_descriptor_profile_t kernel_descriptor_profile;
  // Matrix instruction feature profile implemented for this processor.
  loom_amdgpu_matrix_feature_profile_t matrix_feature_profile;
  // VGPR encoding granule when wavefront-size-32 mode is enabled.
  uint32_t kernel_descriptor_vgpr_encoding_granule_wave32;
  // VGPR encoding granule when wavefront-size-64 mode is enabled.
  uint32_t kernel_descriptor_vgpr_encoding_granule_wave64;
  // True when flat scratch is architected and legacy user SGPRs are invalid.
  bool kernel_descriptor_has_architected_flat_scratch;
  // True when the target uses the GFX10+ SGPR resource encoding rule.
  bool kernel_descriptor_uses_gfx10_sgpr_encoding;
  // True when DX10 clamp and IEEE mode defaults are supported.
  bool kernel_descriptor_has_dx10_clamp_and_ieee_mode;
} loom_amdgpu_processor_info_t;

typedef struct loom_amdgpu_amdhsa_target_id_t {
  // Processor row selected by the target-id processor component.
  const loom_amdgpu_processor_info_t* processor;
  // Target-id feature suffix after ':', or empty when no suffix is present.
  iree_string_view_t feature_suffix;
} loom_amdgpu_amdhsa_target_id_t;

// Returns the number of known AMDGPU processor fact rows.
iree_host_size_t loom_amdgpu_target_info_processor_count(void);

// Returns the |index|-th known AMDGPU processor fact row, or NULL.
const loom_amdgpu_processor_info_t* loom_amdgpu_target_info_processor_at(
    iree_host_size_t index);

// Looks up known AMDGPU processor facts by target CPU name.
//
// Some known processors do not yet have target-low or HSACO support.
iree_status_t loom_amdgpu_target_info_lookup_processor(
    iree_string_view_t target_cpu,
    const loom_amdgpu_processor_info_t** out_processor);

// Looks up a supported AMDGPU target-low descriptor set by key.
iree_status_t loom_amdgpu_target_info_lookup_descriptor_set(
    iree_string_view_t descriptor_set_key,
    const loom_amdgpu_descriptor_set_info_t** out_descriptor_set);

// Looks up a supported AMDGPU target-low descriptor set by stable ID.
iree_status_t loom_amdgpu_target_info_lookup_descriptor_set_by_id(
    uint64_t descriptor_set_stable_id,
    const loom_amdgpu_descriptor_set_info_t** out_descriptor_set);

// Returns descriptor-set facts by stable ID, or NULL when unsupported.
const loom_amdgpu_descriptor_set_info_t*
loom_amdgpu_target_info_descriptor_set_by_id(uint64_t descriptor_set_stable_id);

// Parses an AMDHSA target ID such as `amdgcn-amd-amdhsa--gfx1100`.
iree_status_t loom_amdgpu_target_info_parse_amdhsa_target_id(
    iree_string_view_t target_id,
    loom_amdgpu_amdhsa_target_id_t* out_target_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_TARGET_INFO_DEFS_H_
