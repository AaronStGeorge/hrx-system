// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_SPIRV_PROFILE_H_
#define LOOMC_TARGET_SPIRV_PROFILE_H_

#include "loomc/result.h"
#include "loomc/target/spirv/base.h"

/// @file
/// SPIR-V target profile facts.
///
/// This leaf owns the normalized, header-light SPIR-V profile API. It does not
/// include Vulkan, IREE HAL, or platform probe headers. Live adapters and saved
/// profile importers should normalize their observations into these facts and
/// then create ordinary `loomc_target_profile_t` handles.
///
/// Profiles are partial by construction. Unknown feature facts remain unknown
/// until a caller supplies a stronger observation; false means a feature is
/// known to be unavailable. Contradictory true/false facts are reported through
/// the returned `loomc_result_t` with provenance strings preserved in the
/// diagnostic text.
///
/// @par Example
/// Create a reusable Vulkan-style SPIR-V profile and target selection:
///
/// @code{.c}
/// loomc_spirv_feature_fact_t facts[] = {
///     {
///         .feature = LOOMC_SPIRV_FEATURE_FLOAT16,
///         .state = LOOMC_TARGET_FACT_STATE_TRUE,
///         .provenance = loomc_make_cstring_view("vulkaninfo:shaderFloat16"),
///     },
/// };
/// loomc_spirv_profile_options_t options = {
///     .type = LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
///     .structure_size = sizeof(loomc_spirv_profile_options_t),
///     .identifier = loomc_make_cstring_view("offline-vulkan13"),
///     .preset = LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA,
///     .feature_facts = facts,
///     .feature_fact_count = 1,
/// };
/// loomc_target_profile_t* profile = NULL;
/// loomc_result_t* result = NULL;
/// loomc_status_t status = loomc_target_profile_create_spirv(
///     target_environment, &options, loomc_allocator_system(), &profile,
///     &result);
/// if (!loomc_status_is_ok(status)) return status;
/// if (!loomc_result_succeeded(result)) {
///   // Inspect structured diagnostics; profile is NULL on profile failure.
/// }
/// loomc_result_release(result);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Stable SPIR-V feature fact identifier.
///
/// These identifiers are Loom profile facts, not Vulkan feature struct fields.
/// A live Vulkan probe, saved vulkaninfo profile, or non-Vulkan driver may all
/// map their own capability model into the same feature IDs.
typedef enum loomc_spirv_feature_e {
  /// Unknown or uninitialized feature.
  LOOMC_SPIRV_FEATURE_UNKNOWN = 0,

  /// Vulkan shader-module baseline.
  LOOMC_SPIRV_FEATURE_VULKAN_SHADER = 1,

  /// SPV_KHR_physical_storage_buffer support.
  LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER = 2,

  /// 16-bit floating-point scalar support.
  LOOMC_SPIRV_FEATURE_FLOAT16 = 3,

  /// 64-bit floating-point scalar support.
  LOOMC_SPIRV_FEATURE_FLOAT64 = 4,

  /// 8-bit integer scalar support.
  LOOMC_SPIRV_FEATURE_INT8 = 5,

  /// 16-bit integer scalar support.
  LOOMC_SPIRV_FEATURE_INT16 = 6,

  /// 8-bit storage-buffer access support.
  LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_8BIT_ACCESS = 7,

  /// 16-bit storage-buffer access support.
  LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS = 8,

  /// SPV_NV_cooperative_vector support.
  LOOMC_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV = 9,

  /// SPV_NV_cooperative_vector training-operation support.
  LOOMC_SPIRV_FEATURE_COOPERATIVE_VECTOR_TRAINING_NV = 10,

  /// SPV_KHR_cooperative_matrix support.
  LOOMC_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR = 11,

  /// SPV_KHR_bfloat16 scalar type support.
  LOOMC_SPIRV_FEATURE_BFLOAT16_TYPE_KHR = 12,

  /// SPV_KHR_bfloat16 dot-product support.
  LOOMC_SPIRV_FEATURE_BFLOAT16_DOT_PRODUCT_KHR = 13,

  /// SPV_KHR_bfloat16 cooperative-matrix support.
  LOOMC_SPIRV_FEATURE_BFLOAT16_COOPERATIVE_MATRIX_KHR = 14,

  /// 64-bit integer scalar support.
  LOOMC_SPIRV_FEATURE_INT64 = 15,

  /// Number of public SPIR-V feature identifiers.
  LOOMC_SPIRV_FEATURE_COUNT = 16,
} loomc_spirv_feature_t;

/// Bitset of `loomc_spirv_feature_t` values.
typedef uint64_t loomc_spirv_feature_bits_t;

/// Returns the direct bit for a SPIR-V feature.
static inline loomc_spirv_feature_bits_t loomc_spirv_feature_bit(
    loomc_spirv_feature_t feature) {
  return feature > LOOMC_SPIRV_FEATURE_UNKNOWN &&
                 feature < LOOMC_SPIRV_FEATURE_COUNT
             ? (UINT64_C(1) << feature)
             : 0u;
}

/// Built-in SPIR-V profile preset.
typedef enum loomc_spirv_profile_preset_e {
  /// No preset facts.
  LOOMC_SPIRV_PROFILE_PRESET_NONE = 0,

  /// Vulkan 1.3 shader profile with physical-storage-buffer addressing.
  LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA = 1,
} loomc_spirv_profile_preset_t;

/// One SPIR-V feature observation.
typedef struct loomc_spirv_feature_fact_t {
  /// Feature being observed.
  loomc_spirv_feature_t feature;

  /// Observed state for the feature.
  loomc_target_fact_state_t state;

  /// Borrowed provenance string used in diagnostics.
  loomc_string_view_t provenance;
} loomc_spirv_feature_fact_t;

/// SPIR-V target profile creation options.
typedef struct loomc_spirv_profile_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  const void* next;

  /// Stable profile identifier used in diagnostics.
  loomc_string_view_t identifier;

  /// Built-in preset facts applied before explicit feature facts.
  loomc_spirv_profile_preset_t preset;

  /// Borrowed feature fact array.
  const loomc_spirv_feature_fact_t* feature_facts;

  /// Number of entries in `feature_facts`.
  loomc_host_size_t feature_fact_count;
} loomc_spirv_profile_options_t;

/// Prepared SPIR-V profile summary.
typedef struct loomc_spirv_profile_info_t {
  /// Minimum SPIR-V binary version selected by known-true features.
  uint32_t minimum_spirv_version;

  /// SPIR-V addressing model numeric value.
  uint32_t addressing_model;

  /// SPIR-V memory model numeric value.
  uint32_t memory_model;

  /// Number of OpExtension rows selected by known-true features.
  loomc_host_size_t extension_count;

  /// Number of OpCapability rows selected by known-true features.
  loomc_host_size_t capability_count;

  /// Number of opcode rows selected by known-true features.
  loomc_host_size_t opcode_count;

  /// Number of storage-class rows selected by known-true features.
  loomc_host_size_t storage_class_count;

  /// Number of decoration rows selected by known-true features.
  loomc_host_size_t decoration_count;
} loomc_spirv_profile_info_t;

/// Creates a reusable SPIR-V target profile.
///
/// @param target_environment SPIR-V-capable target environment.
/// @param options Profile facts, or `NULL` for an empty partial profile.
/// @param allocator Host allocator used for profile and result storage.
/// @param out_profile Receives one retained profile when `out_result`
/// succeeds.
/// @param out_result Receives one retained result describing profile
/// preparation.
/// @return OK when profile preparation completed far enough to report a result.
/// Non-OK statuses represent API misuse or infrastructure failures before a
/// result could be produced.
///
/// @ownership
/// The caller owns `out_profile` when the returned result succeeds and releases
/// it with `loomc_target_profile_release`. The caller always owns `out_result`
/// on an OK return and releases it with `loomc_result_release`.
///
/// @thread_safety
/// Created profiles are immutable and may be shared across threads.
LOOMC_API_EXPORT loomc_status_t loomc_target_profile_create_spirv(
    loomc_target_environment_t* target_environment,
    const loomc_spirv_profile_options_t* options, loomc_allocator_t allocator,
    loomc_target_profile_t** out_profile, loomc_result_t** out_result);

/// Returns the known state for one SPIR-V feature fact.
LOOMC_API_EXPORT loomc_status_t loomc_spirv_target_profile_query_feature(
    const loomc_target_profile_t* profile, loomc_spirv_feature_t feature,
    loomc_target_fact_state_t* out_state);

/// Returns prepared SPIR-V profile summary rows.
LOOMC_API_EXPORT loomc_status_t
loomc_spirv_target_profile_query_info(const loomc_target_profile_t* profile,
                                      loomc_spirv_profile_info_t* out_info);

/// Returns an OpExtension row by index.
///
/// @lifetime
/// The returned string view remains valid until `profile` is released.
LOOMC_API_EXPORT loomc_status_t loomc_spirv_target_profile_extension_at(
    const loomc_target_profile_t* profile, loomc_host_size_t index,
    loomc_string_view_t* out_extension);

/// Returns an OpCapability numeric row by index.
LOOMC_API_EXPORT loomc_status_t loomc_spirv_target_profile_capability_at(
    const loomc_target_profile_t* profile, loomc_host_size_t index,
    uint32_t* out_capability);

/// Returns an opcode numeric row by index.
LOOMC_API_EXPORT loomc_status_t loomc_spirv_target_profile_opcode_at(
    const loomc_target_profile_t* profile, loomc_host_size_t index,
    uint32_t* out_opcode);

/// Returns a storage-class numeric row by index.
LOOMC_API_EXPORT loomc_status_t loomc_spirv_target_profile_storage_class_at(
    const loomc_target_profile_t* profile, loomc_host_size_t index,
    uint32_t* out_storage_class);

/// Returns a decoration numeric row by index.
LOOMC_API_EXPORT loomc_status_t loomc_spirv_target_profile_decoration_at(
    const loomc_target_profile_t* profile, loomc_host_size_t index,
    uint32_t* out_decoration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_SPIRV_PROFILE_H_
