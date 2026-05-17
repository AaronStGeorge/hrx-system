// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/spirv/vulkan_profile.h"

#include <stdint.h>

#include "loom/target/arch/spirv/features.h"
#include "loom/target/arch/spirv/target_records.h"

static iree_status_t loom_spirv_vulkan_hal_profile_query_u32(
    iree_hal_device_t* device, iree_string_view_t category,
    iree_string_view_t key, uint32_t* out_value) {
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_device_query_i64(device, category, key, &value));
  if (value < 0 || value > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan profile query '%.*s :: %.*s' returned value out of uint32_t "
        "range",
        (int)category.size, category.data, (int)key.size, key.data);
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t loom_spirv_vulkan_hal_profile_query_feature_flag(
    iree_hal_device_t* device, iree_string_view_t key,
    loom_spirv_vulkan_hal_profile_flag_bits_t flag,
    loom_spirv_vulkan_hal_profile_facts_t* facts) {
  uint32_t value = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
      device, IREE_SV("vulkan.feature"), key, &value));
  if (value != 0) {
    facts->flags |= flag;
  }
  return iree_ok_status();
}

iree_status_t loom_spirv_vulkan_hal_profile_query(
    iree_hal_device_t* device, iree_hal_executable_cache_t* executable_cache,
    loom_spirv_vulkan_hal_profile_facts_t* out_facts) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(executable_cache);
  IREE_ASSERT_ARGUMENT(out_facts);

  *out_facts = (loom_spirv_vulkan_hal_profile_facts_t){0};
  if (iree_hal_executable_cache_can_prepare_format(
          executable_cache, IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION,
          IREE_SV("vulkan-spirv-bda-raw"))) {
    out_facts->flags |= LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_RAW_BDA_EXECUTABLE;
  }

  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
      device, IREE_SV("vulkan.device"), IREE_SV("api_version"),
      &out_facts->api_version));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
      device, IREE_SV("vulkan.device"), IREE_SV("subgroup_size"),
      &out_facts->subgroup_size));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
      device, IREE_SV("vulkan.device"),
      IREE_SV("max_compute_workgroup_invocations"),
      &out_facts->max_compute_workgroup_invocations));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
      device, IREE_SV("vulkan.device"), IREE_SV("max_compute_workgroup_size_x"),
      &out_facts->max_compute_workgroup_size.x));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
      device, IREE_SV("vulkan.device"), IREE_SV("max_compute_workgroup_size_y"),
      &out_facts->max_compute_workgroup_size.y));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
      device, IREE_SV("vulkan.device"), IREE_SV("max_compute_workgroup_size_z"),
      &out_facts->max_compute_workgroup_size.z));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
      device, IREE_SV("vulkan.device"),
      IREE_SV("max_compute_workgroup_count_x"),
      &out_facts->max_compute_workgroup_count.x));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
      device, IREE_SV("vulkan.device"),
      IREE_SV("max_compute_workgroup_count_y"),
      &out_facts->max_compute_workgroup_count.y));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
      device, IREE_SV("vulkan.device"),
      IREE_SV("max_compute_workgroup_count_z"),
      &out_facts->max_compute_workgroup_count.z));

  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_feature_flag(
      device, IREE_SV("buffer_device_address"),
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_BUFFER_DEVICE_ADDRESS, out_facts));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_feature_flag(
      device, IREE_SV("subgroup_size_control"),
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SUBGROUP_SIZE_CONTROL, out_facts));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_feature_flag(
      device, IREE_SV("cooperative_matrix_khr"),
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_COOPERATIVE_MATRIX_KHR, out_facts));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_feature_flag(
      device, IREE_SV("storage_buffer_8bit_access"),
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_STORAGE_BUFFER_8BIT_ACCESS,
      out_facts));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_feature_flag(
      device, IREE_SV("shader_float16"),
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT16, out_facts));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_feature_flag(
      device, IREE_SV("shader_float64"),
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_FLOAT64, out_facts));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_feature_flag(
      device, IREE_SV("shader_int8"),
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT8, out_facts));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_feature_flag(
      device, IREE_SV("shader_int16"),
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT16, out_facts));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_feature_flag(
      device, IREE_SV("shader_int64"),
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INT64, out_facts));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_feature_flag(
      device, IREE_SV("shader_integer_dot_product"),
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_SHADER_INTEGER_DOT_PRODUCT,
      out_facts));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_feature_flag(
      device, IREE_SV("vulkan_memory_model"),
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_VULKAN_MEMORY_MODEL, out_facts));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_feature_flag(
      device, IREE_SV("vulkan_memory_model_device_scope"),
      LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_VULKAN_MEMORY_MODEL_DEVICE_SCOPE,
      out_facts));

  if (iree_any_bit_set(
          out_facts->flags,
          LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_COOPERATIVE_MATRIX_KHR)) {
    IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
        device, IREE_SV("vulkan.cooperative_matrix"), IREE_SV("property_count"),
        &out_facts->cooperative_matrix_property_count));
    IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_query_u32(
        device, IREE_SV("vulkan.cooperative_matrix"),
        IREE_SV("supported_stages"),
        &out_facts->cooperative_matrix_supported_stages));
  }

  return iree_ok_status();
}

static iree_status_t loom_spirv_vulkan_hal_profile_require_flag(
    const loom_spirv_vulkan_hal_profile_facts_t* facts,
    loom_spirv_vulkan_hal_profile_flag_bits_t flag,
    iree_string_view_t message) {
  if (iree_all_bits_set(facts->flags, flag)) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNAVAILABLE, "%.*s", (int)message.size,
                          message.data);
}

iree_status_t loom_spirv_vulkan_hal_profile_initialize_target_bundle(
    const loom_spirv_vulkan_hal_profile_facts_t* facts,
    loom_target_bundle_storage_t* out_storage) {
  IREE_ASSERT_ARGUMENT(facts);
  IREE_ASSERT_ARGUMENT(out_storage);

  *out_storage = (loom_target_bundle_storage_t){0};

  if (facts->api_version < LOOM_SPIRV_VULKAN_API_VERSION_1_3) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "Vulkan SPIR-V raw-BDA profile requires Vulkan 1.3");
  }
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_require_flag(
      facts, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_RAW_BDA_EXECUTABLE,
      IREE_SV("Vulkan HAL executable cache does not support "
              "vulkan-spirv-bda-raw")));
  IREE_RETURN_IF_ERROR(loom_spirv_vulkan_hal_profile_require_flag(
      facts, LOOM_SPIRV_VULKAN_HAL_PROFILE_FLAG_BUFFER_DEVICE_ADDRESS,
      IREE_SV("Vulkan device does not expose buffer device addresses")));

  *out_storage = (loom_target_bundle_storage_t){
      .snapshot = *loom_spirv_low_target_bundle_vulkan1_3.snapshot,
      .export_plan = *loom_spirv_low_target_bundle_vulkan1_3.export_plan,
      .config = *loom_spirv_low_target_bundle_vulkan1_3.config,
      .bundle = loom_spirv_low_target_bundle_vulkan1_3,
  };
  loom_target_bundle_storage_rebind(out_storage);

  out_storage->bundle.name = IREE_SV("spirv-vulkan1.3-bda-hal");
  out_storage->snapshot.name = IREE_SV("spirv-vulkan1.3-bda");
  out_storage->snapshot.max_workgroup_size = facts->max_compute_workgroup_size;
  out_storage->snapshot.max_flat_workgroup_size =
      facts->max_compute_workgroup_invocations;
  out_storage->snapshot.subgroup_size = facts->subgroup_size;
  out_storage->snapshot.max_workgroup_count =
      facts->max_compute_workgroup_count;
  out_storage->export_plan.name = IREE_SV("spirv-hal-kernel");
  out_storage->export_plan.abi_kind = LOOM_TARGET_ABI_HAL_KERNEL;
  out_storage->config.name = IREE_SV("spirv.logical.core.vulkan1.3.bda");
  out_storage->config.contract_feature_bits =
      LOOM_SPIRV_FEATURE_PROFILE_VULKAN_1_3_BDA;
  return iree_ok_status();
}
