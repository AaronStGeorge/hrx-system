// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/loom_check_requirements.h"

#include "loom/target/arch/amdgpu/amdgpu_hal_backend.h"
#include "loom/tooling/execution/hal_runtime.h"

static bool loom_amdgpu_loom_check_requirement_provider_matches(
    const loom_check_requirement_provider_t* provider,
    iree_string_view_t requirement) {
  (void)provider;
  return iree_string_view_equal(requirement, IREE_SV("amdgpu-hal")) ||
         iree_string_view_equal(requirement, IREE_SV("amdgpu-hal-b128")) ||
         iree_string_view_equal(requirement, IREE_SV("amdgpu-hal-gfx11"));
}

static iree_status_t loom_amdgpu_loom_check_select_hal_target(
    const loom_check_environment_t* environment, iree_allocator_t allocator,
    loom_run_hal_selected_target_t* out_target) {
  (void)environment;
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = (loom_run_hal_selected_target_t){0};

  loom_run_hal_runtime_t runtime = {0};
  iree_status_t status = loom_run_hal_runtime_initialize(
      &iree_run_loom_amdgpu_hal_backend, allocator, &runtime);
  if (iree_status_is_ok(status)) {
    status = iree_run_loom_amdgpu_hal_backend.select_target(
        &iree_run_loom_amdgpu_hal_backend, &runtime, allocator, out_target);
  }
  loom_run_hal_runtime_deinitialize(&runtime);
  return status;
}

static iree_status_t loom_amdgpu_loom_check_query_hal(
    const loom_check_environment_t* environment, iree_allocator_t allocator) {
  loom_run_hal_selected_target_t target = {0};
  return loom_amdgpu_loom_check_select_hal_target(environment, allocator,
                                                  &target);
}

static iree_status_t loom_amdgpu_loom_check_query_hal_b128(
    const loom_check_environment_t* environment, iree_allocator_t allocator) {
  loom_run_hal_selected_target_t target = {0};
  iree_status_t status =
      loom_amdgpu_loom_check_select_hal_target(environment, allocator, &target);
  if (iree_status_is_ok(status)) {
    const bool supports_b128 =
        iree_string_view_starts_with(target.preset_key,
                                     IREE_SV("amdgpu-gfx11")) ||
        iree_string_view_starts_with(target.preset_key,
                                     IREE_SV("amdgpu-gfx12")) ||
        iree_string_view_starts_with(target.preset_key,
                                     IREE_SV("amdgpu-gfx1250"));
    if (!supports_b128) {
      status = iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "current AMDGPU HAL target does not use b128 low memory descriptors: "
          "%.*s",
          (int)target.preset_key.size, target.preset_key.data);
    }
  }
  return status;
}

static iree_status_t loom_amdgpu_loom_check_query_hal_gfx11(
    const loom_check_environment_t* environment, iree_allocator_t allocator) {
  loom_run_hal_selected_target_t target = {0};
  iree_status_t status =
      loom_amdgpu_loom_check_select_hal_target(environment, allocator, &target);
  if (iree_status_is_ok(status) &&
      !iree_string_view_starts_with(target.preset_key,
                                    IREE_SV("amdgpu-gfx11"))) {
    status =
        iree_make_status(IREE_STATUS_UNAVAILABLE,
                         "current AMDGPU HAL target is not gfx11-family: %.*s",
                         (int)target.preset_key.size, target.preset_key.data);
  }
  return status;
}

static iree_status_t loom_amdgpu_loom_check_requirement_provider_query(
    const loom_check_requirement_provider_t* provider,
    const loom_check_environment_t* environment, iree_string_view_t requirement,
    iree_allocator_t allocator) {
  (void)provider;
  if (iree_string_view_equal(requirement, IREE_SV("amdgpu-hal"))) {
    return loom_amdgpu_loom_check_query_hal(environment, allocator);
  }
  if (iree_string_view_equal(requirement, IREE_SV("amdgpu-hal-b128"))) {
    return loom_amdgpu_loom_check_query_hal_b128(environment, allocator);
  }
  if (iree_string_view_equal(requirement, IREE_SV("amdgpu-hal-gfx11"))) {
    return loom_amdgpu_loom_check_query_hal_gfx11(environment, allocator);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown AMDGPU loom-check requirement '%.*s'",
                          (int)requirement.size, requirement.data);
}

static iree_status_t loom_amdgpu_loom_check_requirement_provider_append_names(
    const loom_check_requirement_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder,
                                            "amdgpu-hal, amdgpu-hal-b128, "
                                            "amdgpu-hal-gfx11");
}

const loom_check_requirement_provider_t
    loom_amdgpu_loom_check_requirement_provider = {
        .name = IREE_SVL("amdgpu"),
        .match = loom_amdgpu_loom_check_requirement_provider_matches,
        .query = loom_amdgpu_loom_check_requirement_provider_query,
        .append_names =
            loom_amdgpu_loom_check_requirement_provider_append_names,
};
