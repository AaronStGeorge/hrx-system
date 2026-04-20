// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/loom_check_requirements.h"

#include "loom/target/tool/process.h"
#include "loom/tools/loom-check/requirements.h"

static bool loom_amdgpu_loom_check_requirement_provider_matches(
    const loom_check_requirement_provider_t* provider,
    iree_string_view_t requirement) {
  (void)provider;
  return iree_string_view_equal(requirement, IREE_SV("amdgpu-hal")) ||
         iree_string_view_equal(requirement, IREE_SV("amdgpu-hal-b128"));
}

static iree_status_t loom_amdgpu_loom_check_run_hal_probe(
    const loom_check_environment_t* environment, iree_allocator_t allocator,
    loom_tool_process_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(out_result);
  iree_string_view_t path =
      loom_check_environment_iree_run_loom_path(environment);
  iree_string_view_t arguments[] = {IREE_SV("--loom_backend=amdgpu-hal"),
                                    IREE_SV("--probe_hal")};
  return loom_tool_process_run(
      path, loom_check_process_path_searches_path(path), arguments,
      IREE_ARRAYSIZE(arguments), allocator, out_result);
}

static iree_status_t loom_amdgpu_loom_check_hal_probe_failure_status(
    const loom_tool_process_result_t* result) {
  iree_status_t status = iree_ok_status();
  if (!loom_tool_process_result_succeeded(result)) {
    iree_string_view_t detail = iree_make_string_view(
        result->stderr_text.data, result->stderr_text.length);
    if (iree_string_view_is_empty(detail)) {
      detail = iree_make_string_view(result->stdout_text.data,
                                     result->stdout_text.length);
    }
    detail = iree_string_view_trim(detail);
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "iree-run-loom --loom_backend=amdgpu-hal --probe_hal exited with code "
        "%d%s%.*s",
        result->exit_code, iree_string_view_is_empty(detail) ? "" : ": ",
        (int)iree_min(detail.size, (iree_host_size_t)2048), detail.data);
  }
  return status;
}

static iree_status_t loom_amdgpu_loom_check_query_hal(
    const loom_check_environment_t* environment, iree_allocator_t allocator) {
  loom_tool_process_result_t result = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_loom_check_run_hal_probe(environment, allocator, &result));
  iree_status_t status =
      loom_amdgpu_loom_check_hal_probe_failure_status(&result);
  loom_tool_process_result_deinitialize(&result, allocator);
  return status;
}

static bool loom_amdgpu_loom_check_hal_probe_has_preset(
    iree_string_view_t stdout_text, iree_string_view_t preset) {
  iree_string_view_t needle = IREE_SV("hal preset: ");
  iree_host_size_t position = iree_string_view_find(stdout_text, needle, 0);
  while (position != IREE_STRING_VIEW_NPOS) {
    iree_string_view_t suffix = iree_string_view_substr(
        stdout_text, position + needle.size, IREE_HOST_SIZE_MAX);
    if (iree_string_view_starts_with(suffix, preset)) {
      return true;
    }
    position = iree_string_view_find(stdout_text, needle, position + 1);
  }
  return false;
}

static iree_status_t loom_amdgpu_loom_check_query_hal_b128(
    const loom_check_environment_t* environment, iree_allocator_t allocator) {
  loom_tool_process_result_t result = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_loom_check_run_hal_probe(environment, allocator, &result));

  iree_status_t status =
      loom_amdgpu_loom_check_hal_probe_failure_status(&result);
  if (iree_status_is_ok(status)) {
    iree_string_view_t stdout_text = iree_make_string_view(
        result.stdout_text.data, result.stdout_text.length);
    const bool supports_b128 = loom_amdgpu_loom_check_hal_probe_has_preset(
                                   stdout_text, IREE_SV("amdgpu-gfx11")) ||
                               loom_amdgpu_loom_check_hal_probe_has_preset(
                                   stdout_text, IREE_SV("amdgpu-gfx12")) ||
                               loom_amdgpu_loom_check_hal_probe_has_preset(
                                   stdout_text, IREE_SV("amdgpu-gfx1250"));
    if (!supports_b128) {
      iree_string_view_t detail = iree_string_view_trim(stdout_text);
      status = iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "current AMDGPU HAL target does not use b128 low memory descriptors: "
          "%.*s",
          (int)iree_min(detail.size, (iree_host_size_t)2048), detail.data);
    }
  }

  loom_tool_process_result_deinitialize(&result, allocator);
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
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown AMDGPU loom-check requirement '%.*s'",
                          (int)requirement.size, requirement.data);
}

static iree_status_t loom_amdgpu_loom_check_requirement_provider_append_names(
    const loom_check_requirement_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder,
                                            "amdgpu-hal, amdgpu-hal-b128");
}

const loom_check_requirement_provider_t
    loom_amdgpu_loom_check_requirement_provider = {
        .name = IREE_SVL("amdgpu"),
        .match = loom_amdgpu_loom_check_requirement_provider_matches,
        .query = loom_amdgpu_loom_check_requirement_provider_query,
        .append_names =
            loom_amdgpu_loom_check_requirement_provider_append_names,
};
