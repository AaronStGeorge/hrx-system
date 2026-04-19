// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/requirements.h"

#include "loom/target/emit/llvmir/target_registry.h"
#include "loom/target/emit/llvmir/tool.h"
#include "loom/tools/loom-check/status_util.h"

static bool loom_check_case_has_requirement(const loom_check_case_t* test_case,
                                            iree_string_view_t requirement) {
  for (iree_host_size_t i = 0; i < test_case->requirement_count; ++i) {
    if (iree_string_view_equal(test_case->requirements[i], requirement)) {
      return true;
    }
  }
  return false;
}

static char loom_check_ascii_lower(char value) {
  return value >= 'A' && value <= 'Z' ? (char)(value + 'a' - 'A') : value;
}

static bool loom_check_string_contains_case_insensitive(
    iree_string_view_t string, iree_string_view_t needle) {
  if (iree_string_view_is_empty(needle)) return true;
  if (needle.size > string.size) return false;
  for (iree_host_size_t i = 0; i <= string.size - needle.size; ++i) {
    bool matches = true;
    for (iree_host_size_t j = 0; j < needle.size; ++j) {
      if (loom_check_ascii_lower(string.data[i + j]) !=
          loom_check_ascii_lower(needle.data[j])) {
        matches = false;
        break;
      }
    }
    if (matches) return true;
  }
  return false;
}

static bool loom_check_requirement_name_is_known(
    iree_string_view_t requirement) {
  loom_llvmir_target_registry_t target_registry;
  loom_llvmir_target_registry_initialize(&target_registry);
  return iree_string_view_equal(requirement, IREE_SV("llvm-as")) ||
         iree_string_view_equal(requirement, IREE_SV("llvm-dis")) ||
         iree_string_view_equal(requirement, IREE_SV("opt")) ||
         iree_string_view_equal(requirement, IREE_SV("llc")) ||
         loom_llvmir_target_registry_llc_requirement_provider(
             &target_registry, requirement, NULL) ||
         iree_string_view_equal(requirement,
                                IREE_SV("loom-check-test-unavailable"));
}

static iree_status_t loom_check_query_llvm_tool(
    loom_llvmir_tool_kind_t tool_kind, iree_allocator_t allocator) {
  loom_llvmir_toolchain_t toolchain;
  loom_llvmir_toolchain_initialize_from_environment(&toolchain);
  loom_llvmir_tool_output_t version_text = {0};
  iree_status_t status = loom_llvmir_tool_query_version(
      &toolchain, tool_kind, allocator, &version_text);
  loom_llvmir_tool_output_deinitialize(&version_text, allocator);
  return status;
}

static iree_status_t loom_check_query_llc_provider(
    iree_string_view_t requirement,
    const loom_llvmir_target_profile_provider_t* provider,
    iree_allocator_t allocator) {
  loom_llvmir_toolchain_t toolchain;
  loom_llvmir_toolchain_initialize_from_environment(&toolchain);
  loom_llvmir_tool_output_t version_text = {0};
  iree_status_t status = loom_llvmir_tool_query_version(
      &toolchain, LOOM_LLVMIR_TOOL_LLC, allocator, &version_text);

  if (iree_status_is_ok(status)) {
    iree_string_view_t version =
        iree_make_string_view(version_text.data, version_text.length);
    if (!loom_check_string_contains_case_insensitive(
            version, provider->llc_target_name) &&
        !loom_check_string_contains_case_insensitive(version, provider->name)) {
      status =
          iree_make_status(IREE_STATUS_UNAVAILABLE,
                           "llc is available but does not report %.*s support",
                           (int)requirement.size, requirement.data);
    }
  }

  loom_llvmir_tool_output_deinitialize(&version_text, allocator);
  return status;
}

static iree_status_t loom_check_query_requirement(
    iree_string_view_t requirement, iree_allocator_t allocator) {
  if (iree_string_view_equal(requirement, IREE_SV("llvm-as"))) {
    return loom_check_query_llvm_tool(LOOM_LLVMIR_TOOL_LLVM_AS, allocator);
  }
  if (iree_string_view_equal(requirement, IREE_SV("llvm-dis"))) {
    return loom_check_query_llvm_tool(LOOM_LLVMIR_TOOL_LLVM_DIS, allocator);
  }
  if (iree_string_view_equal(requirement, IREE_SV("opt"))) {
    return loom_check_query_llvm_tool(LOOM_LLVMIR_TOOL_OPT, allocator);
  }
  if (iree_string_view_equal(requirement, IREE_SV("llc"))) {
    return loom_check_query_llvm_tool(LOOM_LLVMIR_TOOL_LLC, allocator);
  }
  loom_llvmir_target_registry_t target_registry;
  loom_llvmir_target_registry_initialize(&target_registry);
  const loom_llvmir_target_profile_provider_t* provider = NULL;
  if (loom_llvmir_target_registry_llc_requirement_provider(
          &target_registry, requirement, &provider)) {
    return loom_check_query_llc_provider(requirement, provider, allocator);
  }
  if (iree_string_view_equal(requirement,
                             IREE_SV("loom-check-test-unavailable"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "deterministic unavailable test requirement");
  }
  return iree_ok_status();
}

static iree_status_t loom_check_append_status_name(
    iree_status_t source_status, iree_string_builder_t* builder) {
  iree_string_view_t status_name = loom_check_status_name(source_status);
  iree_status_t status =
      iree_string_builder_append_string(builder, status_name);
  iree_status_ignore(source_status);
  return status;
}

static iree_status_t loom_check_append_supported_requirement_names(
    loom_check_result_t* result) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      &result->detail, "llvm-as, llvm-dis, opt, llc"));

  loom_llvmir_target_registry_t target_registry;
  loom_llvmir_target_registry_initialize(&target_registry);
  for (iree_host_size_t i = 0;
       i < target_registry.profile_registry.provider_count; ++i) {
    const loom_llvmir_target_profile_provider_t* provider =
        target_registry.profile_registry.providers[i];
    if (iree_string_view_is_empty(provider->llc_target_name)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        &result->detail, ", llc-%.*s", (int)provider->name.size,
        provider->name.data));
  }
  return iree_string_builder_append_cstring(&result->detail, "\n");
}

static iree_status_t loom_check_fail_unknown_requirement(
    iree_string_view_t requirement, loom_check_result_t* result) {
  result->raw_outcome = LOOM_CHECK_FAIL;
  result->final_outcome = LOOM_CHECK_FAIL;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      &result->detail,
      "unknown REQUIRES requirement '%.*s'; supported external requirements "
      "are ",
      (int)requirement.size, requirement.data));
  return loom_check_append_supported_requirement_names(result);
}

static iree_status_t loom_check_fail_missing_requirement_declaration(
    iree_string_view_t emit_target, iree_string_view_t requirement,
    loom_check_result_t* result) {
  result->raw_outcome = LOOM_CHECK_FAIL;
  result->final_outcome = LOOM_CHECK_FAIL;
  return iree_string_builder_append_format(
      &result->detail,
      "RUN: emit %.*s requires '// REQUIRES: %.*s'; external tool "
      "dependencies must be declared even when they are available\n",
      (int)emit_target.size, emit_target.data, (int)requirement.size,
      requirement.data);
}

static iree_status_t loom_check_skip_unavailable_requirement(
    iree_string_view_t requirement, iree_status_t availability_status,
    loom_check_result_t* result) {
  result->raw_outcome = LOOM_CHECK_SKIP;
  result->final_outcome = LOOM_CHECK_SKIP;
  iree_status_t status = iree_string_builder_append_format(
      &result->detail,
      "skipped: requirement '%.*s' unavailable: ", (int)requirement.size,
      requirement.data);
  if (iree_status_is_ok(status)) {
    status =
        loom_check_append_status_name(availability_status, &result->detail);
  } else {
    iree_status_ignore(availability_status);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&result->detail, "\n");
  }
  return status;
}

static iree_status_t loom_check_require_declared_requirement(
    const loom_check_case_t* test_case, iree_string_view_t requirement,
    loom_check_result_t* result, bool* out_continue_execution) {
  if (loom_check_case_has_requirement(test_case, requirement)) {
    return iree_ok_status();
  }
  *out_continue_execution = false;
  return loom_check_fail_missing_requirement_declaration(test_case->emit_target,
                                                         requirement, result);
}

static bool loom_check_case_has_llc_provider_requirement(
    const loom_check_case_t* test_case,
    const loom_llvmir_target_profile_provider_t* expected_provider) {
  loom_llvmir_target_registry_t target_registry;
  loom_llvmir_target_registry_initialize(&target_registry);
  for (iree_host_size_t i = 0; i < test_case->requirement_count; ++i) {
    const loom_llvmir_target_profile_provider_t* provider = NULL;
    if (!loom_llvmir_target_registry_llc_requirement_provider(
            &target_registry, test_case->requirements[i], &provider)) {
      continue;
    }
    if (provider == expected_provider ||
        iree_string_view_equal(provider->name, expected_provider->name)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_check_fail_missing_llc_provider_requirement(
    iree_string_view_t emit_target,
    const loom_llvmir_target_profile_provider_t* provider,
    loom_check_result_t* result) {
  result->raw_outcome = LOOM_CHECK_FAIL;
  result->final_outcome = LOOM_CHECK_FAIL;
  return iree_string_builder_append_format(
      &result->detail,
      "RUN: emit %.*s requires '// REQUIRES: llc-%.*s'; external tool "
      "dependencies must be declared even when they are available\n",
      (int)emit_target.size, emit_target.data, (int)provider->name.size,
      provider->name.data);
}

static iree_status_t loom_check_require_declared_llc_provider_requirement(
    const loom_check_case_t* test_case,
    const loom_llvmir_target_profile_provider_t* provider,
    loom_check_result_t* result, bool* out_continue_execution) {
  if (iree_string_view_is_empty(provider->llc_target_name) ||
      loom_check_case_has_llc_provider_requirement(test_case, provider)) {
    return iree_ok_status();
  }
  *out_continue_execution = false;
  return loom_check_fail_missing_llc_provider_requirement(
      test_case->emit_target, provider, result);
}

static iree_status_t loom_check_require_emit_tool_declarations(
    const loom_check_case_t* test_case, loom_check_result_t* result,
    bool* out_continue_execution) {
  if (test_case->mode != LOOM_CHECK_MODE_EMIT) {
    return iree_ok_status();
  }

  iree_string_view_t emit_target =
      iree_string_view_trim(test_case->emit_target);
  iree_string_view_t target_name = iree_string_view_empty();
  iree_string_view_t profile_name = iree_string_view_empty();
  iree_string_view_split(emit_target, ' ', &target_name, &profile_name);
  target_name = iree_string_view_trim(target_name);
  profile_name = iree_string_view_trim(profile_name);

  if (iree_string_view_equal(target_name, IREE_SV("llvmir-bitcode"))) {
    IREE_RETURN_IF_ERROR(loom_check_require_declared_requirement(
        test_case, IREE_SV("llvm-dis"), result, out_continue_execution));
  } else if (iree_string_view_equal(target_name, IREE_SV("llvmir-object")) ||
             iree_string_view_equal(target_name,
                                    IREE_SV("llvmir-assembly-mnemonics")) ||
             iree_string_view_equal(target_name,
                                    IREE_SV("llvmir-asm-mnemonics"))) {
    IREE_RETURN_IF_ERROR(loom_check_require_declared_requirement(
        test_case, IREE_SV("llc"), result, out_continue_execution));
    if (!*out_continue_execution) {
      return iree_ok_status();
    }
    loom_llvmir_target_registry_t target_registry;
    loom_llvmir_target_registry_initialize(&target_registry);
    const loom_llvmir_target_profile_provider_t* provider = NULL;
    iree_status_t status = loom_llvmir_target_registry_lookup_profile_provider(
        &target_registry, profile_name, NULL, &provider);
    if (iree_status_is_ok(status)) {
      IREE_RETURN_IF_ERROR(loom_check_require_declared_llc_provider_requirement(
          test_case, provider, result, out_continue_execution));
    } else {
      iree_status_ignore(status);
    }
  }

  return iree_ok_status();
}

iree_status_t loom_check_preflight_requirements(
    const loom_check_case_t* test_case, iree_allocator_t allocator,
    loom_check_result_t* result, bool* out_continue_execution) {
  IREE_ASSERT_ARGUMENT(test_case);
  IREE_ASSERT_ARGUMENT(result);
  IREE_ASSERT_ARGUMENT(out_continue_execution);

  *out_continue_execution = true;
  for (iree_host_size_t i = 0; i < test_case->requirement_count; ++i) {
    iree_string_view_t requirement = test_case->requirements[i];
    if (!loom_check_requirement_name_is_known(requirement)) {
      *out_continue_execution = false;
      return loom_check_fail_unknown_requirement(requirement, result);
    }
  }

  IREE_RETURN_IF_ERROR(loom_check_require_emit_tool_declarations(
      test_case, result, out_continue_execution));
  if (!*out_continue_execution) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < test_case->requirement_count; ++i) {
    iree_string_view_t requirement = test_case->requirements[i];
    iree_status_t status = loom_check_query_requirement(requirement, allocator);
    if (!iree_status_is_ok(status)) {
      *out_continue_execution = false;
      return loom_check_skip_unavailable_requirement(requirement, status,
                                                     result);
    }
  }

  return iree_ok_status();
}
