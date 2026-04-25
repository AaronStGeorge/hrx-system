// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstring>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/amdgpu/execution_provider.h"
#include "loom/target/arch/amdgpu/amdgpu_hal_testdata.h"
#include "loom/target/arch/amdgpu/amdgpu_hal_backend.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"
#include "loom/tooling/execution/compile_options.h"
#include "loom/tooling/execution/execution_backend.h"
#include "loom/tooling/execution/execution_provider.h"
#include "loom/tooling/execution/hal_runtime.h"
#include "loom/tooling/execution/one_shot.h"
#include "loom/tooling/execution/session.h"

namespace loom {
namespace {

iree_status_t RegisterAllDialects(void* user_data, loom_context_t* context) {
  (void)user_data;
  return loom_op_registry_register_all_dialects(context);
}

std::string StringViewToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

std::string StatusToStringAndFree(iree_status_t status) {
  iree_allocator_t allocator = iree_allocator_system();
  char* buffer = nullptr;
  iree_host_size_t buffer_length = 0;
  std::string result = iree_status_code_string(iree_status_code(status));
  if (iree_status_to_string(status, &allocator, &buffer, &buffer_length)) {
    result.assign(buffer, buffer_length);
    iree_allocator_free(allocator, buffer);
  }
  iree_status_free(status);
  return result;
}

bool StatusMeansAmdgpuUnavailable(iree_status_code_t code) {
  return code == IREE_STATUS_NOT_FOUND || code == IREE_STATUS_UNAVAILABLE ||
         code == IREE_STATUS_FAILED_PRECONDITION;
}

iree_string_view_t EmbeddedLoomSource(const char* name) {
  const struct iree_file_toc_t* file_toc =
      loom_amdgpu_amdgpu_hal_testdata_create();
  for (size_t i = 0; i < loom_amdgpu_amdgpu_hal_testdata_size(); ++i) {
    if (strcmp(file_toc[i].name, name) == 0) {
      return iree_make_string_view(file_toc[i].data, file_toc[i].size);
    }
  }
  return iree_string_view_empty();
}

iree_status_t SelectCurrentAmdgpuProcessor(
    const loom_amdgpu_processor_info_t** out_processor) {
  IREE_ASSERT_ARGUMENT(out_processor);
  *out_processor = nullptr;
  loom_run_hal_runtime_t runtime = {};
  iree_status_t status = loom_run_hal_runtime_initialize(
      &iree_run_loom_amdgpu_hal_backend, iree_allocator_system(), &runtime);

  loom_run_hal_selected_target_t target = {};
  if (iree_status_is_ok(status)) {
    status = iree_run_loom_amdgpu_hal_backend.select_target(
        &iree_run_loom_amdgpu_hal_backend, &runtime, iree_allocator_system(),
        &target);
  }
  if (iree_status_is_ok(status)) {
    *out_processor =
        static_cast<const loom_amdgpu_processor_info_t*>(target.data);
  }
  loom_run_hal_runtime_deinitialize(&runtime);
  return status;
}

void RunAmdgpuHalFixture(
    const char* source_file_name, const iree_string_view_t* bindings,
    const char* binding_conventions, iree_host_size_t binding_count,
    const iree_string_view_t* expected_bindings,
    const char* expected_binding_conventions,
    iree_host_size_t expected_binding_count,
    loom_amdgpu_matrix_feature_profile_t required_matrix_profile =
        LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_NONE) {
  if (required_matrix_profile != LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_NONE) {
    const loom_amdgpu_processor_info_t* processor = nullptr;
    iree_status_t status = SelectCurrentAmdgpuProcessor(&processor);
    const iree_status_code_t status_code = iree_status_code(status);
    std::string status_message;
    if (!iree_status_is_ok(status)) {
      status_message = StatusToStringAndFree(status);
    }
    if (StatusMeansAmdgpuUnavailable(status_code)) {
      GTEST_SKIP() << status_message;
    }
    ASSERT_EQ(status_code, IREE_STATUS_OK) << status_message;
    ASSERT_NE(processor, nullptr);
    if (processor->matrix_feature_profile != required_matrix_profile) {
      GTEST_SKIP() << "selected AMDGPU processor '"
                   << StringViewToString(processor->target_cpu)
                   << "' does not match the matrix feature profile required "
                      "by this target-specific low fixture";
    }
  }

  const iree_string_view_t source = EmbeddedLoomSource(source_file_name);
  ASSERT_FALSE(iree_string_view_is_empty(source)) << source_file_name;

  const loom_run_execution_provider_t* providers[] = {
      &loom_amdgpu_target_provider,
  };
  const loom_run_execution_provider_set_t provider_set = {
      .providers = providers,
      .provider_count = IREE_ARRAYSIZE(providers),
  };
  loom_run_execution_environment_t environment = {};
  IREE_ASSERT_OK(
      loom_run_execution_environment_initialize(&provider_set, &environment));

  loom_run_session_options_t session_options = {};
  loom_run_session_options_initialize(&session_options);
  session_options.register_context = {
      .fn = RegisterAllDialects,
  };
  session_options.initialize_low_descriptor_registry =
      loom_run_execution_environment_low_descriptor_registry_callback(
          &environment);

  loom_run_session_t session = {};
  IREE_ASSERT_OK(loom_run_session_initialize(&session_options, &session));

  loom_run_module_parse_options_t parse_options = {};
  loom_run_module_parse_options_initialize(&parse_options);
  parse_options.filename = iree_make_cstring_view(source_file_name);
  parse_options.source = source;

  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(loom_run_module_parse(&session, &parse_options, &run_module));

  const loom_run_execution_backend_t* backend =
      loom_run_execution_backend_registry_lookup(
          loom_run_execution_environment_execution_backend_registry(
              &environment),
          IREE_SV("amdgpu-hal"));
  ASSERT_NE(backend, nullptr);

  loom_run_candidate_compile_options_t compile_options = {};
  loom_run_candidate_compile_options_initialize(&compile_options);
  compile_options.source_resolver =
      loom_run_module_source_resolver(&run_module);

  loom_run_one_shot_options_t one_shot_options = {};
  loom_run_one_shot_options_initialize(&one_shot_options);
  one_shot_options.hal_bindings = {
      .values = bindings,
      .conventions = binding_conventions,
      .count = binding_count,
  };
  one_shot_options.hal_expected_bindings = {
      .values = expected_bindings,
      .conventions = expected_binding_conventions,
      .count = expected_binding_count,
  };
  one_shot_options.hal_max_output_element_count = 1024;

  loom_run_one_shot_result_t result = {};
  loom_run_one_shot_result_initialize(iree_allocator_system(), &result);
  const loom_run_one_shot_request_t request = {
      .run_module = &run_module,
      .compile_options = &compile_options,
      .options = &one_shot_options,
      .host_allocator = iree_allocator_system(),
      .result = &result,
  };

  iree_status_t run_status = backend->run_one_shot(backend, &request);
  const iree_status_code_t run_status_code = iree_status_code(run_status);
  std::string run_status_message;
  if (!iree_status_is_ok(run_status)) {
    run_status_message = StatusToStringAndFree(run_status);
  }
  const int exit_code = result.exit_code;
  const std::string output =
      StringViewToString(iree_string_builder_view(&result.output));

  loom_run_one_shot_result_deinitialize(&result);
  loom_run_module_deinitialize(&run_module);
  loom_run_session_deinitialize(&session);
  loom_run_execution_environment_deinitialize(&environment);

  if (StatusMeansAmdgpuUnavailable(run_status_code)) {
    GTEST_SKIP() << run_status_message;
  }
  ASSERT_EQ(run_status_code, IREE_STATUS_OK) << run_status_message;
  EXPECT_EQ(exit_code, 0) << output;
}

TEST(AmdgpuHalExecutionTest, RunsCurrentTargetB128VectorAdd) {
  const iree_string_view_t bindings[] = {
      IREE_SV("64x4xi32=1"),
      IREE_SV("64x4xi32=2"),
      IREE_SV("64x4xi32=0"),
  };
  const char binding_conventions[] = {'r', 'r', 'r'};
  const iree_string_view_t expected_bindings[] = {
      IREE_SV("64x4xi32=1"),
      IREE_SV("64x4xi32=2"),
      IREE_SV("64x4xi32=3"),
  };
  const char expected_binding_conventions[] = {'r', 'r', 'r'};
  RunAmdgpuHalFixture("b128_vector_add.loom", bindings, binding_conventions,
                      IREE_ARRAYSIZE(bindings), expected_bindings,
                      expected_binding_conventions,
                      IREE_ARRAYSIZE(expected_bindings));
}

TEST(AmdgpuHalExecutionTest, RunsGfx11WmmaZeroLowPacket) {
  const iree_string_view_t bindings[] = {
      IREE_SV("4xi32=7"),
  };
  const char binding_conventions[] = {'r'};
  const iree_string_view_t expected_bindings[] = {
      IREE_SV("4xi32=0"),
  };
  const char expected_binding_conventions[] = {'r'};
  RunAmdgpuHalFixture("gfx11_wmma_zero.loom", bindings, binding_conventions,
                      IREE_ARRAYSIZE(bindings), expected_bindings,
                      expected_binding_conventions,
                      IREE_ARRAYSIZE(expected_bindings),
                      LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11);
}

}  // namespace
}  // namespace loom
