// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/amdgpu/execution_provider.h"
#include "loom/tooling/execution/compile_options.h"
#include "loom/tooling/execution/execution_backend.h"
#include "loom/tooling/execution/execution_provider.h"
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

TEST(AmdgpuHalExecutionTest, RunsCurrentTargetB128VectorAdd) {
  static const char kSource[] =
      "target.profile @gfx_target preset(\"amdgpu-current\")\n"
      "func.def target(@gfx_target) @loom_kernel(%lhs: buffer, %rhs: buffer, "
      "%output: buffer) {\n"
      "  %tid = kernel.workitem.id<x> : index\n"
      "  %zero = index.constant 0 : offset\n"
      "  %lhs_view = buffer.view %lhs[%zero] : buffer -> "
      "view<64x4xi32, #dense>\n"
      "  %rhs_view = buffer.view %rhs[%zero] : buffer -> "
      "view<64x4xi32, #dense>\n"
      "  %output_view = buffer.view %output[%zero] : buffer -> "
      "view<64x4xi32, #dense>\n"
      "  %lhs_loaded = vector.load %lhs_view[%tid, 0] : "
      "view<64x4xi32, #dense> -> vector<4xi32>\n"
      "  %rhs_loaded = vector.load %rhs_view[%tid, 0] : "
      "view<64x4xi32, #dense> -> vector<4xi32>\n"
      "  %sum = vector.addi %lhs_loaded, %rhs_loaded : vector<4xi32>\n"
      "  vector.store %sum, %output_view[%tid, 0] : vector<4xi32>, "
      "view<64x4xi32, #dense>\n"
      "  func.return\n"
      "}\n";

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
  parse_options.filename = IREE_SV("amdgpu_hal_execution_test.loom");
  parse_options.source = iree_make_cstring_view(kSource);

  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(loom_run_module_parse(&session, &parse_options, &run_module));

  const loom_run_execution_backend_t* backend =
      loom_run_execution_backend_registry_lookup(
          loom_run_execution_environment_execution_backend_registry(
              &environment),
          IREE_SV("amdgpu-hal"));
  ASSERT_NE(backend, nullptr);

  iree_string_view_t bindings[] = {
      IREE_SV("64x4xi32=1"),
      IREE_SV("64x4xi32=2"),
      IREE_SV("64x4xi32=0"),
  };
  const char binding_conventions[] = {'r', 'r', 'r'};
  iree_string_view_t expected_bindings[] = {
      IREE_SV("64x4xi32=1"),
      IREE_SV("64x4xi32=2"),
      IREE_SV("64x4xi32=3"),
  };
  const char expected_binding_conventions[] = {'r', 'r', 'r'};

  loom_run_candidate_compile_options_t compile_options = {};
  loom_run_candidate_compile_options_initialize(&compile_options);
  compile_options.source_resolver =
      loom_run_module_source_resolver(&run_module);

  loom_run_one_shot_options_t one_shot_options = {};
  loom_run_one_shot_options_initialize(&one_shot_options);
  one_shot_options.hal_bindings = {
      .values = bindings,
      .conventions = binding_conventions,
      .count = IREE_ARRAYSIZE(bindings),
  };
  one_shot_options.hal_expected_bindings = {
      .values = expected_bindings,
      .conventions = expected_binding_conventions,
      .count = IREE_ARRAYSIZE(expected_bindings),
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

}  // namespace
}  // namespace loom
