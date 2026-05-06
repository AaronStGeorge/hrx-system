// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal_candidate.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ops/func/ops.h"
#include "loom/target/low_descriptor_registry_core_test.h"

namespace loom {
namespace {

using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t*);

iree_status_t RegisterDialect(loom_context_t* context, uint8_t dialect_id,
                              DialectVtablesFn dialect_vtables_fn) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
  return loom_context_register_dialect(context, dialect_id, vtables,
                                       (uint16_t)count);
}

iree_status_t RegisterContext(void* user_data, loom_context_t* context) {
  (void)user_data;
  return RegisterDialect(context, LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
}

iree_status_t InitializeLowDescriptorRegistry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  (void)user_data;
  loom_target_core_test_low_descriptor_registry_initialize(out_registry);
  return iree_ok_status();
}

constexpr char kHalSource[] =
    "func.def @empty() {\n"
    "  func.return\n"
    "}\n";

int kFakeHalTarget = 0;
int kFakeHalRuntime = 0;
bool g_fake_hal_emit_was_called = false;
loom_target_compile_report_t* g_fake_hal_emit_report = nullptr;
const uint8_t kFakeHalExecutableData[] = {0x7F, 'E', 'L', 'F'};
const uint8_t kFakeHalTargetArtifactData[] = {'h', 's', 'a', 'c', 'o'};
static const loom_target_snapshot_t kFakeSnapshot = {
    .name = IREE_SVL("fake-snapshot"),
};
static const loom_target_export_plan_t kFakeExportPlan = {
    .name = IREE_SVL("fake-export"),
    .abi_kind = LOOM_TARGET_ABI_HAL_KERNEL,
};
static const loom_target_bundle_t kFakeTargetBundle = {
    .name = IREE_SVL("fake-bundle"),
    .snapshot = &kFakeSnapshot,
    .export_plan = &kFakeExportPlan,
};

const loom_run_hal_runtime_t* FakeHalRuntime() {
  return reinterpret_cast<const loom_run_hal_runtime_t*>(&kFakeHalRuntime);
}

iree_status_t FakeHalSelectTarget(const loom_run_hal_backend_t* backend,
                                  const loom_run_hal_runtime_t* runtime,
                                  iree_allocator_t allocator,
                                  loom_run_hal_selected_target_t* out_target) {
  (void)backend;
  (void)runtime;
  (void)allocator;
  *out_target = (loom_run_hal_selected_target_t){
      .data = &kFakeHalTarget,
      .target_bundle = &kFakeTargetBundle,
      .target_key = IREE_SVL("fake-hal"),
  };
  return iree_ok_status();
}

iree_status_t FakeHalFormatTarget(const loom_run_hal_backend_t* backend,
                                  const loom_run_hal_selected_target_t* target,
                                  iree_string_builder_t* output) {
  (void)backend;
  (void)target;
  return iree_string_builder_append_cstring(output, "fake-hal");
}

iree_status_t FakeHalEmitExecutable(
    const loom_run_hal_backend_t* backend, loom_module_t* module,
    const loom_run_hal_selected_target_t* target,
    iree_string_view_t entry_symbol, loom_diagnostic_sink_t diagnostic_sink,
    loom_source_resolver_t source_resolver, uint32_t max_errors,
    loom_target_compile_report_t* report, iree_allocator_t allocator,
    bool* out_emitted, loom_run_hal_executable_t* out_executable) {
  (void)backend;
  (void)module;
  (void)entry_symbol;
  (void)diagnostic_sink;
  (void)source_resolver;
  (void)max_errors;
  (void)allocator;
  g_fake_hal_emit_was_called = true;
  g_fake_hal_emit_report = report;
  *out_emitted = false;
  if (target->data != &kFakeHalTarget) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected fake HAL target payload");
  }
  *out_executable = (loom_run_hal_executable_t){
      .executable_format = IREE_SVL("fake-hal-format"),
      .target_artifact_format = IREE_SVL("fake-native-format"),
      .target_artifact_data = iree_make_const_byte_span(
          kFakeHalTargetArtifactData, sizeof(kFakeHalTargetArtifactData)),
      .executable_data = iree_make_const_byte_span(
          kFakeHalExecutableData, sizeof(kFakeHalExecutableData)),
      .storage = &kFakeHalTarget,
  };
  *out_emitted = true;
  return iree_ok_status();
}

void FakeHalDeinitializeExecutable(const loom_run_hal_backend_t* backend,
                                   loom_run_hal_executable_t* executable,
                                   iree_allocator_t allocator) {
  (void)backend;
  (void)allocator;
  *executable = {};
}

const loom_run_hal_backend_t kFakeHalBackend = {
    .name = IREE_SVL("fake-hal"),
    .hal_driver_name = IREE_SVL("fake"),
    .target_family_name = IREE_SVL("fake"),
    .select_target = FakeHalSelectTarget,
    .format_target = FakeHalFormatTarget,
    .emit_executable = FakeHalEmitExecutable,
    .deinitialize_executable = FakeHalDeinitializeExecutable,
};

class HalCandidateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_run_session_options_t options = {};
    loom_run_session_options_initialize(&options);
    options.register_context = (loom_run_register_context_callback_t){
        .fn = RegisterContext,
    };
    options.initialize_low_descriptor_registry =
        (loom_run_initialize_low_descriptor_registry_callback_t){
            .fn = InitializeLowDescriptorRegistry,
        };
    IREE_ASSERT_OK(loom_run_session_initialize(&options, &session_));
  }

  void TearDown() override { loom_run_session_deinitialize(&session_); }

  iree_status_t Parse(iree_string_view_t source,
                      loom_run_module_t* out_module) {
    loom_run_module_parse_options_t options = {};
    loom_run_module_parse_options_initialize(&options);
    options.filename = IREE_SV("hal_candidate_test.loom");
    options.source = source;
    return loom_run_module_parse(&session_, &options, out_module);
  }

  void InitializeCompileOptions(
      loom_run_module_t* run_module,
      loom_run_candidate_compile_options_t* out_options) {
    loom_run_candidate_compile_options_initialize(out_options);
    out_options->source_resolver = loom_run_module_source_resolver(run_module);
  }

  loom_run_session_t session_ = {};
};

TEST_F(HalCandidateTest, CompileHalExecutableCandidate) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kHalSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCompileOptions(&run_module, &options);
  loom_target_compile_report_t report = {};
  options.report = &report;

  loom_run_hal_candidate_t candidate = {};
  g_fake_hal_emit_was_called = false;
  g_fake_hal_emit_report = nullptr;
  IREE_ASSERT_OK(loom_run_hal_candidate_compile(
      &kFakeHalBackend, FakeHalRuntime(), &run_module, &options,
      iree_allocator_system(), &candidate));
  EXPECT_TRUE(g_fake_hal_emit_was_called);
  EXPECT_EQ(g_fake_hal_emit_report, &candidate.compile_report);
  EXPECT_EQ(candidate.backend, &kFakeHalBackend);
  EXPECT_EQ(candidate.target.data, &kFakeHalTarget);
  EXPECT_EQ(candidate.target.target_bundle, &kFakeTargetBundle);
  EXPECT_EQ(candidate.executable.target_bundle, &kFakeTargetBundle);
  EXPECT_TRUE(iree_string_view_equal(candidate.executable.executable_format,
                                     IREE_SV("fake-hal-format")));
  EXPECT_TRUE(
      iree_string_view_equal(candidate.executable.target_artifact_format,
                             IREE_SV("fake-native-format")));
  EXPECT_EQ(candidate.executable.target_artifact_data.data,
            kFakeHalTargetArtifactData);
  EXPECT_EQ(candidate.executable.target_artifact_data.data_length,
            sizeof(kFakeHalTargetArtifactData));
  EXPECT_EQ(candidate.executable.executable_data.data, kFakeHalExecutableData);
  EXPECT_EQ(candidate.executable.executable_data.data_length,
            sizeof(kFakeHalExecutableData));
  EXPECT_EQ(candidate.compile_report.artifact_kind,
            LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE);
  EXPECT_EQ(candidate.compile_report.status_code, IREE_STATUS_OK);
  EXPECT_TRUE(iree_string_view_equal(candidate.compile_report.backend_name,
                                     IREE_SV("fake-hal")));
  EXPECT_TRUE(iree_string_view_equal(candidate.compile_report.target_key,
                                     IREE_SV("fake-hal")));
  EXPECT_TRUE(iree_string_view_equal(candidate.compile_report.executable_format,
                                     IREE_SV("fake-hal-format")));
  EXPECT_EQ(candidate.compile_report.artifact_size,
            sizeof(kFakeHalExecutableData));
  EXPECT_EQ(report.artifact_size, candidate.compile_report.artifact_size);

  loom_run_hal_candidate_deinitialize(&candidate);
  EXPECT_EQ(candidate.backend, nullptr);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(HalCandidateTest, CompileHalExecutableCandidateWithoutReport) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kHalSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCompileOptions(&run_module, &options);

  loom_run_hal_candidate_t candidate = {};
  g_fake_hal_emit_was_called = false;
  g_fake_hal_emit_report = nullptr;
  IREE_ASSERT_OK(loom_run_hal_candidate_compile(
      &kFakeHalBackend, FakeHalRuntime(), &run_module, &options,
      iree_allocator_system(), &candidate));
  EXPECT_TRUE(g_fake_hal_emit_was_called);
  EXPECT_EQ(g_fake_hal_emit_report, nullptr);
  EXPECT_EQ(candidate.compile_report.detail_flags,
            LOOM_TARGET_COMPILE_REPORT_DETAIL_NONE);
  EXPECT_EQ(candidate.compile_report.artifact_size, 0u);

  loom_run_hal_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(HalCandidateTest, CompileHalRequiresHooks) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kHalSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCompileOptions(&run_module, &options);
  loom_target_compile_report_t report = {};
  options.report = &report;

  const loom_run_hal_backend_t backend = {
      .name = IREE_SVL("missing-hooks"),
  };
  loom_run_hal_candidate_t candidate = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_run_hal_candidate_compile(
                            &backend, FakeHalRuntime(), &run_module, &options,
                            iree_allocator_system(), &candidate));
  EXPECT_EQ(candidate.backend, nullptr);
  EXPECT_EQ(report.artifact_kind,
            LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE);
  EXPECT_EQ(report.status_code, IREE_STATUS_INVALID_ARGUMENT);

  loom_run_module_deinitialize(&run_module);
}

}  // namespace
}  // namespace loom
