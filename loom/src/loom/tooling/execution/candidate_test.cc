// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/candidate.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ops/op_registry.h"
#include "loom/target/emit/ireevm/low_registry.h"
#include "loom/tooling/execution/hal_runtime.h"

namespace loom {
namespace {

iree_status_t RegisterContext(void* user_data, loom_context_t* context) {
  (void)user_data;
  return loom_op_registry_register_all_dialects(context);
}

iree_status_t InitializeLowDescriptorRegistry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  (void)user_data;
  loom_ireevm_low_descriptor_registry_initialize(out_registry);
  return iree_ok_status();
}

constexpr char kVmSource[] =
    "target.preset @vm_target {key = \"iree-vm\", source = @branchy}\n"
    "\n"
    "func.def @branchy(%lhs: i32, %rhs: i32) -> (i32) {\n"
    "  %c0 = scalar.constant 0 : i32\n"
    "  %is_zero = scalar.cmpi eq, %lhs, %c0 : i32\n"
    "  cfg.cond_br %is_zero, ^then, ^else : i1\n"
    "^then:\n"
    "  %sum = scalar.addi %rhs, %rhs : i32\n"
    "  func.return %sum : i32\n"
    "^else:\n"
    "  %diff = scalar.subi %lhs, %rhs : i32\n"
    "  func.return %diff : i32\n"
    "}\n";

constexpr char kHalSource[] =
    "func.def @empty() {\n"
    "  func.return\n"
    "}\n";

int kFakeHalTarget = 0;
const uint8_t kFakeHalExecutableData[] = {0x7F, 'E', 'L', 'F'};

iree_status_t FakeHalSelectTarget(const loom_run_hal_backend_t* backend,
                                  const loom_run_hal_runtime_t* runtime,
                                  iree_allocator_t allocator,
                                  loom_run_hal_selected_target_t* out_target) {
  (void)backend;
  (void)runtime;
  (void)allocator;
  *out_target = (loom_run_hal_selected_target_t){
      .data = &kFakeHalTarget,
      .preset_key = IREE_SVL("fake-hal"),
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

iree_status_t FakeHalCompile(
    const loom_run_hal_backend_t* backend, loom_module_t* module,
    const loom_run_hal_selected_target_t* target,
    iree_string_view_t target_symbol, loom_diagnostic_sink_t diagnostic_sink,
    loom_source_resolver_t source_resolver, uint32_t max_errors,
    loom_target_compile_report_t* report, iree_allocator_t allocator,
    loom_run_hal_executable_t* out_executable) {
  (void)backend;
  (void)module;
  (void)target_symbol;
  (void)diagnostic_sink;
  (void)source_resolver;
  (void)max_errors;
  (void)report;
  (void)allocator;
  if (target->data != &kFakeHalTarget) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected fake HAL target payload");
  }
  *out_executable = (loom_run_hal_executable_t){
      .executable_format = IREE_SVL("fake-hal-format"),
      .executable_data = iree_make_const_byte_span(
          kFakeHalExecutableData, sizeof(kFakeHalExecutableData)),
      .storage = &kFakeHalTarget,
  };
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
    .compile = FakeHalCompile,
    .deinitialize_executable = FakeHalDeinitializeExecutable,
};

class CandidateTest : public ::testing::Test {
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
    options.filename = IREE_SV("candidate_test.loom");
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

TEST_F(CandidateTest, CompileVmArchiveCandidate) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kVmSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCompileOptions(&run_module, &options);
  loom_target_compile_report_pressure_row_t pressure_rows[4] = {};
  options.report_row_storage = {
      .pressure_rows = pressure_rows,
      .pressure_row_capacity = IREE_ARRAYSIZE(pressure_rows),
  };
  loom_target_compile_report_t report = {};
  options.report = &report;

  loom_run_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_run_candidate_compile_vm(
      &run_module, &options, iree_allocator_system(), &candidate));
  EXPECT_EQ(candidate.kind, LOOM_RUN_CANDIDATE_KIND_VM_ARCHIVE);
  EXPECT_GT(candidate.vm_archive.data_length, 0u);
  EXPECT_EQ(candidate.compile_report.artifact_kind,
            LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE);
  EXPECT_EQ(candidate.compile_report.status_code, IREE_STATUS_OK);
  EXPECT_TRUE(
      iree_all_bits_set(candidate.compile_report.detail_flags,
                        LOOM_TARGET_COMPILE_REPORT_DETAIL_ARTIFACT_SIZE));
  EXPECT_TRUE(iree_all_bits_set(candidate.compile_report.detail_flags,
                                LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE));
  EXPECT_TRUE(iree_all_bits_set(candidate.compile_report.detail_flags,
                                LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION));
  EXPECT_TRUE(iree_all_bits_set(candidate.compile_report.detail_flags,
                                LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION));
  EXPECT_TRUE(
      iree_all_bits_set(candidate.compile_report.detail_flags,
                        LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS));
  EXPECT_FALSE(
      iree_string_view_is_empty(candidate.compile_report.target_bundle_name));
  EXPECT_FALSE(
      iree_string_view_is_empty(candidate.compile_report.lowered_symbol));
  EXPECT_GT(candidate.compile_report.schedule_node_count, 0u);
  EXPECT_GT(candidate.compile_report.scheduled_node_count, 0u);
  EXPECT_GT(candidate.compile_report.allocation_assignment_count, 0u);
  EXPECT_GT(candidate.compile_report.emitted_instruction_count, 0u);
  EXPECT_GT(candidate.compile_report.emitted_code_byte_count, 0u);
  EXPECT_GE(candidate.compile_report.emitted_code_storage_byte_count,
            candidate.compile_report.emitted_code_byte_count);
  EXPECT_EQ(candidate.compile_report.pressure_rows, pressure_rows);
  EXPECT_GT(candidate.compile_report.pressure_row_total_count, 0u);
  EXPECT_GT(candidate.compile_report.pressure_row_count, 0u);
  EXPECT_GT(candidate.compile_report.pressure_rows[0].peak_live_units, 0u);
  EXPECT_EQ(candidate.compile_report.artifact_size,
            candidate.vm_archive.data_length);
  EXPECT_EQ(report.artifact_kind, candidate.compile_report.artifact_kind);
  EXPECT_EQ(report.artifact_size, candidate.compile_report.artifact_size);
  EXPECT_EQ(report.pressure_row_total_count,
            candidate.compile_report.pressure_row_total_count);

  loom_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(CandidateTest, CompileHalExecutableCandidate) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kHalSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCompileOptions(&run_module, &options);
  loom_target_compile_report_t report = {};
  options.report = &report;

  loom_run_hal_runtime_t runtime = {};
  loom_run_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_run_candidate_compile_hal(
      &kFakeHalBackend, &runtime, &run_module, &options,
      iree_allocator_system(), &candidate));
  EXPECT_EQ(candidate.kind, LOOM_RUN_CANDIDATE_KIND_HAL_EXECUTABLE);
  EXPECT_EQ(candidate.hal_backend, &kFakeHalBackend);
  EXPECT_EQ(candidate.hal_target.data, &kFakeHalTarget);
  EXPECT_TRUE(iree_string_view_equal(candidate.hal_executable.executable_format,
                                     IREE_SV("fake-hal-format")));
  EXPECT_EQ(candidate.hal_executable.executable_data.data,
            kFakeHalExecutableData);
  EXPECT_EQ(candidate.hal_executable.executable_data.data_length,
            sizeof(kFakeHalExecutableData));
  EXPECT_EQ(candidate.compile_report.artifact_kind,
            LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE);
  EXPECT_EQ(candidate.compile_report.status_code, IREE_STATUS_OK);
  EXPECT_TRUE(iree_string_view_equal(candidate.compile_report.backend_name,
                                     IREE_SV("fake-hal")));
  EXPECT_TRUE(iree_string_view_equal(candidate.compile_report.target_preset_key,
                                     IREE_SV("fake-hal")));
  EXPECT_TRUE(iree_string_view_equal(candidate.compile_report.executable_format,
                                     IREE_SV("fake-hal-format")));
  EXPECT_EQ(candidate.compile_report.artifact_size,
            sizeof(kFakeHalExecutableData));
  EXPECT_EQ(report.artifact_size, candidate.compile_report.artifact_size);

  loom_run_candidate_deinitialize(&candidate);
  EXPECT_EQ(candidate.kind, LOOM_RUN_CANDIDATE_KIND_UNINITIALIZED);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(CandidateTest, CompileHalRequiresHooks) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kHalSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCompileOptions(&run_module, &options);
  loom_target_compile_report_t report = {};
  options.report = &report;

  const loom_run_hal_backend_t backend = {
      .name = IREE_SVL("missing-hooks"),
  };
  loom_run_hal_runtime_t runtime = {};
  loom_run_candidate_t candidate = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_candidate_compile_hal(&backend, &runtime, &run_module, &options,
                                     iree_allocator_system(), &candidate));
  EXPECT_EQ(candidate.kind, LOOM_RUN_CANDIDATE_KIND_UNINITIALIZED);
  EXPECT_EQ(report.artifact_kind,
            LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE);
  EXPECT_EQ(report.status_code, IREE_STATUS_INVALID_ARGUMENT);

  loom_run_module_deinitialize(&run_module);
}

}  // namespace
}  // namespace loom
