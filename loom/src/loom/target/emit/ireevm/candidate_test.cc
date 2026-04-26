// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/candidate.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/target/emit/ireevm/low_registry.h"

namespace loom {
namespace {

using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t*);
using DialectSemanticsFn = const loom_op_semantics_t* (*)(iree_host_size_t*);

iree_status_t RegisterDialect(loom_context_t* context, uint8_t dialect_id,
                              DialectVtablesFn dialect_vtables_fn,
                              DialectSemanticsFn dialect_semantics_fn) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
  iree_host_size_t semantics_count = 0;
  const loom_op_semantics_t* semantics = dialect_semantics_fn(&semantics_count);
  if (semantics_count != count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "dialect %u semantics count %" PRIhsz
                            " does not match vtable count %" PRIhsz,
                            (unsigned)dialect_id, semantics_count, count);
  }
  IREE_RETURN_IF_ERROR(loom_context_register_dialect(context, dialect_id,
                                                     vtables, (uint16_t)count));
  return loom_context_register_dialect_semantics(context, dialect_id, semantics,
                                                 (uint16_t)count);
}

iree_status_t RegisterContext(void* user_data, loom_context_t* context) {
  (void)user_data;
  IREE_RETURN_IF_ERROR(RegisterDialect(context, LOOM_DIALECT_TARGET,
                                       loom_target_dialect_vtables,
                                       loom_target_dialect_op_semantics));
  IREE_RETURN_IF_ERROR(RegisterDialect(context, LOOM_DIALECT_FUNC,
                                       loom_func_dialect_vtables,
                                       loom_func_dialect_op_semantics));
  IREE_RETURN_IF_ERROR(RegisterDialect(context, LOOM_DIALECT_SCALAR,
                                       loom_scalar_dialect_vtables,
                                       loom_scalar_dialect_op_semantics));
  IREE_RETURN_IF_ERROR(RegisterDialect(context, LOOM_DIALECT_CFG,
                                       loom_cfg_dialect_vtables,
                                       loom_cfg_dialect_op_semantics));
  return RegisterDialect(context, LOOM_DIALECT_LOW, loom_low_dialect_vtables,
                         loom_low_dialect_op_semantics);
}

iree_status_t InitializeLowDescriptorRegistry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  (void)user_data;
  loom_ireevm_low_descriptor_registry_initialize(out_registry);
  return iree_ok_status();
}

constexpr char kVmSource[] =
    "target.profile @vm_target preset(\"iree-vm\")\n"
    "\n"
    "func.def target(@vm_target) @branchy(%lhs: i32, %rhs: i32) -> (i32) {\n"
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

class IreeVmCandidateTest : public ::testing::Test {
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
    options.filename = IREE_SV("ireevm_candidate_test.loom");
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

TEST_F(IreeVmCandidateTest, CompileVmArchiveCandidate) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kVmSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCompileOptions(&run_module, &options);
  loom_target_compile_report_pressure_row_t pressure_rows[4] = {};
  loom_target_compile_report_source_low_row_t source_low_rows[4] = {};
  options.report_row_storage = {
      .pressure_rows = pressure_rows,
      .pressure_row_capacity = IREE_ARRAYSIZE(pressure_rows),
      .source_low_rows = source_low_rows,
      .source_low_row_capacity = IREE_ARRAYSIZE(source_low_rows),
  };
  loom_target_compile_report_t report = {};
  options.report = &report;

  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_ireevm_run_candidate_compile(
      &run_module, &options, iree_allocator_system(), &candidate));
  EXPECT_GT(candidate.archive.data_length, 0u);
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
  EXPECT_TRUE(
      iree_all_bits_set(candidate.compile_report.detail_flags,
                        LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS));
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
  EXPECT_EQ(candidate.compile_report.source_low_rows, source_low_rows);
  EXPECT_GT(candidate.compile_report.source_low_selected_op_count, 0u);
  EXPECT_GT(candidate.compile_report.source_low_emitted_op_count, 0u);
  EXPECT_GT(candidate.compile_report.source_low_row_total_count, 0u);
  EXPECT_GT(candidate.compile_report.source_low_row_count, 0u);
  EXPECT_GT(candidate.compile_report.source_low_rows[0].emitted_low_op_count,
            0u);
  EXPECT_EQ(candidate.compile_report.artifact_size,
            candidate.archive.data_length);
  EXPECT_EQ(report.artifact_kind, candidate.compile_report.artifact_kind);
  EXPECT_EQ(report.artifact_size, candidate.compile_report.artifact_size);
  EXPECT_EQ(report.pressure_row_total_count,
            candidate.compile_report.pressure_row_total_count);
  EXPECT_EQ(report.source_low_row_total_count,
            candidate.compile_report.source_low_row_total_count);

  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

}  // namespace
}  // namespace loom
