// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/candidate.h"

#include <algorithm>

#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/bytecode_module_def_reader.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/bytecode/archive.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/ireevm/low_registry.h"
#include "loom/target/arch/ireevm/ops/registry.h"

namespace loom {
namespace {

iree_status_t RegisterContext(void* user_data, loom_context_t* context) {
  (void)user_data;
  IREE_RETURN_IF_ERROR(loom_op_registry_register_all_dialects(context));
  return loom_ireevm_ops_register_dialect(context);
}

iree_status_t InitializeLowDescriptorRegistry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  (void)user_data;
  loom_ireevm_low_descriptor_registry_initialize(out_registry);
  return iree_ok_status();
}

constexpr char kPreparedVmSource[] =
    "ireevm.target<core> @vm_target\n"
    "\n"
    "low.func.def target(@vm_target) abi(vm_module_function) "
    "@double(%value: reg<ireevm.i32>) -> (reg<ireevm.i32>) {\n"
    "  %sum = low.op<ireevm.add.i32>(%value, %value) : "
    "(reg<ireevm.i32>, reg<ireevm.i32>) -> reg<ireevm.i32>\n"
    "  low.return %sum : reg<ireevm.i32>\n"
    "}\n"
    "\n"
    "low.func.def target(@vm_target) abi(vm_module_function) "
    "export(\"branchy\") "
    "@branchy(%lhs: reg<ireevm.i32>, %rhs: reg<ireevm.i32>) -> "
    "(reg<ireevm.i32>) {\n"
    "  %c0 = low.const<ireevm.const.i32> {i32_value = 0} : reg<ireevm.i32>\n"
    "  %is_zero = low.op<ireevm.cmp.eq.i32>(%lhs, %c0) : "
    "(reg<ireevm.i32>, reg<ireevm.i32>) -> reg<ireevm.i32>\n"
    "  low.cond_br %is_zero, ^then, ^else : reg<ireevm.i32>\n"
    "^then:\n"
    "  %sum = low.func.call @double(%rhs) : "
    "(reg<ireevm.i32>) -> (reg<ireevm.i32>)\n"
    "  low.return %sum : reg<ireevm.i32>\n"
    "^else:\n"
    "  %diff = low.op<ireevm.sub.i32>(%lhs, %rhs) : "
    "(reg<ireevm.i32>, reg<ireevm.i32>) -> reg<ireevm.i32>\n"
    "  low.return %diff : reg<ireevm.i32>\n"
    "}\n";

constexpr char kPreparedVmImportSource[] =
    "ireevm.target<core> @vm_target\n"
    "\n"
    "low.func.decl import(vm, \"hal.buffer.length\") target(@vm_target) "
    "abi(vm_module_function) "
    "@hal_buffer_length(%value: reg<ireevm.i32>) -> (reg<ireevm.i32>)\n"
    "\n"
    "low.func.def target(@vm_target) abi(vm_module_function) "
    "export(\"length_identity\") "
    "@length_identity(%value: reg<ireevm.i32>) -> (reg<ireevm.i32>) {\n"
    "  %length = low.func.call @hal_buffer_length(%value) : "
    "(reg<ireevm.i32>) -> (reg<ireevm.i32>)\n"
    "  low.return %length : reg<ireevm.i32>\n"
    "}\n";

constexpr char kPreparedVmWideSource[] =
    "ireevm.target<core> @vm_target\n"
    "\n"
    "low.func.def target(@vm_target) abi(vm_module_function) "
    "export(\"wide_numeric\") "
    "@wide_numeric(%lhs: reg<ireevm.i64 x2>, %rhs: reg<ireevm.i64 x2>, "
    "%x: reg<ireevm.f32>, %y: reg<ireevm.f32>, "
    "%z: reg<ireevm.f64 x2>) -> "
    "(reg<ireevm.i64 x2>, reg<ireevm.f32>, reg<ireevm.f64 x2>, "
    "reg<ireevm.i32>) {\n"
    "  %sum = low.op<ireevm.add.i64>(%lhs, %rhs) : "
    "(reg<ireevm.i64 x2>, reg<ireevm.i64 x2>) -> reg<ireevm.i64 x2>\n"
    "  %product = low.op<ireevm.mul.f32>(%x, %y) : "
    "(reg<ireevm.f32>, reg<ireevm.f32>) -> reg<ireevm.f32>\n"
    "  %one_point_five = low.const<ireevm.const.f64> "
    "{f64_bits = 4609434218613702656} : reg<ireevm.f64 x2>\n"
    "  %wide_sum = low.op<ireevm.add.f64>(%z, %one_point_five) : "
    "(reg<ireevm.f64 x2>, reg<ireevm.f64 x2>) -> reg<ireevm.f64 x2>\n"
    "  %less = low.op<ireevm.cmp.lt.o.f64>(%z, %wide_sum) : "
    "(reg<ireevm.f64 x2>, reg<ireevm.f64 x2>) -> reg<ireevm.i32>\n"
    "  low.return %sum, %product, %wide_sum, %less : "
    "reg<ireevm.i64 x2>, reg<ireevm.f32>, reg<ireevm.f64 x2>, "
    "reg<ireevm.i32>\n"
    "}\n";

constexpr char kPreparedVmRefSource[] =
    "ireevm.target<core> @vm_target\n"
    "\n"
    "low.func.def target(@vm_target) abi(vm_module_function) "
    "export(\"retain_release\") "
    "@retain_release(%value: reg<ireevm.ref>) -> (reg<ireevm.ref>) {\n"
    "  %owned = low.op<ireevm.ref.retain>(%value) : "
    "(reg<ireevm.ref>) -> reg<ireevm.ref>\n"
    "  low.op<ireevm.ref.release>(%owned) : (reg<ireevm.ref>)\n"
    "  low.return %value : reg<ireevm.ref>\n"
    "}\n";

constexpr char kPreparedVmRefBranchSource[] =
    "ireevm.target<core> @vm_target\n"
    "\n"
    "low.func.def target(@vm_target) abi(vm_module_function) "
    "export(\"ref_branch\") "
    "@ref_branch(%value: reg<ireevm.ref>) {\n"
    "  low.br ^join(%value: reg<ireevm.ref>)\n"
    "^join(%arg: reg<ireevm.ref>):\n"
    "  low.op<ireevm.ref.release>(%arg) : (reg<ireevm.ref>)\n"
    "  low.return\n"
    "}\n";

bool FlatbufferStringEquals(flatbuffers_string_t value,
                            iree_string_view_t expected) {
  return iree_string_view_equal(
      iree_make_string_view(value, flatbuffers_string_len(value)), expected);
}

uint16_t ReadU16LE(const uint8_t* data) {
  return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

void ExpectSingleBranchRemapCount(
    iree_vm_BytecodeModuleDef_table_t module_def,
    iree_vm_FunctionDescriptor_struct_t function_descriptor,
    uint16_t expected_remap_count) {
  constexpr uint8_t kBranchOp = 0x56;
  flatbuffers_uint8_vec_t bytecode_data =
      iree_vm_BytecodeModuleDef_bytecode_data(module_def);
  ASSERT_NE(bytecode_data, nullptr);
  const size_t bytecode_data_length = flatbuffers_uint8_vec_len(bytecode_data);
  const size_t bytecode_offset = function_descriptor->bytecode_offset;
  const size_t bytecode_length = function_descriptor->bytecode_length;
  ASSERT_LE(bytecode_offset, bytecode_data_length);
  ASSERT_LE(bytecode_length, bytecode_data_length - bytecode_offset);
  const uint8_t* function_bytecode = bytecode_data + bytecode_offset;
  const uint8_t* function_end = function_bytecode + bytecode_length;
  const uint8_t* branch_op =
      std::find(function_bytecode, function_end, kBranchOp);
  ASSERT_NE(branch_op, function_end);

  size_t remap_count_offset = (size_t)(branch_op - function_bytecode) + 1 + 4;
  if ((remap_count_offset & 1u) != 0) {
    ++remap_count_offset;
  }
  ASSERT_LE(remap_count_offset + sizeof(uint16_t), bytecode_length);
  EXPECT_EQ(ReadU16LE(function_bytecode + remap_count_offset),
            expected_remap_count);
}

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

  void InitializeCandidateOptions(
      loom_run_module_t* run_module,
      loom_run_candidate_compile_options_t* out_options) {
    loom_run_candidate_compile_options_initialize(out_options);
    out_options->source_resolver = loom_run_module_source_resolver(run_module);
  }

  void ParseArchive(const loom_ireevm_module_archive_t* archive,
                    iree_vm_BytecodeModuleDef_table_t* out_module_def) {
    iree_const_byte_span_t flatbuffer = iree_const_byte_span_empty();
    IREE_ASSERT_OK(iree_vm_bytecode_archive_parse_header(
        iree_make_const_byte_span(archive->data, archive->data_length),
        &flatbuffer, nullptr));
    *out_module_def = iree_vm_BytecodeModuleDef_as_root(flatbuffer.data);
    ASSERT_NE(*out_module_def, nullptr);
  }

  loom_run_session_t session_ = {};
};

TEST_F(IreeVmCandidateTest, EmitVmArchiveCandidate) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kPreparedVmSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCandidateOptions(&run_module, &options);
  loom_target_compile_report_pressure_row_t pressure_rows[4] = {};
  options.report_row_storage = {
      .pressure_rows = pressure_rows,
      .pressure_row_capacity = IREE_ARRAYSIZE(pressure_rows),
  };
  loom_target_compile_report_t report = {};
  options.report = &report;

  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_ireevm_run_candidate_emit(
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
  EXPECT_EQ(candidate.compile_report.source_low_selected_op_count, 0u);
  EXPECT_EQ(candidate.compile_report.source_low_emitted_op_count, 0u);
  EXPECT_EQ(candidate.compile_report.source_low_row_total_count, 0u);
  EXPECT_EQ(candidate.compile_report.source_low_row_count, 0u);
  EXPECT_EQ(candidate.compile_report.artifact_size,
            candidate.archive.data_length);
  EXPECT_EQ(report.artifact_kind, candidate.compile_report.artifact_kind);
  EXPECT_EQ(report.artifact_size, candidate.compile_report.artifact_size);
  EXPECT_EQ(report.pressure_row_total_count,
            candidate.compile_report.pressure_row_total_count);
  EXPECT_EQ(report.source_low_row_total_count, 0u);

  iree_vm_BytecodeModuleDef_table_t module_def = nullptr;
  ParseArchive(&candidate.archive, &module_def);
  iree_vm_FunctionDescriptor_vec_t function_descriptors =
      iree_vm_BytecodeModuleDef_function_descriptors(module_def);
  EXPECT_EQ(iree_vm_FunctionDescriptor_vec_len(function_descriptors), 2u);
  iree_vm_ImportFunctionDef_vec_t imported_functions =
      iree_vm_BytecodeModuleDef_imported_functions(module_def);
  EXPECT_EQ(iree_vm_ImportFunctionDef_vec_len(imported_functions), 0u);
  iree_vm_ExportFunctionDef_vec_t exported_functions =
      iree_vm_BytecodeModuleDef_exported_functions(module_def);
  ASSERT_EQ(iree_vm_ExportFunctionDef_vec_len(exported_functions), 1u);
  iree_vm_ExportFunctionDef_table_t export_def =
      iree_vm_ExportFunctionDef_vec_at(exported_functions, 0);
  EXPECT_TRUE(FlatbufferStringEquals(
      iree_vm_ExportFunctionDef_local_name(export_def), IREE_SV("branchy")));
  EXPECT_EQ(iree_vm_ExportFunctionDef_internal_ordinal(export_def), 1);

  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(IreeVmCandidateTest, EmitVmArchiveCandidateWithoutReport) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kPreparedVmSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCandidateOptions(&run_module, &options);

  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_ireevm_run_candidate_emit(
      &run_module, &options, iree_allocator_system(), &candidate));
  EXPECT_GT(candidate.archive.data_length, 0u);
  EXPECT_EQ(candidate.compile_report.detail_flags,
            LOOM_TARGET_COMPILE_REPORT_DETAIL_NONE);
  EXPECT_EQ(candidate.compile_report.artifact_size, 0u);
  EXPECT_EQ(candidate.compile_report.pressure_rows, nullptr);
  EXPECT_EQ(candidate.compile_report.source_low_rows, nullptr);

  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(IreeVmCandidateTest, EmitVmArchiveCandidateWithImport) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kPreparedVmImportSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCandidateOptions(&run_module, &options);

  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_ireevm_run_candidate_emit(
      &run_module, &options, iree_allocator_system(), &candidate));
  EXPECT_GT(candidate.archive.data_length, 0u);

  iree_vm_BytecodeModuleDef_table_t module_def = nullptr;
  ParseArchive(&candidate.archive, &module_def);
  iree_vm_FunctionDescriptor_vec_t function_descriptors =
      iree_vm_BytecodeModuleDef_function_descriptors(module_def);
  EXPECT_EQ(iree_vm_FunctionDescriptor_vec_len(function_descriptors), 1u);
  iree_vm_ImportFunctionDef_vec_t imported_functions =
      iree_vm_BytecodeModuleDef_imported_functions(module_def);
  ASSERT_EQ(iree_vm_ImportFunctionDef_vec_len(imported_functions), 1u);
  iree_vm_ImportFunctionDef_table_t import_def =
      iree_vm_ImportFunctionDef_vec_at(imported_functions, 0);
  EXPECT_TRUE(
      FlatbufferStringEquals(iree_vm_ImportFunctionDef_full_name(import_def),
                             IREE_SV("hal.buffer.length")));
  iree_vm_ExportFunctionDef_vec_t exported_functions =
      iree_vm_BytecodeModuleDef_exported_functions(module_def);
  ASSERT_EQ(iree_vm_ExportFunctionDef_vec_len(exported_functions), 1u);
  iree_vm_ExportFunctionDef_table_t export_def =
      iree_vm_ExportFunctionDef_vec_at(exported_functions, 0);
  EXPECT_TRUE(
      FlatbufferStringEquals(iree_vm_ExportFunctionDef_local_name(export_def),
                             IREE_SV("length_identity")));

  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(IreeVmCandidateTest, EmitVmArchiveCandidateWithWideScalars) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kPreparedVmWideSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCandidateOptions(&run_module, &options);

  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_ireevm_run_candidate_emit(
      &run_module, &options, iree_allocator_system(), &candidate));
  EXPECT_GT(candidate.archive.data_length, 0u);

  iree_vm_BytecodeModuleDef_table_t module_def = nullptr;
  ParseArchive(&candidate.archive, &module_def);
  constexpr iree_vm_FeatureBits_enum_t kFeatureRequirements =
      (iree_vm_FeatureBits_enum_t)(iree_vm_FeatureBits_EXT_F32 |
                                   iree_vm_FeatureBits_EXT_F64);
  EXPECT_EQ(iree_vm_BytecodeModuleDef_requirements(module_def),
            kFeatureRequirements);
  iree_vm_FunctionDescriptor_vec_t function_descriptors =
      iree_vm_BytecodeModuleDef_function_descriptors(module_def);
  ASSERT_EQ(iree_vm_FunctionDescriptor_vec_len(function_descriptors), 1u);
  iree_vm_FunctionDescriptor_struct_t function_descriptor =
      iree_vm_FunctionDescriptor_vec_at(function_descriptors, 0);
  EXPECT_EQ(function_descriptor->requirements, kFeatureRequirements);
  iree_vm_FunctionSignatureDef_vec_t function_signatures =
      iree_vm_BytecodeModuleDef_function_signatures(module_def);
  ASSERT_EQ(iree_vm_FunctionSignatureDef_vec_len(function_signatures), 1u);
  iree_vm_FunctionSignatureDef_table_t signature_def =
      iree_vm_FunctionSignatureDef_vec_at(function_signatures, 0);
  EXPECT_TRUE(FlatbufferStringEquals(
      iree_vm_FunctionSignatureDef_calling_convention(signature_def),
      IREE_SV("0IIffF_IfFi")));

  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(IreeVmCandidateTest, EmitVmArchiveCandidateWithRefs) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kPreparedVmRefSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCandidateOptions(&run_module, &options);

  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_ireevm_run_candidate_emit(
      &run_module, &options, iree_allocator_system(), &candidate));
  EXPECT_GT(candidate.archive.data_length, 0u);

  iree_vm_BytecodeModuleDef_table_t module_def = nullptr;
  ParseArchive(&candidate.archive, &module_def);
  iree_vm_FunctionDescriptor_vec_t function_descriptors =
      iree_vm_BytecodeModuleDef_function_descriptors(module_def);
  ASSERT_EQ(iree_vm_FunctionDescriptor_vec_len(function_descriptors), 1u);
  iree_vm_FunctionDescriptor_struct_t function_descriptor =
      iree_vm_FunctionDescriptor_vec_at(function_descriptors, 0);
  EXPECT_GT(function_descriptor->bytecode_length, 0);
  EXPECT_EQ(function_descriptor->i32_register_count, 0);
  EXPECT_EQ(function_descriptor->ref_register_count, 2);
  iree_vm_FunctionSignatureDef_vec_t function_signatures =
      iree_vm_BytecodeModuleDef_function_signatures(module_def);
  ASSERT_EQ(iree_vm_FunctionSignatureDef_vec_len(function_signatures), 1u);
  iree_vm_FunctionSignatureDef_table_t signature_def =
      iree_vm_FunctionSignatureDef_vec_at(function_signatures, 0);
  EXPECT_TRUE(FlatbufferStringEquals(
      iree_vm_FunctionSignatureDef_calling_convention(signature_def),
      IREE_SV("0r_r")));

  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(IreeVmCandidateTest, EmitVmArchiveCandidateWithRefBranch) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kPreparedVmRefBranchSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCandidateOptions(&run_module, &options);

  loom_ireevm_run_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_ireevm_run_candidate_emit(
      &run_module, &options, iree_allocator_system(), &candidate));
  EXPECT_GT(candidate.archive.data_length, 0u);

  iree_vm_BytecodeModuleDef_table_t module_def = nullptr;
  ParseArchive(&candidate.archive, &module_def);
  iree_vm_FunctionDescriptor_vec_t function_descriptors =
      iree_vm_BytecodeModuleDef_function_descriptors(module_def);
  ASSERT_EQ(iree_vm_FunctionDescriptor_vec_len(function_descriptors), 1u);
  iree_vm_FunctionDescriptor_struct_t function_descriptor =
      iree_vm_FunctionDescriptor_vec_at(function_descriptors, 0);
  EXPECT_GT(function_descriptor->bytecode_length, 0);
  EXPECT_EQ(function_descriptor->i32_register_count, 0);
  EXPECT_EQ(function_descriptor->ref_register_count, 1);
  ExpectSingleBranchRemapCount(module_def, function_descriptor,
                               /*expected_remap_count=*/0);

  loom_ireevm_run_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

}  // namespace
}  // namespace loom
