// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/verify/verify.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/target/ops.h"
#include "loom/testing/diagnostic_matchers.h"

namespace loom {
namespace {

using ::loom::testing::CapturedDiagnostic;
using ::loom::testing::DiagnosticCapture;
using ::loom::testing::ExpectError;
using ::loom::testing::ExpectI64Param;
using ::loom::testing::FindDiagnostic;
using ::loom::testing::GetStringParam;

class TargetVerifyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_op_registry_initialize_context(iree_allocator_system(),
                                                       &context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_verify_result_t VerifySource(const char* source,
                                    DiagnosticCapture* verify_capture) {
    const char* filename = "target_verify_test.loom";
    loom_module_t* module = ParseSource(source, filename);
    if (!module) return {};

    loom_source_entry_t source_entries[] = {{
        .source_id = FindContextSourceId(filename),
        .source = iree_make_cstring_view(source),
        .filename = iree_make_cstring_view(filename),
    }};
    EXPECT_NE(source_entries[0].source_id, LOOM_SOURCE_ID_INVALID);
    loom_source_table_resolver_t resolver_data = {
        .entries = source_entries,
        .count = IREE_ARRAYSIZE(source_entries),
    };

    verify_capture->Reset();
    loom_verify_options_t verify_options = {};
    verify_options.sink = verify_capture->sink();
    verify_options.max_errors = 20;
    verify_options.source_resolver = {loom_source_table_resolve,
                                      &resolver_data};
    loom_verify_result_t result = {};
    IREE_EXPECT_OK(loom_verify_module(module, &verify_options, &result));
    loom_module_free(module);
    return result;
  }

  loom_module_t* ParseSource(const char* source, const char* filename) {
    DiagnosticCapture parse_capture;
    loom_text_parse_options_t parse_options = {};
    parse_options.diagnostic_sink = parse_capture.sink();
    parse_options.max_errors = 20;

    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source),
                                   iree_make_cstring_view(filename), &context_,
                                   &block_pool_, &parse_options, &module));
    if (!parse_capture.diagnostics.empty()) {
      ADD_FAILURE() << "expected parser success, got "
                    << parse_capture.diagnostics.size() << " diagnostic(s)";
      if (module) loom_module_free(module);
      return nullptr;
    }
    EXPECT_NE(module, nullptr);
    return module;
  }

  loom_op_t* FindFirstMutableOp(loom_module_t* module, loom_op_kind_t kind) {
    loom_op_t* op = nullptr;
    loom_block_for_each_op(loom_module_block(module), op) {
      if (op->kind == kind) return op;
    }
    return nullptr;
  }

  loom_source_id_t FindContextSourceId(const char* filename) const {
    iree_string_view_t source_name = iree_make_cstring_view(filename);
    for (iree_host_size_t i = 0; i < context_.sources.count; ++i) {
      if (iree_string_view_equal(context_.sources.entries[i], source_name)) {
        return (loom_source_id_t)i;
      }
    }
    ADD_FAILURE() << "expected context source table to contain " << filename;
    return LOOM_SOURCE_ID_INVALID;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

static const char* kValidTargetRecords =
    "func.def @matmul() {\n"
    "  func.return\n"
    "}\n"
    "target.snapshot @x86_64 {codegen_format = llvmir, target_triple = "
    "\"x86_64-pc-linux-gnu\", data_layout = \"e-m:e-p:64:64\", "
    "artifact_format = elf, target_cpu = \"x86-64-v3\", target_features = "
    "\"+avx2,+fma\", default_pointer_bitwidth = 64, index_bitwidth = 64, "
    "offset_bitwidth = 64, memory_space_generic = 0, memory_space_global = 0, "
    "memory_space_workgroup = 4294967295, memory_space_constant = 0, "
    "memory_space_private = 0, memory_space_host = 0, "
    "memory_space_descriptor = 4294967295}\n"
    "target.export @matmul_cpu {source = @matmul, export_symbol = \"matmul\", "
    "abi = object_function, linkage = dso_local, hal_binding_alignment = 0, "
    "hal_workgroup_size_x = 0, hal_workgroup_size_y = 0, "
    "hal_workgroup_size_z = 0, hal_flat_workgroup_size_min = 0, "
    "hal_flat_workgroup_size_max = 0, hal_buffer_resource_flags = 0}\n"
    "target.config @generic {contract_set_key = \"default\", "
    "contract_feature_bits = 0}\n"
    "target.bundle @matmul_x86 {snapshot = @x86_64, export_plan = "
    "@matmul_cpu, config = @generic}\n"
    "target.snapshot @gfx1100 {codegen_format = llvmir, target_triple = "
    "\"amdgcn-amd-amdhsa\", data_layout = \"e-p:64:64\", artifact_format = "
    "elf, target_cpu = \"gfx1100\", target_features = \"+wavefrontsize32\", "
    "default_pointer_bitwidth = 64, index_bitwidth = 64, offset_bitwidth = 64, "
    "memory_space_generic = 0, memory_space_global = 1, "
    "memory_space_workgroup = 3, memory_space_constant = 4, "
    "memory_space_private = 5, memory_space_host = 4294967295, "
    "memory_space_descriptor = 4294967295}\n"
    "target.export @matmul_hal {source = @matmul, export_symbol = \"matmul\", "
    "abi = hal_kernel, linkage = default, hal_binding_alignment = 16, "
    "hal_workgroup_size_x = 16, hal_workgroup_size_y = 4, "
    "hal_workgroup_size_z = 1, hal_flat_workgroup_size_min = 64, "
    "hal_flat_workgroup_size_max = 64, hal_buffer_resource_flags = 0}\n"
    "target.config @gfx11_config {contract_set_key = \"amdgpu.gfx11\", "
    "contract_feature_bits = 1}\n"
    "target.bundle @matmul_gfx1100 {snapshot = @gfx1100, export_plan = "
    "@matmul_hal, config = @gfx11_config}\n"
    "target.snapshot @wasm32 {codegen_format = wasm, target_triple = "
    "\"wasm32-unknown-unknown\", data_layout = \"\", artifact_format = "
    "wasm_binary, target_cpu = \"\", target_features = \"+simd128\", "
    "default_pointer_bitwidth = 32, index_bitwidth = 32, offset_bitwidth = 32, "
    "memory_space_generic = 0, memory_space_global = 0, "
    "memory_space_workgroup = 4294967295, memory_space_constant = 0, "
    "memory_space_private = 4294967295, memory_space_host = 4294967295, "
    "memory_space_descriptor = 4294967295}\n"
    "target.export @wasm_export {source = @matmul, export_symbol = \"matmul\", "
    "abi = wasm_function, linkage = default, hal_binding_alignment = 0, "
    "hal_workgroup_size_x = 0, hal_workgroup_size_y = 0, "
    "hal_workgroup_size_z = 0, hal_flat_workgroup_size_min = 0, "
    "hal_flat_workgroup_size_max = 0, hal_buffer_resource_flags = 0}\n"
    "target.config @wasm_config {contract_set_key = \"wasm.core.simd128\", "
    "contract_feature_bits = 0}\n"
    "target.bundle @matmul_wasm32 {snapshot = @wasm32, export_plan = "
    "@wasm_export, config = @wasm_config}\n";

TEST_F(TargetVerifyTest, CpuAmdgpuAndWasmRecordsVerify) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(kValidTargetRecords, &capture);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(TargetVerifyTest, FutureTargetEnumOrdinalsVerifyAsOpenEnums) {
  loom_module_t* module =
      ParseSource(kValidTargetRecords, "target_verify_test.loom");
  ASSERT_NE(module, nullptr);
  loom_op_t* snapshot = FindFirstMutableOp(module, LOOM_OP_TARGET_SNAPSHOT);
  ASSERT_NE(snapshot, nullptr);
  loom_op_attrs(snapshot)[loom_target_snapshot_codegen_format_ATTR_INDEX] =
      loom_attr_enum(250);
  loom_op_attrs(snapshot)[loom_target_snapshot_artifact_format_ATTR_INDEX] =
      loom_attr_enum(251);
  loom_op_t* export_op = FindFirstMutableOp(module, LOOM_OP_TARGET_EXPORT);
  ASSERT_NE(export_op, nullptr);
  loom_op_attrs(export_op)[loom_target_export_abi_ATTR_INDEX] =
      loom_attr_enum(252);
  loom_op_attrs(export_op)[loom_target_export_linkage_ATTR_INDEX] =
      loom_attr_enum(253);

  DiagnosticCapture capture;
  loom_verify_options_t verify_options = {};
  verify_options.sink = capture.sink();
  verify_options.max_errors = 20;
  loom_verify_result_t result = {};
  IREE_ASSERT_OK(loom_verify_module(module, &verify_options, &result));

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
  loom_module_free(module);
}

TEST_F(TargetVerifyTest, SnapshotRejectsInvalidBitwidth) {
  DiagnosticCapture capture;
  VerifySource(
      "target.snapshot @bad {codegen_format = llvmir, target_triple = \"\", "
      "data_layout = \"\", artifact_format = elf, target_cpu = \"\", "
      "target_features = \"\", default_pointer_bitwidth = 0, "
      "index_bitwidth = 64, offset_bitwidth = 64, memory_space_generic = 0, "
      "memory_space_global = 0, memory_space_workgroup = 0, "
      "memory_space_constant = 0, memory_space_private = 0, "
      "memory_space_host = 0, memory_space_descriptor = 0}\n",
      &capture);

  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 14));
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic,
              loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 14),
              LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "default_pointer_bitwidth");
  ExpectI64Param(*diagnostic, 1, 0);
}

TEST_F(TargetVerifyTest, ObjectExportRejectsIgnoredHalFields) {
  DiagnosticCapture capture;
  VerifySource(
      "func.def @matmul() {\n"
      "  func.return\n"
      "}\n"
      "target.export @bad {source = @matmul, export_symbol = \"matmul\", "
      "abi = object_function, linkage = default, hal_binding_alignment = 16, "
      "hal_workgroup_size_x = 0, hal_workgroup_size_y = 0, "
      "hal_workgroup_size_z = 0, hal_flat_workgroup_size_min = 0, "
      "hal_flat_workgroup_size_max = 0, hal_buffer_resource_flags = 0}\n",
      &capture);

  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 14));
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "hal_binding_alignment");
  ExpectI64Param(*diagnostic, 1, 16);
}

TEST_F(TargetVerifyTest, BundleRejectsWrongRecordClass) {
  DiagnosticCapture capture;
  VerifySource(
      "target.config @not_snapshot {contract_set_key = \"default\", "
      "contract_feature_bits = 0}\n"
      "target.config @config {contract_set_key = \"default\", "
      "contract_feature_bits = 0}\n"
      "target.export @export {export_symbol = \"\", abi = object_function, "
      "linkage = default, hal_binding_alignment = 0, hal_workgroup_size_x = 0, "
      "hal_workgroup_size_y = 0, hal_workgroup_size_z = 0, "
      "hal_flat_workgroup_size_min = 0, hal_flat_workgroup_size_max = 0, "
      "hal_buffer_resource_flags = 0}\n"
      "target.bundle @bundle {snapshot = @not_snapshot, export_plan = @export, "
      "config = @config}\n",
      &capture);

  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3));
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3),
              LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "not_snapshot");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "target config");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "target snapshot");
  ASSERT_EQ(diagnostic->related_locations.size(), 1u);
  EXPECT_EQ(diagnostic->related_locations[0].label, "defined here");
  EXPECT_TRUE(diagnostic->related_locations[0].has_source_range);
}

TEST_F(TargetVerifyTest, RejectsDuplicateTargetRecordDefinition) {
  DiagnosticCapture capture;
  VerifySource(
      "target.config @duplicate {contract_set_key = \"default\", "
      "contract_feature_bits = 0}\n"
      "target.config @duplicate {contract_set_key = \"other\", "
      "contract_feature_bits = 1}\n",
      &capture);

  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 5));
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "duplicate");
  ExpectError(*diagnostic, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 5),
              LOOM_EMITTER_VERIFIER);
  ASSERT_EQ(diagnostic->related_locations.size(), 1u);
  EXPECT_EQ(diagnostic->related_locations[0].label, "first definition here");
  EXPECT_TRUE(diagnostic->related_locations[0].has_source_range);
}

}  // namespace
}  // namespace loom
