// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/verify.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/target/emit/ireevm/descriptors.h"

namespace loom {
namespace {

struct CollectedEmission {
  const loom_error_def_t* error = nullptr;
  const loom_op_t* op = nullptr;
  std::vector<std::string> string_params;
  std::vector<loom_type_t> type_params;
  std::vector<int64_t> i64_params;
  std::vector<uint32_t> u32_params;
  std::vector<uint64_t> u64_params;
  std::vector<loom_diagnostic_field_ref_t> field_refs;
};

struct EmissionCollector {
  std::vector<CollectedEmission> emissions;

  iree_diagnostic_emitter_t emitter() {
    return iree_diagnostic_emitter_t{
        .fn = Collect,
        .user_data = this,
    };
  }

 private:
  static std::string CopyString(iree_string_view_t value) {
    return std::string(value.data, value.size);
  }

  static iree_status_t Collect(void* user_data,
                               const loom_diagnostic_emission_t* emission) {
    auto* collector = static_cast<EmissionCollector*>(user_data);
    CollectedEmission entry;
    entry.error = emission->error;
    entry.op = emission->op;
    for (iree_host_size_t i = 0; i < emission->param_count; ++i) {
      const loom_diagnostic_param_t* param = &emission->params[i];
      entry.field_refs.push_back(param->field_ref);
      if (param->kind == LOOM_PARAM_STRING) {
        entry.string_params.push_back(CopyString(param->string));
      } else if (param->kind == LOOM_PARAM_I64) {
        entry.i64_params.push_back(param->i64);
      } else if (param->kind == LOOM_PARAM_TYPE) {
        entry.type_params.push_back(param->type);
      } else if (param->kind == LOOM_PARAM_U32) {
        entry.u32_params.push_back(param->u32);
      } else if (param->kind == LOOM_PARAM_U64) {
        entry.u64_params.push_back(param->u64);
      }
    }
    collector->emissions.push_back(std::move(entry));
    return iree_ok_status();
  }
};

class LowDescriptorVerifyTest : public ::testing::Test {
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

  loom_module_t* ParseSource(const std::string& source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;

    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("low_descriptor_verify_test.loom"), &context_,
                        &block_pool_, &parse_options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  loom_low_verify_result_t VerifyModule(
      loom_module_t* module, const loom_low_descriptor_registry_t* registry,
      EmissionCollector* collector) {
    loom_low_verify_options_t options = {
        .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
        .descriptor_registry = registry,
        .emitter = collector->emitter(),
        .max_errors = 20,
    };
    loom_low_verify_result_t result = {};
    IREE_EXPECT_OK(loom_low_verify_module(module, &options, &result));
    return result;
  }

  static loom_low_descriptor_registry_t IreeVmRegistry() {
    static const loom_low_descriptor_set_t* descriptor_sets[] = {
        loom_ireevm_core_descriptor_set(),
    };
    return loom_low_descriptor_registry_t{
        .descriptor_sets = descriptor_sets,
        .descriptor_set_count = IREE_ARRAYSIZE(descriptor_sets),
    };
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

static const char kVmTargetRecords[] =
    "target.snapshot @vm_snapshot {codegen_format = vm, target_triple = "
    "\"iree-vm\", data_layout = \"\", artifact_format = vm_bytecode, "
    "target_cpu = \"\", target_features = \"\", default_pointer_bitwidth = "
    "64, index_bitwidth = 64, offset_bitwidth = 64, memory_space_generic = 0, "
    "memory_space_global = 0, memory_space_workgroup = 0, "
    "memory_space_constant = 0, memory_space_private = 0, memory_space_host = "
    "0, memory_space_descriptor = 0}\n"
    "target.export @vm_export {export_symbol = \"add\", abi = "
    "vm_module_function, linkage = default, hal_binding_alignment = 0, "
    "hal_workgroup_size_x = 0, hal_workgroup_size_y = 0, "
    "hal_workgroup_size_z = 0, hal_flat_workgroup_size_min = 0, "
    "hal_flat_workgroup_size_max = 0, hal_buffer_resource_flags = 0}\n"
    "target.config @vm_config {contract_set_key = \"iree.vm.core\", "
    "contract_feature_bits = 0}\n"
    "target.bundle @vm_target {snapshot = @vm_snapshot, export_plan = "
    "@vm_export, config = @vm_config}\n";

TEST_F(LowDescriptorVerifyTest, ValidVmPacketsPass) {
  std::string source =
      std::string(kVmTargetRecords) +
      "low.func.def target(@vm_target) @add(%lhs : reg<vm.i32>, %rhs : "
      "reg<vm.i32>) -> (reg<vm.i32>) {\n"
      "  %sum = low.op<iree.vm.add.i32>(%lhs, %rhs) : (reg<vm.i32>, "
      "reg<vm.i32>) -> reg<vm.i32>\n"
      "  low.return %sum : reg<vm.i32>\n"
      "}\n"
      "low.func.def target(@vm_target) @constant() -> (reg<vm.i32>) {\n"
      "  %c0 = low.const<iree.vm.const.i32> {i32_value = 7} : reg<vm.i32>\n"
      "  low.return %c0 : reg<vm.i32>\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(collector.emissions.empty());

  loom_module_free(module);
}

TEST_F(LowDescriptorVerifyTest, ValidSymbolicImmediatePasses) {
  std::string source =
      std::string(kVmTargetRecords) +
      "test.record @target_block {}\n"
      "low.func.def target(@vm_target) @branch() {\n"
      "  low.op<iree.vm.br>() {target_block = @target_block} : ()\n"
      "  low.return\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(collector.emissions.empty());

  loom_module_free(module);
}

TEST_F(LowDescriptorVerifyTest, RejectsMissingImmediateAttr) {
  std::string source =
      std::string(kVmTargetRecords) +
      "low.func.def target(@vm_target) @constant() -> (reg<vm.i32>) {\n"
      "  %c0 = low.const<iree.vm.const.i32> : reg<vm.i32>\n"
      "  low.return %c0 : reg<vm.i32>\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 7));
  ASSERT_GE(collector.emissions[0].string_params.size(), 3u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "constant");
  EXPECT_EQ(collector.emissions[0].string_params[1], "iree.vm.const.i32");
  EXPECT_EQ(collector.emissions[0].string_params[2], "i32_value");
  ASSERT_GE(collector.emissions[0].field_refs.size(), 3u);
  EXPECT_EQ(collector.emissions[0].field_refs[1].kind,
            LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE);
  EXPECT_EQ(collector.emissions[0].field_refs[2].kind,
            LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE);

  loom_module_free(module);
}

TEST_F(LowDescriptorVerifyTest, RejectsUnexpectedImmediateAttr) {
  std::string source =
      std::string(kVmTargetRecords) +
      "low.func.def target(@vm_target) @add(%lhs : reg<vm.i32>, %rhs : "
      "reg<vm.i32>) -> (reg<vm.i32>) {\n"
      "  %sum = low.op<iree.vm.add.i32>(%lhs, %rhs) {junk = 1} : "
      "(reg<vm.i32>, reg<vm.i32>) -> reg<vm.i32>\n"
      "  low.return %sum : reg<vm.i32>\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 8));
  ASSERT_GE(collector.emissions[0].string_params.size(), 3u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "add");
  EXPECT_EQ(collector.emissions[0].string_params[1], "iree.vm.add.i32");
  EXPECT_EQ(collector.emissions[0].string_params[2], "junk");

  loom_module_free(module);
}

TEST_F(LowDescriptorVerifyTest, RejectsImmediateKindMismatch) {
  std::string source =
      std::string(kVmTargetRecords) +
      "low.func.def target(@vm_target) @constant() -> (reg<vm.i32>) {\n"
      "  %c0 = low.const<iree.vm.const.i32> {i32_value = \"wrong\"} : "
      "reg<vm.i32>\n"
      "  low.return %c0 : reg<vm.i32>\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 9));
  ASSERT_GE(collector.emissions[0].string_params.size(), 4u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "constant");
  EXPECT_EQ(collector.emissions[0].string_params[1], "iree.vm.const.i32");
  EXPECT_EQ(collector.emissions[0].string_params[2], "i32_value");
  EXPECT_EQ(collector.emissions[0].string_params[3], "i64 signed integer");
  ASSERT_GE(collector.emissions[0].u32_params.size(), 1u);
  EXPECT_EQ(collector.emissions[0].u32_params[0],
            static_cast<uint32_t>(LOOM_ATTR_STRING));

  loom_module_free(module);
}

TEST_F(LowDescriptorVerifyTest, RejectsImmediateRangeMismatch) {
  std::string source =
      std::string(kVmTargetRecords) +
      "low.func.def target(@vm_target) @constant() -> (reg<vm.i32>) {\n"
      "  %c0 = low.const<iree.vm.const.i32> {i32_value = 2147483648} : "
      "reg<vm.i32>\n"
      "  low.return %c0 : reg<vm.i32>\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 10));
  ASSERT_GE(collector.emissions[0].string_params.size(), 4u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "constant");
  EXPECT_EQ(collector.emissions[0].string_params[1], "iree.vm.const.i32");
  EXPECT_EQ(collector.emissions[0].string_params[2], "i32_value");
  EXPECT_EQ(collector.emissions[0].string_params[3], "-2147483648..2147483647");
  ASSERT_GE(collector.emissions[0].i64_params.size(), 1u);
  EXPECT_EQ(collector.emissions[0].i64_params[0], INT64_C(2147483648));

  loom_module_free(module);
}

TEST_F(LowDescriptorVerifyTest, RejectsUnknownDescriptor) {
  std::string source =
      std::string(kVmTargetRecords) +
      "low.func.def target(@vm_target) @add(%lhs : reg<vm.i32>, %rhs : "
      "reg<vm.i32>) -> (reg<vm.i32>) {\n"
      "  %sum = low.op<iree.vm.missing.i32>(%lhs, %rhs) : (reg<vm.i32>, "
      "reg<vm.i32>) -> reg<vm.i32>\n"
      "  low.return %sum : reg<vm.i32>\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 4));
  ASSERT_GE(collector.emissions[0].string_params.size(), 3u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "add");
  EXPECT_EQ(collector.emissions[0].string_params[1], "iree.vm.missing.i32");
  EXPECT_EQ(collector.emissions[0].string_params[2], "iree.vm.core");
  ASSERT_GE(collector.emissions[0].field_refs.size(), 2u);
  EXPECT_EQ(collector.emissions[0].field_refs[1].kind,
            LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE);

  loom_module_free(module);
}

TEST_F(LowDescriptorVerifyTest, RejectsResultRegisterClassMismatch) {
  std::string source =
      std::string(kVmTargetRecords) +
      "low.func.def target(@vm_target) @add(%lhs : reg<vm.i32>, %rhs : "
      "reg<vm.i32>) -> (reg<vm.i64>) {\n"
      "  %sum = low.op<iree.vm.add.i32>(%lhs, %rhs) : (reg<vm.i32>, "
      "reg<vm.i32>) -> reg<vm.i64>\n"
      "  low.return %sum : reg<vm.i64>\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 6));
  ASSERT_GE(collector.emissions[0].string_params.size(), 5u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "add");
  EXPECT_EQ(collector.emissions[0].string_params[1], "iree.vm.add.i32");
  EXPECT_EQ(collector.emissions[0].string_params[2], "result");
  EXPECT_EQ(collector.emissions[0].string_params[3], "dst");
  EXPECT_EQ(collector.emissions[0].string_params[4], "vm.i32");
  ASSERT_GE(collector.emissions[0].type_params.size(), 1u);
  EXPECT_TRUE(loom_type_is_register(collector.emissions[0].type_params[0]));
  ASSERT_GE(collector.emissions[0].u32_params.size(), 1u);
  EXPECT_EQ(collector.emissions[0].u32_params[0], 1u);
  ASSERT_GE(collector.emissions[0].field_refs.size(), 4u);
  EXPECT_EQ(collector.emissions[0].field_refs[1].kind,
            LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE);
  EXPECT_EQ(collector.emissions[0].field_refs[3].kind,
            LOOM_DIAGNOSTIC_FIELD_RESULT);
  EXPECT_EQ(collector.emissions[0].field_refs[3].index, 0u);

  loom_module_free(module);
}

TEST_F(LowDescriptorVerifyTest, RejectsOperandRegisterClassMismatch) {
  std::string source =
      std::string(kVmTargetRecords) +
      "low.func.def target(@vm_target) @add(%lhs : reg<vm.i64>, %rhs : "
      "reg<vm.i32>) -> (reg<vm.i32>) {\n"
      "  %sum = low.op<iree.vm.add.i32>(%lhs, %rhs) : (reg<vm.i64>, "
      "reg<vm.i32>) -> reg<vm.i32>\n"
      "  low.return %sum : reg<vm.i32>\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 6));
  ASSERT_GE(collector.emissions[0].string_params.size(), 5u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "add");
  EXPECT_EQ(collector.emissions[0].string_params[1], "iree.vm.add.i32");
  EXPECT_EQ(collector.emissions[0].string_params[2], "operand");
  EXPECT_EQ(collector.emissions[0].string_params[3], "lhs");
  EXPECT_EQ(collector.emissions[0].string_params[4], "vm.i32");
  ASSERT_GE(collector.emissions[0].u32_params.size(), 1u);
  EXPECT_EQ(collector.emissions[0].u32_params[0], 1u);
  ASSERT_GE(collector.emissions[0].field_refs.size(), 4u);
  EXPECT_EQ(collector.emissions[0].field_refs[3].kind,
            LOOM_DIAGNOSTIC_FIELD_OPERAND);
  EXPECT_EQ(collector.emissions[0].field_refs[3].index, 0u);

  loom_module_free(module);
}

TEST_F(LowDescriptorVerifyTest, RejectsOperandRegisterUnitMismatch) {
  std::string source =
      std::string(kVmTargetRecords) +
      "low.func.def target(@vm_target) @add(%lhs : reg<vm.i32 x2>, %rhs : "
      "reg<vm.i32>) -> (reg<vm.i32>) {\n"
      "  %sum = low.op<iree.vm.add.i32>(%lhs, %rhs) : (reg<vm.i32 x2>, "
      "reg<vm.i32>) -> reg<vm.i32>\n"
      "  low.return %sum : reg<vm.i32>\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 6));
  ASSERT_GE(collector.emissions[0].string_params.size(), 5u);
  EXPECT_EQ(collector.emissions[0].string_params[2], "operand");
  EXPECT_EQ(collector.emissions[0].string_params[3], "lhs");
  EXPECT_EQ(collector.emissions[0].string_params[4], "vm.i32");
  ASSERT_GE(collector.emissions[0].u32_params.size(), 1u);
  EXPECT_EQ(collector.emissions[0].u32_params[0], 1u);
  ASSERT_GE(collector.emissions[0].type_params.size(), 1u);
  EXPECT_EQ(
      loom_type_register_unit_count(collector.emissions[0].type_params[0]), 2u);

  loom_module_free(module);
}

TEST_F(LowDescriptorVerifyTest, RejectsOperandCountMismatch) {
  std::string source =
      std::string(kVmTargetRecords) +
      "low.func.def target(@vm_target) @add(%lhs : reg<vm.i32>) -> "
      "(reg<vm.i32>) {\n"
      "  %sum = low.op<iree.vm.add.i32>(%lhs) : (reg<vm.i32>) -> reg<vm.i32>\n"
      "  low.return %sum : reg<vm.i32>\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 1));
  ASSERT_GE(collector.emissions[0].string_params.size(), 1u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "iree.vm.add.i32");
  ASSERT_GE(collector.emissions[0].u32_params.size(), 2u);
  EXPECT_EQ(collector.emissions[0].u32_params[0], 1u);
  EXPECT_EQ(collector.emissions[0].u32_params[1], 2u);

  loom_module_free(module);
}

TEST_F(LowDescriptorVerifyTest, RejectsResultCountMismatch) {
  std::string source =
      std::string(kVmTargetRecords) +
      "low.func.def target(@vm_target) @add(%lhs : reg<vm.i32>, %rhs : "
      "reg<vm.i32>) {\n"
      "  low.op<iree.vm.add.i32>(%lhs, %rhs) : (reg<vm.i32>, reg<vm.i32>)\n"
      "  low.return\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 2));
  ASSERT_GE(collector.emissions[0].string_params.size(), 1u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "iree.vm.add.i32");
  ASSERT_GE(collector.emissions[0].u32_params.size(), 2u);
  EXPECT_EQ(collector.emissions[0].u32_params[0], 0u);
  EXPECT_EQ(collector.emissions[0].u32_params[1], 1u);

  loom_module_free(module);
}

// clang-format off
static const uint8_t kFeatureTestStrings[] =
    LOOM_BSTRING_LITERAL("\x00", "")
    LOOM_BSTRING_LITERAL("\x09", "test.core")
    LOOM_BSTRING_LITERAL("\x0b", "test.target")
    LOOM_BSTRING_LITERAL("\x0d", "test.features")
    LOOM_BSTRING_LITERAL("\x08", "test.gpr")
    LOOM_BSTRING_LITERAL("\x03", "dst")
    LOOM_BSTRING_LITERAL("\x03", "lhs")
    LOOM_BSTRING_LITERAL("\x03", "rhs")
    LOOM_BSTRING_LITERAL("\x08", "test.alu")
    LOOM_BSTRING_LITERAL("\x0c", "test.add.i32")
    LOOM_BSTRING_LITERAL("\x07", "add.i32")
    LOOM_BSTRING_LITERAL("\x0f", "integer.add.i32");
// clang-format on

enum {
  FEATURE_STRING_empty = 0,
  FEATURE_STRING_set_key = FEATURE_STRING_empty + sizeof(""),
  FEATURE_STRING_target_key = FEATURE_STRING_set_key + sizeof("test.core"),
  FEATURE_STRING_feature_key =
      FEATURE_STRING_target_key + sizeof("test.target"),
  FEATURE_STRING_reg_gpr = FEATURE_STRING_feature_key + sizeof("test.features"),
  FEATURE_STRING_field_dst = FEATURE_STRING_reg_gpr + sizeof("test.gpr"),
  FEATURE_STRING_field_lhs = FEATURE_STRING_field_dst + sizeof("dst"),
  FEATURE_STRING_field_rhs = FEATURE_STRING_field_lhs + sizeof("lhs"),
  FEATURE_STRING_schedule_alu = FEATURE_STRING_field_rhs + sizeof("rhs"),
  FEATURE_STRING_descriptor_add =
      FEATURE_STRING_schedule_alu + sizeof("test.alu"),
  FEATURE_STRING_mnemonic_add =
      FEATURE_STRING_descriptor_add + sizeof("test.add.i32"),
  FEATURE_STRING_semantic_add = FEATURE_STRING_mnemonic_add + sizeof("add.i32"),
  FEATURE_STRING_END = FEATURE_STRING_semantic_add + sizeof("integer.add.i32"),
};

static_assert(FEATURE_STRING_END == sizeof(kFeatureTestStrings) - 1,
              "feature descriptor string offsets must cover the table payload");

#define FEATURE_STRING_OFFSET(field) \
  static_cast<loom_bstring_table_offset_t>(FEATURE_STRING_##field)

struct FeatureTestTables {
  loom_low_descriptor_t descriptors[1];
  loom_low_descriptor_ref_t descriptor_refs[1];
  loom_low_operand_t operands[3];
  loom_low_reg_class_t reg_classes[1];
  loom_low_reg_class_alt_t reg_class_alts[1];
  loom_low_schedule_class_t schedule_classes[1];
  uint64_t feature_mask_words[1];
  loom_low_descriptor_set_t set;
};

void InitializeFeatureTestTables(FeatureTestTables* tables) {
  *tables = {};
  tables->reg_classes[0].name_string_offset = FEATURE_STRING_OFFSET(reg_gpr);
  tables->reg_classes[0].flags = LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY;
  tables->reg_classes[0].alloc_unit_bits = 32;
  tables->reg_classes[0].spill_class_id = LOOM_LOW_REG_CLASS_NONE;

  tables->reg_class_alts[0].reg_class_id = 0;
  tables->reg_class_alts[0].flags = LOOM_LOW_REG_CLASS_ALT_FLAG_PREFERRED;

  tables->operands[0].field_name_string_offset =
      FEATURE_STRING_OFFSET(field_dst);
  tables->operands[0].role = LOOM_LOW_OPERAND_ROLE_RESULT;
  tables->operands[0].reg_class_alt_count = 1;
  tables->operands[0].unit_count = 1;
  tables->operands[1].field_name_string_offset =
      FEATURE_STRING_OFFSET(field_lhs);
  tables->operands[1].role = LOOM_LOW_OPERAND_ROLE_OPERAND;
  tables->operands[1].reg_class_alt_count = 1;
  tables->operands[1].unit_count = 1;
  tables->operands[2].field_name_string_offset =
      FEATURE_STRING_OFFSET(field_rhs);
  tables->operands[2].role = LOOM_LOW_OPERAND_ROLE_OPERAND;
  tables->operands[2].reg_class_alt_count = 1;
  tables->operands[2].unit_count = 1;

  tables->schedule_classes[0].name_string_offset =
      FEATURE_STRING_OFFSET(schedule_alu);
  tables->schedule_classes[0].latency_kind = LOOM_LOW_LATENCY_KIND_EXACT;
  tables->schedule_classes[0].model_quality = LOOM_LOW_MODEL_QUALITY_EXACT;

  tables->feature_mask_words[0] = UINT64_C(0x5);

  tables->descriptors[0].key_string_offset =
      FEATURE_STRING_OFFSET(descriptor_add);
  tables->descriptors[0].mnemonic_string_offset =
      FEATURE_STRING_OFFSET(mnemonic_add);
  tables->descriptors[0].semantic_tag_string_offset =
      FEATURE_STRING_OFFSET(semantic_add);
  tables->descriptors[0].feature_mask_word_count = 1;
  tables->descriptors[0].operand_count = 3;
  tables->descriptors[0].result_count = 1;
  tables->descriptors[0].schedule_class_id = 0;
  tables->descriptors[0].flags = LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE;

  tables->descriptor_refs[0].key_string_offset =
      FEATURE_STRING_OFFSET(descriptor_add);
  tables->descriptor_refs[0].descriptor_ordinal = 0;

  tables->set.abi_version = LOOM_LOW_DESCRIPTOR_SET_ABI_VERSION;
  tables->set.generator_version = 1;
  tables->set.key_string_offset = FEATURE_STRING_OFFSET(set_key);
  tables->set.target_key_string_offset = FEATURE_STRING_OFFSET(target_key);
  tables->set.feature_key_string_offset = FEATURE_STRING_OFFSET(feature_key);
  tables->set.string_table.data = kFeatureTestStrings;
  tables->set.string_table.data_length = sizeof(kFeatureTestStrings) - 1;
  tables->set.descriptors = tables->descriptors;
  tables->set.descriptor_count = IREE_ARRAYSIZE(tables->descriptors);
  tables->set.descriptor_refs = tables->descriptor_refs;
  tables->set.descriptor_ref_count = IREE_ARRAYSIZE(tables->descriptor_refs);
  tables->set.operands = tables->operands;
  tables->set.operand_count = IREE_ARRAYSIZE(tables->operands);
  tables->set.reg_classes = tables->reg_classes;
  tables->set.reg_class_count = IREE_ARRAYSIZE(tables->reg_classes);
  tables->set.reg_class_alts = tables->reg_class_alts;
  tables->set.reg_class_alt_count = IREE_ARRAYSIZE(tables->reg_class_alts);
  tables->set.schedule_classes = tables->schedule_classes;
  tables->set.schedule_class_count = IREE_ARRAYSIZE(tables->schedule_classes);
  tables->set.feature_mask_words = tables->feature_mask_words;
  tables->set.feature_mask_word_count =
      IREE_ARRAYSIZE(tables->feature_mask_words);
}

TEST_F(LowDescriptorVerifyTest, RejectsMissingFeatureBits) {
  FeatureTestTables tables;
  InitializeFeatureTestTables(&tables);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
  const loom_low_descriptor_set_t* descriptor_sets[] = {&tables.set};
  loom_low_descriptor_registry_t registry = {
      .descriptor_sets = descriptor_sets,
      .descriptor_set_count = IREE_ARRAYSIZE(descriptor_sets),
  };

  std::string source =
      "test.record @snapshot {}\n"
      "test.record @export {}\n"
      "target.config @test_config {contract_set_key = \"test.core\", "
      "contract_feature_bits = 1}\n"
      "target.bundle @test_target {snapshot = @snapshot, export_plan = "
      "@export, config = @test_config}\n"
      "low.func.def target(@test_target) @add(%lhs : reg<test.gpr x1>, "
      "%rhs : reg<test.gpr x1>) -> (reg<test.gpr x1>) {\n"
      "  %sum = low.op<test.add.i32>(%lhs, %rhs) : (reg<test.gpr x1>, "
      "reg<test.gpr x1>) -> reg<test.gpr x1>\n"
      "  low.return %sum : reg<test.gpr x1>\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 5));
  ASSERT_GE(collector.emissions[0].string_params.size(), 3u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "add");
  EXPECT_EQ(collector.emissions[0].string_params[1], "test.add.i32");
  EXPECT_EQ(collector.emissions[0].string_params[2], "test.core");
  ASSERT_GE(collector.emissions[0].u32_params.size(), 1u);
  EXPECT_EQ(collector.emissions[0].u32_params[0], 0u);
  ASSERT_GE(collector.emissions[0].u64_params.size(), 1u);
  EXPECT_EQ(collector.emissions[0].u64_params[0], UINT64_C(0x4));

  loom_module_free(module);
}

#undef FEATURE_STRING_OFFSET

}  // namespace
}  // namespace loom
