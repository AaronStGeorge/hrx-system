// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/verify.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/error_defs.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/emit/ireevm/descriptors.h"
#include "loom/testing/context.h"

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
    IREE_ASSERT_OK(loom_testing_context_initialize_all(iree_allocator_system(),
                                                       &context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* ParseSource(const std::string& source) {
    return ParseSource(source, nullptr);
  }

  loom_module_t* ParseSource(
      const std::string& source,
      const loom_text_low_asm_environment_t* low_asm_environment) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;
    if (low_asm_environment) {
      parse_options.low_asm_environment = *low_asm_environment;
    }

    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("low_descriptor_verify_test.loom"), &context_,
                        &block_pool_, &parse_options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  iree_status_t PrintModule(const loom_module_t* module,
                            const loom_text_print_options_t* print_options,
                            std::string* out_text) {
    out_text->clear();
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    iree_status_t status = loom_text_print_module_to_builder_with_options(
        module, &builder, print_options);
    if (iree_status_is_ok(status)) {
      *out_text = std::string(iree_string_builder_buffer(&builder),
                              iree_string_builder_size(&builder));
    }
    iree_string_builder_deinitialize(&builder);
    return status;
  }

  iree_status_t WriteModule(const loom_module_t* module,
                            std::vector<uint8_t>* out_bytes) {
    iree_io_stream_t* stream = nullptr;
    IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    iree_status_t status =
        loom_bytecode_write_module(module, stream, nullptr, &block_pool_);

    if (iree_status_is_ok(status)) {
      iree_io_stream_pos_t length = iree_io_stream_length(stream);
      out_bytes->resize((size_t)length);
      status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0);
    }
    if (iree_status_is_ok(status)) {
      status = iree_io_stream_read(stream, out_bytes->size(), out_bytes->data(),
                                   nullptr);
    }
    iree_io_stream_release(stream);
    return status;
  }

  iree_status_t ReadModule(const std::vector<uint8_t>& bytes,
                           loom_module_t** out_module) {
    loom_bytecode_read_options_t options = {
        .verify_module = false,
        .verify_max_errors = 20,
    };
    loom_bytecode_read_result_t result = {};
    IREE_RETURN_IF_ERROR(loom_bytecode_read_module(
        iree_make_const_byte_span(bytes.data(), bytes.size()),
        IREE_SV("low_descriptor_verify_test.loombc"), &context_, &block_pool_,
        &options, &result, out_module, iree_allocator_system()));
    if (result.error_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "bytecode read emitted %u errors",
                              result.error_count);
    }
    return iree_ok_status();
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

  static loom_text_low_asm_environment_t LowAsmEnvironment(
      const loom_low_descriptor_registry_t* registry) {
    loom_text_low_asm_environment_t environment = {};
    loom_low_descriptor_text_asm_environment_initialize(registry, &environment);
    return environment;
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

TEST_F(LowDescriptorVerifyTest, LowFuncDefAsmBodyParsesAndPrintsByPolicy) {
  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  loom_text_low_asm_environment_t environment = LowAsmEnvironment(&registry);
  std::string source =
      std::string(kVmTargetRecords) +
      "low.func.def target(@vm_target) @add(%lhs : reg<vm.i32>, %rhs : "
      "reg<vm.i32>) -> (reg<vm.i32>) asm<iree.vm.core> {\n"
      "  %sum = vm.add.i32 %lhs, %rhs\n"
      "  return %sum\n"
      "}\n";
  loom_module_t* module = ParseSource(source, &environment);
  ASSERT_NE(module, nullptr);

  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(collector.emissions.empty());

  loom_text_print_options_t canonical_print_options = {
      .flags = LOOM_TEXT_PRINT_DEFAULT,
  };
  std::string canonical_text;
  IREE_ASSERT_OK(
      PrintModule(module, &canonical_print_options, &canonical_text));
  EXPECT_EQ(canonical_text.find("asm<iree.vm.core>"), std::string::npos);
  EXPECT_NE(canonical_text.find("low.op<iree.vm.add.i32>"), std::string::npos);

  loom_text_print_options_t missing_environment_options = {
      .flags = LOOM_TEXT_PRINT_DEFAULT,
      .low_asm_descriptor_set_key = IREE_SV("iree.vm.core"),
  };
  std::string missing_environment_text;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        PrintModule(module, &missing_environment_options,
                                    &missing_environment_text));

  loom_text_print_options_t asm_print_options = {
      .flags = LOOM_TEXT_PRINT_DEFAULT,
      .low_asm_environment = environment,
      .low_asm_descriptor_set_key = IREE_SV("iree.vm.core"),
  };
  std::string asm_text;
  IREE_ASSERT_OK(PrintModule(module, &asm_print_options, &asm_text));
  EXPECT_NE(asm_text.find("asm<iree.vm.core>"), std::string::npos);
  EXPECT_NE(asm_text.find("vm.add.i32 %lhs, %rhs"), std::string::npos);
  EXPECT_EQ(asm_text.find("low.op<iree.vm.add.i32>"), std::string::npos);

  std::vector<uint8_t> bytecode;
  IREE_ASSERT_OK(WriteModule(module, &bytecode));
  loom_module_free(module);

  loom_module_t* bytecode_module = nullptr;
  IREE_ASSERT_OK(ReadModule(bytecode, &bytecode_module));
  ASSERT_NE(bytecode_module, nullptr);

  std::string bytecode_canonical_text;
  IREE_ASSERT_OK(PrintModule(bytecode_module, &canonical_print_options,
                             &bytecode_canonical_text));
  EXPECT_NE(bytecode_canonical_text.find("low.op<iree.vm.add.i32>"),
            std::string::npos);

  std::string bytecode_asm_text;
  IREE_ASSERT_OK(
      PrintModule(bytecode_module, &asm_print_options, &bytecode_asm_text));
  EXPECT_NE(bytecode_asm_text.find("asm<iree.vm.core>"), std::string::npos);
  EXPECT_NE(bytecode_asm_text.find("vm.add.i32 %lhs, %rhs"), std::string::npos);

  loom_module_free(bytecode_module);
}

TEST_F(LowDescriptorVerifyTest, AsmAuthoredPacketErrorsKeepSourceLocation) {
  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  loom_text_low_asm_environment_t environment = LowAsmEnvironment(&registry);
  std::string source =
      std::string(kVmTargetRecords) +
      "low.func.def target(@vm_target) @constant() -> (reg<vm.i32>) "
      "asm<iree.vm.core> {\n"
      "  %c0 = vm.const.i32 2147483648\n"
      "  return %c0\n"
      "}\n";
  loom_module_t* module = ParseSource(source, &environment);
  ASSERT_NE(module, nullptr);

  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 10));
  ASSERT_NE(collector.emissions[0].op, nullptr);
  EXPECT_TRUE(loom_low_const_isa(collector.emissions[0].op));

  ASSERT_LT(collector.emissions[0].op->location, module->locations.count);
  const loom_location_entry_t location =
      module->locations.entries[collector.emissions[0].op->location];
  ASSERT_EQ(location.kind, LOOM_LOCATION_FILE);
  EXPECT_EQ(location.file.start_line, 6u);
  EXPECT_EQ(location.file.start_col, 3u);

  bool found_immediate_span = false;
  for (uint16_t i = 0; i < location.file.field_span_count; ++i) {
    const loom_location_field_span_t span = location.file.field_spans[i];
    if (span.kind == LOOM_LOCATION_FIELD_ATTRIBUTE &&
        span.index == loom_low_const_attrs_ATTR_INDEX) {
      found_immediate_span = true;
      EXPECT_EQ(span.start_line, 6u);
      EXPECT_EQ(span.start_col, 22u);
      EXPECT_EQ(span.end_line, 6u);
      break;
    }
  }
  EXPECT_TRUE(found_immediate_span);

  loom_module_free(module);
}

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

TEST_F(LowDescriptorVerifyTest, ValidVmPacketsPassWithFoundationRequirements) {
  std::string source =
      std::string(kVmTargetRecords) +
      "low.func.def target(@vm_target) @add(%lhs : reg<vm.i32>, %rhs : "
      "reg<vm.i32>) -> (reg<vm.i32>) {\n"
      "  %sum = low.op<iree.vm.add.i32>(%lhs, %rhs) : (reg<vm.i32>, "
      "reg<vm.i32>) -> reg<vm.i32>\n"
      "  low.return %sum : reg<vm.i32>\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_verify_options_t options = {
      .descriptor_registry = &registry,
      .descriptor_requirements =
          LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
      .emitter = collector.emitter(),
      .max_errors = 20,
  };
  loom_low_verify_result_t result = {};
  IREE_ASSERT_OK(loom_low_verify_module(module, &options, &result));

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
    LOOM_BSTRING_LITERAL("\x08", "test.alt")
    LOOM_BSTRING_LITERAL("\x03", "dst")
    LOOM_BSTRING_LITERAL("\x03", "lhs")
    LOOM_BSTRING_LITERAL("\x03", "rhs")
    LOOM_BSTRING_LITERAL("\x04", "exec")
    LOOM_BSTRING_LITERAL("\x04", "mode")
    LOOM_BSTRING_LITERAL("\x08", "test.alu")
    LOOM_BSTRING_LITERAL("\x09", "test.mode")
    LOOM_BSTRING_LITERAL("\x04", "fast")
    LOOM_BSTRING_LITERAL("\x04", "slow")
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
  FEATURE_STRING_reg_alt = FEATURE_STRING_reg_gpr + sizeof("test.gpr"),
  FEATURE_STRING_field_dst = FEATURE_STRING_reg_alt + sizeof("test.alt"),
  FEATURE_STRING_field_lhs = FEATURE_STRING_field_dst + sizeof("dst"),
  FEATURE_STRING_field_rhs = FEATURE_STRING_field_lhs + sizeof("lhs"),
  FEATURE_STRING_field_exec = FEATURE_STRING_field_rhs + sizeof("rhs"),
  FEATURE_STRING_field_mode = FEATURE_STRING_field_exec + sizeof("exec"),
  FEATURE_STRING_schedule_alu = FEATURE_STRING_field_mode + sizeof("mode"),
  FEATURE_STRING_enum_mode = FEATURE_STRING_schedule_alu + sizeof("test.alu"),
  FEATURE_STRING_enum_fast = FEATURE_STRING_enum_mode + sizeof("test.mode"),
  FEATURE_STRING_enum_slow = FEATURE_STRING_enum_fast + sizeof("fast"),
  FEATURE_STRING_descriptor_add = FEATURE_STRING_enum_slow + sizeof("slow"),
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
  loom_low_operand_t operands[4];
  loom_low_immediate_t immediates[1];
  loom_low_enum_domain_t enum_domains[1];
  loom_low_enum_value_t enum_values[2];
  loom_low_constraint_t constraints[1];
  loom_low_reg_class_t reg_classes[2];
  loom_low_reg_class_alt_t reg_class_alts[2];
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
  tables->reg_classes[0].spill_slot_space = LOOM_LOW_SPILL_SLOT_SPACE_STACK;
  tables->reg_classes[1].name_string_offset = FEATURE_STRING_OFFSET(reg_alt);
  tables->reg_classes[1].flags = LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY;
  tables->reg_classes[1].alloc_unit_bits = 32;
  tables->reg_classes[1].spill_class_id = LOOM_LOW_REG_CLASS_NONE;
  tables->reg_classes[1].spill_slot_space = LOOM_LOW_SPILL_SLOT_SPACE_STACK;

  tables->reg_class_alts[0].reg_class_id = 0;
  tables->reg_class_alts[0].flags = LOOM_LOW_REG_CLASS_ALT_FLAG_PREFERRED;
  tables->reg_class_alts[1].reg_class_id = 1;
  tables->reg_class_alts[1].flags = 0;

  tables->operands[0].field_name_string_offset =
      FEATURE_STRING_OFFSET(field_dst);
  tables->operands[0].role = LOOM_LOW_OPERAND_ROLE_RESULT;
  tables->operands[0].reg_class_alt_count = 2;
  tables->operands[0].unit_count = 1;
  tables->operands[1].field_name_string_offset =
      FEATURE_STRING_OFFSET(field_lhs);
  tables->operands[1].role = LOOM_LOW_OPERAND_ROLE_OPERAND;
  tables->operands[1].reg_class_alt_count = 2;
  tables->operands[1].unit_count = 1;
  tables->operands[2].field_name_string_offset =
      FEATURE_STRING_OFFSET(field_rhs);
  tables->operands[2].role = LOOM_LOW_OPERAND_ROLE_OPERAND;
  tables->operands[2].reg_class_alt_count = 2;
  tables->operands[2].unit_count = 1;
  tables->operands[3].field_name_string_offset =
      FEATURE_STRING_OFFSET(field_exec);
  tables->operands[3].role = LOOM_LOW_OPERAND_ROLE_IMPLICIT;
  tables->operands[3].flags = LOOM_LOW_OPERAND_FLAG_IMPLICIT;
  tables->operands[3].reg_class_alt_count = 2;
  tables->operands[3].unit_count = 1;

  tables->immediates[0].field_name_string_offset =
      FEATURE_STRING_OFFSET(field_mode);
  tables->immediates[0].kind = LOOM_LOW_IMMEDIATE_KIND_ENUM;
  tables->immediates[0].enum_domain_id = 0;

  tables->enum_domains[0].name_string_offset = FEATURE_STRING_OFFSET(enum_mode);
  tables->enum_domains[0].value_start = 0;
  tables->enum_domains[0].value_count = 2;

  tables->enum_values[0].token_string_offset = FEATURE_STRING_OFFSET(enum_fast);
  tables->enum_values[0].value = 0;
  tables->enum_values[1].token_string_offset = FEATURE_STRING_OFFSET(enum_slow);
  tables->enum_values[1].value = 1;

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
  tables->descriptors[0].operand_count = 4;
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
  tables->set.immediates = tables->immediates;
  tables->set.enum_domains = tables->enum_domains;
  tables->set.enum_values = tables->enum_values;
  tables->set.constraints = tables->constraints;
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

void EnableFeatureTestEnumImmediate(FeatureTestTables* tables) {
  tables->descriptors[0].immediate_start = 0;
  tables->descriptors[0].immediate_count = 1;
  tables->set.immediate_count = 1;
  tables->set.enum_domain_count = 1;
  tables->set.enum_value_count = 2;
}

void AddFeatureTestConstraint(FeatureTestTables* tables,
                              loom_low_constraint_kind_t kind,
                              uint16_t lhs_operand_index,
                              uint16_t rhs_operand_index) {
  tables->constraints[0].kind = kind;
  tables->constraints[0].lhs_operand_index = lhs_operand_index;
  tables->constraints[0].rhs_operand_index = rhs_operand_index;
  tables->descriptors[0].constraint_start = 0;
  tables->descriptors[0].constraint_count = 1;
  tables->set.constraint_count = 1;
}

std::string FeatureTargetRecords(const char* feature_bits) {
  return std::string(
             "target.snapshot @test_snapshot {codegen_format = low_native, "
             "target_triple = \"test\", data_layout = \"\", artifact_format = "
             "elf, target_cpu = \"generic\", target_features = \"\", "
             "default_pointer_bitwidth = 64, index_bitwidth = 64, "
             "offset_bitwidth = 64, memory_space_generic = 0, "
             "memory_space_global = 0, memory_space_workgroup = 0, "
             "memory_space_constant = 0, memory_space_private = 0, "
             "memory_space_host = 0, memory_space_descriptor = 0}\n"
             "target.export @test_export {export_symbol = \"add\", abi = "
             "object_function, linkage = default, hal_binding_alignment = 0, "
             "hal_workgroup_size_x = 0, hal_workgroup_size_y = 0, "
             "hal_workgroup_size_z = 0, hal_flat_workgroup_size_min = 0, "
             "hal_flat_workgroup_size_max = 0, hal_buffer_resource_flags = "
             "0}\n"
             "target.config @test_config {contract_set_key = \"test.core\", "
             "contract_feature_bits = ") +
         feature_bits +
         "}\n"
         "target.bundle @test_target {snapshot = @test_snapshot, export_plan = "
         "@test_export, config = @test_config}\n";
}

std::string FeatureEnumImmediateSource(const char* mode_value) {
  return FeatureTargetRecords("5") +
         std::string(
             "low.func.def target(@test_target) @add(%lhs : reg<test.gpr x1>, "
             "%rhs : reg<test.gpr x1>) -> (reg<test.gpr x1>) {\n"
             "  %sum = low.op<test.add.i32>(%lhs, %rhs) {mode = ") +
         mode_value +
         "} : (reg<test.gpr x1>, reg<test.gpr x1>) -> reg<test.gpr x1>\n"
         "  low.return %sum : reg<test.gpr x1>\n"
         "}\n";
}

TEST_F(LowDescriptorVerifyTest,
       RejectsRegistryMissingRequestedFoundationRequirements) {
  FeatureTestTables tables;
  InitializeFeatureTestTables(&tables);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
  const loom_low_descriptor_set_t* descriptor_sets[] = {&tables.set};
  loom_low_descriptor_registry_t registry = {
      .descriptor_sets = descriptor_sets,
      .descriptor_set_count = IREE_ARRAYSIZE(descriptor_sets),
  };

  loom_module_t* module = ParseSource("func.def @unused() {\n}\n");
  ASSERT_NE(module, nullptr);

  EmissionCollector collector;
  loom_low_verify_options_t options = {
      .descriptor_registry = &registry,
      .descriptor_requirements =
          LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
      .emitter = collector.emitter(),
      .max_errors = 20,
  };
  loom_low_verify_result_t result = {};
  iree_status_t status = loom_low_verify_module(module, &options, &result);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, status);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(collector.emissions.empty());

  loom_module_free(module);
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
      FeatureTargetRecords("1") +
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

TEST_F(LowDescriptorVerifyTest, ValidEnumImmediateTokenPasses) {
  FeatureTestTables tables;
  InitializeFeatureTestTables(&tables);
  EnableFeatureTestEnumImmediate(&tables);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
  const loom_low_descriptor_set_t* descriptor_sets[] = {&tables.set};
  loom_low_descriptor_registry_t registry = {
      .descriptor_sets = descriptor_sets,
      .descriptor_set_count = IREE_ARRAYSIZE(descriptor_sets),
  };

  loom_module_t* module = ParseSource(FeatureEnumImmediateSource("fast"));
  ASSERT_NE(module, nullptr);

  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(collector.emissions.empty());

  loom_module_free(module);
}

TEST_F(LowDescriptorVerifyTest, ValidEnumImmediateOrdinalPasses) {
  FeatureTestTables tables;
  InitializeFeatureTestTables(&tables);
  EnableFeatureTestEnumImmediate(&tables);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
  const loom_low_descriptor_set_t* descriptor_sets[] = {&tables.set};
  loom_low_descriptor_registry_t registry = {
      .descriptor_sets = descriptor_sets,
      .descriptor_set_count = IREE_ARRAYSIZE(descriptor_sets),
  };

  loom_module_t* module = ParseSource(FeatureEnumImmediateSource("1"));
  ASSERT_NE(module, nullptr);

  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(collector.emissions.empty());

  loom_module_free(module);
}

TEST_F(LowDescriptorVerifyTest, RejectsUnknownEnumImmediateToken) {
  FeatureTestTables tables;
  InitializeFeatureTestTables(&tables);
  EnableFeatureTestEnumImmediate(&tables);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
  const loom_low_descriptor_set_t* descriptor_sets[] = {&tables.set};
  loom_low_descriptor_registry_t registry = {
      .descriptor_sets = descriptor_sets,
      .descriptor_set_count = IREE_ARRAYSIZE(descriptor_sets),
  };

  loom_module_t* module = ParseSource(FeatureEnumImmediateSource("missing"));
  ASSERT_NE(module, nullptr);

  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 12));
  ASSERT_GE(collector.emissions[0].string_params.size(), 5u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "add");
  EXPECT_EQ(collector.emissions[0].string_params[1], "test.add.i32");
  EXPECT_EQ(collector.emissions[0].string_params[2], "mode");
  EXPECT_EQ(collector.emissions[0].string_params[3], "missing");
  EXPECT_EQ(collector.emissions[0].string_params[4], "test.mode");

  loom_module_free(module);
}

TEST_F(LowDescriptorVerifyTest, RejectsUnknownEnumImmediateOrdinal) {
  FeatureTestTables tables;
  InitializeFeatureTestTables(&tables);
  EnableFeatureTestEnumImmediate(&tables);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
  const loom_low_descriptor_set_t* descriptor_sets[] = {&tables.set};
  loom_low_descriptor_registry_t registry = {
      .descriptor_sets = descriptor_sets,
      .descriptor_set_count = IREE_ARRAYSIZE(descriptor_sets),
  };

  loom_module_t* module = ParseSource(FeatureEnumImmediateSource("7"));
  ASSERT_NE(module, nullptr);

  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 12));
  ASSERT_GE(collector.emissions[0].string_params.size(), 5u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "add");
  EXPECT_EQ(collector.emissions[0].string_params[1], "test.add.i32");
  EXPECT_EQ(collector.emissions[0].string_params[2], "mode");
  EXPECT_EQ(collector.emissions[0].string_params[3], "7");
  EXPECT_EQ(collector.emissions[0].string_params[4], "test.mode");

  loom_module_free(module);
}

TEST_F(LowDescriptorVerifyTest, IgnoresImplicitDescriptorOperandRows) {
  FeatureTestTables tables;
  InitializeFeatureTestTables(&tables);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
  const loom_low_descriptor_set_t* descriptor_sets[] = {&tables.set};
  loom_low_descriptor_registry_t registry = {
      .descriptor_sets = descriptor_sets,
      .descriptor_set_count = IREE_ARRAYSIZE(descriptor_sets),
  };

  std::string source =
      FeatureTargetRecords("5") +
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

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(collector.emissions.empty());

  loom_module_free(module);
}

TEST_F(LowDescriptorVerifyTest, AcceptsMatchingTiedConstraintTypes) {
  FeatureTestTables tables;
  InitializeFeatureTestTables(&tables);
  AddFeatureTestConstraint(&tables, LOOM_LOW_CONSTRAINT_KIND_TIED, 0, 1);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
  const loom_low_descriptor_set_t* descriptor_sets[] = {&tables.set};
  loom_low_descriptor_registry_t registry = {
      .descriptor_sets = descriptor_sets,
      .descriptor_set_count = IREE_ARRAYSIZE(descriptor_sets),
  };

  std::string source =
      FeatureTargetRecords("5") +
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

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(collector.emissions.empty());

  loom_module_free(module);
}

TEST_F(LowDescriptorVerifyTest, RejectsTiedConstraintTypeMismatch) {
  FeatureTestTables tables;
  InitializeFeatureTestTables(&tables);
  AddFeatureTestConstraint(&tables, LOOM_LOW_CONSTRAINT_KIND_TIED, 0, 1);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
  const loom_low_descriptor_set_t* descriptor_sets[] = {&tables.set};
  loom_low_descriptor_registry_t registry = {
      .descriptor_sets = descriptor_sets,
      .descriptor_set_count = IREE_ARRAYSIZE(descriptor_sets),
  };

  std::string source =
      FeatureTargetRecords("5") +
      "low.func.def target(@test_target) @add(%lhs : reg<test.gpr x1>, "
      "%rhs : reg<test.gpr x1>) -> (reg<test.alt x1>) {\n"
      "  %sum = low.op<test.add.i32>(%lhs, %rhs) : (reg<test.gpr x1>, "
      "reg<test.gpr x1>) -> reg<test.alt x1>\n"
      "  low.return %sum : reg<test.alt x1>\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 11));
  ASSERT_GE(collector.emissions[0].string_params.size(), 7u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "add");
  EXPECT_EQ(collector.emissions[0].string_params[1], "test.add.i32");
  EXPECT_EQ(collector.emissions[0].string_params[2], "tied");
  EXPECT_EQ(collector.emissions[0].string_params[3], "result");
  EXPECT_EQ(collector.emissions[0].string_params[4], "dst");
  EXPECT_EQ(collector.emissions[0].string_params[5], "operand");
  EXPECT_EQ(collector.emissions[0].string_params[6], "lhs");
  ASSERT_GE(collector.emissions[0].type_params.size(), 2u);
  EXPECT_FALSE(loom_type_equal(collector.emissions[0].type_params[0],
                               collector.emissions[0].type_params[1]));
  ASSERT_GE(collector.emissions[0].field_refs.size(), 8u);
  EXPECT_EQ(collector.emissions[0].field_refs[4].kind,
            LOOM_DIAGNOSTIC_FIELD_RESULT);
  EXPECT_EQ(collector.emissions[0].field_refs[4].index, 0u);
  EXPECT_EQ(collector.emissions[0].field_refs[7].kind,
            LOOM_DIAGNOSTIC_FIELD_OPERAND);
  EXPECT_EQ(collector.emissions[0].field_refs[7].index, 0u);

  loom_module_free(module);
}

TEST_F(LowDescriptorVerifyTest, RejectsDestructiveConstraintTypeMismatch) {
  FeatureTestTables tables;
  InitializeFeatureTestTables(&tables);
  AddFeatureTestConstraint(&tables, LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE, 0, 1);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
  const loom_low_descriptor_set_t* descriptor_sets[] = {&tables.set};
  loom_low_descriptor_registry_t registry = {
      .descriptor_sets = descriptor_sets,
      .descriptor_set_count = IREE_ARRAYSIZE(descriptor_sets),
  };

  std::string source =
      FeatureTargetRecords("5") +
      "low.func.def target(@test_target) @add(%lhs : reg<test.gpr x1>, "
      "%rhs : reg<test.gpr x1>) -> (reg<test.alt x1>) {\n"
      "  %sum = low.op<test.add.i32>(%lhs, %rhs) : (reg<test.gpr x1>, "
      "reg<test.gpr x1>) -> reg<test.alt x1>\n"
      "  low.return %sum : reg<test.alt x1>\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 11));
  ASSERT_GE(collector.emissions[0].string_params.size(), 3u);
  EXPECT_EQ(collector.emissions[0].string_params[2], "destructive");

  loom_module_free(module);
}

TEST_F(LowDescriptorVerifyTest, RejectsCommutableConstraintTypeMismatch) {
  FeatureTestTables tables;
  InitializeFeatureTestTables(&tables);
  AddFeatureTestConstraint(&tables, LOOM_LOW_CONSTRAINT_KIND_COMMUTABLE, 1, 2);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
  const loom_low_descriptor_set_t* descriptor_sets[] = {&tables.set};
  loom_low_descriptor_registry_t registry = {
      .descriptor_sets = descriptor_sets,
      .descriptor_set_count = IREE_ARRAYSIZE(descriptor_sets),
  };

  std::string source =
      FeatureTargetRecords("5") +
      "low.func.def target(@test_target) @add(%lhs : reg<test.gpr x1>, "
      "%rhs : reg<test.alt x1>) -> (reg<test.gpr x1>) {\n"
      "  %sum = low.op<test.add.i32>(%lhs, %rhs) : (reg<test.gpr x1>, "
      "reg<test.alt x1>) -> reg<test.gpr x1>\n"
      "  low.return %sum : reg<test.gpr x1>\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  EmissionCollector collector;
  loom_low_verify_result_t result = VerifyModule(module, &registry, &collector);

  EXPECT_EQ(result.error_count, 1u);
  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 11));
  ASSERT_GE(collector.emissions[0].string_params.size(), 7u);
  EXPECT_EQ(collector.emissions[0].string_params[2], "commutable");
  EXPECT_EQ(collector.emissions[0].string_params[4], "lhs");
  EXPECT_EQ(collector.emissions[0].string_params[6], "rhs");
  ASSERT_GE(collector.emissions[0].field_refs.size(), 8u);
  EXPECT_EQ(collector.emissions[0].field_refs[4].kind,
            LOOM_DIAGNOSTIC_FIELD_OPERAND);
  EXPECT_EQ(collector.emissions[0].field_refs[4].index, 0u);
  EXPECT_EQ(collector.emissions[0].field_refs[7].kind,
            LOOM_DIAGNOSTIC_FIELD_OPERAND);
  EXPECT_EQ(collector.emissions[0].field_refs[7].index, 1u);

  loom_module_free(module);
}

#undef FEATURE_STRING_OFFSET

}  // namespace
}  // namespace loom
