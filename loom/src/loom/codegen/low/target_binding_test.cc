// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/target_binding.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/target/emit/ireevm/descriptors.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

static std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

struct CollectedEmission {
  const loom_error_def_t* error = nullptr;
  const loom_op_t* op = nullptr;
  std::vector<std::string> string_params;
  std::vector<loom_diagnostic_field_ref_t> field_refs;
  iree_host_size_t related_count = 0;
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
    entry.related_count = emission->related_op_count;
    for (iree_host_size_t i = 0; i < emission->param_count; ++i) {
      entry.field_refs.push_back(emission->params[i].field_ref);
      if (emission->params[i].kind == LOOM_PARAM_STRING) {
        entry.string_params.push_back(CopyString(emission->params[i].string));
      }
    }
    collector->emissions.push_back(std::move(entry));
    return iree_ok_status();
  }
};

class LowTargetBindingTest : public ::testing::Test {
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

  loom_module_t* ParseSource(const char* source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;

    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(
        iree_make_cstring_view(source), IREE_SV("low_target_binding_test.loom"),
        &context_, &block_pool_, &parse_options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  loom_module_t* ParseSource(const std::string& source) {
    return ParseSource(source.c_str());
  }

  const loom_op_t* FindFirstOp(loom_module_t* module, loom_op_kind_t kind) {
    loom_block_t* block = loom_module_block(module);
    const loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (op->kind == kind) return op;
    }
    return nullptr;
  }

  const loom_op_t* FindFirstBodyOp(const loom_op_t* low_func,
                                   loom_op_kind_t kind) {
    loom_region_t* body = loom_low_func_def_body(low_func);
    if (!body || body->block_count == 0) {
      return nullptr;
    }
    loom_block_t* block = loom_region_entry_block(body);
    const loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (op->kind == kind) return op;
    }
    return nullptr;
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

static const char kValidVmSnapshotAndExportRecords[] =
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
    "hal_flat_workgroup_size_max = 0, hal_buffer_resource_flags = 0}\n";

static const char kValidVmConfigRecord[] =
    "target.config @vm_config {contract_set_key = \"iree.vm.core\", "
    "contract_feature_bits = 0}\n";

static std::string WithValidVmTargetRecords(const char* suffix) {
  return std::string(kValidVmSnapshotAndExportRecords) + kValidVmConfigRecord +
         suffix;
}

static std::string WithValidVmSnapshotAndExportRecords(const char* suffix) {
  return std::string(kValidVmSnapshotAndExportRecords) + suffix;
}

static std::string ValidVmLowFunction() {
  return WithValidVmTargetRecords(
      "target.bundle @vm_target {snapshot = @vm_snapshot, export_plan = "
      "@vm_export, config = @vm_config}\n"
      "low.func.def target(@vm_target) @add(%lhs : reg<vm.i32>, %rhs : "
      "reg<vm.i32>) -> (reg<vm.i32>) {\n"
      "  %sum = low.op<iree.vm.add.i32>(%lhs, %rhs) : (reg<vm.i32>, "
      "reg<vm.i32>) -> reg<vm.i32>\n"
      "  low.return %sum : reg<vm.i32>\n"
      "}\n");
}

TEST_F(LowTargetBindingTest, RegistryVerifiesAndFindsDescriptorSet) {
  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  IREE_ASSERT_OK(loom_low_descriptor_registry_verify(&registry));

  const loom_low_descriptor_set_t* descriptor_set = nullptr;
  IREE_ASSERT_OK(loom_low_descriptor_registry_lookup(
      &registry, IREE_SV("iree.vm.core"), &descriptor_set));
  EXPECT_EQ(descriptor_set, loom_ireevm_core_descriptor_set());

  IREE_ASSERT_OK(loom_low_descriptor_registry_lookup(
      &registry, IREE_SV("missing"), &descriptor_set));
  EXPECT_EQ(descriptor_set, nullptr);
}

TEST_F(LowTargetBindingTest, RegistryRejectsDuplicateKeys) {
  const loom_low_descriptor_set_t* descriptor_sets[] = {
      loom_ireevm_core_descriptor_set(),
      loom_ireevm_core_descriptor_set(),
  };
  loom_low_descriptor_registry_t registry = {
      .descriptor_sets = descriptor_sets,
      .descriptor_set_count = IREE_ARRAYSIZE(descriptor_sets),
  };
  iree_status_t status = loom_low_descriptor_registry_verify(&registry);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS, status);
}

TEST_F(LowTargetBindingTest, RegistryRejectsNullEntries) {
  const loom_low_descriptor_set_t* descriptor_sets[] = {
      loom_ireevm_core_descriptor_set(),
      nullptr,
  };
  loom_low_descriptor_registry_t registry = {
      .descriptor_sets = descriptor_sets,
      .descriptor_set_count = IREE_ARRAYSIZE(descriptor_sets),
  };
  iree_status_t status = loom_low_descriptor_registry_verify(&registry);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  const loom_low_descriptor_set_t* descriptor_set = nullptr;
  status = loom_low_descriptor_registry_lookup(&registry, IREE_SV("missing"),
                                               &descriptor_set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(descriptor_set, nullptr);
}

TEST_F(LowTargetBindingTest, ResolvesBundleConfigAndDescriptorSet) {
  loom_module_t* module = ParseSource(ValidVmLowFunction());
  ASSERT_NE(module, nullptr);
  const loom_op_t* low_func = FindFirstOp(module, LOOM_OP_LOW_FUNC_DEF);
  ASSERT_NE(low_func, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_resolved_target_t target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module, low_func, &registry, collector.emitter(), &target));

  EXPECT_TRUE(collector.emissions.empty());
  EXPECT_EQ(target.target_op->kind, LOOM_OP_TARGET_BUNDLE);
  EXPECT_EQ(target.snapshot_op->kind, LOOM_OP_TARGET_SNAPSHOT);
  EXPECT_EQ(target.export_plan_op->kind, LOOM_OP_TARGET_EXPORT);
  EXPECT_EQ(target.config_op->kind, LOOM_OP_TARGET_CONFIG);
  EXPECT_EQ(target.descriptor_set, loom_ireevm_core_descriptor_set());
  EXPECT_EQ(target.bundle_storage.bundle.snapshot->codegen_format,
            LOOM_TARGET_CODEGEN_FORMAT_VM);
  EXPECT_EQ(target.bundle_storage.bundle.export_plan->abi_kind,
            LOOM_TARGET_ABI_VM_MODULE_FUNCTION);
  EXPECT_EQ(ToString(target.bundle_storage.bundle.config->contract_set_key),
            "iree.vm.core");
  EXPECT_TRUE(iree_string_view_equal(target.target_name, IREE_SV("vm_target")));
  EXPECT_TRUE(iree_string_view_equal(target.descriptor_set_key,
                                     IREE_SV("iree.vm.core")));
  EXPECT_EQ(target.feature_bits, 0u);

  loom_module_free(module);
}

TEST_F(LowTargetBindingTest, ResolvesDescriptorPacketToDenseOrdinal) {
  loom_module_t* module = ParseSource(ValidVmLowFunction());
  ASSERT_NE(module, nullptr);
  const loom_op_t* low_func = FindFirstOp(module, LOOM_OP_LOW_FUNC_DEF);
  ASSERT_NE(low_func, nullptr);
  const loom_op_t* low_op = FindFirstBodyOp(low_func, LOOM_OP_LOW_OP);
  ASSERT_NE(low_op, nullptr);
  const loom_op_t* low_return = FindFirstBodyOp(low_func, LOOM_OP_LOW_RETURN);
  ASSERT_NE(low_return, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_resolved_target_t target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module, low_func, &registry, collector.emitter(), &target));
  ASSERT_NE(target.descriptor_set, nullptr);

  loom_low_resolved_descriptor_packet_t packet = {};
  IREE_ASSERT_OK(
      loom_low_resolve_descriptor_packet(module, &target, low_op, &packet));
  EXPECT_EQ(packet.op, low_op);
  EXPECT_EQ(packet.kind, LOOM_LOW_DESCRIPTOR_PACKET_OP);
  EXPECT_EQ(ToString(packet.key), "iree.vm.add.i32");
  EXPECT_EQ(packet.key_attr_index, loom_low_op_opcode_ATTR_INDEX);
  ASSERT_NE(packet.descriptor, nullptr);
  EXPECT_EQ(packet.stable_id, packet.descriptor->stable_id);
  EXPECT_EQ(packet.descriptor,
            loom_low_descriptor_set_descriptor_at(target.descriptor_set,
                                                  packet.descriptor_ordinal));

  IREE_ASSERT_OK(
      loom_low_resolve_descriptor_packet(module, &target, low_return, &packet));
  EXPECT_EQ(packet.op, low_return);
  EXPECT_EQ(packet.kind, LOOM_LOW_DESCRIPTOR_PACKET_NONE);
  EXPECT_EQ(packet.descriptor, nullptr);
  EXPECT_EQ(packet.descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);

  loom_module_free(module);
}

TEST_F(LowTargetBindingTest, ResolvesMissingDescriptorPacketAsUnresolved) {
  std::string source = WithValidVmTargetRecords(
      "target.bundle @vm_target {snapshot = @vm_snapshot, export_plan = "
      "@vm_export, config = @vm_config}\n"
      "low.func.def target(@vm_target) @add(%lhs : reg<vm.i32>, %rhs : "
      "reg<vm.i32>) -> (reg<vm.i32>) {\n"
      "  %sum = low.op<iree.vm.missing.i32>(%lhs, %rhs) : (reg<vm.i32>, "
      "reg<vm.i32>) -> reg<vm.i32>\n"
      "  low.return %sum : reg<vm.i32>\n"
      "}\n");
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);
  const loom_op_t* low_func = FindFirstOp(module, LOOM_OP_LOW_FUNC_DEF);
  ASSERT_NE(low_func, nullptr);
  const loom_op_t* low_op = FindFirstBodyOp(low_func, LOOM_OP_LOW_OP);
  ASSERT_NE(low_op, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_resolved_target_t target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module, low_func, &registry, collector.emitter(), &target));
  ASSERT_NE(target.descriptor_set, nullptr);

  loom_low_resolved_descriptor_packet_t packet = {};
  IREE_ASSERT_OK(
      loom_low_resolve_descriptor_packet(module, &target, low_op, &packet));
  EXPECT_EQ(packet.kind, LOOM_LOW_DESCRIPTOR_PACKET_OP);
  EXPECT_EQ(ToString(packet.key), "iree.vm.missing.i32");
  EXPECT_EQ(packet.descriptor, nullptr);
  EXPECT_EQ(packet.descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);

  loom_module_free(module);
}

TEST_F(LowTargetBindingTest, RejectsTargetRecordThatIsNotBundle) {
  const char source[] =
      "target.config @vm_config {contract_set_key = \"iree.vm.core\", "
      "contract_feature_bits = 0}\n"
      "low.func.def target(@vm_config) @add() {\n"
      "  low.return\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);
  const loom_op_t* low_func = FindFirstOp(module, LOOM_OP_LOW_FUNC_DEF);
  ASSERT_NE(low_func, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_resolved_target_t target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module, low_func, &registry, collector.emitter(), &target));

  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3));
  ASSERT_GE(collector.emissions[0].string_params.size(), 3u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "vm_config");
  EXPECT_EQ(collector.emissions[0].string_params[2], "target bundle");
  EXPECT_EQ(target.descriptor_set, nullptr);

  loom_module_free(module);
}

TEST_F(LowTargetBindingTest, RejectsBundleSnapshotThatIsNotTargetSnapshot) {
  std::string source = WithValidVmSnapshotAndExportRecords(
      "test.record @not_snapshot {}\n"
      "target.config @vm_config {contract_set_key = \"iree.vm.core\", "
      "contract_feature_bits = 0}\n"
      "target.bundle @vm_target {snapshot = @not_snapshot, export_plan = "
      "@vm_export, config = @vm_config}\n"
      "low.func.def target(@vm_target) @add() {\n"
      "  low.return\n"
      "}\n");
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);
  const loom_op_t* low_func = FindFirstOp(module, LOOM_OP_LOW_FUNC_DEF);
  ASSERT_NE(low_func, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_resolved_target_t target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module, low_func, &registry, collector.emitter(), &target));

  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3));
  ASSERT_GE(collector.emissions[0].string_params.size(), 3u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "not_snapshot");
  EXPECT_EQ(collector.emissions[0].string_params[2], "target snapshot");
  EXPECT_EQ(target.descriptor_set, nullptr);

  loom_module_free(module);
}

TEST_F(LowTargetBindingTest, RejectsBundleConfigThatIsNotTargetConfig) {
  std::string source = WithValidVmSnapshotAndExportRecords(
      "test.record @not_config {}\n"
      "target.bundle @vm_target {snapshot = @vm_snapshot, export_plan = "
      "@vm_export, config = @not_config}\n"
      "low.func.def target(@vm_target) @add() {\n"
      "  low.return\n"
      "}\n");
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);
  const loom_op_t* low_func = FindFirstOp(module, LOOM_OP_LOW_FUNC_DEF);
  ASSERT_NE(low_func, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_resolved_target_t target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module, low_func, &registry, collector.emitter(), &target));

  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3));
  ASSERT_GE(collector.emissions[0].string_params.size(), 3u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "not_config");
  EXPECT_EQ(collector.emissions[0].string_params[2], "target config");
  EXPECT_EQ(target.descriptor_set, nullptr);

  loom_module_free(module);
}

TEST_F(LowTargetBindingTest, RejectsBundleExportThatIsNotTargetExport) {
  std::string source = WithValidVmTargetRecords(
      "test.record @not_export {}\n"
      "target.bundle @vm_target {snapshot = @vm_snapshot, export_plan = "
      "@not_export, config = @vm_config}\n"
      "low.func.def target(@vm_target) @add() {\n"
      "  low.return\n"
      "}\n");
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);
  const loom_op_t* low_func = FindFirstOp(module, LOOM_OP_LOW_FUNC_DEF);
  ASSERT_NE(low_func, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_resolved_target_t target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module, low_func, &registry, collector.emitter(), &target));

  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3));
  ASSERT_GE(collector.emissions[0].string_params.size(), 3u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "not_export");
  EXPECT_EQ(collector.emissions[0].string_params[2], "target export plan");
  EXPECT_EQ(target.descriptor_set, nullptr);

  loom_module_free(module);
}

TEST_F(LowTargetBindingTest, RejectsMissingBundleConfig) {
  std::string source = WithValidVmSnapshotAndExportRecords(
      "target.bundle @vm_target {snapshot = @vm_snapshot, export_plan = "
      "@vm_export, config = @missing_config}\n"
      "low.func.def target(@vm_target) @add() {\n"
      "  low.return\n"
      "}\n");
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);
  const loom_op_t* low_func = FindFirstOp(module, LOOM_OP_LOW_FUNC_DEF);
  ASSERT_NE(low_func, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_resolved_target_t target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module, low_func, &registry, collector.emitter(), &target));

  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 2));
  ASSERT_GE(collector.emissions[0].string_params.size(), 1u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "missing_config");
  ASSERT_FALSE(collector.emissions[0].field_refs.empty());
  EXPECT_EQ(collector.emissions[0].field_refs[0].kind,
            LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE);
  EXPECT_EQ(collector.emissions[0].field_refs[0].index,
            loom_target_bundle_config_ATTR_INDEX);
  EXPECT_EQ(target.descriptor_set, nullptr);

  loom_module_free(module);
}

TEST_F(LowTargetBindingTest, RejectsMissingDescriptorSet) {
  std::string source = WithValidVmSnapshotAndExportRecords(
      "target.config @vm_config {contract_set_key = \"missing.vm\", "
      "contract_feature_bits = 0}\n"
      "target.bundle @vm_target {snapshot = @vm_snapshot, export_plan = "
      "@vm_export, config = @vm_config}\n"
      "low.func.def target(@vm_target) @add() {\n"
      "  low.return\n"
      "}\n");
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);
  const loom_op_t* low_func = FindFirstOp(module, LOOM_OP_LOW_FUNC_DEF);
  ASSERT_NE(low_func, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_resolved_target_t target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module, low_func, &registry, collector.emitter(), &target));

  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 3));
  ASSERT_GE(collector.emissions[0].string_params.size(), 3u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "add");
  EXPECT_EQ(collector.emissions[0].string_params[1], "vm_target");
  EXPECT_EQ(collector.emissions[0].string_params[2], "missing.vm");
  EXPECT_EQ(collector.emissions[0].related_count, 1u);
  EXPECT_EQ(target.descriptor_set, nullptr);

  loom_module_free(module);
}

TEST_F(LowTargetBindingTest, RejectsNegativeFeatureBits) {
  std::string source = WithValidVmSnapshotAndExportRecords(
      "target.config @vm_config {contract_set_key = \"iree.vm.core\", "
      "contract_feature_bits = -1}\n"
      "target.bundle @vm_target {snapshot = @vm_snapshot, export_plan = "
      "@vm_export, config = @vm_config}\n"
      "low.func.def target(@vm_target) @add() {\n"
      "  low.return\n"
      "}\n");
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);
  const loom_op_t* low_func = FindFirstOp(module, LOOM_OP_LOW_FUNC_DEF);
  ASSERT_NE(low_func, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistry();
  EmissionCollector collector;
  loom_low_resolved_target_t target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module, low_func, &registry, collector.emitter(), &target));

  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 14));
  EXPECT_EQ(collector.emissions[0].op->kind, LOOM_OP_TARGET_CONFIG);
  ASSERT_GE(collector.emissions[0].string_params.size(), 2u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "contract_feature_bits");
  EXPECT_EQ(collector.emissions[0].string_params[1],
            "a non-negative feature bitset");
  ASSERT_FALSE(collector.emissions[0].field_refs.empty());
  EXPECT_EQ(collector.emissions[0].field_refs[0].kind,
            LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE);
  EXPECT_EQ(collector.emissions[0].field_refs[0].index,
            loom_target_config_contract_feature_bits_ATTR_INDEX);
  EXPECT_EQ(target.descriptor_set, nullptr);

  loom_module_free(module);
}

}  // namespace
}  // namespace loom
