// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/target_binding.h"

#include <stdint.h>

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/descriptors_verify.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/emit/ireevm/descriptors.h"
#include "loom/testing/diagnostic_matchers.h"

namespace loom {
namespace {

static std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

using EmissionCollector = ::loom::testing::DiagnosticEmissionCapture;

class LowTargetBindingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
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

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
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

  static loom_low_descriptor_registry_t IreeVmRegistryWithPreset() {
    static const loom_target_snapshot_t kVmSnapshot = {
        .name = IREE_SVL("test.iree-vm"),
        .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_VM,
        .target_triple = IREE_SVL("iree-vm"),
        .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE,
        .target_cpu = IREE_SVL("iree-vm"),
        .default_pointer_bitwidth = 64,
        .index_bitwidth = 64,
        .offset_bitwidth = 64,
        .memory_spaces =
            {
                .generic = 0,
                .global = 0,
                .workgroup = UINT32_MAX,
                .constant = 0,
                .private_memory = 0,
                .host = 0,
                .descriptor = UINT32_MAX,
            },
    };
    static const loom_target_export_plan_t kVmExportPlan = {
        .name = IREE_SVL("test.iree-vm"),
        .abi_kind = LOOM_TARGET_ABI_VM_MODULE_FUNCTION,
        .linkage = LOOM_TARGET_LINKAGE_DEFAULT,
    };
    static const loom_target_config_t kVmConfig = {
        .name = IREE_SVL("test.iree-vm"),
        .contract_set_key = IREE_SVL("iree.vm.core"),
    };
    static const loom_target_bundle_t kVmBundle = {
        .name = IREE_SVL("test.iree-vm"),
        .snapshot = &kVmSnapshot,
        .export_plan = &kVmExportPlan,
        .config = &kVmConfig,
    };
    static const loom_low_descriptor_set_t* descriptor_sets[] = {
        loom_ireevm_core_descriptor_set(),
    };
    static const loom_target_bundle_t* target_bundles[] = {
        &kVmBundle,
    };
    return loom_low_descriptor_registry_t{
        .descriptor_sets = descriptor_sets,
        .descriptor_set_count = IREE_ARRAYSIZE(descriptor_sets),
        .target_bundles = target_bundles,
        .target_bundle_count = IREE_ARRAYSIZE(target_bundles),
    };
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

static std::string ValidVmLowFunction() {
  return std::string(
      "target.profile @vm_target preset(\"test.iree-vm\")\n"
      "low.func.def target(@vm_target) @add(%lhs: reg<vm.i32>, %rhs: "
      "reg<vm.i32>) -> (reg<vm.i32>) {\n"
      "  %sum = low.op<iree.vm.add.i32>(%lhs, %rhs) : (reg<vm.i32>, "
      "reg<vm.i32>) -> reg<vm.i32>\n"
      "  low.return %sum : reg<vm.i32>\n"
      "}\n");
}

TEST_F(LowTargetBindingTest, RegistryVerifiesAndFindsDescriptorSet) {
  loom_low_descriptor_registry_t registry = IreeVmRegistryWithPreset();
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

TEST_F(LowTargetBindingTest, ResolvesProfilePresetAndDescriptorSet) {
  loom_module_t* module = ParseSource(ValidVmLowFunction());
  ASSERT_NE(module, nullptr);
  const loom_op_t* low_func = FindFirstOp(module, LOOM_OP_LOW_FUNC_DEF);
  ASSERT_NE(low_func, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistryWithPreset();
  EmissionCollector collector;
  loom_low_resolved_target_t target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module, low_func, &registry, collector.emitter(), &target));

  EXPECT_TRUE(collector.emissions.empty());
  EXPECT_EQ(target.target_op->kind, LOOM_OP_TARGET_PROFILE);
  EXPECT_EQ(target.descriptor_set, loom_ireevm_core_descriptor_set());
  EXPECT_EQ(target.bundle_storage.bundle.snapshot,
            &target.bundle_storage.snapshot);
  EXPECT_EQ(target.bundle_storage.bundle.export_plan,
            &target.bundle_storage.export_plan);
  EXPECT_EQ(target.bundle_storage.bundle.config, &target.bundle_storage.config);
  EXPECT_EQ(target.bundle_storage.snapshot.codegen_format,
            LOOM_TARGET_CODEGEN_FORMAT_VM);
  EXPECT_EQ(target.bundle_storage.export_plan.abi_kind,
            LOOM_TARGET_ABI_VM_MODULE_FUNCTION);
  EXPECT_EQ(ToString(target.bundle_storage.config.contract_set_key),
            "iree.vm.core");
  EXPECT_TRUE(iree_string_view_equal(target.target_name, IREE_SV("vm_target")));
  EXPECT_TRUE(iree_string_view_equal(target.descriptor_set_key,
                                     IREE_SV("iree.vm.core")));
  EXPECT_EQ(target.feature_bits, 0u);

  loom_module_free(module);
}

TEST_F(LowTargetBindingTest, FunctionContractOverridesPresetExportPlan) {
  const char source[] =
      "target.profile @vm_target preset(\"test.iree-vm\")\n"
      "low.func.def target(@vm_target) abi(object_function) "
      "export(\"add_export\", {linkage = dso_local}) @add() {\n"
      "  low.return\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);
  const loom_op_t* low_func = FindFirstOp(module, LOOM_OP_LOW_FUNC_DEF);
  ASSERT_NE(low_func, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistryWithPreset();
  EmissionCollector collector;
  loom_low_resolved_target_t target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module, low_func, &registry, collector.emitter(), &target));

  EXPECT_TRUE(collector.emissions.empty());
  EXPECT_EQ(target.bundle_storage.export_plan.abi_kind,
            LOOM_TARGET_ABI_OBJECT_FUNCTION);
  EXPECT_EQ(target.bundle_storage.export_plan.linkage,
            LOOM_TARGET_LINKAGE_DSO_LOCAL);
  EXPECT_EQ(ToString(target.bundle_storage.export_plan.export_symbol),
            "add_export");

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

  loom_low_descriptor_registry_t registry = IreeVmRegistryWithPreset();
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
  std::string source =
      "target.profile @vm_target preset(\"test.iree-vm\")\n"
      "low.func.def target(@vm_target) @add(%lhs: reg<vm.i32>, %rhs: "
      "reg<vm.i32>) -> (reg<vm.i32>) {\n"
      "  %sum = low.op<iree.vm.missing.i32>(%lhs, %rhs) : (reg<vm.i32>, "
      "reg<vm.i32>) -> reg<vm.i32>\n"
      "  low.return %sum : reg<vm.i32>\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);
  const loom_op_t* low_func = FindFirstOp(module, LOOM_OP_LOW_FUNC_DEF);
  ASSERT_NE(low_func, nullptr);
  const loom_op_t* low_op = FindFirstBodyOp(low_func, LOOM_OP_LOW_OP);
  ASSERT_NE(low_op, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistryWithPreset();
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

TEST_F(LowTargetBindingTest, RejectsTargetRecordThatIsNotProfile) {
  const char source[] =
      "test.record @not_target {}\n"
      "low.func.def target(@not_target) @add() {\n"
      "  low.return\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);
  const loom_op_t* low_func = FindFirstOp(module, LOOM_OP_LOW_FUNC_DEF);
  ASSERT_NE(low_func, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistryWithPreset();
  EmissionCollector collector;
  loom_low_resolved_target_t target = {};
  IREE_ASSERT_OK(loom_low_resolve_function_target(
      module, low_func, &registry, collector.emitter(), &target));

  ASSERT_EQ(collector.emissions.size(), 1u);
  EXPECT_EQ(collector.emissions[0].error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3));
  ASSERT_GE(collector.emissions[0].string_params.size(), 3u);
  EXPECT_EQ(collector.emissions[0].string_params[0], "not_target");
  EXPECT_EQ(collector.emissions[0].string_params[2], "target profile");
  EXPECT_EQ(target.descriptor_set, nullptr);

  loom_module_free(module);
}

TEST_F(LowTargetBindingTest, RejectsMissingDescriptorSet) {
  const char source[] =
      "target.profile @vm_target preset(\"test.iree-vm\") "
      "{contract_set_key = \"missing.vm\"}\n"
      "low.func.def target(@vm_target) @add() {\n"
      "  low.return\n"
      "}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);
  const loom_op_t* low_func = FindFirstOp(module, LOOM_OP_LOW_FUNC_DEF);
  ASSERT_NE(low_func, nullptr);

  loom_low_descriptor_registry_t registry = IreeVmRegistryWithPreset();
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

}  // namespace
}  // namespace loom
