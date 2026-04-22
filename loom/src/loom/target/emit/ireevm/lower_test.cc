// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/lower.h"

#include <memory>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"
#include "loom/codegen/low/packetization.h"
#include "loom/codegen/low/source_selection.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/emit/ireevm/function_bytecode.h"
#include "loom/target/emit/ireevm/low_registry.h"
#include "loom/target/emit/ireevm/module_archive.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

struct CollectedEmission {
  const loom_error_def_t* error = nullptr;
  const loom_op_t* op = nullptr;
  std::vector<std::string> string_params;
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
      if (param->kind == LOOM_PARAM_STRING) {
        entry.string_params.push_back(CopyString(param->string));
      }
    }
    collector->emissions.push_back(std::move(entry));
    return iree_ok_status();
  }
};

struct ModuleDeleter {
  void operator()(loom_module_t* module) const { loom_module_free(module); }
};

using ModulePtr = std::unique_ptr<loom_module_t, ModuleDeleter>;

struct BytecodeOwner {
  ~BytecodeOwner() {
    loom_ireevm_function_bytecode_deinitialize(&bytecode,
                                               iree_allocator_system());
  }

  loom_ireevm_function_bytecode_t bytecode = {};
};

struct ModuleArchiveOwner {
  ~ModuleArchiveOwner() {
    loom_ireevm_module_archive_deinitialize(&archive, iree_allocator_system());
  }

  loom_ireevm_module_archive_t archive = {};
};

struct VmInstanceOwner {
  ~VmInstanceOwner() {
    if (instance != nullptr) {
      iree_vm_instance_release(instance);
    }
  }

  iree_vm_instance_t* instance = nullptr;
};

struct VmModuleOwner {
  ~VmModuleOwner() {
    if (module != nullptr) {
      iree_vm_module_release(module);
    }
  }

  iree_vm_module_t* module = nullptr;
};

struct VmContextOwner {
  ~VmContextOwner() {
    if (context != nullptr) {
      iree_vm_context_release(context);
    }
  }

  iree_vm_context_t* context = nullptr;
};

struct VmListOwner {
  ~VmListOwner() {
    if (list != nullptr) {
      iree_vm_list_release(list);
    }
  }

  iree_vm_list_t* list = nullptr;
};

struct StringBuilderOwner {
  StringBuilderOwner() {
    iree_string_builder_initialize(iree_allocator_system(), &builder);
  }

  ~StringBuilderOwner() { iree_string_builder_deinitialize(&builder); }

  iree_string_builder_t builder = {};
};

class IreeVmLowerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_testing_context_initialize_all(iree_allocator_system(),
                                                       &context_));
    loom_ireevm_low_descriptor_registry_initialize(&target_registry_);
  }

  void TearDown() override {
    ReleaseSidecars();
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseSource(const char* source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("ireevm_lower_test.loom"), &context_,
                                  &block_pool_, &parse_options, &module));
    IREE_ASSERT(module != nullptr);
    return ModulePtr(module);
  }

  ModulePtr ParseAndLowerTargetedSource(const char* source,
                                        EmissionCollector* collector,
                                        loom_low_lower_result_t* out_result) {
    ModulePtr module = ParseSource(source);

    loom_low_lower_policy_registry_t policy_registry = {};
    loom_ireevm_low_lower_policy_registry_initialize(&policy_registry);
    iree_arena_allocator_t selection_arena;
    iree_arena_initialize(module->arena.block_pool, &selection_arena);
    const loom_low_source_selection_options_t selection_options = {
        .descriptor_registry = &target_registry_.registry,
        .policy_registry = &policy_registry,
        .lowering_kind = IREE_SVL("IREE VM source-to-low"),
    };
    loom_low_source_selection_t selection = {};
    IREE_CHECK_OK(loom_low_select_source_func(module.get(), &selection_options,
                                              &selection_arena, &selection));
    const loom_low_lower_options_t options = {
        .target_ref = selection.target_ref,
        .bundle = selection.target_bundle,
        .descriptor_registry = &target_registry_.registry,
        .descriptor_requirements =
            LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        .policy = selection.policy,
        .emitter = collector->emitter(),
        .max_errors = 20,
    };
    IREE_CHECK_OK(loom_low_lower_function(module.get(), selection.func,
                                          &options, out_result));
    iree_arena_deinitialize(&selection_arena);
    return module;
  }

  void VerifyLowModule(loom_module_t* module) {
    EmissionCollector collector;
    const loom_low_verify_options_t options = {
        .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
        .descriptor_registry = &target_registry_.registry,
        .descriptor_requirements =
            LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        .emitter = collector.emitter(),
        .max_errors = 20,
    };
    loom_low_verify_result_t result = {};
    IREE_ASSERT_OK(loom_low_verify_module(module, &options, &result));
    EXPECT_EQ(result.error_count, 0u);
    EXPECT_TRUE(collector.emissions.empty());
  }

  void Packetize(loom_module_t* module, loom_op_t* low_function,
                 loom_low_packetization_t* out_packetization) {
    ReleaseSidecars();
    iree_arena_initialize(&block_pool_, &sidecar_arena_);
    sidecar_arena_initialized_ = true;

    loom_low_packetization_options_t options = {
        .descriptor_registry = &target_registry_.registry,
    };
    IREE_ASSERT_OK(loom_low_packetize_function(
        module, low_function, &options, &sidecar_arena_, out_packetization));
  }

  void ReleaseSidecars() {
    if (sidecar_arena_initialized_) {
      iree_arena_deinitialize(&sidecar_arena_);
      sidecar_arena_initialized_ = false;
    }
  }

  iree_status_t InvokeI32I32ToI32(iree_vm_context_t* context,
                                  iree_vm_function_t function, int32_t lhs,
                                  int32_t rhs, int32_t* out_result) {
    VmListOwner input_list;
    IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                             2, iree_allocator_system(),
                                             &input_list.list));
    IREE_RETURN_IF_ERROR(iree_vm_list_resize(input_list.list, 2));
    iree_vm_value_t lhs_value = iree_vm_value_make_i32(lhs);
    iree_vm_value_t rhs_value = iree_vm_value_make_i32(rhs);
    IREE_RETURN_IF_ERROR(
        iree_vm_list_set_value(input_list.list, 0, &lhs_value));
    IREE_RETURN_IF_ERROR(
        iree_vm_list_set_value(input_list.list, 1, &rhs_value));

    VmListOwner output_list;
    IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                             1, iree_allocator_system(),
                                             &output_list.list));
    IREE_RETURN_IF_ERROR(iree_vm_invoke(
        context, function, IREE_VM_INVOCATION_FLAG_NONE, nullptr,
        input_list.list, output_list.list, iree_allocator_system()));
    if (iree_vm_list_size(output_list.list) != 1) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "VM invocation returned the wrong arity");
    }
    iree_vm_value_t result = {};
    IREE_RETURN_IF_ERROR(iree_vm_list_get_value(output_list.list, 0, &result));
    if (result.type != IREE_VM_VALUE_TYPE_I32) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "VM invocation returned the wrong value type");
    }
    *out_result = result.i32;
    return iree_ok_status();
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t target_registry_ = {};
  iree_arena_allocator_t sidecar_arena_ = {};
  bool sidecar_arena_initialized_ = false;
};

TEST_F(IreeVmLowerTest, LowersSemanticCfgFunctionAndExecutesVmArchive) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @vm_target preset(\"iree-vm\")\n"
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
      "}\n",
      &lower_collector, &lower_result);

  EXPECT_EQ(lower_result.error_count, 0u);
  EXPECT_EQ(lower_result.remark_count, 0u);
  ASSERT_NE(lower_result.low_func_op, nullptr);
  EXPECT_TRUE(lower_collector.emissions.empty());
  VerifyLowModule(module.get());

  loom_low_packetization_t packetization = {};
  Packetize(module.get(), lower_result.low_func_op, &packetization);

  BytecodeOwner bytecode_owner;
  IREE_ASSERT_OK(loom_ireevm_emit_function_bytecode(
      &packetization.schedule, &packetization.allocation,
      iree_allocator_system(), &bytecode_owner.bytecode));

  const loom_ireevm_module_archive_function_t functions[] = {
      {
          .export_name = IREE_SV("branchy"),
          .calling_convention = IREE_SV("0ii_i"),
          .bytecode = &bytecode_owner.bytecode,
      },
  };
  ModuleArchiveOwner archive_owner;
  IREE_ASSERT_OK(loom_ireevm_emit_module_archive(
      IREE_SV("loom_ireevm_lower_test"), functions, IREE_ARRAYSIZE(functions),
      iree_allocator_system(), &archive_owner.archive));

  VmInstanceOwner instance_owner;
  IREE_ASSERT_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                         iree_allocator_system(),
                                         &instance_owner.instance));
  VmModuleOwner module_owner;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      instance_owner.instance, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
      iree_make_const_byte_span(archive_owner.archive.data,
                                archive_owner.archive.data_length),
      iree_allocator_null(), iree_allocator_system(), &module_owner.module));

  iree_vm_function_t function = {};
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      module_owner.module, IREE_VM_FUNCTION_LINKAGE_EXPORT, IREE_SV("branchy"),
      &function));

  StringBuilderOwner disassembly;
  IREE_ASSERT_OK(iree_vm_bytecode_module_disassemble_function(
      module_owner.module, function.ordinal, &disassembly.builder));
  const std::string disassembly_string(
      iree_string_builder_buffer(&disassembly.builder),
      iree_string_builder_size(&disassembly.builder));
  SCOPED_TRACE(disassembly_string);
  EXPECT_NE(disassembly_string.find("vm.cmp.eq.i32"), std::string::npos);
  EXPECT_NE(disassembly_string.find("vm.cond_br"), std::string::npos);
  EXPECT_NE(disassembly_string.find("vm.add.i32"), std::string::npos);
  EXPECT_NE(disassembly_string.find("vm.sub.i32"), std::string::npos);

  VmContextOwner context_owner;
  iree_vm_module_t* modules[] = {module_owner.module};
  IREE_ASSERT_OK(iree_vm_context_create_with_modules(
      instance_owner.instance, IREE_VM_CONTEXT_FLAG_NONE,
      IREE_ARRAYSIZE(modules), modules, iree_allocator_system(),
      &context_owner.context));

  int32_t result = 0;
  IREE_ASSERT_OK(
      InvokeI32I32ToI32(context_owner.context, function, 0, 21, &result));
  EXPECT_EQ(result, 42);
  IREE_ASSERT_OK(
      InvokeI32I32ToI32(context_owner.context, function, 50, 8, &result));
  EXPECT_EQ(result, 42);
}

TEST_F(IreeVmLowerTest, UnsupportedSourceOpEmitsDiagnosticAndNoLowFunction) {
  EmissionCollector lower_collector;
  loom_low_lower_result_t lower_result = {};
  ModulePtr module = ParseAndLowerTargetedSource(
      "target.profile @vm_target preset(\"iree-vm\")\n"
      "func.def target(@vm_target) @mul(%lhs: i32, %rhs: i32) -> (i32) {\n"
      "  %product = scalar.muli %lhs, %rhs : i32\n"
      "  func.return %product : i32\n"
      "}\n",
      &lower_collector, &lower_result);

  EXPECT_EQ(lower_result.error_count, 1u);
  EXPECT_EQ(lower_result.remark_count, 0u);
  EXPECT_EQ(lower_result.low_func_op, nullptr);
  EXPECT_FALSE(loom_symbol_ref_is_valid(lower_result.low_func_ref));
  ASSERT_EQ(lower_collector.emissions.size(), 1u);
  const CollectedEmission& emission = lower_collector.emissions[0];
  EXPECT_EQ(emission.error,
            loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 1));
  ASSERT_EQ(emission.string_params.size(), 7u);
  EXPECT_EQ(emission.string_params[0], "vm_target");
  EXPECT_EQ(emission.string_params[3], "mul");
  EXPECT_EQ(emission.string_params[4], "op");
  EXPECT_EQ(emission.string_params[5], "scalar.muli");
  EXPECT_NE(emission.string_params[6].find("descriptor mapping"),
            std::string::npos);

  loom_string_id_t low_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module.get(), IREE_SV("mul__low"),
                                           &low_name_id));
  EXPECT_EQ(loom_module_find_symbol(module.get(), low_name_id),
            LOOM_SYMBOL_ID_INVALID);
}

}  // namespace
}  // namespace loom
