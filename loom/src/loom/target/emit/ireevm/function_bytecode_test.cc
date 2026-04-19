// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/function_bytecode.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"
#include "loom/codegen/low/packetization.h"
#include "loom/codegen/low/schedule.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/emit/ireevm/descriptors.h"
#include "loom/target/emit/ireevm/module_archive.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

constexpr uint8_t kVmOpcodeAddI32 = 0x22;
constexpr uint8_t kVmOpcodeBranch = 0x56;
constexpr uint8_t kVmOpcodeReturn = 0x5A;
constexpr uint8_t kVmOpcodeBlock = 0x79;

static uint16_t ReadLeU16(const uint8_t* data) {
  return (uint16_t)(data[0] | (data[1] << 8));
}

static uint32_t ReadLeU32(const uint8_t* data) {
  return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
         ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

static void ReassignValue(
    std::vector<loom_low_allocation_assignment_t>* assignments,
    loom_value_id_t value_id, uint32_t location_base) {
  for (loom_low_allocation_assignment_t& assignment : *assignments) {
    if (assignment.value_id == value_id) {
      assignment.location_base = location_base;
      return;
    }
  }
  FAIL() << "missing allocation assignment for value " << value_id;
}

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
    if (instance) {
      iree_vm_instance_release(instance);
    }
  }

  iree_vm_instance_t* instance = nullptr;
};

struct VmModuleOwner {
  ~VmModuleOwner() {
    if (module) {
      iree_vm_module_release(module);
    }
  }

  iree_vm_module_t* module = nullptr;
};

struct VmContextOwner {
  ~VmContextOwner() {
    if (context) {
      iree_vm_context_release(context);
    }
  }

  iree_vm_context_t* context = nullptr;
};

struct VmListOwner {
  ~VmListOwner() {
    if (list) {
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

class IreeVmFunctionBytecodeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_testing_context_initialize_all(iree_allocator_system(),
                                                       &context_));
  }

  void TearDown() override {
    ReleaseSidecars();
    if (module_) {
      loom_module_free(module_);
      module_ = nullptr;
    }
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* ParseSource(const char* source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;

    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(
        loom_text_parse(iree_make_cstring_view(source),
                        IREE_SV("ireevm_function_bytecode_test.loom"),
                        &context_, &block_pool_, &parse_options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  const loom_op_t* FindFirstLowFunction(loom_module_t* module) {
    loom_block_t* block = loom_module_block(module);
    const loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (loom_low_func_def_isa(op)) {
        return op;
      }
    }
    return nullptr;
  }

  void BuildSidecars(const char* body,
                     loom_low_schedule_sidecar_t* out_schedule,
                     loom_low_allocation_sidecar_t* out_allocation) {
    std::string source =
        "target.snapshot @vm_snapshot {codegen_format = vm, target_triple = "
        "\"iree-vm\", data_layout = \"\", artifact_format = vm_bytecode, "
        "target_cpu = \"\", target_features = \"\", "
        "default_pointer_bitwidth = 64, index_bitwidth = 64, "
        "offset_bitwidth = 64, memory_space_generic = 0, "
        "memory_space_global = 0, memory_space_workgroup = 0, "
        "memory_space_constant = 0, memory_space_private = 0, "
        "memory_space_host = 0, memory_space_descriptor = 0}\n"
        "target.export @vm_export {export_symbol = \"add\", abi = "
        "vm_module_function, linkage = default, hal_binding_alignment = 0, "
        "hal_workgroup_size_x = 0, hal_workgroup_size_y = 0, "
        "hal_workgroup_size_z = 0, hal_flat_workgroup_size_min = 0, "
        "hal_flat_workgroup_size_max = 0, hal_buffer_resource_flags = 0}\n"
        "target.config @vm_config {contract_set_key = \"iree.vm.core\", "
        "contract_feature_bits = 0}\n"
        "target.bundle @vm_target {snapshot = @vm_snapshot, export_plan = "
        "@vm_export, config = @vm_config}\n";
    source += body;
    module_ = ParseSource(source.c_str());
    ASSERT_NE(module_, nullptr);

    const loom_op_t* low_func = FindFirstLowFunction(module_);
    ASSERT_NE(low_func, nullptr);
    static const loom_low_descriptor_set_t* descriptor_sets[] = {
        loom_ireevm_core_descriptor_set(),
    };
    registry_ = loom_low_descriptor_registry_t{
        .descriptor_sets = descriptor_sets,
        .descriptor_set_count = IREE_ARRAYSIZE(descriptor_sets),
    };

    loom_low_verify_options_t verify_options = {
        .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
        .descriptor_registry = &registry_,
        .descriptor_requirements =
            LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        .max_errors = 20,
    };
    loom_low_verify_result_t verify_result = {};
    IREE_ASSERT_OK(
        loom_low_verify_module(module_, &verify_options, &verify_result));
    EXPECT_EQ(verify_result.error_count, 0u);

    iree_arena_initialize(&block_pool_, &sidecar_arena_);
    sidecar_arena_initialized_ = true;

    loom_low_packetization_options_t packetization_options = {
        .descriptor_registry = &registry_,
    };
    loom_low_packetization_t packetization = {};
    IREE_ASSERT_OK(
        loom_low_packetize_function(module_, low_func, &packetization_options,
                                    &sidecar_arena_, &packetization));
    *out_schedule = packetization.schedule;
    *out_allocation = packetization.allocation;
  }

  void ReleaseSidecars() {
    if (sidecar_arena_initialized_) {
      iree_arena_deinitialize(&sidecar_arena_);
      sidecar_arena_initialized_ = false;
    }
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_low_descriptor_registry_t registry_ = {};
  iree_arena_allocator_t sidecar_arena_ = {};
  bool sidecar_arena_initialized_ = false;
};

TEST_F(IreeVmFunctionBytecodeTest, EmitsStraightLineI32FunctionBody) {
  loom_low_schedule_sidecar_t schedule = {};
  loom_low_allocation_sidecar_t allocation = {};
  BuildSidecars(
      "low.func.def target(@vm_target) @add(%lhs : reg<vm.i32>, %rhs : "
      "reg<vm.i32>) -> (reg<vm.i32>) {\n"
      "  %sum = low.op<iree.vm.add.i32>(%lhs, %rhs) : "
      "(reg<vm.i32>, reg<vm.i32>) -> reg<vm.i32>\n"
      "  low.return %sum : reg<vm.i32>\n"
      "}\n",
      &schedule, &allocation);

  BytecodeOwner owner;
  loom_ireevm_function_bytecode_t& bytecode = owner.bytecode;
  IREE_ASSERT_OK(loom_ireevm_emit_function_bytecode(
      &schedule, &allocation, iree_allocator_system(), &bytecode));

  ASSERT_GE(bytecode.data_length, 16u);
  EXPECT_EQ(bytecode.bytecode_length, 14u);
  EXPECT_EQ(bytecode.block_count, 1u);
  EXPECT_EQ(bytecode.i32_register_count, 2u);
  EXPECT_EQ(bytecode.ref_register_count, 0u);
  EXPECT_EQ(bytecode.data[0], kVmOpcodeBlock);
  EXPECT_EQ(bytecode.data[1], kVmOpcodeAddI32);
  EXPECT_EQ(ReadLeU16(&bytecode.data[2]), 0u);
  EXPECT_EQ(ReadLeU16(&bytecode.data[4]), 1u);
  EXPECT_EQ(ReadLeU16(&bytecode.data[6]), 0u);
  EXPECT_EQ(bytecode.data[8], kVmOpcodeReturn);
  EXPECT_EQ(bytecode.data[9], 0u);
  EXPECT_EQ(ReadLeU16(&bytecode.data[10]), 1u);
  EXPECT_EQ(ReadLeU16(&bytecode.data[12]), 0u);
}

TEST_F(IreeVmFunctionBytecodeTest, EmitsBranchRemapList) {
  loom_low_schedule_sidecar_t schedule = {};
  loom_low_allocation_sidecar_t allocation = {};
  BuildSidecars(
      "low.func.def target(@vm_target) @branch(%arg : reg<vm.i32>) -> "
      "(reg<vm.i32>) {\n"
      "  low.br ^join(%arg : reg<vm.i32>)\n"
      "^join(%result : reg<vm.i32>):\n"
      "  low.return %result : reg<vm.i32>\n"
      "}\n",
      &schedule, &allocation);

  ASSERT_GT(schedule.block_count, 0u);
  const loom_low_schedule_block_t& entry_block = schedule.blocks[0];
  ASSERT_GT(entry_block.scheduled_node_count, 0u);
  const uint32_t terminator_node_index =
      schedule.scheduled_node_indices[entry_block.scheduled_node_start +
                                      entry_block.scheduled_node_count - 1];
  const loom_op_t* terminator = schedule.nodes[terminator_node_index].op;
  ASSERT_TRUE(loom_low_br_isa(terminator));
  const loom_block_t* join_block = loom_low_br_dest(terminator);
  std::vector<loom_low_allocation_assignment_t> assignments(
      allocation.assignments,
      allocation.assignments + allocation.assignment_count);
  ReassignValue(&assignments, loom_block_arg_id(join_block, 0), 1);
  loom_low_allocation_sidecar_t remap_allocation = allocation;
  remap_allocation.assignments = assignments.data();

  BytecodeOwner owner;
  loom_ireevm_function_bytecode_t& bytecode = owner.bytecode;
  IREE_ASSERT_OK(loom_ireevm_emit_function_bytecode(
      &schedule, &remap_allocation, iree_allocator_system(), &bytecode));

  ASSERT_GE(bytecode.data_length, 16u);
  EXPECT_EQ(bytecode.block_count, 2u);
  EXPECT_EQ(bytecode.data[0], kVmOpcodeBlock);
  EXPECT_EQ(bytecode.data[1], kVmOpcodeBranch);
  EXPECT_EQ(ReadLeU32(&bytecode.data[2]), 12u);
  EXPECT_EQ(ReadLeU16(&bytecode.data[6]), 1u);
  EXPECT_EQ(ReadLeU16(&bytecode.data[8]), 0u);
  EXPECT_EQ(ReadLeU16(&bytecode.data[10]), 1u);
  EXPECT_EQ(bytecode.data[12], kVmOpcodeBlock);
  EXPECT_EQ(bytecode.data[13], kVmOpcodeReturn);
}

TEST_F(IreeVmFunctionBytecodeTest, EmitsLoadableExecutableVmModuleArchive) {
  loom_low_schedule_sidecar_t schedule = {};
  loom_low_allocation_sidecar_t allocation = {};
  BuildSidecars(
      "low.func.def target(@vm_target) @add(%lhs : reg<vm.i32>, %rhs : "
      "reg<vm.i32>) -> (reg<vm.i32>) {\n"
      "  %sum = low.op<iree.vm.add.i32>(%lhs, %rhs) : "
      "(reg<vm.i32>, reg<vm.i32>) -> reg<vm.i32>\n"
      "  low.return %sum : reg<vm.i32>\n"
      "}\n",
      &schedule, &allocation);

  BytecodeOwner bytecode_owner;
  loom_ireevm_function_bytecode_t& bytecode = bytecode_owner.bytecode;
  IREE_ASSERT_OK(loom_ireevm_emit_function_bytecode(
      &schedule, &allocation, iree_allocator_system(), &bytecode));

  const loom_ireevm_module_archive_function_t functions[] = {
      {
          .export_name = IREE_SV("add"),
          .calling_convention = IREE_SV("0ii_i"),
          .bytecode = &bytecode,
      },
      {
          .export_name = IREE_SV("add_alt"),
          .calling_convention = IREE_SV("0ii_i"),
          .bytecode = &bytecode,
      },
  };
  ModuleArchiveOwner archive_owner;
  loom_ireevm_module_archive_t& archive = archive_owner.archive;
  IREE_ASSERT_OK(loom_ireevm_emit_module_archive(
      IREE_SV("loom_test"), functions, IREE_ARRAYSIZE(functions),
      iree_allocator_system(), &archive));
  ASSERT_GT(archive.data_length, 16u);

  VmInstanceOwner instance_owner;
  IREE_ASSERT_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                         iree_allocator_system(),
                                         &instance_owner.instance));
  VmModuleOwner module_owner;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      instance_owner.instance, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
      iree_make_const_byte_span(archive.data, archive.data_length),
      iree_allocator_null(), iree_allocator_system(), &module_owner.module));

  StringBuilderOwner disassembly;
  IREE_ASSERT_OK(iree_vm_bytecode_module_disassemble_function(
      module_owner.module, 1, &disassembly.builder));
  const std::string disassembly_string(disassembly.builder.buffer,
                                       disassembly.builder.size);
  EXPECT_NE(disassembly_string.find("vm.add.i32"), std::string::npos);
  EXPECT_NE(disassembly_string.find("vm.return"), std::string::npos);

  VmContextOwner context_owner;
  iree_vm_module_t* modules[] = {module_owner.module};
  IREE_ASSERT_OK(iree_vm_context_create_with_modules(
      instance_owner.instance, IREE_VM_CONTEXT_FLAG_NONE,
      IREE_ARRAYSIZE(modules), modules, iree_allocator_system(),
      &context_owner.context));

  iree_vm_function_t function = {};
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      module_owner.module, IREE_VM_FUNCTION_LINKAGE_EXPORT, IREE_SV("add_alt"),
      &function));

  VmListOwner input_list;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                                     iree_allocator_system(),
                                     &input_list.list));
  IREE_ASSERT_OK(iree_vm_list_resize(input_list.list, 2));
  iree_vm_value_t lhs = iree_vm_value_make_i32(20);
  iree_vm_value_t rhs = iree_vm_value_make_i32(22);
  IREE_ASSERT_OK(iree_vm_list_set_value(input_list.list, 0, &lhs));
  IREE_ASSERT_OK(iree_vm_list_set_value(input_list.list, 1, &rhs));

  VmListOwner output_list;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                     iree_allocator_system(),
                                     &output_list.list));
  IREE_ASSERT_OK(iree_vm_invoke(
      context_owner.context, function, IREE_VM_INVOCATION_FLAG_NONE, nullptr,
      input_list.list, output_list.list, iree_allocator_system()));
  ASSERT_EQ(iree_vm_list_size(output_list.list), 1u);
  iree_vm_value_t result = {};
  IREE_ASSERT_OK(iree_vm_list_get_value(output_list.list, 0, &result));
  EXPECT_EQ(result.type, IREE_VM_VALUE_TYPE_I32);
  EXPECT_EQ(result.i32, 42);
}

}  // namespace
}  // namespace loom
