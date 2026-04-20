// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/test/low_registry.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

class LowAllocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_testing_context_initialize_all(iree_allocator_system(),
                                                       &context_));
    loom_test_low_descriptor_registry_initialize(&target_registry_);
  }

  void TearDown() override {
    if (sidecar_arena_initialized_) {
      iree_arena_deinitialize(&sidecar_arena_);
      sidecar_arena_initialized_ = false;
    }
    if (module_) {
      loom_module_free(module_);
      module_ = nullptr;
    }
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* ParseSource(const std::string& source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;

    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("low_allocation_test.loom"), &context_,
                        &block_pool_, &parse_options, &module));
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

  void ParseAndVerify(const char* body) {
    std::string source =
        "target.snapshot @test_snapshot {codegen_format = low_native, "
        "target_triple = \"test-low\", data_layout = \"\", artifact_format = "
        "elf, target_cpu = \"test-low\", target_features = \"\", "
        "default_pointer_bitwidth = 64, index_bitwidth = 64, "
        "offset_bitwidth = 64, memory_space_generic = 0, "
        "memory_space_global = 0, memory_space_workgroup = 0, "
        "memory_space_constant = 0, memory_space_private = 0, "
        "memory_space_host = 0, memory_space_descriptor = 0}\n"
        "target.export @test_export {export_symbol = \"allocated\", abi = "
        "object_function, linkage = default, hal_binding_alignment = 0, "
        "hal_workgroup_size_x = 0, hal_workgroup_size_y = 0, "
        "hal_workgroup_size_z = 0, hal_flat_workgroup_size_min = 0, "
        "hal_flat_workgroup_size_max = 0, hal_buffer_resource_flags = 0}\n"
        "target.config @test_config {contract_set_key = \"test.low.core\", "
        "contract_feature_bits = 0}\n"
        "target.bundle @test_target {snapshot = @test_snapshot, export_plan = "
        "@test_export, config = @test_config}\n";
    source += body;
    module_ = ParseSource(source);
    ASSERT_NE(module_, nullptr);

    loom_low_verify_options_t verify_options = {
        .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
        .descriptor_registry = &target_registry_.registry,
        .descriptor_requirements =
            LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        .max_errors = 20,
    };
    loom_low_verify_result_t verify_result = {};
    IREE_ASSERT_OK(
        loom_low_verify_module(module_, &verify_options, &verify_result));
    EXPECT_EQ(verify_result.error_count, 0u);
  }

  iree_status_t AllocateFirstLowFunctionWithOptions(
      const loom_low_allocation_options_t* options,
      loom_low_allocation_sidecar_t* out_allocation) {
    const loom_op_t* low_function = FindFirstLowFunction(module_);
    EXPECT_NE(low_function, nullptr);
    EXPECT_FALSE(sidecar_arena_initialized_);

    iree_arena_initialize(&block_pool_, &sidecar_arena_);
    sidecar_arena_initialized_ = true;
    return loom_low_allocate_function(module_, low_function, options,
                                      &sidecar_arena_, out_allocation);
  }

  loom_low_allocation_sidecar_t AllocateFirstLowFunction() {
    loom_low_allocation_options_t options = {
        .descriptor_registry = &target_registry_.registry,
    };
    loom_low_allocation_sidecar_t allocation = {};
    IREE_EXPECT_OK(AllocateFirstLowFunctionWithOptions(&options, &allocation));
    IREE_EXPECT_OK(loom_low_allocation_verify_sidecar(&allocation));
    return allocation;
  }

  const loom_low_allocation_assignment_t* FindAssignmentByName(
      const loom_low_allocation_sidecar_t& allocation, const char* name) {
    iree_string_view_t expected_name = iree_make_cstring_view(name);
    for (iree_host_size_t i = 0; i < allocation.assignment_count; ++i) {
      const loom_low_allocation_assignment_t* assignment =
          &allocation.assignments[i];
      const loom_value_t* value =
          loom_module_value(module_, assignment->value_id);
      if (value->name_id == LOOM_STRING_ID_INVALID ||
          value->name_id >= module_->strings.count) {
        continue;
      }
      iree_string_view_t value_name = module_->strings.entries[value->name_id];
      if (iree_string_view_equal(value_name, expected_name)) {
        return assignment;
      }
    }
    return nullptr;
  }

  loom_value_id_t FindValueIdByName(const char* name) {
    iree_string_view_t expected_name = iree_make_cstring_view(name);
    for (iree_host_size_t i = 0; i < module_->values.count; ++i) {
      const loom_value_t* value =
          loom_module_value(module_, (loom_value_id_t)i);
      if (value->name_id == LOOM_STRING_ID_INVALID ||
          value->name_id >= module_->strings.count) {
        continue;
      }
      iree_string_view_t value_name = module_->strings.entries[value->name_id];
      if (iree_string_view_equal(value_name, expected_name)) {
        return (loom_value_id_t)i;
      }
    }
    return LOOM_VALUE_ID_INVALID;
  }

  void ExpectSameLocation(const loom_low_allocation_assignment_t* lhs,
                          const loom_low_allocation_assignment_t* rhs) {
    ASSERT_NE(lhs, nullptr);
    ASSERT_NE(rhs, nullptr);
    EXPECT_EQ(lhs->value_class.type_kind, rhs->value_class.type_kind);
    EXPECT_EQ(lhs->value_class.register_class_id,
              rhs->value_class.register_class_id);
    EXPECT_EQ(lhs->location_kind, rhs->location_kind);
    EXPECT_EQ(lhs->location_base, rhs->location_base);
    EXPECT_EQ(lhs->location_count, rhs->location_count);
  }

  void ExpectPhysicalLocation(
      const loom_low_allocation_assignment_t* assignment,
      uint32_t location_base, uint32_t location_count) {
    ASSERT_NE(assignment, nullptr);
    EXPECT_EQ(assignment->location_kind,
              LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
    EXPECT_EQ(assignment->location_base, location_base);
    EXPECT_EQ(assignment->location_count, location_count);
  }

  bool AssignmentOverlapsPhysicalLocation(
      const loom_low_allocation_assignment_t* assignment,
      uint32_t location_base, uint32_t location_count) {
    if (assignment->location_kind !=
        LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
      return false;
    }
    uint64_t assignment_end =
        (uint64_t)assignment->location_base + assignment->location_count;
    uint64_t location_end = (uint64_t)location_base + location_count;
    return assignment->location_base < location_end &&
           location_base < assignment_end;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t target_registry_ = {};
  iree_arena_allocator_t sidecar_arena_ = {};
  bool sidecar_arena_initialized_ = false;
};

TEST_F(LowAllocationTest, CoalescesTiedResultWithOperand) {
  ParseAndVerify(
      "low.func.def target(@test_target) @allocated(%acc : reg<test.i32>) -> "
      "(reg<test.i32>) {\n"
      "  %out = low.op<test.tied.any>(%acc) : (reg<test.i32>) -> %acc as "
      "reg<test.i32>\n"
      "  low.return %out : reg<test.i32>\n"
      "}\n");
  loom_low_allocation_sidecar_t allocation = AllocateFirstLowFunction();

  ExpectSameLocation(FindAssignmentByName(allocation, "acc"),
                     FindAssignmentByName(allocation, "out"));
}

TEST_F(LowAllocationTest, RejectsTiedResultWhenOperandSpills) {
  ParseAndVerify(
      "low.func.def target(@test_target) @allocated(%acc : reg<test.i32>) -> "
      "(reg<test.i32>) {\n"
      "  %out = low.op<test.tied.any>(%acc) : (reg<test.i32>) -> %acc as "
      "reg<test.i32>\n"
      "  low.return %out : reg<test.i32>\n"
      "}\n");
  const loom_low_allocation_budget_t budget = {
      .register_class = IREE_SV("test.i32"),
      .max_units = 0,
  };
  loom_low_allocation_options_t options = {
      .descriptor_registry = &target_registry_.registry,
      .budgets = &budget,
      .budget_count = 1,
  };
  loom_low_allocation_sidecar_t allocation = {};
  iree_status_t status =
      AllocateFirstLowFunctionWithOptions(&options, &allocation);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, status);
}

TEST_F(LowAllocationTest, FixedValueLocationIsReusableAfterLastUse) {
  ParseAndVerify(
      "low.func.def target(@test_target) @allocated(%fixed : reg<test.phys>, "
      "%rhs : reg<test.phys>) -> (reg<test.phys>) {\n"
      "  %dead = low.op<test.add.phys>(%fixed, %rhs) : (reg<test.phys>, "
      "reg<test.phys>) -> reg<test.phys>\n"
      "  %later = low.op<test.add.phys>(%rhs, %rhs) : (reg<test.phys>, "
      "reg<test.phys>) -> reg<test.phys>\n"
      "  low.return %later : reg<test.phys>\n"
      "}\n");
  loom_value_id_t fixed_value_id = FindValueIdByName("fixed");
  ASSERT_NE(fixed_value_id, LOOM_VALUE_ID_INVALID);
  const loom_low_allocation_fixed_value_t fixed_value = {
      .value_id = fixed_value_id,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = 0,
      .location_count = 1,
  };
  loom_low_allocation_options_t options = {
      .descriptor_registry = &target_registry_.registry,
      .fixed_values = &fixed_value,
      .fixed_value_count = 1,
  };
  loom_low_allocation_sidecar_t allocation = {};
  IREE_ASSERT_OK(AllocateFirstLowFunctionWithOptions(&options, &allocation));
  IREE_ASSERT_OK(loom_low_allocation_verify_sidecar(&allocation));

  ExpectPhysicalLocation(FindAssignmentByName(allocation, "fixed"), 0, 1);
  ExpectPhysicalLocation(FindAssignmentByName(allocation, "later"), 0, 1);
}

TEST_F(LowAllocationTest, FutureFixedValueBlocksOverlappingOrdinaryValue) {
  ParseAndVerify(
      "low.func.def target(@test_target) @allocated(%lhs : reg<test.phys>, "
      "%rhs : reg<test.phys>) -> (reg<test.phys>, reg<test.phys>) {\n"
      "  %late = low.op<test.add.phys>(%rhs, %rhs) : (reg<test.phys>, "
      "reg<test.phys>) -> reg<test.phys>\n"
      "  low.return %lhs, %late : reg<test.phys>, reg<test.phys>\n"
      "}\n");
  loom_value_id_t late_value_id = FindValueIdByName("late");
  ASSERT_NE(late_value_id, LOOM_VALUE_ID_INVALID);
  const loom_low_allocation_fixed_value_t fixed_value = {
      .value_id = late_value_id,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = 0,
      .location_count = 1,
  };
  loom_low_allocation_options_t options = {
      .descriptor_registry = &target_registry_.registry,
      .fixed_values = &fixed_value,
      .fixed_value_count = 1,
  };
  loom_low_allocation_sidecar_t allocation = {};
  IREE_ASSERT_OK(AllocateFirstLowFunctionWithOptions(&options, &allocation));
  IREE_ASSERT_OK(loom_low_allocation_verify_sidecar(&allocation));

  ExpectPhysicalLocation(FindAssignmentByName(allocation, "late"), 0, 1);
  const loom_low_allocation_assignment_t* lhs_assignment =
      FindAssignmentByName(allocation, "lhs");
  ASSERT_NE(lhs_assignment, nullptr);
  EXPECT_FALSE(AssignmentOverlapsPhysicalLocation(lhs_assignment, 0, 1));
}

TEST_F(LowAllocationTest, ReservedRangeBlocksWholeFunction) {
  ParseAndVerify(
      "low.func.def target(@test_target) @allocated(%lhs : reg<test.phys>, "
      "%rhs : reg<test.phys>) -> (reg<test.phys>) {\n"
      "  %sum = low.op<test.add.phys>(%lhs, %rhs) : (reg<test.phys>, "
      "reg<test.phys>) -> reg<test.phys>\n"
      "  low.return %sum : reg<test.phys>\n"
      "}\n");
  const loom_low_allocation_reserved_range_t reserved_range = {
      .register_class = IREE_SV("test.phys"),
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = 0,
      .location_count = 1,
  };
  loom_low_allocation_options_t options = {
      .descriptor_registry = &target_registry_.registry,
      .reserved_ranges = &reserved_range,
      .reserved_range_count = 1,
  };
  loom_low_allocation_sidecar_t allocation = {};
  IREE_ASSERT_OK(AllocateFirstLowFunctionWithOptions(&options, &allocation));
  IREE_ASSERT_OK(loom_low_allocation_verify_sidecar(&allocation));

  for (iree_host_size_t i = 0; i < allocation.assignment_count; ++i) {
    EXPECT_FALSE(
        AssignmentOverlapsPhysicalLocation(&allocation.assignments[i], 0, 1));
  }
}

TEST_F(LowAllocationTest, RejectsOverlappingReservedRanges) {
  ParseAndVerify(
      "low.func.def target(@test_target) @allocated(%lhs : reg<test.phys>, "
      "%rhs : reg<test.phys>) -> (reg<test.phys>) {\n"
      "  %sum = low.op<test.add.phys>(%lhs, %rhs) : (reg<test.phys>, "
      "reg<test.phys>) -> reg<test.phys>\n"
      "  low.return %sum : reg<test.phys>\n"
      "}\n");
  const loom_low_allocation_reserved_range_t reserved_ranges[] = {
      {
          .register_class = IREE_SV("test.phys"),
          .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          .location_base = 0,
          .location_count = 2,
      },
      {
          .register_class = IREE_SV("test.phys"),
          .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          .location_base = 1,
          .location_count = 1,
      },
  };
  loom_low_allocation_options_t options = {
      .descriptor_registry = &target_registry_.registry,
      .reserved_ranges = reserved_ranges,
      .reserved_range_count = IREE_ARRAYSIZE(reserved_ranges),
  };
  loom_low_allocation_sidecar_t allocation = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      AllocateFirstLowFunctionWithOptions(&options, &allocation));
}

}  // namespace
}  // namespace loom
