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
    std::string source = "target.profile @test_target preset(\"test-low\")\n";
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

  void ExpectSpillSlot(const loom_low_allocation_assignment_t* assignment,
                       uint32_t slot_index, uint32_t slot_count) {
    ASSERT_NE(assignment, nullptr);
    EXPECT_EQ(assignment->location_kind,
              LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT);
    EXPECT_EQ(assignment->location_base, slot_index);
    EXPECT_EQ(assignment->location_count, slot_count);
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
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      AllocateFirstLowFunctionWithOptions(&options, &allocation));
}

TEST_F(LowAllocationTest, RejectsUnspillableRegisterClassExhaustion) {
  ParseAndVerify(
      "low.func.def target(@test_target) @allocated(%lhs : "
      "reg<test.special>, %rhs : reg<test.special>) -> "
      "(reg<test.special>) {\n"
      "  %sum = low.op<test.add.special>(%lhs, %rhs) : "
      "(reg<test.special>, reg<test.special>) -> reg<test.special>\n"
      "  low.return %sum : reg<test.special>\n"
      "}\n");
  loom_low_allocation_options_t options = {
      .descriptor_registry = &target_registry_.registry,
  };
  loom_low_allocation_sidecar_t allocation = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      AllocateFirstLowFunctionWithOptions(&options, &allocation));
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

TEST_F(LowAllocationTest, AlignsPowerOfTwoRegisterTuples) {
  ParseAndVerify(
      "low.func.def target(@test_target) @allocated(%scalar : "
      "reg<test.phys>, %wide : reg<test.phys x4>) -> (reg<test.phys>, "
      "reg<test.phys x4>) {\n"
      "  low.return %scalar, %wide : reg<test.phys>, reg<test.phys x4>\n"
      "}\n");
  loom_low_allocation_sidecar_t allocation = AllocateFirstLowFunction();

  ExpectPhysicalLocation(FindAssignmentByName(allocation, "scalar"), 0, 1);
  ExpectPhysicalLocation(FindAssignmentByName(allocation, "wide"), 4, 4);
}

TEST_F(LowAllocationTest, RejectsMisalignedFixedRegisterTuple) {
  ParseAndVerify(
      "low.func.def target(@test_target) @allocated(%wide : "
      "reg<test.phys x4>) -> (reg<test.phys x4>) {\n"
      "  low.return %wide : reg<test.phys x4>\n"
      "}\n");
  loom_value_id_t wide_value_id = FindValueIdByName("wide");
  ASSERT_NE(wide_value_id, LOOM_VALUE_ID_INVALID);
  const loom_low_allocation_fixed_value_t fixed_value = {
      .value_id = wide_value_id,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = 2,
      .location_count = 4,
  };
  loom_low_allocation_options_t options = {
      .descriptor_registry = &target_registry_.registry,
      .fixed_values = &fixed_value,
      .fixed_value_count = 1,
  };
  loom_low_allocation_sidecar_t allocation = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      AllocateFirstLowFunctionWithOptions(&options, &allocation));
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

TEST_F(LowAllocationTest, AliasedRegisterClassesDoNotOverlap) {
  ParseAndVerify(
      "low.func.def target(@test_target) @allocated(%narrow : "
      "reg<test.alias32>, %wide : reg<test.alias64>) -> "
      "(reg<test.alias32>, reg<test.alias64>) {\n"
      "  low.return %narrow, %wide : reg<test.alias32>, "
      "reg<test.alias64>\n"
      "}\n");
  loom_low_allocation_sidecar_t allocation = AllocateFirstLowFunction();

  ExpectPhysicalLocation(FindAssignmentByName(allocation, "narrow"), 0, 1);
  ExpectSpillSlot(FindAssignmentByName(allocation, "wide"), 0, 1);
}

TEST_F(LowAllocationTest, BudgetAppliesToAliasedRegisterClass) {
  ParseAndVerify(
      "low.func.def target(@test_target) @allocated(%narrow : "
      "reg<test.alias32>) -> (reg<test.alias32>) {\n"
      "  low.return %narrow : reg<test.alias32>\n"
      "}\n");
  const loom_low_allocation_budget_t budget = {
      .register_class = IREE_SV("test.alias64"),
      .max_units = 0,
  };
  loom_low_allocation_options_t options = {
      .descriptor_registry = &target_registry_.registry,
      .budgets = &budget,
      .budget_count = 1,
  };
  loom_low_allocation_sidecar_t allocation = {};
  IREE_ASSERT_OK(AllocateFirstLowFunctionWithOptions(&options, &allocation));
  IREE_ASSERT_OK(loom_low_allocation_verify_sidecar(&allocation));

  ExpectSpillSlot(FindAssignmentByName(allocation, "narrow"), 0, 1);
}

TEST_F(LowAllocationTest, RejectsDuplicateAliasedRegisterBudgets) {
  ParseAndVerify(
      "low.func.def target(@test_target) @allocated(%narrow : "
      "reg<test.alias32>) -> (reg<test.alias32>) {\n"
      "  low.return %narrow : reg<test.alias32>\n"
      "}\n");
  const loom_low_allocation_budget_t budgets[] = {
      {
          .register_class = IREE_SV("test.alias32"),
          .max_units = 1,
      },
      {
          .register_class = IREE_SV("test.alias64"),
          .max_units = 1,
      },
  };
  loom_low_allocation_options_t options = {
      .descriptor_registry = &target_registry_.registry,
      .budgets = budgets,
      .budget_count = IREE_ARRAYSIZE(budgets),
  };
  loom_low_allocation_sidecar_t allocation = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      AllocateFirstLowFunctionWithOptions(&options, &allocation));
}

TEST_F(LowAllocationTest, FutureFixedValueBlocksAliasedRegisterClass) {
  ParseAndVerify(
      "low.func.def target(@test_target) @allocated(%narrow : "
      "reg<test.alias32>, %wide : reg<test.alias64>) -> "
      "(reg<test.alias32>, reg<test.alias64>) {\n"
      "  low.return %narrow, %wide : reg<test.alias32>, "
      "reg<test.alias64>\n"
      "}\n");
  loom_value_id_t wide_value_id = FindValueIdByName("wide");
  ASSERT_NE(wide_value_id, LOOM_VALUE_ID_INVALID);
  const loom_low_allocation_fixed_value_t fixed_value = {
      .value_id = wide_value_id,
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

  ExpectSpillSlot(FindAssignmentByName(allocation, "narrow"), 0, 1);
  ExpectPhysicalLocation(FindAssignmentByName(allocation, "wide"), 0, 1);
}

TEST_F(LowAllocationTest, ReservedRangeBlocksAliasedRegisterClass) {
  ParseAndVerify(
      "low.func.def target(@test_target) @allocated(%narrow : "
      "reg<test.alias32>) -> (reg<test.alias32>) {\n"
      "  low.return %narrow : reg<test.alias32>\n"
      "}\n");
  const loom_low_allocation_reserved_range_t reserved_range = {
      .register_class = IREE_SV("test.alias64"),
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

  ExpectSpillSlot(FindAssignmentByName(allocation, "narrow"), 0, 1);
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

TEST_F(LowAllocationTest, RejectsOverlappingAliasedReservedRanges) {
  ParseAndVerify(
      "low.func.def target(@test_target) @allocated(%narrow : "
      "reg<test.alias32>) -> (reg<test.alias32>) {\n"
      "  low.return %narrow : reg<test.alias32>\n"
      "}\n");
  const loom_low_allocation_reserved_range_t reserved_ranges[] = {
      {
          .register_class = IREE_SV("test.alias32"),
          .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          .location_base = 0,
          .location_count = 1,
      },
      {
          .register_class = IREE_SV("test.alias64"),
          .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
          .location_base = 0,
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

TEST_F(LowAllocationTest, VerifierRejectsAliasedRegisterClassOverlap) {
  ParseAndVerify(
      "low.func.def target(@test_target) @allocated(%narrow : "
      "reg<test.alias32>, %wide : reg<test.alias64>) -> "
      "(reg<test.alias32>, reg<test.alias64>) {\n"
      "  low.return %narrow, %wide : reg<test.alias32>, "
      "reg<test.alias64>\n"
      "}\n");
  loom_low_allocation_sidecar_t allocation = AllocateFirstLowFunction();
  ASSERT_EQ(allocation.assignment_count, 2u);

  loom_low_allocation_assignment_t assignments[2] = {
      allocation.assignments[0],
      allocation.assignments[1],
  };
  assignments[0].location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  assignments[0].location_base = 0;
  assignments[0].location_count = 1;
  assignments[1].location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  assignments[1].location_base = 0;
  assignments[1].location_count = 1;

  loom_low_allocation_sidecar_t invalid_allocation = allocation;
  invalid_allocation.assignments = assignments;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_low_allocation_verify_sidecar(&invalid_allocation));
}

}  // namespace
}  // namespace loom
