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
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/schedule/run.h"
#include "loom/codegen/low/storage_lease.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"
#include "loom/target/test/descriptors.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ::iree::StatusCode;
using ModulePtr = ::loom::testing::ModulePtr;

static const loom_low_descriptor_set_provider_t kDescriptorSetProviders[] = {
    loom_test_low_core_descriptor_set,
};

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

struct CapturedAllocationDiagnostic {
  // Number of capacity diagnostics observed by the capture callback.
  uint32_t count = 0;
  // Error domain of the most recently captured diagnostic.
  loom_error_domain_t domain = LOOM_ERROR_DOMAIN_COUNT_;
  // Error code of the most recently captured diagnostic.
  uint16_t code = 0;
  // Kind of allocator input that exceeded capacity.
  std::string subject_kind;
  // Register class reported by the capacity diagnostic.
  std::string register_class;
  // First requested location in the reported range.
  uint32_t location_base = UINT32_MAX;
  // Number of requested units in the reported range.
  uint32_t location_count = UINT32_MAX;
  // Exclusive end location in the reported range.
  uint64_t location_end = UINT64_MAX;
  // Effective allocation capacity for the reported register class.
  uint32_t allocation_capacity = UINT32_MAX;
};

iree_status_t CaptureAllocationDiagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  CapturedAllocationDiagnostic* captured =
      static_cast<CapturedAllocationDiagnostic*>(user_data);
  ++captured->count;
  captured->domain = emission->error->domain;
  captured->code = emission->error->code;
  if (emission->param_count < 10u) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation diagnostic missing capacity params");
  }
  captured->subject_kind = ToString(emission->params[4].string);
  captured->register_class = ToString(emission->params[5].string);
  captured->location_base = emission->params[6].u32;
  captured->location_count = emission->params[7].u32;
  captured->location_end = emission->params[8].u64;
  captured->allocation_capacity = emission->params[9].u32;
  return iree_ok_status();
}

struct TestStorageLeaseConfig {
  // First leased unit within the matched operand assignment.
  uint32_t unit_offset = 0;
  // Number of leased units within the matched operand assignment.
  uint32_t unit_count = 1;
  // Number of leading operands to lease on each matched node.
  uint16_t operand_count = 1;
  // Source-order ordinal to match, or UINT32_MAX to match any node.
  uint32_t source_ordinal = UINT32_MAX;
  // True when only terminator nodes should be matched.
  bool require_terminator = true;
};

enum TestStorageLeaseIdentity {
  kTestStorageLeaseReleaseClass = 7,
  kTestStorageLeaseReleaseAction = 1,
  kTestStorageLeaseReleaseReason = 8,
};

iree_status_t EmitOperandStorageLease(void* user_data,
                                      const loom_low_schedule_table_t* schedule,
                                      const loom_low_schedule_node_t* node,
                                      loom_low_storage_lease_emit_fn_t emit,
                                      void* emit_user_data) {
  (void)schedule;
  const TestStorageLeaseConfig* config =
      static_cast<const TestStorageLeaseConfig*>(user_data);
  if (config->require_terminator &&
      node->kind != LOOM_LOW_SCHEDULE_NODE_TERMINATOR) {
    return iree_ok_status();
  }
  if (config->source_ordinal != UINT32_MAX &&
      node->source_ordinal != config->source_ordinal) {
    return iree_ok_status();
  }
  if (node->operand_count == 0) {
    return iree_ok_status();
  }
  const uint16_t operand_count = config->operand_count < node->operand_count
                                     ? config->operand_count
                                     : node->operand_count;
  for (uint16_t i = 0; i < operand_count; ++i) {
    const loom_low_storage_lease_event_t event = {
        .kind = LOOM_LOW_STORAGE_LEASE_SOURCE_READ,
        .attachment = LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND,
        .attachment_index = i,
        .unit_offset = config->unit_offset,
        .unit_count = config->unit_count,
        .release_scope = LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS,
        .release_class_id = kTestStorageLeaseReleaseClass,
        .release_class_name = IREE_SV("test.return"),
        .release_action_id = kTestStorageLeaseReleaseAction,
        .release_action_name = IREE_SV("test.release-storage"),
        .release_reason_id = kTestStorageLeaseReleaseReason,
        .release_reason_name = IREE_SV("test.storage-release"),
        .flags = LOOM_LOW_STORAGE_LEASE_FLAG_STARTS_AT_ISSUE,
    };
    IREE_RETURN_IF_ERROR(emit(emit_user_data, &event));
  }
  return iree_ok_status();
}

class LowAllocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    descriptor_registry_ = {
        .descriptor_set_providers = kDescriptorSetProviders,
        .descriptor_set_provider_count =
            IREE_ARRAYSIZE(kDescriptorSetProviders),
    };
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
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

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    loom_low_descriptor_text_asm_environment_initialize(
        &descriptor_registry_, &options.low_asm_environment);
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("allocation_test.loom"), &context_,
                                  &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_op_t* FindLowFunction(loom_module_t* module, iree_string_view_t name) {
    loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    uint16_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    loom_op_t* op = module->symbols.entries[symbol_id].defining_op;
    IREE_ASSERT(loom_low_func_def_isa(op));
    return op;
  }

  loom_value_id_t FindValueByName(loom_module_t* module,
                                  iree_string_view_t name) {
    for (iree_host_size_t i = 0; i < module->values.count; ++i) {
      if (iree_string_view_equal(
              loom_low_diagnostic_value_name(module, (loom_value_id_t)i),
              name)) {
        return (loom_value_id_t)i;
      }
    }
    IREE_ASSERT(false, "value name not found");
    return LOOM_VALUE_ID_INVALID;
  }

  iree_status_t Allocate(
      loom_module_t* module, loom_op_t* function_op,
      const loom_low_allocation_reserved_range_t* reserved_ranges,
      iree_host_size_t reserved_range_count,
      CapturedAllocationDiagnostic* captured_diagnostic) {
    const iree_diagnostic_emitter_t emitter = {
        .fn = CaptureAllocationDiagnostic,
        .user_data = captured_diagnostic,
    };
    loom_low_allocation_options_t options = {
        .descriptor_registry = &descriptor_registry_,
        .reserved_ranges = reserved_ranges,
        .reserved_range_count = reserved_range_count,
        .emitter = emitter,
    };
    loom_low_allocation_table_t table = {};
    return loom_low_allocate_function(module, function_op, &options,
                                      &analysis_arena_, &table);
  }

  iree_status_t ScheduleStorageLeases(
      loom_module_t* module, loom_op_t* function_op,
      TestStorageLeaseConfig* config, loom_low_schedule_table_t* out_schedule,
      loom_low_storage_lease_table_t* out_storage_leases) {
    loom_low_schedule_options_t schedule_options = {
        .descriptor_registry = &descriptor_registry_,
    };
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_function(module, function_op, &schedule_options,
                                   &analysis_arena_, out_schedule));
    const loom_low_storage_lease_provider_t provider = {
        .user_data = config,
        .query = EmitOperandStorageLease,
    };
    return loom_low_storage_lease_build(out_schedule, &provider,
                                        &analysis_arena_, out_storage_leases);
  }

  iree_status_t AllocateWithStorageLeases(
      loom_module_t* module, loom_op_t* function_op,
      loom_low_storage_lease_table_t storage_leases,
      loom_low_allocation_table_t* out_table) {
    return AllocateWithStorageLeasesAndBudgets(
        module, function_op, storage_leases, NULL, 0, out_table);
  }

  iree_status_t AllocateWithFixedValuesAndBudgets(
      loom_module_t* module, loom_op_t* function_op,
      const loom_low_allocation_fixed_value_t* fixed_values,
      iree_host_size_t fixed_value_count,
      const loom_low_allocation_budget_t* budgets,
      iree_host_size_t budget_count, loom_low_allocation_table_t* out_table) {
    loom_low_allocation_options_t options = {};
    options.descriptor_registry = &descriptor_registry_;
    options.budgets = budgets;
    options.budget_count = budget_count;
    options.fixed_values = fixed_values;
    options.fixed_value_count = fixed_value_count;
    return loom_low_allocate_function(module, function_op, &options,
                                      &analysis_arena_, out_table);
  }

  iree_status_t AllocateWithStorageLeasesAndBudgets(
      loom_module_t* module, loom_op_t* function_op,
      loom_low_storage_lease_table_t storage_leases,
      const loom_low_allocation_budget_t* budgets,
      iree_host_size_t budget_count, loom_low_allocation_table_t* out_table) {
    loom_low_allocation_options_t options = {
        .descriptor_registry = &descriptor_registry_,
        .budgets = budgets,
        .budget_count = budget_count,
        .storage_leases = storage_leases,
    };
    return loom_low_allocate_function(module, function_op, &options,
                                      &analysis_arena_, out_table);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  iree_arena_allocator_t analysis_arena_;
  loom_low_descriptor_registry_t descriptor_registry_;
};

static const char kSinglePhysFunction[] = R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @single(%value: reg<test.phys>) -> (reg<test.phys>) {
  low.return %value : reg<test.phys>
}
)";

static const char kCopyReuseFunction[] = R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @copy_reuse(%value: reg<test.phys>) -> (reg<test.phys>) {
  %copy = low.copy %value : reg<test.phys> -> reg<test.phys>
  low.return %copy : reg<test.phys>
}
)";

static const char kDeadResultClobberWindowFunction[] = R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @dead_result_clobber_window(%leader: reg<test.phys>, %lhs: reg<test.phys>, %rhs: reg<test.phys>) -> (reg<test.phys>) asm<test.low.core> {
  %dead = test.add.phys %lhs, %rhs
  return %leader
}
)";

static const char kPacketMoveConcatCycleFunction[] = R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @packet_move_concat_cycle(%lhs: reg<test.phys>, %rhs: reg<test.phys>) -> (reg<test.phys x2>) asm<test.low.core> {
  %pair = concat(%rhs, %lhs) : (reg<test.phys>, reg<test.phys>) -> reg<test.phys x2>
  return %pair
}
)";

static const char kEdgeCopyBranchCycleFunction[] = R"(
test.target<low_core> @test_target

low.func.def target(@test_target) @edge_copy_branch_cycle(%cond: reg<test.i32>, %lhs: reg<test.phys>, %rhs: reg<test.phys>) -> (reg<test.phys x2>) asm<test.low.core> {
  low.cond_br %cond, ^then, ^else : reg<test.i32>
^then:
  low.br ^join(%rhs: reg<test.phys>, %lhs: reg<test.phys>)
^else:
  low.br ^join(%lhs: reg<test.phys>, %rhs: reg<test.phys>)
^join(%a: reg<test.phys>, %b: reg<test.phys>):
  %pair = concat(%a, %b) : (reg<test.phys>, reg<test.phys>) -> reg<test.phys x2>
  return %pair
}
)";

TEST_F(LowAllocationTest, EmitsDiagnosticForReservedRangeBeyondCapacity) {
  ModulePtr module = ParseModule(kSinglePhysFunction);
  loom_op_t* function_op = FindLowFunction(module.get(), IREE_SV("single"));
  const loom_low_allocation_reserved_range_t reserved_range = {
      .register_class = IREE_SV("test.phys"),
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = 32,
      .location_count = 1,
  };

  CapturedAllocationDiagnostic captured;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kOutOfRange,
      Allocate(module.get(), function_op, &reserved_range, 1, &captured));

  EXPECT_EQ(captured.count, 1u);
  EXPECT_EQ(captured.domain, LOOM_ERROR_DOMAIN_BACKEND);
  EXPECT_EQ(captured.code, 22u);
  EXPECT_EQ(captured.subject_kind, "reserved range");
  EXPECT_EQ(captured.register_class, "test.phys");
  EXPECT_EQ(captured.location_base, 32u);
  EXPECT_EQ(captured.location_count, 1u);
  EXPECT_EQ(captured.location_end, 33u);
  EXPECT_EQ(captured.allocation_capacity, 32u);
}

TEST_F(LowAllocationTest, RecordsStorageLeaseInstances) {
  ModulePtr module = ParseModule(kSinglePhysFunction);
  loom_op_t* function_op = FindLowFunction(module.get(), IREE_SV("single"));

  TestStorageLeaseConfig config = {};
  loom_low_schedule_table_t schedule = {};
  loom_low_storage_lease_table_t storage_leases = {};
  IREE_ASSERT_OK(ScheduleStorageLeases(module.get(), function_op, &config,
                                       &schedule, &storage_leases));
  ASSERT_EQ(storage_leases.record_count, 1u);

  loom_low_allocation_table_t table = {};
  IREE_ASSERT_OK(AllocateWithStorageLeases(module.get(), function_op,
                                           storage_leases, &table));

  ASSERT_EQ(table.storage_leases.record_count, 1u);
  ASSERT_EQ(table.storage_lease_instance_count, 1u);
  const loom_low_allocation_storage_lease_t* instance =
      &table.storage_lease_instances[0];
  EXPECT_EQ(instance->lease_record_index, 0u);
  ASSERT_LT(instance->assignment_index, table.assignment_count);
  const loom_low_allocation_assignment_t* assignment =
      &table.assignments[instance->assignment_index];
  EXPECT_EQ(instance->value_id, assignment->value_id);
  EXPECT_EQ(instance->descriptor_reg_class_id,
            assignment->descriptor_reg_class_id);
  EXPECT_EQ(instance->location_kind, assignment->location_kind);
  EXPECT_EQ(instance->location_base, assignment->location_base);
  EXPECT_EQ(instance->location_count, 1u);
  EXPECT_EQ(instance->start_point, table.liveness.blocks[0].start_point);
  EXPECT_EQ(instance->end_point, table.liveness.blocks[0].end_point);
}

TEST_F(LowAllocationTest, StorageLeaseBlocksPhysicalReuse) {
  ModulePtr module = ParseModule(kCopyReuseFunction);
  loom_op_t* function_op = FindLowFunction(module.get(), IREE_SV("copy_reuse"));

  TestStorageLeaseConfig config = {
      .source_ordinal = 0,
      .require_terminator = false,
  };
  loom_low_schedule_table_t schedule = {};
  loom_low_storage_lease_table_t storage_leases = {};
  IREE_ASSERT_OK(ScheduleStorageLeases(module.get(), function_op, &config,
                                       &schedule, &storage_leases));
  ASSERT_EQ(storage_leases.record_count, 1u);

  const loom_low_allocation_budget_t budget = {
      .register_class = IREE_SV("test.phys"),
      .max_units = 2,
  };
  loom_low_allocation_table_t table = {};
  IREE_ASSERT_OK(AllocateWithStorageLeasesAndBudgets(
      module.get(), function_op, storage_leases, &budget, 1, &table));

  ASSERT_EQ(table.storage_lease_instance_count, 1u);
  ASSERT_EQ(table.copy_decision_count, 1u);
  const loom_low_allocation_copy_decision_t* copy_decision =
      &table.copy_decisions[0];
  EXPECT_EQ(copy_decision->kind, LOOM_LOW_ALLOCATION_COPY_MATERIALIZED);
  const loom_low_allocation_assignment_t* source_assignment =
      &table.assignments[copy_decision->source_assignment_index];
  const loom_low_allocation_assignment_t* result_assignment =
      &table.assignments[copy_decision->result_assignment_index];
  EXPECT_FALSE(loom_low_allocation_storage_assignment_ranges_equal(
      table.target.descriptor_set, source_assignment, result_assignment));
}

TEST_F(LowAllocationTest, ReservesPacketMoveTemporaryForConcatCycle) {
  ModulePtr module = ParseModule(kPacketMoveConcatCycleFunction);
  loom_op_t* function_op =
      FindLowFunction(module.get(), IREE_SV("packet_move_concat_cycle"));

  loom_low_allocation_fixed_value_t fixed_values[3] = {};
  fixed_values[0].value_id = FindValueByName(module.get(), IREE_SV("lhs"));
  fixed_values[0].location_kind =
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  fixed_values[0].location_base = 0;
  fixed_values[0].location_count = 1;
  fixed_values[1].value_id = FindValueByName(module.get(), IREE_SV("rhs"));
  fixed_values[1].location_kind =
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  fixed_values[1].location_base = 1;
  fixed_values[1].location_count = 1;
  fixed_values[2].value_id = FindValueByName(module.get(), IREE_SV("pair"));
  fixed_values[2].location_kind =
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  fixed_values[2].location_base = 0;
  fixed_values[2].location_count = 2;

  loom_low_allocation_budget_t budget = {};
  budget.register_class = IREE_SV("test.phys");
  budget.max_units = 3;

  loom_low_allocation_table_t table = {};
  IREE_ASSERT_OK(AllocateWithFixedValuesAndBudgets(
      module.get(), function_op, fixed_values, IREE_ARRAYSIZE(fixed_values),
      &budget, 1, &table));

  ASSERT_EQ(table.packet_move_temporary_group_count, 1u);
  const loom_low_allocation_packet_move_temporary_group_t* group =
      &table.packet_move_temporary_groups[0];
  EXPECT_EQ(group->source_ordinal, 0u);
  EXPECT_EQ(group->temporary_start, 0u);
  EXPECT_EQ(group->temporary_count, 1u);

  ASSERT_EQ(table.packet_move_temporary_count, 1u);
  const loom_low_allocation_packet_move_temporary_t* temporary =
      &table.packet_move_temporaries[0];
  EXPECT_EQ(temporary->location_kind,
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
  EXPECT_EQ(temporary->descriptor_reg_class_id,
            table.assignments[0].descriptor_reg_class_id);
  EXPECT_EQ(temporary->location, 2u);
}

TEST_F(LowAllocationTest, ReservesEdgeCopyTemporaryForBranchCycle) {
  ModulePtr module = ParseModule(kEdgeCopyBranchCycleFunction);
  loom_op_t* function_op =
      FindLowFunction(module.get(), IREE_SV("edge_copy_branch_cycle"));

  loom_low_allocation_fixed_value_t fixed_values[5] = {};
  fixed_values[0].value_id = FindValueByName(module.get(), IREE_SV("lhs"));
  fixed_values[0].location_kind =
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  fixed_values[0].location_base = 0;
  fixed_values[0].location_count = 1;
  fixed_values[1].value_id = FindValueByName(module.get(), IREE_SV("rhs"));
  fixed_values[1].location_kind =
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  fixed_values[1].location_base = 1;
  fixed_values[1].location_count = 1;
  fixed_values[2].value_id = FindValueByName(module.get(), IREE_SV("a"));
  fixed_values[2].location_kind =
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  fixed_values[2].location_base = 0;
  fixed_values[2].location_count = 1;
  fixed_values[3].value_id = FindValueByName(module.get(), IREE_SV("b"));
  fixed_values[3].location_kind =
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  fixed_values[3].location_base = 1;
  fixed_values[3].location_count = 1;
  fixed_values[4].value_id = FindValueByName(module.get(), IREE_SV("pair"));
  fixed_values[4].location_kind =
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  fixed_values[4].location_base = 0;
  fixed_values[4].location_count = 2;

  loom_low_allocation_budget_t budget = {};
  budget.register_class = IREE_SV("test.phys");
  budget.max_units = 3;

  loom_low_allocation_table_t table = {};
  IREE_ASSERT_OK(AllocateWithFixedValuesAndBudgets(
      module.get(), function_op, fixed_values, IREE_ARRAYSIZE(fixed_values),
      &budget, 1, &table));

  ASSERT_EQ(table.edge_copy_group_count, 2u);
  ASSERT_EQ(table.edge_copy_count, 4u);
  EXPECT_EQ(table.edge_copy_groups[0].temporary_start, 0u);
  EXPECT_EQ(table.edge_copy_groups[0].temporary_count, 1u);
  EXPECT_EQ(table.edge_copy_groups[1].temporary_start, 1u);
  EXPECT_EQ(table.edge_copy_groups[1].temporary_count, 0u);

  ASSERT_EQ(table.edge_copy_temporary_count, 1u);
  const loom_low_allocation_edge_copy_temporary_t* temporary =
      &table.edge_copy_temporaries[0];
  EXPECT_EQ(temporary->location_kind,
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
  EXPECT_EQ(temporary->descriptor_reg_class_id,
            table.assignments[table.edge_copies[0].source_assignment_index]
                .descriptor_reg_class_id);
  EXPECT_EQ(temporary->location, 2u);
}

TEST_F(LowAllocationTest, RecordsStorageReleaseActionForForcedReuse) {
  ModulePtr module = ParseModule(kDeadResultClobberWindowFunction);
  loom_op_t* function_op =
      FindLowFunction(module.get(), IREE_SV("dead_result_clobber_window"));

  TestStorageLeaseConfig config = {
      .operand_count = 2,
      .source_ordinal = 0,
      .require_terminator = false,
  };
  loom_low_schedule_table_t schedule = {};
  loom_low_storage_lease_table_t storage_leases = {};
  IREE_ASSERT_OK(ScheduleStorageLeases(module.get(), function_op, &config,
                                       &schedule, &storage_leases));
  ASSERT_EQ(storage_leases.record_count, 2u);

  const loom_low_allocation_budget_t budget = {
      .register_class = IREE_SV("test.phys"),
      .max_units = 3,
  };
  loom_low_allocation_table_t table = {};
  IREE_ASSERT_OK(AllocateWithStorageLeasesAndBudgets(
      module.get(), function_op, storage_leases, &budget, 1, &table));

  ASSERT_EQ(table.storage_lease_instance_count, 2u);
  ASSERT_EQ(table.storage_release_action_count, 1u);
  const loom_low_allocation_storage_lease_t* lease =
      &table.storage_lease_instances[0];
  EXPECT_EQ(lease->release_action_index, 0u);
  EXPECT_EQ(lease->end_point, 1u);

  const loom_low_storage_release_action_t* action =
      &table.storage_release_actions[0];
  EXPECT_EQ(action->insertion_packet_index, 1u);
  EXPECT_EQ(action->insertion_node_index, 1u);
  EXPECT_EQ(action->release_class_id, kTestStorageLeaseReleaseClass);
  EXPECT_TRUE(iree_string_view_equal(action->release_class_name,
                                     IREE_SV("test.return")));
  EXPECT_EQ(action->release_action_id, kTestStorageLeaseReleaseAction);
  EXPECT_TRUE(iree_string_view_equal(action->release_action_name,
                                     IREE_SV("test.release-storage")));
  EXPECT_EQ(action->release_reason_id, kTestStorageLeaseReleaseReason);
  EXPECT_TRUE(iree_string_view_equal(action->release_reason_name,
                                     IREE_SV("test.storage-release")));
  EXPECT_EQ(action->required_progress, 1u);
  EXPECT_EQ(action->lease_record_index, 0u);
}

TEST_F(LowAllocationTest, RejectsStorageLeaseBeyondAssignedUnits) {
  ModulePtr module = ParseModule(kSinglePhysFunction);
  loom_op_t* function_op = FindLowFunction(module.get(), IREE_SV("single"));

  TestStorageLeaseConfig config = {
      .unit_offset = 1,
      .unit_count = 1,
  };
  loom_low_schedule_table_t schedule = {};
  loom_low_storage_lease_table_t storage_leases = {};
  IREE_ASSERT_OK(ScheduleStorageLeases(module.get(), function_op, &config,
                                       &schedule, &storage_leases));

  loom_low_allocation_table_t table = {};
  IREE_EXPECT_STATUS_IS(StatusCode::kOutOfRange,
                        AllocateWithStorageLeases(module.get(), function_op,
                                                  storage_leases, &table));
}

}  // namespace
}  // namespace loom
