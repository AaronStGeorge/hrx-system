// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/types.h"
#include "loom/testing/module_ptr.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

static const loom_target_snapshot_t kHalSnapshot = {
    .name = IREE_SVL("test-hal"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
    .target_triple = IREE_SVL("test-hal"),
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 32,
    .offset_bitwidth = 64,
    .max_workgroup_size =
        {
            .x = 1024,
            .y = 512,
            .z = 64,
        },
    .max_workgroup_count =
        {
            .x = 1024,
            .y = 2048,
            .z = 4096,
        },
};

static const loom_target_export_plan_t kHalExportPlan = {
    .name = IREE_SVL("test-hal-kernel"),
    .abi_kind = LOOM_TARGET_ABI_HAL_KERNEL,
    .hal_kernel =
        {
            .required_workgroup_size = {.x = 64, .y = 8, .z = 1},
        },
};

static const loom_target_export_plan_t kOpenHalExportPlan = {
    .name = IREE_SVL("test-hal-open-kernel"),
    .abi_kind = LOOM_TARGET_ABI_HAL_KERNEL,
};

static const loom_target_config_t kHalConfig = {
    .name = IREE_SVL("test-hal-config"),
};

static const loom_target_bundle_t kHalBundle = {
    .name = IREE_SVL("test-hal-bundle"),
    .snapshot = &kHalSnapshot,
    .export_plan = &kHalExportPlan,
    .config = &kHalConfig,
};

static const loom_target_bundle_t kOpenHalBundle = {
    .name = IREE_SVL("test-hal-open-bundle"),
    .snapshot = &kHalSnapshot,
    .export_plan = &kOpenHalExportPlan,
    .config = &kHalConfig,
};

class KernelFactsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_KERNEL, loom_kernel_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &fact_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&fact_arena_);
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
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("kernel_facts_test.loom"), &context_,
                                  &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_func_like_t FindFunction(const loom_module_t* module,
                                iree_string_view_t name) {
    loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    loom_func_like_t function = loom_func_like_cast(
        module, module->symbols.entries[symbol_id].defining_op);
    IREE_ASSERT(function.op != NULL);
    return function;
  }

  loom_value_fact_table_t ComputeFacts(
      const loom_module_t* module, loom_func_like_t function,
      const loom_target_bundle_t* target_bundle = &kHalBundle) {
    loom_value_fact_table_t table = {0};
    IREE_CHECK_OK(loom_value_fact_table_initialize(&table, &fact_arena_,
                                                   module->values.count));
    table.context.target_bundle = target_bundle;
    IREE_CHECK_OK(loom_value_fact_table_compute(&table, module, function));
    return table;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t fact_arena_;
  loom_context_t context_;
};

TEST_F(KernelFactsTest, TargetBundleRefinesCoordinateFactsBeforePropagation) {
  ModulePtr module = ParseModule(R"(
func.def @kernel_coords() -> (index, index, index) {
  %tid_x = kernel.workitem.id<x> : index
  %wg_y = kernel.workgroup.id<y> : index
  %c4 = index.constant 4 : index
  %byte_offset = index.mul %tid_x, %c4 : index
  func.return %tid_x, %wg_y, %byte_offset : index, index, index
}
)");
  loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("kernel_coords"));
  loom_value_fact_table_t table = ComputeFacts(module.get(), function);
  loom_block_t* block = loom_region_entry_block(loom_func_like_body(function));

  loom_op_t* workitem_id = block->first_op;
  ASSERT_TRUE(loom_kernel_workitem_id_isa(workitem_id));
  loom_value_facts_t workitem_facts = loom_value_fact_table_lookup(
      &table, loom_kernel_workitem_id_result(workitem_id));
  EXPECT_EQ(workitem_facts.range_lo, 0);
  EXPECT_EQ(workitem_facts.range_hi, 63);

  loom_op_t* workgroup_id = workitem_id->next_op;
  ASSERT_TRUE(loom_kernel_workgroup_id_isa(workgroup_id));
  loom_value_facts_t workgroup_facts = loom_value_fact_table_lookup(
      &table, loom_kernel_workgroup_id_result(workgroup_id));
  EXPECT_EQ(workgroup_facts.range_lo, 0);
  EXPECT_EQ(workgroup_facts.range_hi, 2047);

  loom_op_t* byte_offset = workgroup_id->next_op->next_op;
  ASSERT_TRUE(loom_index_mul_isa(byte_offset));
  loom_value_facts_t byte_offset_facts =
      loom_value_fact_table_lookup(&table, loom_index_mul_result(byte_offset));
  EXPECT_EQ(byte_offset_facts.range_lo, 0);
  EXPECT_EQ(byte_offset_facts.range_hi, 252);
}

TEST_F(KernelFactsTest, OpenTargetUsesCapabilityBoundsWithoutFixedWorkgroup) {
  ModulePtr module = ParseModule(R"(
func.def @kernel_coords() -> (index, index) {
  %tid_y = kernel.workitem.id<y> : index
  %wg_y = kernel.workgroup.id<y> : index
  func.return %tid_y, %wg_y : index, index
}
)");
  loom_func_like_t function =
      FindFunction(module.get(), IREE_SV("kernel_coords"));
  loom_value_fact_table_t table =
      ComputeFacts(module.get(), function, &kOpenHalBundle);
  loom_block_t* block = loom_region_entry_block(loom_func_like_body(function));

  loom_op_t* workitem_id = block->first_op;
  ASSERT_TRUE(loom_kernel_workitem_id_isa(workitem_id));
  loom_value_facts_t workitem_facts = loom_value_fact_table_lookup(
      &table, loom_kernel_workitem_id_result(workitem_id));
  EXPECT_EQ(workitem_facts.range_lo, 0);
  EXPECT_EQ(workitem_facts.range_hi, 511);

  loom_op_t* workgroup_id = workitem_id->next_op;
  ASSERT_TRUE(loom_kernel_workgroup_id_isa(workgroup_id));
  loom_value_facts_t workgroup_facts = loom_value_fact_table_lookup(
      &table, loom_kernel_workgroup_id_result(workgroup_id));
  EXPECT_EQ(workgroup_facts.range_lo, 0);
  EXPECT_EQ(workgroup_facts.range_hi, 2047);
}

}  // namespace
}  // namespace loom
