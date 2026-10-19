// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/workgroup_collective_legalization.h"

#include <tuple>
#include <utility>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/facts.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

// Invoke only the workgroup rewrite. Subgroup collectives remain opaque, so
// their implementation cannot change the publication protocol observed here.
class WorkgroupCollectiveLegalizationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    Register(LOOM_DIALECT_BUFFER, loom_buffer_dialect_vtables);
    Register(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    Register(LOOM_DIALECT_KERNEL, loom_kernel_dialect_vtables);
    Register(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    Register(LOOM_DIALECT_SCF, loom_scf_dialect_vtables);
    Register(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    Register(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables);
    Register(LOOM_DIALECT_VIEW, loom_view_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("collective"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_rewriter_initialize(&rewriter_, module_, &arena_);
  }

  void TearDown() override {
    loom_rewriter_deinitialize(&rewriter_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Register(uint8_t dialect,
                const loom_op_vtable_t* const* (*get)(iree_host_size_t*)) {
    iree_host_size_t count = 0;
    const auto* vtables = get(&count);
    IREE_ASSERT_OK(
        loom_context_register_dialect(&context_, dialect, vtables, count));
  }

  void CreateKernel(uint32_t workgroup_size, uint32_t subgroup_size,
                    std::vector<loom_type_t> types) {
    types_ = std::move(types);
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    loom_string_id_t name;
    loom_symbol_id_t symbol;
    IREE_ASSERT_OK(
        loom_module_intern_string(module_, IREE_SV("collective"), &name));
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
    IREE_ASSERT_OK(loom_kernel_def_build(
        &builder_, 0, 0, 0, loom_symbol_ref_null(), LOOM_STRING_ID_INVALID, 0,
        {0, symbol}, nullptr, 0, types_.data(), types_.size(), nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &kernel_));
    loom_builder_enter_region(&builder_, kernel_,
                              loom_kernel_def_config(kernel_));
    const auto one = Constant(1);
    const auto size = Constant(workgroup_size);
    loom_op_t* launch;
    IREE_ASSERT_OK(loom_kernel_launch_config_build(
        &builder_, 0, one, one, one, size, one, one, LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, LOOM_LOCATION_UNKNOWN,
        &launch));
    loom_builder_enter_region(&builder_, kernel_,
                              loom_kernel_def_body(kernel_));
    arguments_ =
        loom_region_entry_block(loom_kernel_def_body(kernel_))->arg_ids;
    facts_.storage.snapshot.subgroup_size = subgroup_size;
    loom_target_bundle_storage_rebind(&facts_.storage);
  }

  loom_value_id_t Constant(int64_t value) {
    loom_op_t* op;
    IREE_CHECK_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN, &op));
    return loom_index_constant_result(op);
  }

  loom_op_t* Reduce(loom_combining_kind_t kind = LOOM_COMBINING_KIND_ADDI,
                    uint8_t math = 0) {
    loom_op_t* op;
    IREE_CHECK_OK(loom_kernel_workgroup_reduce_build(
        &builder_, kind, math, arguments_, types_.size(), types_.data(),
        types_.size(), LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_op_t* Scan(loom_kernel_workgroup_scan_mode_t mode =
                      LOOM_KERNEL_WORKGROUP_SCAN_MODE_INCLUSIVE,
                  loom_kernel_workgroup_scan_direction_t direction =
                      LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_FORWARD,
                  loom_combining_kind_t kind = LOOM_COMBINING_KIND_ADDI,
                  uint8_t math = 0) {
    loom_op_t* op;
    IREE_CHECK_OK(loom_kernel_workgroup_scan_build(
        &builder_, kind, math, arguments_[0], mode, direction, types_[0],
        LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  void Rewrite(loom_op_t* op, loom_target_legalizer_action_t expected =
                                  LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN) {
    IREE_ASSERT_OK(loom_test_use_build(&builder_, loom_op_results(op),
                                       op->result_count, LOOM_LOCATION_UNKNOWN,
                                       &use_));
    loom_op_t* terminator;
    IREE_ASSERT_OK(loom_kernel_return_build(&builder_, LOOM_LOCATION_UNKNOWN,
                                            &terminator));
    ASSERT_NO_FATAL_FAILURE(Verify());
    ASSERT_NO_FATAL_FAILURE(Legalize(op, expected));
    ASSERT_NO_FATAL_FAILURE(Verify());
  }

  void Legalize(loom_op_t* op, loom_target_legalizer_action_t expected) {
    loom_target_legalization_context_t context = {
        .module = module_,
        .function = loom_func_like_cast(module_, kernel_),
        .target_facts = &facts_,
        .rewriter = &rewriter_,
        .mode = LOOM_TARGET_LEGALIZATION_MODE_FINAL,
        .arena = &arena_,
    };
    const auto legalize = loom_kernel_workgroup_reduce_isa(op)
                              ? loom_kernel_target_legalize_workgroup_reduce
                              : loom_kernel_target_legalize_workgroup_scan;
    loom_target_legalizer_result_t result = {};
    IREE_ASSERT_OK(legalize(nullptr, &context, op, &result));
    ASSERT_EQ(result.action, expected);
  }

  void Verify() {
    loom_verify_options_t options = {};
    options.sink.fn = loom_diagnostic_stderr_sink;
    loom_verify_result_t result;
    IREE_ASSERT_OK(loom_verify_module(module_, &options, &result));
    ASSERT_EQ(result.error_count, 0u);
  }

  static void Gather(loom_region_t* region, loom_op_kind_t kind,
                     std::vector<loom_op_t*>& matches) {
    loom_block_t* block;
    loom_region_for_each_block(region, block) {
      loom_op_t* op;
      loom_block_for_each_op(block, op) {
        if (op->kind == kind) {
          matches.push_back(op);
        }
        for (uint16_t i = 0; i < op->region_count; ++i) {
          Gather(loom_op_regions(op)[i], kind, matches);
        }
      }
    }
  }

  std::vector<loom_op_t*> Ops(loom_op_kind_t kind) {
    std::vector<loom_op_t*> matches;
    Gather(loom_kernel_def_body(kernel_), kind, matches);
    return matches;
  }

  loom_op_t* Definition(loom_value_id_t value) {
    return loom_value_def_op(loom_module_value(module_, value));
  }

  void ExpectConstant(loom_value_id_t value, int64_t expected) {
    const auto* op = Definition(value);
    ASSERT_NE(op, nullptr);
    ASSERT_TRUE(loom_index_constant_isa(op));
    EXPECT_EQ(loom_attr_as_i64(loom_index_constant_value(op)), expected);
  }

  // Slot addresses are in payload units; vector memory ops use element units.
  void ExpectAddress(loom_value_id_t index, loom_value_id_t slot,
                     int64_t offset, int64_t components = 1) {
    if (components != 1) {
      const auto* multiply = Definition(index);
      ASSERT_TRUE(loom_index_mul_isa(multiply));
      ExpectConstant(loom_index_mul_rhs(multiply), components);
      index = loom_index_mul_lhs(multiply);
    }
    if (slot == LOOM_VALUE_ID_INVALID) {
      ExpectConstant(index, offset);
    } else if (offset == 0) {
      EXPECT_EQ(index, slot);
    } else {
      const auto* add = Definition(index);
      ASSERT_TRUE(loom_index_add_isa(add));
      ExpectConstant(loom_index_add_lhs(add), offset);
      EXPECT_EQ(loom_index_add_rhs(add), slot);
    }
  }

  void ExpectScratch(loom_value_id_t view, loom_scalar_type_t element_type,
                     int64_t element_count, int64_t byte_count) {
    const auto type = loom_module_value_type(module_, view);
    EXPECT_TRUE(loom_type_equal(
        type, loom_type_shaped_1d(LOOM_TYPE_VIEW, element_type,
                                  loom_dim_pack_static(element_count), 0)));
    const auto* view_op = Definition(view);
    ASSERT_TRUE(loom_buffer_view_isa(view_op));
    ExpectConstant(loom_buffer_view_byte_offset(view_op), 0);
    const auto* allocation = Definition(loom_buffer_view_buffer(view_op));
    ASSERT_TRUE(loom_buffer_alloca_isa(allocation));
    EXPECT_EQ(loom_buffer_alloca_memory_space(allocation),
              LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP);
    ExpectConstant(loom_buffer_alloca_byte_length(allocation), byte_count);
  }

  void ExpectComparison(loom_value_id_t condition,
                        loom_index_cmp_predicate_t predicate,
                        loom_value_id_t lhs, int64_t rhs) {
    const auto* compare = Definition(condition);
    ASSERT_TRUE(loom_index_cmp_isa(compare));
    EXPECT_EQ(loom_index_cmp_predicate(compare), predicate);
    EXPECT_EQ(loom_index_cmp_lhs(compare), lhs);
    ExpectConstant(loom_index_cmp_rhs(compare), rhs);
  }

  void ExpectGuard(loom_op_t* op, loom_index_cmp_predicate_t predicate,
                   loom_value_id_t lhs, int64_t rhs) {
    ASSERT_TRUE(loom_scf_if_isa(op->parent_op));
    EXPECT_EQ(op->parent_block,
              loom_region_entry_block(loom_scf_if_then_region(op->parent_op)));
    ExpectComparison(loom_scf_if_condition(op->parent_op), predicate, lhs, rhs);
  }

  void ExpectPartialRead(loom_op_t* load, loom_value_id_t index,
                         loom_value_id_t subgroup, loom_value_id_t lane,
                         int64_t components = 1) {
    if (components != 1) {
      const auto* multiply = Definition(index);
      ASSERT_TRUE(loom_index_mul_isa(multiply));
      ExpectConstant(loom_index_mul_rhs(multiply), components);
      index = loom_index_mul_lhs(multiply);
    }
    // Inactive fold lanes still access a valid partial; only the first
    // subgroup participates in this phase.
    const auto* select = Definition(index);
    ASSERT_TRUE(loom_scf_select_isa(select));
    ExpectComparison(loom_scf_select_condition(select),
                     LOOM_INDEX_CMP_PREDICATE_ULT, lane, 3);
    EXPECT_EQ(loom_scf_select_true_value(select), lane);
    ExpectConstant(loom_scf_select_false_value(select), 0);
    ExpectGuard(load, LOOM_INDEX_CMP_PREDICATE_EQ, subgroup, 0);
  }

  // Compare a nested access's enclosing control with unconditional barriers.
  // No absolute operation positions or ordering of pure arithmetic is pinned.
  void ExpectPhase(loom_op_t* access, unsigned phase) {
    const auto barriers = Ops(LOOM_OP_KERNEL_BARRIER);
    ASSERT_EQ(barriers.size(), 2u);
    while (access->parent_op != kernel_) {
      access = access->parent_op;
    }
    for (unsigned i = 0; i < barriers.size(); ++i) {
      ASSERT_EQ(barriers[i]->parent_op, kernel_);
      EXPECT_EQ(loom_kernel_barrier_memory_space(barriers[i]),
                LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP);
      EXPECT_EQ(loom_kernel_barrier_scope(barriers[i]),
                LOOM_ATOMIC_SCOPE_WORKGROUP);
      EXPECT_EQ(loom_kernel_barrier_ordering(barriers[i]),
                LOOM_ATOMIC_ORDERING_ACQ_REL);
      EXPECT_EQ(access->block_ordinal > barriers[i]->block_ordinal, phase > i);
    }
  }

  // Module and rewrite storage owned by this fixture.
  iree_arena_block_pool_t pool_ = {};
  // Temporary storage for the direct legalization callback.
  iree_arena_allocator_t arena_ = {};
  // Minimal dialect registry, with no target provider or pass pipeline.
  loom_context_t context_ = {};
  // IR owner for one kernel.
  loom_module_t* module_ = nullptr;
  // Fixture builder, independent of the legalizer's builder.
  loom_builder_t builder_ = {};
  // Mutation context consumed by the production legalizer.
  loom_rewriter_t rewriter_ = {};
  // Target topology supplied at the legalizer boundary.
  loom_target_facts_t facts_ = {};
  // Source kernel enclosing the collective.
  loom_op_t* kernel_ = nullptr;
  // Consumer that observes result replacement.
  loom_op_t* use_ = nullptr;
  // Borrowed launch arguments carrying independent payloads.
  const loom_value_id_t* arguments_ = nullptr;
  // Logical types of the source payloads.
  std::vector<loom_type_t> types_;
};

TEST_F(WorkgroupCollectiveLegalizationTest,
       VariadicReductionPublishesOutsideImmutablePartials) {
  // Three subgroups, with both unused fold lanes and a partial last subgroup.
  CreateKernel(9, 4,
               {loom_type_scalar(LOOM_SCALAR_TYPE_I64),
                loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                                    loom_dim_pack_static(2), 0),
                loom_type_scalar(LOOM_SCALAR_TYPE_I1)});
  ASSERT_NO_FATAL_FAILURE(Rewrite(Reduce()));
  const auto stores = Ops(LOOM_OP_VIEW_STORE);
  const auto loads = Ops(LOOM_OP_VIEW_LOAD);
  const auto vector_stores = Ops(LOOM_OP_VECTOR_STORE);
  const auto vector_loads = Ops(LOOM_OP_VECTOR_LOAD);
  ASSERT_EQ(stores.size(), 4u);
  ASSERT_EQ(loads.size(), 4u);
  ASSERT_EQ(vector_stores.size(), 2u);
  ASSERT_EQ(vector_loads.size(), 2u);
  ASSERT_EQ(Ops(LOOM_OP_BUFFER_ALLOCA).size(), 3u);
  const auto subgroup_ids = Ops(LOOM_OP_INDEX_ASSUME);
  const auto lane_ids = Ops(LOOM_OP_KERNEL_SUBGROUP_LANE_ID);
  ASSERT_EQ(subgroup_ids.size(), 1u);
  ASSERT_EQ(lane_ids.size(), 1u);
  const auto subgroup = loom_index_assume_results(subgroup_ids[0]).values[0];
  const auto lane = loom_kernel_subgroup_lane_id_result(lane_ids[0]);
  loom_op_t* partial_stores[] = {stores[0], vector_stores[0], stores[1]};
  loom_op_t* result_stores[] = {stores[2], vector_stores[1], stores[3]};
  loom_op_t* partial_loads[] = {loads[0], vector_loads[0], loads[1]};
  loom_op_t* result_loads[] = {loads[2], vector_loads[1], loads[3]};
  const loom_scalar_type_t storage_types[] = {
      LOOM_SCALAR_TYPE_I64, LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I8};
  const int64_t components[] = {1, 2, 1};
  const int64_t bytes[] = {32, 32, 4};
  for (unsigned i = 0; i < 3; ++i) {
    SCOPED_TRACE(i);
    // View and vector memory operations share these operand positions.
    const auto view = loom_op_operands(partial_stores[i])[1];
    ExpectScratch(view, storage_types[i], 4 * components[i], bytes[i]);
    EXPECT_EQ(loom_op_operands(result_stores[i])[1], view);
    EXPECT_EQ(loom_op_operands(partial_loads[i])[0], view);
    EXPECT_EQ(loom_op_operands(result_loads[i])[0], view);
    ExpectAddress(loom_op_operands(partial_stores[i])[2], subgroup, 0,
                  components[i]);
    ExpectPartialRead(partial_loads[i], loom_op_operands(partial_loads[i])[1],
                      subgroup, lane, components[i]);
    ExpectAddress(loom_op_operands(result_stores[i])[2], LOOM_VALUE_ID_INVALID,
                  3, components[i]);
    ExpectAddress(loom_op_operands(result_loads[i])[1], LOOM_VALUE_ID_INVALID,
                  3, components[i]);
    ExpectGuard(partial_stores[i], LOOM_INDEX_CMP_PREDICATE_EQ, lane, 0);
    ExpectGuard(result_stores[i], LOOM_INDEX_CMP_PREDICATE_EQ, lane, 0);
    ExpectGuard(result_stores[i]->parent_op, LOOM_INDEX_CMP_PREDICATE_EQ,
                subgroup, 0);
    ExpectPhase(partial_stores[i], 0);
    ExpectPhase(partial_loads[i], 1);
    ExpectPhase(result_stores[i], 1);
    ExpectPhase(result_loads[i], 2);
  }
  // Predicate storage widens to a byte and restores the caller's logical type.
  const auto results = loom_test_use_values(use_);
  EXPECT_EQ(results.values[0], loom_view_load_result(loads[2]));
  EXPECT_EQ(results.values[1], loom_vector_load_result(vector_loads[1]));
  const auto* truncate = Definition(results.values[2]);
  ASSERT_TRUE(loom_scalar_trunci_isa(truncate));
  EXPECT_EQ(loom_scalar_trunci_input(truncate),
            loom_view_load_result(loads[3]));
}

class WorkgroupScanPublicationTest
    : public WorkgroupCollectiveLegalizationTest,
      public ::testing::WithParamInterface<
          std::tuple<loom_kernel_workgroup_scan_mode_t,
                     loom_kernel_workgroup_scan_direction_t>> {};

TEST_P(WorkgroupScanPublicationTest, AdjustmentsOccupyDisjointSubgroupSlots) {
  const auto [mode, direction] = GetParam();
  const auto subgroup_mode = mode == LOOM_KERNEL_WORKGROUP_SCAN_MODE_EXCLUSIVE
                                 ? LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE
                                 : LOOM_KERNEL_SUBGROUP_SCAN_MODE_INCLUSIVE;
  const auto subgroup_direction =
      direction == LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_REVERSE
          ? LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_REVERSE
          : LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD;
  CreateKernel(9, 4,
               {loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                                    loom_dim_pack_static(2), 0)});
  ASSERT_NO_FATAL_FAILURE(Rewrite(Scan(mode, direction)));
  const auto stores = Ops(LOOM_OP_VECTOR_STORE);
  const auto loads = Ops(LOOM_OP_VECTOR_LOAD);
  const auto scans = Ops(LOOM_OP_KERNEL_SUBGROUP_SCAN);
  ASSERT_EQ(stores.size(), 2u);
  ASSERT_EQ(loads.size(), 2u);
  ASSERT_EQ(scans.size(), 2u);
  ASSERT_EQ(Ops(LOOM_OP_BUFFER_ALLOCA).size(), 1u);
  const auto subgroup_ids = Ops(LOOM_OP_INDEX_ASSUME);
  const auto lane_ids = Ops(LOOM_OP_KERNEL_SUBGROUP_LANE_ID);
  ASSERT_EQ(subgroup_ids.size(), 1u);
  ASSERT_EQ(lane_ids.size(), 1u);
  const auto subgroup = loom_index_assume_results(subgroup_ids[0]).values[0];
  const auto lane = loom_kernel_subgroup_lane_id_result(lane_ids[0]);
  const auto view = loom_vector_store_view(stores[0]);
  ExpectScratch(view, LOOM_SCALAR_TYPE_I32, 12, 48);
  EXPECT_EQ(loom_vector_store_view(stores[1]), view);
  EXPECT_EQ(loom_vector_load_view(loads[0]), view);
  EXPECT_EQ(loom_vector_load_view(loads[1]), view);
  ExpectAddress(loom_vector_store_indices(stores[0]).values[0], subgroup, 0, 2);
  ExpectAddress(loom_vector_store_indices(stores[1]).values[0], lane, 3, 2);
  ExpectAddress(loom_vector_load_indices(loads[1]).values[0], subgroup, 3, 2);
  ExpectPartialRead(loads[0], loom_vector_load_indices(loads[0]).values[0],
                    subgroup, lane, 2);
  ExpectGuard(stores[0], LOOM_INDEX_CMP_PREDICATE_EQ, lane, 0);
  ExpectGuard(stores[1], LOOM_INDEX_CMP_PREDICATE_ULT, lane, 3);
  ExpectGuard(stores[1]->parent_op, LOOM_INDEX_CMP_PREDICATE_EQ, subgroup, 0);
  ExpectPhase(stores[0], 0);
  ExpectPhase(loads[0], 1);
  ExpectPhase(stores[1], 1);
  ExpectPhase(loads[1], 2);
  EXPECT_EQ(loom_kernel_subgroup_scan_mode(scans[0]), subgroup_mode);
  EXPECT_EQ(loom_kernel_subgroup_scan_direction(scans[0]), subgroup_direction);
  EXPECT_EQ(loom_kernel_subgroup_scan_value(scans[0]), arguments_[0]);
  EXPECT_EQ(loom_kernel_subgroup_scan_mode(scans[1]),
            LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE);
  EXPECT_EQ(loom_kernel_subgroup_scan_direction(scans[1]), subgroup_direction);
  EXPECT_EQ(loom_kernel_subgroup_scan_cluster_size(scans[1]), 3u);
  EXPECT_EQ(loom_kernel_subgroup_scan_value(scans[1]),
            loom_vector_load_result(loads[0]));
  EXPECT_EQ(loom_vector_store_value(stores[1]),
            loom_kernel_subgroup_scan_result(scans[1]));
  const auto* combine = Definition(loom_test_use_values(use_).values[0]);
  ASSERT_TRUE(loom_vector_addi_isa(combine));
  const auto prefix = loom_kernel_subgroup_scan_result(scans[0]);
  const auto adjustment = loom_vector_load_result(loads[1]);
  const bool forward =
      direction == LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_FORWARD;
  EXPECT_EQ(loom_vector_addi_lhs(combine), forward ? adjustment : prefix);
  EXPECT_EQ(loom_vector_addi_rhs(combine), forward ? prefix : adjustment);
}

INSTANTIATE_TEST_SUITE_P(
    Modes, WorkgroupScanPublicationTest,
    ::testing::Combine(
        ::testing::Values(LOOM_KERNEL_WORKGROUP_SCAN_MODE_INCLUSIVE,
                          LOOM_KERNEL_WORKGROUP_SCAN_MODE_EXCLUSIVE),
        ::testing::Values(LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_FORWARD,
                          LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_REVERSE)));

class WorkgroupCollectiveTopologyTest
    : public WorkgroupCollectiveLegalizationTest,
      public ::testing::WithParamInterface<bool> {};

TEST_P(WorkgroupCollectiveTopologyTest, SingleSubgroupNeedsNoScratchOrBarrier) {
  CreateKernel(3, 4, {loom_type_scalar(LOOM_SCALAR_TYPE_I64)});
  ASSERT_NO_FATAL_FAILURE(Rewrite(GetParam() ? Scan() : Reduce()));
  EXPECT_TRUE(Ops(LOOM_OP_BUFFER_ALLOCA).empty());
  EXPECT_TRUE(Ops(LOOM_OP_KERNEL_BARRIER).empty());
  const auto* result = Definition(loom_test_use_values(use_).values[0]);
  EXPECT_EQ(result->kind, GetParam() ? LOOM_OP_KERNEL_SUBGROUP_SCAN
                                     : LOOM_OP_KERNEL_SUBGROUP_REDUCE);
  EXPECT_EQ(loom_op_operands(result)[0], arguments_[0]);
}

TEST_P(WorkgroupCollectiveTopologyTest, LinearFoldNeverRewritesPartials) {
  // More subgroups than lanes: every lane reads the immutable partials itself.
  CreateKernel(5, 2,
               {loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                                    loom_dim_pack_static(2), 0)});
  ASSERT_NO_FATAL_FAILURE(Rewrite(GetParam() ? Scan() : Reduce()));
  const auto stores = Ops(LOOM_OP_VECTOR_STORE);
  const auto loads = Ops(LOOM_OP_VECTOR_LOAD);
  const auto barriers = Ops(LOOM_OP_KERNEL_BARRIER);
  ASSERT_EQ(stores.size(), 1u);
  ASSERT_EQ(loads.size(), 3u);
  ASSERT_EQ(barriers.size(), 1u);
  ASSERT_EQ(Ops(LOOM_OP_BUFFER_ALLOCA).size(), 1u);
  const auto view = loom_vector_store_view(stores[0]);
  ExpectScratch(view, LOOM_SCALAR_TYPE_I32, 6, 24);
  ASSERT_EQ(barriers[0]->parent_op, kernel_);
  EXPECT_LT(stores[0]->parent_op->block_ordinal, barriers[0]->block_ordinal);
  for (unsigned i = 0; i < loads.size(); ++i) {
    EXPECT_EQ(loom_vector_load_view(loads[i]), view);
    ExpectAddress(loom_vector_load_indices(loads[i]).values[0],
                  LOOM_VALUE_ID_INVALID, i, 2);
    EXPECT_EQ(loads[i]->parent_op, kernel_);
    EXPECT_GT(loads[i]->block_ordinal, barriers[0]->block_ordinal);
  }
}

TEST_P(WorkgroupCollectiveTopologyTest, PreservedSumsStayExplicit) {
  CreateKernel(GetParam() ? 3 : 9, 4, {loom_type_scalar(LOOM_SCALAR_TYPE_F32)});
  auto* reduction = Reduce(LOOM_COMBINING_KIND_ADDF,
                           LOOM_KERNEL_COLLECTIVEMATHFLAGS_PRESERVE);
  auto* scan =
      Scan(LOOM_KERNEL_WORKGROUP_SCAN_MODE_INCLUSIVE,
           LOOM_KERNEL_WORKGROUP_SCAN_DIRECTION_FORWARD,
           LOOM_COMBINING_KIND_ADDF, LOOM_KERNEL_COLLECTIVEMATHFLAGS_PRESERVE);
  ASSERT_NO_FATAL_FAILURE(
      Rewrite(reduction, LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT));
  ASSERT_NO_FATAL_FAILURE(
      Legalize(scan, LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT));
  ASSERT_NO_FATAL_FAILURE(Verify());
  EXPECT_TRUE(Ops(LOOM_OP_BUFFER_ALLOCA).empty());
  EXPECT_TRUE(Ops(LOOM_OP_KERNEL_BARRIER).empty());
  EXPECT_EQ(Ops(LOOM_OP_KERNEL_WORKGROUP_REDUCE).size(), 1u);
  EXPECT_EQ(Ops(LOOM_OP_KERNEL_WORKGROUP_SCAN).size(), 1u);
}

INSTANTIATE_TEST_SUITE_P(Kinds, WorkgroupCollectiveTopologyTest,
                         ::testing::Bool());

}  // namespace
}  // namespace loom
