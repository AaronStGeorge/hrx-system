// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include <memory>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/packetization.h"
#include "loom/codegen/low/source_selection.h"
#include "loom/codegen/low/verify.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"
#include "loom/testing/context.h"
#include "loom/testing/gen.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

struct ModuleDeleter {
  void operator()(loom_module_t* module) const { loom_module_free(module); }
};

using ModulePtr = std::unique_ptr<loom_module_t, ModuleDeleter>;

struct SourceCounts {
  uint32_t scalar_integer_ops = 0;
  uint32_t scalar_constants = 0;
  uint32_t vector_integer_ops = 0;
  uint32_t index_madd_ops = 0;
};

static loom_type_t I32Type() { return loom_type_scalar(LOOM_SCALAR_TYPE_I32); }

static loom_type_t IndexType() {
  return loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
}

static loom_type_t Vector4xi32Type() {
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                             loom_dim_pack_static(4), 0);
}

static iree_status_t AddSymbol(loom_builder_t* builder, iree_string_view_t name,
                               loom_symbol_ref_t* out_ref) {
  IREE_ASSERT_ARGUMENT(out_ref);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, name, &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_add_symbol(builder->module, name_id, &symbol_id));
  *out_ref = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = symbol_id,
  };
  return iree_ok_status();
}

static loom_value_id_t PickLatestExactType(const loom_test_gen_values_t* values,
                                           loom_type_t type) {
  for (uint16_t i = values->count; i > 0; --i) {
    const uint16_t index = (uint16_t)(i - 1);
    if (loom_type_equal(values->types[index], type)) {
      return values->entries[index];
    }
  }
  return LOOM_VALUE_ID_INVALID;
}

static iree_status_t GenerateScalarI32Constant(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_op_t* op = nullptr;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      context->builder,
      loom_attr_i64((int64_t)loom_test_gen_next_range(context->gen, 256)),
      I32Type(), LOOM_LOCATION_UNKNOWN, &op));
  loom_test_gen_values_add(context->values, loom_scalar_constant_result(op),
                           I32Type());
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

static iree_status_t GenerateScalarI32Binary(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_value_id_t lhs = loom_test_gen_values_pick_typed(
      context->gen, context->values, LOOM_SCALAR_TYPE_I32);
  loom_value_id_t rhs = loom_test_gen_values_pick_typed(
      context->gen, context->values, LOOM_SCALAR_TYPE_I32);
  if (lhs == LOOM_VALUE_ID_INVALID || rhs == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }
  loom_op_t* op = nullptr;
  if (loom_test_gen_next_bool(context->gen)) {
    IREE_RETURN_IF_ERROR(loom_scalar_addi_build(
        context->builder, 0, lhs, rhs, I32Type(), LOOM_LOCATION_UNKNOWN, &op));
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_subi_build(
        context->builder, 0, lhs, rhs, I32Type(), LOOM_LOCATION_UNKNOWN, &op));
  }
  loom_test_gen_values_add(context->values, loom_op_results(op)[0], I32Type());
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

static iree_status_t GenerateVector4xi32Binary(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  const loom_type_t vector_type = Vector4xi32Type();
  loom_value_id_t lhs = loom_test_gen_values_pick_exact_type(
      context->gen, context->values, vector_type);
  loom_value_id_t rhs = loom_test_gen_values_pick_exact_type(
      context->gen, context->values, vector_type);
  if (lhs == LOOM_VALUE_ID_INVALID || rhs == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }
  loom_op_t* op = nullptr;
  if (loom_test_gen_next_bool(context->gen)) {
    IREE_RETURN_IF_ERROR(loom_vector_addi_build(context->builder, 0, lhs, rhs,
                                                vector_type,
                                                LOOM_LOCATION_UNKNOWN, &op));
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_subi_build(context->builder, 0, lhs, rhs,
                                                vector_type,
                                                LOOM_LOCATION_UNKNOWN, &op));
  }
  loom_test_gen_values_add(context->values, loom_op_results(op)[0],
                           vector_type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

static iree_status_t GenerateIndexMadd(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_value_id_t a = loom_test_gen_values_pick_typed(
      context->gen, context->values, LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t b = loom_test_gen_values_pick_typed(
      context->gen, context->values, LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t c = loom_test_gen_values_pick_typed(
      context->gen, context->values, LOOM_SCALAR_TYPE_INDEX);
  if (a == LOOM_VALUE_ID_INVALID || b == LOOM_VALUE_ID_INVALID ||
      c == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }
  loom_op_t* op = nullptr;
  IREE_RETURN_IF_ERROR(loom_index_madd_build(
      context->builder, a, b, c, IndexType(), LOOM_LOCATION_UNKNOWN, &op));
  loom_test_gen_values_add(context->values, loom_index_madd_result(op),
                           IndexType());
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

static const loom_test_gen_op_hook_t kSourceLowStressHooks[] = {
    {1, GenerateScalarI32Constant, nullptr, nullptr},
    {4, GenerateScalarI32Binary, nullptr, nullptr},
    {4, GenerateVector4xi32Binary, nullptr, nullptr},
    {3, GenerateIndexMadd, nullptr, nullptr},
};

static void CountSourceOp(const loom_op_t* op, SourceCounts* counts) {
  switch (op->kind) {
    case LOOM_OP_SCALAR_ADDI:
    case LOOM_OP_SCALAR_SUBI:
      ++counts->scalar_integer_ops;
      break;
    case LOOM_OP_SCALAR_CONSTANT:
      ++counts->scalar_constants;
      break;
    case LOOM_OP_VECTOR_ADDI:
    case LOOM_OP_VECTOR_SUBI:
      ++counts->vector_integer_ops;
      break;
    case LOOM_OP_INDEX_MADD:
      ++counts->index_madd_ops;
      break;
    default:
      break;
  }
}

static SourceCounts CountSourceBodyOps(const loom_op_t* func_op) {
  SourceCounts counts = {};
  const loom_region_t* body = loom_func_def_body(func_op);
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) { CountSourceOp(op, &counts); }
  }
  return counts;
}

static uint32_t CountLowDescriptorOps(const loom_op_t* low_func_op) {
  uint32_t count = 0;
  const loom_region_t* body = loom_low_func_def_body(low_func_op);
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (loom_low_op_isa(op) || loom_low_const_isa(op)) {
        ++count;
      }
    }
  }
  return count;
}

class SourceLoweringStressTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_testing_context_initialize_all(iree_allocator_system(),
                                                       &context_));
    loom_test_low_descriptor_registry_initialize(&descriptor_registry_);
    loom_test_low_lower_policy_registry_initialize(&policy_registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t GenerateTargetedModule(uint64_t seed, uint16_t op_count,
                                       loom_module_t** out_module,
                                       loom_symbol_ref_t* out_func_ref) {
    IREE_ASSERT_ARGUMENT(out_module);
    IREE_ASSERT_ARGUMENT(out_func_ref);
    *out_module = nullptr;
    *out_func_ref = loom_symbol_ref_null();

    loom_module_t* raw_module = nullptr;
    IREE_RETURN_IF_ERROR(loom_module_allocate(&context_, IREE_SV("stress"),
                                              &block_pool_, nullptr,
                                              context_.allocator, &raw_module));
    ModulePtr module(raw_module);

    loom_builder_t builder;
    loom_builder_initialize(module.get(), &module->arena,
                            loom_module_block(module.get()), &builder);

    loom_symbol_ref_t target_ref = loom_symbol_ref_null();
    IREE_RETURN_IF_ERROR(
        AddSymbol(&builder, IREE_SV("test_target"), &target_ref));
    loom_string_id_t preset_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_builder_intern_string(&builder, IREE_SV("test-low"), &preset_id));
    loom_op_t* target_op = nullptr;
    IREE_RETURN_IF_ERROR(loom_target_profile_build(
        &builder, target_ref, preset_id, loom_named_attr_slice_empty(),
        LOOM_LOCATION_UNKNOWN, &target_op));

    loom_symbol_ref_t func_ref = loom_symbol_ref_null();
    IREE_RETURN_IF_ERROR(AddSymbol(&builder, IREE_SV("generated"), &func_ref));
    const loom_type_t arg_types[] = {
        I32Type(),   I32Type(),   Vector4xi32Type(), Vector4xi32Type(),
        IndexType(), IndexType(), IndexType(),
    };
    const loom_type_t result_types[] = {
        I32Type(),
        Vector4xi32Type(),
        IndexType(),
    };
    loom_op_t* func_op = nullptr;
    IREE_RETURN_IF_ERROR(loom_func_def_build(
        &builder, LOOM_FUNC_DEF_BUILD_FLAG_HAS_TARGET, 0, 0, 0, target_ref, 0,
        loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_empty(), func_ref, arg_types,
        IREE_ARRAYSIZE(arg_types), result_types, IREE_ARRAYSIZE(result_types),
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));

    loom_region_t* body = loom_func_def_body(func_op);
    loom_builder_ip_t saved_ip =
        loom_builder_enter_region(&builder, func_op, body);

    loom_test_gen_values_t values;
    loom_test_gen_values_initialize(&values);
    loom_block_t* entry_block = loom_region_entry_block(body);
    for (uint16_t i = 0; i < entry_block->arg_count; ++i) {
      const loom_value_id_t arg_id = loom_block_arg_id(entry_block, i);
      loom_test_gen_values_add(&values, arg_id,
                               loom_module_value_type(module.get(), arg_id));
    }

    loom_test_gen_t gen;
    loom_test_gen_initialize_seeded(seed, &gen);
    loom_test_gen_body_config_t body_config = {};
    body_config.op_count = op_count;
    body_config.max_nesting_depth = 0;
    body_config.dead_op_probability = 0;
    body_config.value_fan_out = 4;
    loom_test_gen_type_palette_default(&body_config.palette);
    memcpy(body_config.hooks, kSourceLowStressHooks,
           sizeof(kSourceLowStressHooks));
    body_config.hook_count = IREE_ARRAYSIZE(kSourceLowStressHooks);
    IREE_RETURN_IF_ERROR(
        loom_test_gen_body_internal(&gen, &body_config, &builder, &values, 0));

    loom_value_id_t returns[] = {
        PickLatestExactType(&values, I32Type()),
        PickLatestExactType(&values, Vector4xi32Type()),
        PickLatestExactType(&values, IndexType()),
    };
    for (loom_value_id_t return_value : returns) {
      IREE_ASSERT(return_value != LOOM_VALUE_ID_INVALID);
    }
    loom_op_t* return_op = nullptr;
    IREE_RETURN_IF_ERROR(
        loom_func_return_build(&builder, returns, IREE_ARRAYSIZE(returns),
                               LOOM_LOCATION_UNKNOWN, &return_op));
    loom_builder_restore(&builder, saved_ip);

    *out_func_ref = func_ref;
    *out_module = module.release();
    return iree_ok_status();
  }

  iree_status_t VerifySourceModule(loom_module_t* module) {
    loom_verify_options_t options = {};
    loom_verify_result_t result = {};
    IREE_RETURN_IF_ERROR(loom_verify_module(module, &options, &result));
    if (result.error_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "generated source failed verification");
    }
    return iree_ok_status();
  }

  iree_status_t VerifyLowModule(loom_module_t* module) {
    const loom_low_verify_options_t options = {
        .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
        .descriptor_registry = &descriptor_registry_.registry,
        .descriptor_requirements =
            LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        .max_errors = 20,
    };
    loom_low_verify_result_t result = {};
    IREE_RETURN_IF_ERROR(loom_low_verify_module(module, &options, &result));
    if (result.error_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "generated low function failed verification");
    }
    return iree_ok_status();
  }

  iree_status_t LowerGeneratedModule(loom_module_t* module,
                                     loom_symbol_ref_t func_ref,
                                     iree_arena_allocator_t* arena,
                                     loom_low_lower_result_t* out_result) {
    const iree_string_view_t func_name =
        module->strings
            .entries[module->symbols.entries[func_ref.symbol_id].name_id];
    const loom_low_source_selection_options_t selection_options = {
        .func_symbol_name = func_name,
        .descriptor_registry = &descriptor_registry_.registry,
        .policy_registry = &policy_registry_,
    };
    loom_low_source_selection_t selection = {};
    IREE_RETURN_IF_ERROR(loom_low_select_source_func(module, &selection_options,
                                                     arena, &selection));
    const loom_low_lower_options_t lower_options = {
        .target_ref = selection.target_ref,
        .bundle = selection.target_bundle,
        .descriptor_registry = &descriptor_registry_.registry,
        .descriptor_requirements =
            LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        .policy = selection.policy,
        .max_errors = 20,
    };
    return loom_low_lower_function(module, selection.func, &lower_options,
                                   out_result);
  }

  iree_status_t PacketizeGeneratedLowFunction(
      loom_module_t* module, loom_op_t* low_func_op,
      iree_arena_allocator_t* arena,
      loom_low_packetization_t* out_packetization) {
    const loom_low_packetization_options_t options = {
        .descriptor_registry = &descriptor_registry_.registry,
        .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE,
    };
    IREE_RETURN_IF_ERROR(loom_low_packetize_function(
        module, low_func_op, &options, arena, out_packetization));
    return loom_low_packet_validate_sidecars(&out_packetization->schedule,
                                             &out_packetization->allocation);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t descriptor_registry_ = {};
  loom_low_lower_policy_registry_t policy_registry_ = {};
};

TEST_F(SourceLoweringStressTest, GeneratedSupportedSourceLowersAndPacketizes) {
  SourceCounts aggregate_source_counts = {};
  uint32_t aggregate_low_descriptor_ops = 0;
  uint32_t aggregate_schedule_nodes = 0;
  uint32_t aggregate_assignments = 0;

  for (uint64_t seed = 0; seed < 16; ++seed) {
    SCOPED_TRACE(::testing::Message() << "seed=" << seed);
    loom_module_t* module_raw = nullptr;
    loom_symbol_ref_t func_ref = loom_symbol_ref_null();
    IREE_ASSERT_OK(GenerateTargetedModule(seed, 64, &module_raw, &func_ref));
    ModulePtr module(module_raw);

    IREE_ASSERT_OK(VerifySourceModule(module.get()));
    ASSERT_LT(func_ref.symbol_id, module->symbols.count);
    const loom_symbol_t* func_symbol =
        &module->symbols.entries[func_ref.symbol_id];
    ASSERT_TRUE(loom_func_def_isa(func_symbol->defining_op));
    SourceCounts source_counts = CountSourceBodyOps(func_symbol->defining_op);
    aggregate_source_counts.scalar_integer_ops +=
        source_counts.scalar_integer_ops;
    aggregate_source_counts.scalar_constants += source_counts.scalar_constants;
    aggregate_source_counts.vector_integer_ops +=
        source_counts.vector_integer_ops;
    aggregate_source_counts.index_madd_ops += source_counts.index_madd_ops;

    iree_arena_allocator_t lowering_arena;
    iree_arena_initialize(&block_pool_, &lowering_arena);
    loom_low_lower_result_t lower_result = {};
    IREE_ASSERT_OK(LowerGeneratedModule(module.get(), func_ref, &lowering_arena,
                                        &lower_result));
    EXPECT_EQ(lower_result.error_count, 0u);
    EXPECT_EQ(lower_result.remark_count, 0u);
    ASSERT_NE(lower_result.low_func_op, nullptr);
    EXPECT_TRUE(loom_symbol_ref_is_valid(lower_result.low_func_ref));
    IREE_ASSERT_OK(VerifyLowModule(module.get()));

    aggregate_low_descriptor_ops +=
        CountLowDescriptorOps(lower_result.low_func_op);

    iree_arena_allocator_t packet_arena;
    iree_arena_initialize(&block_pool_, &packet_arena);
    loom_low_packetization_t packetization = {};
    IREE_ASSERT_OK(PacketizeGeneratedLowFunction(
        module.get(), lower_result.low_func_op, &packet_arena, &packetization));
    EXPECT_GT(packetization.schedule.scheduled_node_count, 0u);
    EXPECT_GT(packetization.allocation.assignment_count, 0u);
    aggregate_schedule_nodes +=
        (uint32_t)packetization.schedule.scheduled_node_count;
    aggregate_assignments +=
        (uint32_t)packetization.allocation.assignment_count;
    iree_arena_deinitialize(&packet_arena);
    iree_arena_deinitialize(&lowering_arena);
  }

  EXPECT_GT(aggregate_source_counts.scalar_integer_ops, 0u);
  EXPECT_GT(aggregate_source_counts.scalar_constants, 0u);
  EXPECT_GT(aggregate_source_counts.vector_integer_ops, 0u);
  EXPECT_GT(aggregate_source_counts.index_madd_ops, 0u);
  EXPECT_GT(aggregate_low_descriptor_ops, 0u);
  EXPECT_GT(aggregate_schedule_nodes, 0u);
  EXPECT_GT(aggregate_assignments, 0u);
}

}  // namespace
}  // namespace loom
