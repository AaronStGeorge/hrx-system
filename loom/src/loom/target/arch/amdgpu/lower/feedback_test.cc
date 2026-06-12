// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/feedback.h"

#include <stdint.h>

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/feedback_abi.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

class AmdgpuFeedbackTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    loom_amdgpu_low_descriptor_registry_initialize(&low_registry_);
    descriptor_set_ = loom_low_descriptor_registry_lookup(
        &low_registry_.registry, IREE_SV("amdgpu.rdna3.core"));
    if (descriptor_set_ == NULL) {
      GTEST_SKIP() << "RDNA3 descriptor set is not linked.";
    }
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    BuildFunctionBody();
  }

  void TearDown() override {
    if (module_ != NULL) {
      loom_module_free(module_);
    }
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_symbol_ref_t AddSymbol(iree_string_view_t name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(&builder_, name, &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return (loom_symbol_ref_t){
        .module_id = 0,
        .symbol_id = symbol_id,
    };
  }

  void BuildFunctionBody() {
    loom_symbol_ref_t target = AddSymbol(IREE_SV("gfx_target"));
    loom_symbol_ref_t callee = AddSymbol(IREE_SV("test_fn"));
    loom_op_t* function_op = NULL;
    IREE_ASSERT_OK(loom_low_func_def_build(
        &builder_, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0,
        /*purity=*/0, /*allocation=*/0, /*schedule=*/0, target, /*abi=*/0,
        loom_make_named_attr_slice(NULL, 0),
        loom_make_named_attr_slice(NULL, 0),
        /*export_symbol=*/LOOM_STRING_ID_INVALID,
        loom_make_named_attr_slice(NULL, 0), callee, /*arg_types=*/NULL,
        /*arg_types_count=*/0, /*result_types=*/NULL, /*result_count=*/0,
        /*tied_results=*/NULL, /*tied_result_count=*/0, /*predicates=*/NULL,
        /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN, &function_op));
    body_block_ = loom_region_entry_block(loom_low_func_def_body(function_op));
    loom_builder_initialize(module_, &module_->arena, body_block_, &builder_);
    builder_.ip.parent_op = function_op;
  }

  std::vector<loom_op_t*> Ops() const {
    std::vector<loom_op_t*> ops;
    loom_op_t* op = NULL;
    loom_block_for_each_op(body_block_, op) { ops.push_back(op); }
    return ops;
  }

  iree_string_view_t String(loom_string_id_t string_id) const {
    return module_->strings.entries[string_id];
  }

  const loom_low_descriptor_t* DescriptorForRef(
      loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    const uint32_t descriptor_ordinal =
        loom_amdgpu_descriptor_ref_ordinal(descriptor_set_, descriptor_ref);
    return loom_low_descriptor_set_descriptor_at(descriptor_set_,
                                                 descriptor_ordinal);
  }

  void ExpectLowOpDescriptorRef(
      const loom_op_t* op, loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    ASSERT_TRUE(loom_low_op_isa(op));
    const loom_low_descriptor_t* descriptor = DescriptorForRef(descriptor_ref);
    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(loom_low_op_descriptor_ordinal(op),
              loom_low_descriptor_set_descriptor_ordinal(descriptor_set_,
                                                         descriptor));
    EXPECT_EQ(ToString(String(loom_low_op_opcode(op))),
              ToString(loom_low_descriptor_set_string(
                  descriptor_set_, descriptor->key_string_offset)));
  }

  void ExpectRel32Attrs(const loom_op_t* op, loom_symbol_ref_t expected_symbol,
                        int64_t expected_byte_offset) const {
    loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
    ASSERT_EQ(attrs.count, 2u);
    ASSERT_EQ(ToString(String(attrs.entries[0].name_id)), "byte_offset");
    EXPECT_EQ(loom_attr_as_i64(attrs.entries[0].value), expected_byte_offset);
    ASSERT_EQ(ToString(String(attrs.entries[1].name_id)), "symbol");
    loom_symbol_ref_t actual_symbol =
        loom_attr_as_symbol(attrs.entries[1].value);
    EXPECT_EQ(actual_symbol.module_id, expected_symbol.module_id);
    EXPECT_EQ(actual_symbol.symbol_id, expected_symbol.symbol_id);
  }

  void ExpectLoadOp(const loom_op_t* op,
                    loom_amdgpu_descriptor_ref_t descriptor_ref,
                    loom_value_id_t expected_address,
                    uint32_t expected_byte_offset,
                    loom_value_id_t expected_result) const {
    ExpectLowOpDescriptorRef(op, descriptor_ref);
    ASSERT_EQ(loom_low_op_operands(op).count, 1u);
    EXPECT_EQ(loom_low_op_operands(op).values[0], expected_address);
    ASSERT_EQ(loom_low_op_attrs(op).count, 1u);
    const loom_named_attr_t attr = loom_low_op_attrs(op).entries[0];
    EXPECT_EQ(ToString(String(attr.name_id)), "offset");
    EXPECT_EQ(loom_attr_as_i64(attr.value), expected_byte_offset);
    EXPECT_EQ(loom_value_slice_get(loom_low_op_results(op), 0),
              expected_result);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t low_registry_ = {};
  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
  loom_block_t* body_block_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(AmdgpuFeedbackTest, LoadsCommonFeedbackConfigValues) {
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &values));

  std::vector<loom_op_t*> ops = Ops();
  ASSERT_EQ(ops.size(), 9u);
  ExpectLowOpDescriptorRef(ops[0], LOOM_AMDGPU_DESCRIPTOR_REF_S_GETPC_B64);
  ASSERT_TRUE(loom_low_slice_isa(ops[1]));
  ASSERT_TRUE(loom_low_slice_isa(ops[2]));
  ExpectLowOpDescriptorRef(
      ops[3], LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32_RHS_SYMBOL_REL32_LO);
  ExpectRel32Attrs(ops[3], config_symbol, 0);
  ExpectLowOpDescriptorRef(
      ops[4], LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32_RHS_SYMBOL_REL32_HI);
  ExpectRel32Attrs(ops[4], config_symbol, 0);
  ASSERT_TRUE(loom_low_concat_isa(ops[5]));
  EXPECT_EQ(values.address, loom_low_concat_result(ops[5]));

  ExpectLoadOp(ops[6], LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY,
               values.address, LOOM_AMDGPU_FEEDBACK_CONFIG_FLAGS_OFFSET,
               values.flags);
  ExpectLoadOp(ops[7], LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX2_OFFSET_ONLY,
               values.address, LOOM_AMDGPU_FEEDBACK_CONFIG_CHANNEL_BASE_OFFSET,
               values.channel_base);
  ExpectLoadOp(ops[8], LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX2_OFFSET_ONLY,
               values.address, LOOM_AMDGPU_FEEDBACK_CONFIG_NOTIFY_SIGNAL_OFFSET,
               values.notify_signal);

  loom_type_t sgpr_type = loom_low_register_type(
      descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  EXPECT_TRUE(loom_type_equal(sgpr_type,
                              loom_module_value_type(module_, values.flags)));
  loom_type_t sgpr_x2_type = loom_low_register_type(
      descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  EXPECT_TRUE(loom_type_equal(sgpr_x2_type,
                              loom_module_value_type(module_, values.address)));
  EXPECT_TRUE(loom_type_equal(
      sgpr_x2_type, loom_module_value_type(module_, values.channel_base)));
  EXPECT_TRUE(loom_type_equal(
      sgpr_x2_type, loom_module_value_type(module_, values.notify_signal)));
}

}  // namespace
