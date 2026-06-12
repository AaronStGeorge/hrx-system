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

  bool LowOpHasDescriptorRef(
      const loom_op_t* op, loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    if (!loom_low_op_isa(op)) {
      return false;
    }
    const loom_low_descriptor_t* descriptor = DescriptorForRef(descriptor_ref);
    if (descriptor == nullptr) {
      return false;
    }
    return loom_low_op_descriptor_ordinal(op) ==
           loom_low_descriptor_set_descriptor_ordinal(descriptor_set_,
                                                      descriptor);
  }

  std::vector<loom_op_t*> OpsForDescriptorRef(
      loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    std::vector<loom_op_t*> filtered_ops;
    for (loom_op_t* op : Ops()) {
      if (LowOpHasDescriptorRef(op, descriptor_ref)) {
        filtered_ops.push_back(op);
      }
    }
    return filtered_ops;
  }

  void ExpectLowConstU32(loom_value_id_t value, uint32_t expected_value) const {
    const loom_value_t* ir_value = loom_module_value(module_, value);
    ASSERT_FALSE(loom_value_is_block_arg(ir_value));
    const loom_op_t* op = loom_value_def_op(ir_value);
    ASSERT_TRUE(loom_low_const_isa(op));
    ExpectLowConstDescriptorRef(op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32);
    loom_named_attr_slice_t attrs = loom_low_const_attrs(op);
    ASSERT_EQ(attrs.count, 1u);
    EXPECT_EQ(ToString(String(attrs.entries[0].name_id)), "imm32");
    EXPECT_EQ(loom_attr_as_i64(attrs.entries[0].value), expected_value);
  }

  void ExpectLowConstDescriptorRef(
      const loom_op_t* op, loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    ASSERT_TRUE(loom_low_const_isa(op));
    const loom_low_descriptor_t* descriptor = DescriptorForRef(descriptor_ref);
    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(loom_low_const_descriptor_ordinal(op),
              loom_low_descriptor_set_descriptor_ordinal(descriptor_set_,
                                                         descriptor));
    EXPECT_EQ(ToString(String(loom_low_const_opcode(op))),
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

  void ExpectControlOp(const loom_op_t* op,
                       loom_amdgpu_descriptor_ref_t descriptor_ref,
                       iree_string_view_t expected_attr_name,
                       int64_t expected_attr_value,
                       loom_value_id_t expected_result) const {
    ExpectLowOpDescriptorRef(op, descriptor_ref);
    EXPECT_EQ(loom_low_op_operands(op).count, 0u);
    loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
    ASSERT_EQ(attrs.count, 1u);
    EXPECT_EQ(ToString(String(attrs.entries[0].name_id)),
              ToString(expected_attr_name));
    EXPECT_EQ(loom_attr_as_i64(attrs.entries[0].value), expected_attr_value);

    loom_value_slice_t results = loom_low_op_results(op);
    if (expected_result == LOOM_VALUE_ID_INVALID) {
      EXPECT_EQ(results.count, 0u);
    } else {
      ASSERT_EQ(results.count, 1u);
      EXPECT_EQ(loom_value_slice_get(results, 0), expected_result);
    }
  }

  void ExpectStoreOp(const loom_op_t* op,
                     loom_amdgpu_descriptor_ref_t descriptor_ref,
                     loom_value_id_t expected_packet_base,
                     uint32_t expected_byte_offset,
                     uint32_t expected_value_unit_count) const {
    ExpectLowOpDescriptorRef(op, descriptor_ref);
    loom_value_slice_t operands = loom_low_op_operands(op);
    ASSERT_EQ(operands.count, 3u);
    EXPECT_EQ(operands.values[2], expected_packet_base);
    loom_type_t expected_vaddr_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);
    EXPECT_TRUE(
        loom_type_equal(expected_vaddr_type,
                        loom_module_value_type(module_, operands.values[0])));
    loom_type_t expected_value_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
        expected_value_unit_count);
    EXPECT_TRUE(
        loom_type_equal(expected_value_type,
                        loom_module_value_type(module_, operands.values[1])));
    ASSERT_EQ(loom_low_op_results(op).count, 0u);
    loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
    ASSERT_EQ(attrs.count, 1u);
    EXPECT_EQ(ToString(String(attrs.entries[0].name_id)), "offset");
    EXPECT_EQ(loom_attr_as_i64(attrs.entries[0].value), expected_byte_offset);
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

TEST_F(AmdgpuFeedbackTest, LoadsFeedbackChannelHeaderValues) {
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));

  std::vector<loom_op_t*> ops = Ops();
  ASSERT_EQ(ops.size(), 14u);
  EXPECT_EQ(channel_values.address, config_values.channel_base);
  ExpectLoadOp(ops[9], LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY,
               channel_values.address,
               LOOM_AMDGPU_FEEDBACK_CHANNEL_RECORD_LENGTH_OFFSET,
               channel_values.record_length);
  ExpectLoadOp(ops[10], LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY,
               channel_values.address,
               LOOM_AMDGPU_FEEDBACK_CHANNEL_ABI_VERSION_OFFSET,
               channel_values.abi_version);
  ExpectLoadOp(ops[11], LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY,
               channel_values.address,
               LOOM_AMDGPU_FEEDBACK_CHANNEL_FLAGS_OFFSET, channel_values.flags);
  ExpectLoadOp(ops[12], LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX2_OFFSET_ONLY,
               channel_values.address,
               LOOM_AMDGPU_FEEDBACK_CHANNEL_RING_BASE_OFFSET,
               channel_values.ring_base);
  ExpectLoadOp(ops[13], LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX2_OFFSET_ONLY,
               channel_values.address,
               LOOM_AMDGPU_FEEDBACK_CHANNEL_RING_CAPACITY_OFFSET,
               channel_values.ring_capacity);

  loom_type_t sgpr_type = loom_low_register_type(
      descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  EXPECT_TRUE(loom_type_equal(
      sgpr_type,
      loom_module_value_type(module_, channel_values.record_length)));
  EXPECT_TRUE(loom_type_equal(
      sgpr_type, loom_module_value_type(module_, channel_values.abi_version)));
  EXPECT_TRUE(loom_type_equal(
      sgpr_type, loom_module_value_type(module_, channel_values.flags)));
  loom_type_t sgpr_x2_type = loom_low_register_type(
      descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  EXPECT_TRUE(loom_type_equal(
      sgpr_x2_type, loom_module_value_type(module_, channel_values.ring_base)));
  EXPECT_TRUE(loom_type_equal(
      sgpr_x2_type,
      loom_module_value_type(module_, channel_values.ring_capacity)));
}

TEST_F(AmdgpuFeedbackTest, EmitsReservedPacketHeaderStores) {
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));

  loom_amdgpu_feedback_packet_header_t header = {
      .record_length = (uint32_t)loom_amdgpu_feedback_packet_length(
          /*payload_length=*/64),
      .kind = LOOM_AMDGPU_FEEDBACK_PACKET_KIND_ASAN,
      .flags = LOOM_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC,
      .sequence = channel_values.ring_capacity,
      .source_dispatch_ptr = config_values.notify_signal,
      .source_workgroup_id_x = config_values.flags,
      .source_workitem_id_x = channel_values.flags,
  };
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_packet_header(
      &builder_, descriptor_set_, channel_values.ring_base, &header,
      LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> b32_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR);
  ASSERT_EQ(b32_stores.size(), 8u);
  ExpectStoreOp(b32_stores[0],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                channel_values.ring_base,
                LOOM_AMDGPU_FEEDBACK_PACKET_RECORD_LENGTH_OFFSET, 1);
  ExpectLowConstU32(loom_low_op_operands(b32_stores[0]).values[1],
                    header.record_length);
  ExpectStoreOp(b32_stores[1],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                channel_values.ring_base,
                LOOM_AMDGPU_FEEDBACK_PACKET_HEADER_LENGTH_OFFSET, 1);
  ExpectLowConstU32(
      loom_low_op_operands(b32_stores[1]).values[1],
      LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH |
          ((uint32_t)LOOM_AMDGPU_FEEDBACK_PACKET_KIND_ASAN << 16));
  ExpectStoreOp(
      b32_stores[2], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      channel_values.ring_base, LOOM_AMDGPU_FEEDBACK_PACKET_FLAGS_OFFSET, 1);
  ExpectLowConstU32(loom_low_op_operands(b32_stores[2]).values[1],
                    LOOM_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC);
  ExpectStoreOp(
      b32_stores[3], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      channel_values.ring_base, LOOM_AMDGPU_FEEDBACK_PACKET_STATE_OFFSET, 1);
  ExpectStoreOp(b32_stores[4],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                channel_values.ring_base,
                LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_WORKGROUP_ID_X_OFFSET, 1);
  EXPECT_NE(loom_low_op_operands(b32_stores[4]).values[1],
            header.source_workgroup_id_x);
  ExpectStoreOp(b32_stores[5],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                channel_values.ring_base,
                LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_WORKITEM_ID_X_OFFSET, 1);
  EXPECT_NE(loom_low_op_operands(b32_stores[5]).values[1],
            header.source_workitem_id_x);
  ExpectStoreOp(b32_stores[6],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                channel_values.ring_base,
                LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED0_OFFSET, 1);
  ExpectStoreOp(b32_stores[7],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                channel_values.ring_base,
                LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED1_OFFSET, 1);

  std::vector<loom_op_t*> b64_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR);
  ASSERT_EQ(b64_stores.size(), 4u);
  ExpectStoreOp(
      b64_stores[0], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
      channel_values.ring_base, LOOM_AMDGPU_FEEDBACK_PACKET_SEQUENCE_OFFSET, 2);
  EXPECT_NE(loom_low_op_operands(b64_stores[0]).values[1], header.sequence);
  ExpectStoreOp(b64_stores[1],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
                channel_values.ring_base,
                LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_DISPATCH_PTR_OFFSET, 2);
  EXPECT_NE(loom_low_op_operands(b64_stores[1]).values[1],
            header.source_dispatch_ptr);
  ExpectStoreOp(b64_stores[2],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
                channel_values.ring_base,
                LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED_ARRAY_0_OFFSET, 2);
  ExpectStoreOp(b64_stores[3],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
                channel_values.ring_base,
                LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED_ARRAY_1_OFFSET, 2);
}

TEST_F(AmdgpuFeedbackTest, EmitsFeedbackControlPrimitives) {
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_send_message(
      &builder_, descriptor_set_, /*message=*/1, LOOM_LOCATION_UNKNOWN));
  loom_value_id_t message_result = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_send_message_rtn_b32(
      &builder_, descriptor_set_, /*message=*/2, LOOM_LOCATION_UNKNOWN,
      &message_result));
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_halt(
      &builder_, descriptor_set_, /*reason=*/5, LOOM_LOCATION_UNKNOWN));
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_trap(
      &builder_, descriptor_set_, /*trap_id=*/3, LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> ops = Ops();
  ASSERT_EQ(ops.size(), 4u);
  ExpectControlOp(ops[0], LOOM_AMDGPU_DESCRIPTOR_REF_S_SENDMSG,
                  IREE_SV("message"), 1, LOOM_VALUE_ID_INVALID);
  ExpectControlOp(ops[1], LOOM_AMDGPU_DESCRIPTOR_REF_S_SENDMSG_RTN_B32,
                  IREE_SV("message"), 2, message_result);
  ExpectControlOp(ops[2], LOOM_AMDGPU_DESCRIPTOR_REF_S_SETHALT,
                  IREE_SV("reason"), 5, LOOM_VALUE_ID_INVALID);
  ExpectControlOp(ops[3], LOOM_AMDGPU_DESCRIPTOR_REF_S_TRAP, IREE_SV("trapid"),
                  3, LOOM_VALUE_ID_INVALID);

  loom_type_t sgpr_type = loom_low_register_type(
      descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  EXPECT_TRUE(loom_type_equal(sgpr_type,
                              loom_module_value_type(module_, message_result)));
}

}  // namespace
