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
#include "loom/ops/cache.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/feedback_abi.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/signal_abi.h"
#include "loom/target/registers.h"

namespace {

constexpr int64_t kSystemCacheScope = LOOM_CACHE_SCOPE_SYSTEM;

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

  void UseDescriptorSet(iree_string_view_t key) {
    descriptor_set_ =
        loom_low_descriptor_registry_lookup(&low_registry_.registry, key);
    if (descriptor_set_ == NULL) {
      GTEST_SKIP() << "AMDGPU descriptor set is not linked: " << ToString(key);
    }
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

  iree_host_size_t PacketOperandCountForRef(
      loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    const loom_low_descriptor_t* descriptor = DescriptorForRef(descriptor_ref);
    EXPECT_NE(descriptor, nullptr);
    if (descriptor == nullptr) {
      return 0;
    }
    EXPECT_LT(descriptor->canonical_asm_form_ordinal,
              descriptor_set_->asm_form_count);
    if (descriptor->canonical_asm_form_ordinal >=
        descriptor_set_->asm_form_count) {
      return 0;
    }
    return descriptor_set_->asm_forms[descriptor->canonical_asm_form_ordinal]
        .operand_index_count;
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

  iree_status_t BuildFeedbackPacketAddress(
      loom_amdgpu_feedback_packet_address_t* out_packet_address) {
    *out_packet_address = {};
    loom_amdgpu_feedback_config_values_t config_values = {};
    loom_amdgpu_feedback_channel_header_values_t channel_values = {};
    IREE_RETURN_IF_ERROR(
        BuildFeedbackChannelValues(&config_values, &channel_values));
    return loom_amdgpu_build_feedback_uniform_packet_address(
        &builder_, descriptor_set_, channel_values.ring_base,
        LOOM_LOCATION_UNKNOWN, out_packet_address);
  }

  iree_status_t BuildFeedbackChannelValues(
      loom_amdgpu_feedback_config_values_t* out_config_values,
      loom_amdgpu_feedback_channel_header_values_t* out_channel_values) {
    loom_symbol_ref_t config_symbol =
        AddSymbol(IREE_SV("iree_feedback_config"));
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_config_values(
        &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
        out_config_values));
    return loom_amdgpu_build_feedback_channel_header_values(
        &builder_, descriptor_set_, out_config_values->channel_base,
        LOOM_LOCATION_UNKNOWN, out_channel_values);
  }

  void ExpectAttrI64(loom_named_attr_slice_t attrs, iree_string_view_t name,
                     int64_t expected_value) const {
    for (iree_host_size_t i = 0; i < attrs.count; ++i) {
      if (iree_string_view_equal(String(attrs.entries[i].name_id), name)) {
        EXPECT_EQ(loom_attr_as_i64(attrs.entries[i].value), expected_value);
        return;
      }
    }
    FAIL() << "expected attr not found: " << ToString(name);
  }

  const loom_op_t* FindLoadOp(loom_amdgpu_descriptor_ref_t descriptor_ref,
                              loom_value_id_t expected_address,
                              uint32_t expected_byte_offset) const {
    for (loom_op_t* op : OpsForDescriptorRef(descriptor_ref)) {
      loom_value_slice_t operands = loom_low_op_operands(op);
      if (operands.count != 1 || operands.values[0] != expected_address) {
        continue;
      }
      loom_named_attr_slice_t attrs = loom_low_op_attrs(op);
      if (attrs.count != 1 ||
          !iree_string_view_equal(String(attrs.entries[0].name_id),
                                  IREE_SV("offset")) ||
          loom_attr_as_i64(attrs.entries[0].value) != expected_byte_offset) {
        continue;
      }
      return op;
    }
    return nullptr;
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

  void ExpectStoreOpWithBase(const loom_op_t* op,
                             loom_amdgpu_descriptor_ref_t descriptor_ref,
                             loom_value_id_t expected_base,
                             uint32_t expected_byte_offset,
                             uint32_t expected_value_unit_count) const {
    ExpectLowOpDescriptorRef(op, descriptor_ref);
    loom_value_slice_t operands = loom_low_op_operands(op);
    ASSERT_EQ(operands.count, PacketOperandCountForRef(descriptor_ref));
    EXPECT_EQ(operands.values[2], expected_base);
    if (operands.count == 4) {
      const loom_value_t* m0_value =
          loom_module_value(module_, operands.values[3]);
      ASSERT_FALSE(loom_value_is_block_arg(m0_value));
      ExpectLowConstDescriptorRef(loom_value_def_op(m0_value),
                                  LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM);
    }
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
    ExpectAttrI64(loom_low_op_attrs(op), IREE_SV("offset"),
                  expected_byte_offset);
  }

  void ExpectStoreOp(const loom_op_t* op,
                     loom_amdgpu_descriptor_ref_t descriptor_ref,
                     const loom_amdgpu_feedback_packet_address_t& address,
                     uint32_t expected_byte_offset,
                     uint32_t expected_value_unit_count) const {
    ExpectStoreOpWithBase(op, descriptor_ref, address.base,
                          expected_byte_offset, expected_value_unit_count);
    EXPECT_EQ(loom_low_op_operands(op).values[0], address.byte_offset);
  }

  void ExpectPublishStateStore(
      const loom_op_t* op,
      const loom_amdgpu_feedback_packet_address_t& address) const {
    ExpectStoreOp(op, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                  address, LOOM_AMDGPU_FEEDBACK_PACKET_STATE_OFFSET, 1);
    ExpectLowConstU32(loom_low_op_operands(op).values[1],
                      LOOM_AMDGPU_FEEDBACK_PACKET_STATE_READY);
  }

  void ExpectGlobalAtomicAddU64(const loom_op_t* op,
                                loom_value_id_t expected_base,
                                uint32_t expected_byte_offset) const {
    ExpectLowOpDescriptorRef(
        op, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR);
    loom_value_slice_t operands = loom_low_op_operands(op);
    ASSERT_EQ(operands.count,
              PacketOperandCountForRef(
                  LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR));
    EXPECT_EQ(operands.values[2], expected_base);
    if (operands.count == 4) {
      const loom_value_t* m0_value =
          loom_module_value(module_, operands.values[3]);
      ASSERT_FALSE(loom_value_is_block_arg(m0_value));
      ExpectLowConstDescriptorRef(loom_value_def_op(m0_value),
                                  LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM);
    }
    loom_type_t expected_vaddr_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);
    EXPECT_TRUE(
        loom_type_equal(expected_vaddr_type,
                        loom_module_value_type(module_, operands.values[0])));
    loom_type_t expected_value_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
    EXPECT_TRUE(
        loom_type_equal(expected_value_type,
                        loom_module_value_type(module_, operands.values[1])));
    ASSERT_EQ(loom_low_op_results(op).count, 0u);
    ExpectAttrI64(loom_low_op_attrs(op), IREE_SV("offset"),
                  expected_byte_offset);
  }

  void ExpectGlobalAtomicCompareExchangeU64(
      const loom_op_t* op, loom_value_id_t expected_base,
      uint32_t expected_byte_offset, loom_value_id_t expected_result) const {
    ExpectLowOpDescriptorRef(
        op, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_CMPSWAP_B64_RTN_SADDR);
    loom_value_slice_t operands = loom_low_op_operands(op);
    ASSERT_EQ(
        operands.count,
        PacketOperandCountForRef(
            LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_CMPSWAP_B64_RTN_SADDR));
    EXPECT_EQ(operands.values[2], expected_base);
    if (operands.count == 4) {
      const loom_value_t* m0_value =
          loom_module_value(module_, operands.values[3]);
      ASSERT_FALSE(loom_value_is_block_arg(m0_value));
      ExpectLowConstDescriptorRef(loom_value_def_op(m0_value),
                                  LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM);
    }
    loom_type_t expected_vaddr_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);
    EXPECT_TRUE(
        loom_type_equal(expected_vaddr_type,
                        loom_module_value_type(module_, operands.values[0])));
    loom_type_t expected_pair_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 4);
    EXPECT_TRUE(
        loom_type_equal(expected_pair_type,
                        loom_module_value_type(module_, operands.values[1])));
    const loom_value_t* pair_value =
        loom_module_value(module_, operands.values[1]);
    ASSERT_FALSE(loom_value_is_block_arg(pair_value));
    ASSERT_TRUE(loom_low_concat_isa(loom_value_def_op(pair_value)));
    ASSERT_EQ(loom_low_op_results(op).count, 1u);
    EXPECT_EQ(loom_value_slice_get(loom_low_op_results(op), 0),
              expected_result);
    loom_type_t expected_result_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
    EXPECT_TRUE(
        loom_type_equal(expected_result_type,
                        loom_module_value_type(module_, expected_result)));
    ExpectAttrI64(loom_low_op_attrs(op), IREE_SV("offset"),
                  expected_byte_offset);
  }

  void ExpectGlobalLoadB64(const loom_op_t* op, loom_value_id_t expected_base,
                           uint32_t expected_byte_offset,
                           loom_value_id_t expected_result) const {
    ExpectLowOpDescriptorRef(op,
                             LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR);
    loom_value_slice_t operands = loom_low_op_operands(op);
    ASSERT_EQ(operands.count,
              PacketOperandCountForRef(
                  LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR));
    EXPECT_EQ(operands.values[1], expected_base);
    if (operands.count == 3) {
      const loom_value_t* m0_value =
          loom_module_value(module_, operands.values[2]);
      ASSERT_FALSE(loom_value_is_block_arg(m0_value));
      ExpectLowConstDescriptorRef(loom_value_def_op(m0_value),
                                  LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM);
    }
    loom_type_t expected_vaddr_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);
    EXPECT_TRUE(
        loom_type_equal(expected_vaddr_type,
                        loom_module_value_type(module_, operands.values[0])));
    ASSERT_EQ(loom_low_op_results(op).count, 1u);
    EXPECT_EQ(loom_value_slice_get(loom_low_op_results(op), 0),
              expected_result);
    loom_type_t expected_result_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
    EXPECT_TRUE(
        loom_type_equal(expected_result_type,
                        loom_module_value_type(module_, expected_result)));
    ExpectAttrI64(loom_low_op_attrs(op), IREE_SV("offset"),
                  expected_byte_offset);
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

TEST_F(AmdgpuFeedbackTest, BuildsUniformPacketAddress) {
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));

  loom_amdgpu_feedback_packet_address_t packet_address = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_uniform_packet_address(
      &builder_, descriptor_set_, channel_values.ring_base,
      LOOM_LOCATION_UNKNOWN, &packet_address));

  EXPECT_EQ(packet_address.base, channel_values.ring_base);
  ExpectLowConstU32(packet_address.byte_offset, 0);
  loom_type_t vgpr_type = loom_low_register_type(
      descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1);
  EXPECT_TRUE(loom_type_equal(
      vgpr_type, loom_module_value_type(module_, packet_address.byte_offset)));
}

TEST_F(AmdgpuFeedbackTest, IncrementsDroppedPacketCount) {
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));

  IREE_ASSERT_OK(loom_amdgpu_build_feedback_dropped_packet_count_increment(
      &builder_, descriptor_set_, channel_values.address,
      LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> atomic_ops = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR);
  ASSERT_EQ(atomic_ops.size(), 1u);
  ExpectGlobalAtomicAddU64(
      atomic_ops[0], channel_values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_DROPPED_PACKET_COUNT_OFFSET);
  ASSERT_EQ(loom_low_op_attrs(atomic_ops[0]).count, 1u);
}

TEST_F(AmdgpuFeedbackTest, IncrementsDroppedPacketCountWithCdnaM0) {
  UseDescriptorSet(IREE_SV("amdgpu.cdna3.core"));
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));

  IREE_ASSERT_OK(loom_amdgpu_build_feedback_dropped_packet_count_increment(
      &builder_, descriptor_set_, channel_values.address,
      LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> atomic_ops = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR);
  ASSERT_EQ(atomic_ops.size(), 1u);
  ExpectGlobalAtomicAddU64(
      atomic_ops[0], channel_values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_DROPPED_PACKET_COUNT_OFFSET);
  ASSERT_EQ(loom_low_op_operands(atomic_ops[0]).count, 4u);
  ASSERT_EQ(loom_low_op_attrs(atomic_ops[0]).count, 2u);
  ExpectAttrI64(loom_low_op_attrs(atomic_ops[0]), IREE_SV("sc1"), 1);
}

TEST_F(AmdgpuFeedbackTest, IncrementsDroppedPacketCountWithGfx12SystemScope) {
  UseDescriptorSet(IREE_SV("amdgpu.rdna4.core"));
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));

  IREE_ASSERT_OK(loom_amdgpu_build_feedback_dropped_packet_count_increment(
      &builder_, descriptor_set_, channel_values.address,
      LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> atomic_ops = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR);
  ASSERT_EQ(atomic_ops.size(), 1u);
  ExpectGlobalAtomicAddU64(
      atomic_ops[0], channel_values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_DROPPED_PACKET_COUNT_OFFSET);
  ExpectAttrI64(loom_low_op_attrs(atomic_ops[0]), IREE_SV("scope"),
                kSystemCacheScope);
}

TEST_F(AmdgpuFeedbackTest, LoadsReservationHead) {
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));

  loom_value_id_t reservation_head = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_reservation_head_load(
      &builder_, descriptor_set_, channel_values.address, LOOM_LOCATION_UNKNOWN,
      &reservation_head));

  std::vector<loom_op_t*> load_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR);
  ASSERT_EQ(load_ops.size(), 1u);
  ExpectGlobalLoadB64(load_ops[0], channel_values.address,
                      LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET,
                      reservation_head);
  ASSERT_EQ(loom_low_op_attrs(load_ops[0]).count, 2u);
  ExpectAttrI64(loom_low_op_attrs(load_ops[0]), IREE_SV("glc"), 1);
}

TEST_F(AmdgpuFeedbackTest, LoadsReservationHeadWithCdnaM0) {
  UseDescriptorSet(IREE_SV("amdgpu.cdna3.core"));
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));

  loom_value_id_t reservation_head = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_reservation_head_load(
      &builder_, descriptor_set_, channel_values.address, LOOM_LOCATION_UNKNOWN,
      &reservation_head));

  std::vector<loom_op_t*> load_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR);
  ASSERT_EQ(load_ops.size(), 1u);
  ExpectGlobalLoadB64(load_ops[0], channel_values.address,
                      LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET,
                      reservation_head);
  ASSERT_EQ(loom_low_op_operands(load_ops[0]).count, 3u);
  ASSERT_EQ(loom_low_op_attrs(load_ops[0]).count, 3u);
  ExpectAttrI64(loom_low_op_attrs(load_ops[0]), IREE_SV("sc0"), 1);
  ExpectAttrI64(loom_low_op_attrs(load_ops[0]), IREE_SV("sc1"), 1);
}

TEST_F(AmdgpuFeedbackTest, LoadsReservationHeadWithGfx12SystemScope) {
  UseDescriptorSet(IREE_SV("amdgpu.rdna4.core"));
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));

  loom_value_id_t reservation_head = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_reservation_head_load(
      &builder_, descriptor_set_, channel_values.address, LOOM_LOCATION_UNKNOWN,
      &reservation_head));

  std::vector<loom_op_t*> load_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR);
  ASSERT_EQ(load_ops.size(), 1u);
  ExpectGlobalLoadB64(load_ops[0], channel_values.address,
                      LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET,
                      reservation_head);
  ExpectAttrI64(loom_low_op_attrs(load_ops[0]), IREE_SV("scope"),
                kSystemCacheScope);
}

TEST_F(AmdgpuFeedbackTest, LoadsReadTailWithRdnaAcquireOrdering) {
  loom_amdgpu_feedback_config_values_t config_values = {};
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(BuildFeedbackChannelValues(&config_values, &channel_values));

  loom_value_id_t read_tail = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_read_tail_acquire_load(
      &builder_, descriptor_set_, channel_values.address, LOOM_LOCATION_UNKNOWN,
      &read_tail));

  std::vector<loom_op_t*> load_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR);
  ASSERT_EQ(load_ops.size(), 1u);
  ExpectGlobalLoadB64(load_ops[0], channel_values.address,
                      LOOM_AMDGPU_FEEDBACK_CHANNEL_READ_TAIL_OFFSET, read_tail);
  ASSERT_EQ(loom_low_op_attrs(load_ops[0]).count, 2u);
  ExpectAttrI64(loom_low_op_attrs(load_ops[0]), IREE_SV("glc"), 1);

  std::vector<loom_op_t*> waitcnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT);
  ASSERT_EQ(waitcnt_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(waitcnt_ops[0]), IREE_SV("vmcnt"), 0);
  ExpectAttrI64(loom_low_op_attrs(waitcnt_ops[0]), IREE_SV("lgkmcnt"), 15);
  ASSERT_EQ(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL1_INV).size(),
      1u);
  ASSERT_EQ(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL0_INV).size(),
      1u);
}

TEST_F(AmdgpuFeedbackTest, LoadsReadTailWithCdnaAcquireOrdering) {
  UseDescriptorSet(IREE_SV("amdgpu.cdna3.core"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(BuildFeedbackChannelValues(&config_values, &channel_values));

  loom_value_id_t read_tail = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_read_tail_acquire_load(
      &builder_, descriptor_set_, channel_values.address, LOOM_LOCATION_UNKNOWN,
      &read_tail));

  std::vector<loom_op_t*> load_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR);
  ASSERT_EQ(load_ops.size(), 1u);
  ExpectGlobalLoadB64(load_ops[0], channel_values.address,
                      LOOM_AMDGPU_FEEDBACK_CHANNEL_READ_TAIL_OFFSET, read_tail);
  ASSERT_EQ(loom_low_op_operands(load_ops[0]).count, 3u);
  ASSERT_EQ(loom_low_op_attrs(load_ops[0]).count, 3u);
  ExpectAttrI64(loom_low_op_attrs(load_ops[0]), IREE_SV("sc0"), 1);
  ExpectAttrI64(loom_low_op_attrs(load_ops[0]), IREE_SV("sc1"), 1);

  std::vector<loom_op_t*> waitcnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT);
  ASSERT_EQ(waitcnt_ops.size(), 1u);
  std::vector<loom_op_t*> invalidate_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_INV);
  ASSERT_EQ(invalidate_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(invalidate_ops[0]), IREE_SV("sc0"), 1);
  ExpectAttrI64(loom_low_op_attrs(invalidate_ops[0]), IREE_SV("sc1"), 1);
}

TEST_F(AmdgpuFeedbackTest, LoadsReadTailWithGfx12AcquireOrdering) {
  UseDescriptorSet(IREE_SV("amdgpu.rdna4.core"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(BuildFeedbackChannelValues(&config_values, &channel_values));

  loom_value_id_t read_tail = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_read_tail_acquire_load(
      &builder_, descriptor_set_, channel_values.address, LOOM_LOCATION_UNKNOWN,
      &read_tail));

  std::vector<loom_op_t*> load_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR);
  ASSERT_EQ(load_ops.size(), 1u);
  ExpectGlobalLoadB64(load_ops[0], channel_values.address,
                      LOOM_AMDGPU_FEEDBACK_CHANNEL_READ_TAIL_OFFSET, read_tail);
  ExpectAttrI64(loom_low_op_attrs(load_ops[0]), IREE_SV("scope"),
                kSystemCacheScope);

  std::vector<loom_op_t*> loadcnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_LOADCNT);
  ASSERT_EQ(loadcnt_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(loadcnt_ops[0]), IREE_SV("loadcnt"), 0);
  std::vector<loom_op_t*> invalidate_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_INV);
  ASSERT_EQ(invalidate_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(invalidate_ops[0]), IREE_SV("scope"),
                kSystemCacheScope);
}

TEST_F(AmdgpuFeedbackTest, CompareExchangesReservationHeadWithRdnaOrdering) {
  loom_amdgpu_feedback_config_values_t config_values = {};
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(BuildFeedbackChannelValues(&config_values, &channel_values));

  loom_value_id_t old_head = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_amdgpu_build_feedback_reservation_head_compare_exchange_acq_rel(
          &builder_, descriptor_set_, channel_values.address,
          channel_values.ring_base, channel_values.ring_capacity,
          LOOM_LOCATION_UNKNOWN, &old_head));

  std::vector<loom_op_t*> atomic_ops = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_CMPSWAP_B64_RTN_SADDR);
  ASSERT_EQ(atomic_ops.size(), 1u);
  ExpectGlobalAtomicCompareExchangeU64(
      atomic_ops[0], channel_values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET, old_head);
  ASSERT_EQ(loom_low_op_attrs(atomic_ops[0]).count, 2u);
  ExpectAttrI64(loom_low_op_attrs(atomic_ops[0]), IREE_SV("glc"), 1);

  std::vector<loom_op_t*> waitcnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT);
  ASSERT_EQ(waitcnt_ops.size(), 2u);
  std::vector<loom_op_t*> vscnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT_VSCNT);
  ASSERT_EQ(vscnt_ops.size(), 1u);
  ASSERT_EQ(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL1_INV).size(),
      1u);
  ASSERT_EQ(
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL0_INV).size(),
      1u);
}

TEST_F(AmdgpuFeedbackTest, CompareExchangesReservationHeadWithCdnaOrdering) {
  UseDescriptorSet(IREE_SV("amdgpu.cdna3.core"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(BuildFeedbackChannelValues(&config_values, &channel_values));

  loom_value_id_t old_head = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_amdgpu_build_feedback_reservation_head_compare_exchange_acq_rel(
          &builder_, descriptor_set_, channel_values.address,
          channel_values.ring_base, channel_values.ring_capacity,
          LOOM_LOCATION_UNKNOWN, &old_head));

  std::vector<loom_op_t*> atomic_ops = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_CMPSWAP_B64_RTN_SADDR);
  ASSERT_EQ(atomic_ops.size(), 1u);
  ExpectGlobalAtomicCompareExchangeU64(
      atomic_ops[0], channel_values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET, old_head);
  ASSERT_EQ(loom_low_op_operands(atomic_ops[0]).count, 4u);
  ASSERT_EQ(loom_low_op_attrs(atomic_ops[0]).count, 3u);
  ExpectAttrI64(loom_low_op_attrs(atomic_ops[0]), IREE_SV("sc0"), 1);
  ExpectAttrI64(loom_low_op_attrs(atomic_ops[0]), IREE_SV("sc1"), 1);

  std::vector<loom_op_t*> writeback_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_WBL2);
  ASSERT_EQ(writeback_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(writeback_ops[0]), IREE_SV("sc0"), 1);
  ExpectAttrI64(loom_low_op_attrs(writeback_ops[0]), IREE_SV("sc1"), 1);
  std::vector<loom_op_t*> waitcnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT);
  ASSERT_EQ(waitcnt_ops.size(), 1u);
  std::vector<loom_op_t*> invalidate_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_INV);
  ASSERT_EQ(invalidate_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(invalidate_ops[0]), IREE_SV("sc0"), 1);
  ExpectAttrI64(loom_low_op_attrs(invalidate_ops[0]), IREE_SV("sc1"), 1);
}

TEST_F(AmdgpuFeedbackTest, CompareExchangesReservationHeadWithGfx12Ordering) {
  UseDescriptorSet(IREE_SV("amdgpu.rdna4.core"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(BuildFeedbackChannelValues(&config_values, &channel_values));

  loom_value_id_t old_head = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_amdgpu_build_feedback_reservation_head_compare_exchange_acq_rel(
          &builder_, descriptor_set_, channel_values.address,
          channel_values.ring_base, channel_values.ring_capacity,
          LOOM_LOCATION_UNKNOWN, &old_head));

  std::vector<loom_op_t*> atomic_ops = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_CMPSWAP_B64_RTN_SADDR);
  ASSERT_EQ(atomic_ops.size(), 1u);
  ExpectGlobalAtomicCompareExchangeU64(
      atomic_ops[0], channel_values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET, old_head);
  ExpectAttrI64(loom_low_op_attrs(atomic_ops[0]), IREE_SV("scope"),
                kSystemCacheScope);

  std::vector<loom_op_t*> writeback_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_WB);
  ASSERT_EQ(writeback_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(writeback_ops[0]), IREE_SV("scope"),
                kSystemCacheScope);
  std::vector<loom_op_t*> storecnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_STORECNT);
  ASSERT_EQ(storecnt_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(storecnt_ops[0]), IREE_SV("storecnt"), 0);
  std::vector<loom_op_t*> loadcnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_LOADCNT);
  ASSERT_EQ(loadcnt_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(loadcnt_ops[0]), IREE_SV("loadcnt"), 0);
  std::vector<loom_op_t*> invalidate_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_INV);
  ASSERT_EQ(invalidate_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(invalidate_ops[0]), IREE_SV("scope"),
                kSystemCacheScope);
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
  loom_amdgpu_feedback_packet_address_t packet_address = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_uniform_packet_address(
      &builder_, descriptor_set_, channel_values.ring_base,
      LOOM_LOCATION_UNKNOWN, &packet_address));

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
      &builder_, descriptor_set_, &packet_address, &header,
      LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> b32_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR);
  ASSERT_EQ(b32_stores.size(), 8u);
  ExpectStoreOp(
      b32_stores[0], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address, LOOM_AMDGPU_FEEDBACK_PACKET_RECORD_LENGTH_OFFSET, 1);
  ExpectLowConstU32(loom_low_op_operands(b32_stores[0]).values[1],
                    header.record_length);
  ExpectStoreOp(
      b32_stores[1], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address, LOOM_AMDGPU_FEEDBACK_PACKET_HEADER_LENGTH_OFFSET, 1);
  ExpectLowConstU32(
      loom_low_op_operands(b32_stores[1]).values[1],
      LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH |
          ((uint32_t)LOOM_AMDGPU_FEEDBACK_PACKET_KIND_ASAN << 16));
  ExpectStoreOp(b32_stores[2],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                packet_address, LOOM_AMDGPU_FEEDBACK_PACKET_FLAGS_OFFSET, 1);
  ExpectLowConstU32(loom_low_op_operands(b32_stores[2]).values[1],
                    LOOM_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC);
  ExpectStoreOp(b32_stores[3],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                packet_address, LOOM_AMDGPU_FEEDBACK_PACKET_STATE_OFFSET, 1);
  ExpectStoreOp(b32_stores[4],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                packet_address,
                LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_WORKGROUP_ID_X_OFFSET, 1);
  EXPECT_NE(loom_low_op_operands(b32_stores[4]).values[1],
            header.source_workgroup_id_x);
  ExpectStoreOp(b32_stores[5],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                packet_address,
                LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_WORKITEM_ID_X_OFFSET, 1);
  EXPECT_NE(loom_low_op_operands(b32_stores[5]).values[1],
            header.source_workitem_id_x);
  ExpectStoreOp(
      b32_stores[6], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address, LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED0_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[7], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address, LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED1_OFFSET, 1);

  std::vector<loom_op_t*> b64_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR);
  ASSERT_EQ(b64_stores.size(), 4u);
  ExpectStoreOp(b64_stores[0],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
                packet_address, LOOM_AMDGPU_FEEDBACK_PACKET_SEQUENCE_OFFSET, 2);
  EXPECT_NE(loom_low_op_operands(b64_stores[0]).values[1], header.sequence);
  ExpectStoreOp(b64_stores[1],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
                packet_address,
                LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_DISPATCH_PTR_OFFSET, 2);
  EXPECT_NE(loom_low_op_operands(b64_stores[1]).values[1],
            header.source_dispatch_ptr);
  ExpectStoreOp(
      b64_stores[2], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
      packet_address, LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED_ARRAY_0_OFFSET, 2);
  ExpectStoreOp(
      b64_stores[3], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
      packet_address, LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED_ARRAY_1_OFFSET, 2);
}

TEST_F(AmdgpuFeedbackTest, PublishesPacketStateWithRdnaReleaseOrdering) {
  loom_amdgpu_feedback_packet_address_t packet_address = {};
  IREE_ASSERT_OK(BuildFeedbackPacketAddress(&packet_address));
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_publish_packet_state(
      &builder_, descriptor_set_, &packet_address, LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> waitcnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT);
  ASSERT_EQ(waitcnt_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(waitcnt_ops[0]), IREE_SV("vmcnt"), 0);
  ExpectAttrI64(loom_low_op_attrs(waitcnt_ops[0]), IREE_SV("lgkmcnt"), 15);

  std::vector<loom_op_t*> vscnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT_VSCNT);
  ASSERT_EQ(vscnt_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(vscnt_ops[0]), IREE_SV("vscnt"), 0);

  std::vector<loom_op_t*> b32_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR);
  ASSERT_EQ(b32_stores.size(), 1u);
  ExpectPublishStateStore(b32_stores[0], packet_address);
}

TEST_F(AmdgpuFeedbackTest, PublishesPacketStateWithCdnaReleaseOrdering) {
  UseDescriptorSet(IREE_SV("amdgpu.cdna3.core"));
  loom_amdgpu_feedback_packet_address_t packet_address = {};
  IREE_ASSERT_OK(BuildFeedbackPacketAddress(&packet_address));
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_publish_packet_state(
      &builder_, descriptor_set_, &packet_address, LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> writeback_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_WBL2);
  ASSERT_EQ(writeback_ops.size(), 1u);
  ASSERT_EQ(loom_low_op_attrs(writeback_ops[0]).count, 2u);
  ExpectAttrI64(loom_low_op_attrs(writeback_ops[0]), IREE_SV("sc0"), 1);
  ExpectAttrI64(loom_low_op_attrs(writeback_ops[0]), IREE_SV("sc1"), 1);

  std::vector<loom_op_t*> b32_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR);
  ASSERT_EQ(b32_stores.size(), 1u);
  ExpectPublishStateStore(b32_stores[0], packet_address);
  ExpectAttrI64(loom_low_op_attrs(b32_stores[0]), IREE_SV("sc0"), 1);
  ExpectAttrI64(loom_low_op_attrs(b32_stores[0]), IREE_SV("sc1"), 1);
}

TEST_F(AmdgpuFeedbackTest, PublishesPacketStateWithGfx12SystemScope) {
  UseDescriptorSet(IREE_SV("amdgpu.rdna4.core"));
  loom_amdgpu_feedback_packet_address_t packet_address = {};
  IREE_ASSERT_OK(BuildFeedbackPacketAddress(&packet_address));
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_publish_packet_state(
      &builder_, descriptor_set_, &packet_address, LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> writeback_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_WB);
  ASSERT_EQ(writeback_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(writeback_ops[0]), IREE_SV("scope"),
                kSystemCacheScope);
  std::vector<loom_op_t*> storecnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAIT_STORECNT);
  ASSERT_EQ(storecnt_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(storecnt_ops[0]), IREE_SV("storecnt"), 0);

  std::vector<loom_op_t*> b32_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR);
  ASSERT_EQ(b32_stores.size(), 1u);
  ExpectPublishStateStore(b32_stores[0], packet_address);
  ExpectAttrI64(loom_low_op_attrs(b32_stores[0]), IREE_SV("scope"),
                kSystemCacheScope);
}

TEST_F(AmdgpuFeedbackTest, PublishesPacketAndNotifiesHost) {
  loom_symbol_ref_t config_symbol = AddSymbol(IREE_SV("iree_feedback_config"));
  loom_amdgpu_feedback_config_values_t config_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_config_values(
      &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
      &config_values));
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_channel_header_values(
      &builder_, descriptor_set_, config_values.channel_base,
      LOOM_LOCATION_UNKNOWN, &channel_values));
  loom_amdgpu_feedback_packet_address_t packet_address = {};
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_uniform_packet_address(
      &builder_, descriptor_set_, channel_values.ring_base,
      LOOM_LOCATION_UNKNOWN, &packet_address));
  IREE_ASSERT_OK(loom_amdgpu_build_feedback_publish_packet(
      &builder_, descriptor_set_, &packet_address, config_values.notify_signal,
      LOOM_LOCATION_UNKNOWN));

  std::vector<loom_op_t*> waitcnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT);
  ASSERT_EQ(waitcnt_ops.size(), 3u);
  std::vector<loom_op_t*> vscnt_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT_VSCNT);
  ASSERT_EQ(vscnt_ops.size(), 3u);

  std::vector<loom_op_t*> b32_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR);
  ASSERT_EQ(b32_stores.size(), 1u);
  ExpectPublishStateStore(b32_stores[0], packet_address);

  std::vector<loom_op_t*> atomic_ops = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR);
  ASSERT_EQ(atomic_ops.size(), 1u);
  ExpectGlobalAtomicAddU64(atomic_ops[0], config_values.notify_signal,
                           LOOM_AMDGPU_SIGNAL_VALUE_OFFSET);

  const loom_op_t* mailbox_load = FindLoadOp(
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX2_OFFSET_ONLY,
      config_values.notify_signal, LOOM_AMDGPU_SIGNAL_EVENT_MAILBOX_PTR_OFFSET);
  ASSERT_NE(mailbox_load, nullptr);
  const loom_value_id_t mailbox_ptr =
      loom_value_slice_get(loom_low_op_results(mailbox_load), 0);
  const loom_op_t* event_id_load = FindLoadOp(
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY,
      config_values.notify_signal, LOOM_AMDGPU_SIGNAL_EVENT_ID_OFFSET);
  ASSERT_NE(event_id_load, nullptr);
  const loom_value_id_t event_id =
      loom_value_slice_get(loom_low_op_results(event_id_load), 0);

  std::vector<loom_op_t*> b64_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR);
  ASSERT_EQ(b64_stores.size(), 1u);
  ExpectStoreOpWithBase(
      b64_stores[0], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
      mailbox_ptr, /*expected_byte_offset=*/0, /*expected_value_unit_count=*/2);

  std::vector<loom_op_t*> and_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32);
  ASSERT_EQ(and_ops.size(), 1u);
  loom_value_slice_t and_operands = loom_low_op_operands(and_ops[0]);
  ASSERT_EQ(and_operands.count, 2u);
  EXPECT_EQ(and_operands.values[0], event_id);

  std::vector<loom_op_t*> m0_moves =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0);
  ASSERT_EQ(m0_moves.size(), 1u);
  EXPECT_EQ(loom_low_op_operands(m0_moves[0]).values[0],
            loom_value_slice_get(loom_low_op_results(and_ops[0]), 0));
  std::vector<loom_op_t*> sendmsg_ops =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_S_SENDMSG);
  ASSERT_EQ(sendmsg_ops.size(), 1u);
  ExpectAttrI64(loom_low_op_attrs(sendmsg_ops[0]), IREE_SV("message"),
                LOOM_AMDGPU_SIGNAL_INTERRUPT_SENDMSG);
  ASSERT_EQ(loom_low_op_operands(sendmsg_ops[0]).count, 1u);
  EXPECT_EQ(loom_low_op_operands(sendmsg_ops[0]).values[0],
            loom_value_slice_get(loom_low_op_results(m0_moves[0]), 0));
}

}  // namespace
