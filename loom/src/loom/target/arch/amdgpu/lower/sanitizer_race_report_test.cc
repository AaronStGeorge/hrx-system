// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer_race_report.h"

#include <stdint.h>

#include <cstdio>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/renderer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cache.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/target/ops.h"
#include "loom/target/arch/amdgpu/abi/feedback.h"
#include "loom/target/arch/amdgpu/abi/tsan.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/lower/feedback.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"
#include "loom/verify/verify.h"

namespace {

using iree::StatusCode;

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

iree_status_t EmitDiagnosticToStderr(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  (void)user_data;
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  loom_type_formatter_t type_formatter = {};
  IREE_RETURN_IF_ERROR(loom_diagnostic_render_message(
      emission->error, emission->params, emission->param_count, type_formatter,
      &stream));
  std::fprintf(stderr, "low verifier: %.*s\n",
               (int)iree_string_builder_size(&builder),
               iree_string_builder_buffer(&builder));
  iree_string_builder_deinitialize(&builder);
  return iree_ok_status();
}

class AmdgpuSanitizerRaceReportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
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
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
  }

  void BuildFunctionBody() {
    loom_symbol_ref_t target = AddSymbol(IREE_SV("gfx_target"));
    loom_string_id_t contract_set_key = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(
        &builder_, IREE_SV("amdgpu.rdna3.core"), &contract_set_key));
    loom_op_t* target_op = NULL;
    IREE_ASSERT_OK(loom_target_generic_build(
        &builder_,
        LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_CODEGEN_FORMAT |
            LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_ARTIFACT_FORMAT |
            LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_ABI |
            LOOM_TARGET_GENERIC_BUILD_FLAG_HAS_CONTRACT_SET_KEY,
        LOOM_TARGET_GENERIC_KIND_REFERENCE, target,
        LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE, LOOM_TARGET_ARTIFACT_FORMAT_ELF,
        /*default_pointer_bitwidth=*/0, /*index_bitwidth=*/0,
        /*offset_bitwidth=*/0, /*max_workgroup_size_x=*/0,
        /*max_workgroup_size_y=*/0, /*max_workgroup_size_z=*/0,
        /*max_flat_workgroup_size=*/0, /*subgroup_size=*/0,
        /*max_grid_size_x=*/0, /*max_grid_size_y=*/0,
        /*max_grid_size_z=*/0, /*max_flat_grid_size=*/0,
        /*max_workgroup_count_x=*/0, /*max_workgroup_count_y=*/0,
        /*max_workgroup_count_z=*/0, /*memory_space_generic=*/0,
        /*memory_space_global=*/0, /*memory_space_workgroup=*/0,
        /*memory_space_constant=*/0, /*memory_space_private=*/0,
        /*memory_space_host=*/0, /*memory_space_descriptor=*/0,
        LOOM_TARGET_ABI_OBJECT_FUNCTION,
        /*export_symbol=*/LOOM_STRING_ID_INVALID,
        /*linkage=*/0, /*hal_buffer_resource_flags=*/0, contract_set_key,
        /*contract_feature_bits=*/0, LOOM_LOCATION_UNKNOWN, &target_op));
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
    loom_region_t* body = body_block_->parent_region;
    loom_block_t* block = NULL;
    loom_region_for_each_block(body, block) {
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) { ops.push_back(op); }
    }
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

  void ExpectLowOpDescriptorRef(
      const loom_op_t* op, loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    ASSERT_TRUE(loom_low_op_isa(op));
    const loom_low_descriptor_t* descriptor = DescriptorForRef(descriptor_ref);
    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(loom_low_op_descriptor_ordinal(op),
              loom_low_descriptor_set_descriptor_ordinal(descriptor_set_,
                                                         descriptor));
  }

  void ExpectLowConstDescriptorRef(
      const loom_op_t* op, loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    ASSERT_TRUE(loom_low_const_isa(op));
    const loom_low_descriptor_t* descriptor = DescriptorForRef(descriptor_ref);
    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(loom_low_const_descriptor_ordinal(op),
              loom_low_descriptor_set_descriptor_ordinal(descriptor_set_,
                                                         descriptor));
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

  void ExpectStoreOp(const loom_op_t* op,
                     loom_amdgpu_descriptor_ref_t descriptor_ref,
                     const loom_amdgpu_feedback_packet_address_t& address,
                     uint32_t expected_byte_offset,
                     uint32_t expected_value_unit_count) const {
    ExpectLowOpDescriptorRef(op, descriptor_ref);
    loom_value_slice_t operands = loom_low_op_operands(op);
    ASSERT_EQ(operands.count, PacketOperandCountForRef(descriptor_ref));
    EXPECT_EQ(operands.values[0], address.byte_offset);
    EXPECT_EQ(operands.values[2], address.base);
    if (operands.count == 4) {
      const loom_value_t* m0_value =
          loom_module_value(module_, operands.values[3]);
      ASSERT_FALSE(loom_value_is_block_arg(m0_value));
      ExpectLowConstDescriptorRef(loom_value_def_op(m0_value),
                                  LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM);
    }
    const loom_type_t expected_value_type = loom_low_register_type(
        descriptor_set_->stable_id, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
        expected_value_unit_count);
    EXPECT_TRUE(
        loom_type_equal(expected_value_type,
                        loom_module_value_type(module_, operands.values[1])));
    ASSERT_EQ(loom_low_op_results(op).count, 0u);
    ExpectAttrI64(loom_low_op_attrs(op), IREE_SV("offset"),
                  expected_byte_offset);
  }

  void VerifyModuleOk() {
    loom_verify_options_t options = {
        /*.sink=*/{loom_diagnostic_stderr_sink, NULL},
        /*.max_errors=*/20,
    };
    loom_verify_result_t result = {};
    IREE_ASSERT_OK(loom_verify_module(module_, &options, &result));
    EXPECT_EQ(result.error_count, 0u);
  }

  void VerifyLowModuleOk() {
    loom_low_verify_options_t options = {
        /*.descriptor_registry=*/&low_registry_.registry,
        /*.target_selection=*/{},
        /*.emitter=*/{EmitDiagnosticToStderr, NULL},
        /*.provider_list=*/{},
        /*.max_errors=*/20,
    };
    loom_low_verify_scratch_t scratch =
        loom_low_verify_scratch_for_module(module_);
    loom_low_verify_result_t result = {};
    IREE_ASSERT_OK(
        loom_low_verify_module(module_, &options, &scratch, &result));
    EXPECT_EQ(result.error_count, 0u);
  }

  iree_status_t BuildFeedbackValues(
      loom_amdgpu_feedback_config_values_t* out_config_values,
      loom_amdgpu_feedback_channel_header_values_t* out_channel_values,
      loom_amdgpu_feedback_packet_address_t* out_packet_address) {
    loom_symbol_ref_t config_symbol =
        AddSymbol(IREE_SV("iree_feedback_config"));
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_config_values(
        &builder_, descriptor_set_, config_symbol, LOOM_LOCATION_UNKNOWN,
        out_config_values));
    IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_channel_header_values(
        &builder_, descriptor_set_, out_config_values->channel_base,
        LOOM_LOCATION_UNKNOWN, out_channel_values));
    return loom_amdgpu_build_feedback_uniform_packet_address(
        &builder_, descriptor_set_, out_channel_values->ring_base,
        LOOM_LOCATION_UNKNOWN, out_packet_address);
  }

  loom_amdgpu_sanitizer_race_report_t MakeReport(
      const loom_amdgpu_feedback_config_values_t& config_values,
      const loom_amdgpu_feedback_channel_header_values_t& channel_values) {
    return {
        /*.check_kind=*/config_values.flags,
        /*.flags=*/channel_values.flags,
        /*.memory_space=*/config_values.flags,
        /*.current_access_kind=*/channel_values.flags,
        /*.prior_access_kind=*/config_values.flags,
        /*.access_size=*/config_values.flags,
        /*.current_site_id=*/config_values.notify_signal,
        /*.prior_site_id=*/config_values.channel_base,
        /*.memory_address=*/channel_values.ring_base,
        /*.shadow_address=*/config_values.address,
        /*.shadow_value=*/channel_values.ring_capacity,
        /*.prior_workgroup_id_x=*/config_values.flags,
        /*.prior_workgroup_id_y=*/channel_values.flags,
        /*.prior_workgroup_id_z=*/config_values.flags,
        /*.prior_workitem_id_x=*/channel_values.flags,
        /*.prior_workitem_id_y=*/config_values.flags,
        /*.prior_workitem_id_z=*/channel_values.flags,
    };
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t low_registry_ = {};
  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
  loom_block_t* body_block_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(AmdgpuSanitizerRaceReportTest, EmitsRaceReportPayloadStores) {
  loom_amdgpu_feedback_config_values_t config_values = {};
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  loom_amdgpu_feedback_packet_address_t packet_address = {};
  IREE_ASSERT_OK(
      BuildFeedbackValues(&config_values, &channel_values, &packet_address));

  const loom_amdgpu_sanitizer_race_report_t report =
      MakeReport(config_values, channel_values);
  IREE_ASSERT_OK(loom_amdgpu_build_sanitizer_race_report_payload(
      &builder_, descriptor_set_, &packet_address, &report,
      LOOM_LOCATION_UNKNOWN));

  loom_op_t* return_op = NULL;
  IREE_ASSERT_OK(loom_low_return_build(&builder_, /*values=*/NULL,
                                       /*value_count=*/0, LOOM_LOCATION_UNKNOWN,
                                       &return_op));
  VerifyModuleOk();
  VerifyLowModuleOk();

  const uint32_t payload_base = LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH;
  std::vector<loom_op_t*> b32_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR);
  ASSERT_EQ(b32_stores.size(), 14u);
  ExpectStoreOp(b32_stores[0],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                packet_address,
                payload_base + LOOM_AMDGPU_TSAN_REPORT_RECORD_LENGTH_OFFSET, 1);
  ExpectLowConstU32(loom_low_op_operands(b32_stores[0]).values[1],
                    LOOM_AMDGPU_TSAN_REPORT_BYTE_LENGTH);
  ExpectStoreOp(b32_stores[1],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                packet_address,
                payload_base + LOOM_AMDGPU_TSAN_REPORT_ABI_VERSION_OFFSET, 1);
  ExpectLowConstU32(loom_low_op_operands(b32_stores[1]).values[1],
                    LOOM_AMDGPU_TSAN_REPORT_ABI_VERSION);
  ExpectStoreOp(b32_stores[2],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                packet_address,
                payload_base + LOOM_AMDGPU_TSAN_REPORT_CHECK_KIND_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[3], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address, payload_base + LOOM_AMDGPU_TSAN_REPORT_FLAGS_OFFSET, 1);
  ExpectStoreOp(b32_stores[4],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                packet_address,
                payload_base + LOOM_AMDGPU_TSAN_REPORT_MEMORY_SPACE_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[5], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_ACCESS_KIND_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[6], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_ACCESS_KIND_OFFSET, 1);
  ExpectStoreOp(b32_stores[7],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                packet_address,
                payload_base + LOOM_AMDGPU_TSAN_REPORT_ACCESS_SIZE_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[8], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_X_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[9], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_Y_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[10], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_Z_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[11], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_X_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[12], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_Y_OFFSET, 1);
  ExpectStoreOp(
      b32_stores[13], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_Z_OFFSET, 1);

  std::vector<loom_op_t*> b64_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR);
  ASSERT_EQ(b64_stores.size(), 5u);
  ExpectStoreOp(
      b64_stores[0], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_CURRENT_SITE_ID_OFFSET, 2);
  ExpectStoreOp(b64_stores[1],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
                packet_address,
                payload_base + LOOM_AMDGPU_TSAN_REPORT_PRIOR_SITE_ID_OFFSET, 2);
  ExpectStoreOp(
      b64_stores[2], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_MEMORY_ADDRESS_OFFSET, 2);
  ExpectStoreOp(
      b64_stores[3], LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
      packet_address,
      payload_base + LOOM_AMDGPU_TSAN_REPORT_SHADOW_ADDRESS_OFFSET, 2);
  ExpectStoreOp(b64_stores[4],
                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
                packet_address,
                payload_base + LOOM_AMDGPU_TSAN_REPORT_SHADOW_VALUE_OFFSET, 2);
}

TEST_F(AmdgpuSanitizerRaceReportTest, RejectsUnsupportedValueShapes) {
  loom_amdgpu_feedback_config_values_t config_values = {};
  loom_amdgpu_feedback_channel_header_values_t channel_values = {};
  loom_amdgpu_feedback_packet_address_t packet_address = {};
  IREE_ASSERT_OK(
      BuildFeedbackValues(&config_values, &channel_values, &packet_address));

  loom_amdgpu_sanitizer_race_report_t report =
      MakeReport(config_values, channel_values);
  report.check_kind = channel_values.ring_capacity;
  IREE_EXPECT_STATUS_IS(StatusCode::kInternal,
                        loom_amdgpu_build_sanitizer_race_report_payload(
                            &builder_, descriptor_set_, &packet_address,
                            &report, LOOM_LOCATION_UNKNOWN));

  report = MakeReport(config_values, channel_values);
  report.current_site_id = config_values.flags;
  IREE_EXPECT_STATUS_IS(StatusCode::kInternal,
                        loom_amdgpu_build_sanitizer_race_report_payload(
                            &builder_, descriptor_set_, &packet_address,
                            &report, LOOM_LOCATION_UNKNOWN));
}

}  // namespace
