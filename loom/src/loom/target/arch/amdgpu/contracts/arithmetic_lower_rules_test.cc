// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/contracts/arithmetic_lower_rules.h"

#include <string>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower/lower_rules.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower/materializers.h"
#include "loom/testing/module_ptr.h"

extern "C" bool loom_amdgpu_value_can_materialize_as_vgpr_i32(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  (void)context;
  (void)value_id;
  IREE_ASSERT_UNREACHABLE("unused test materializer");
  return false;
}

extern "C" bool loom_amdgpu_value_can_materialize_as_vgpr_f32(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  (void)context;
  (void)value_id;
  IREE_ASSERT_UNREACHABLE("unused test materializer");
  return false;
}

extern "C" bool loom_amdgpu_value_can_materialize_as_vgpr_address(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  (void)context;
  (void)value_id;
  IREE_ASSERT_UNREACHABLE("unused test materializer");
  return false;
}

extern "C" bool loom_amdgpu_value_can_materialize_as_native_i1_mask(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  (void)context;
  (void)value_id;
  IREE_ASSERT_UNREACHABLE("unused test materializer");
  return false;
}

extern "C" iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_i32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  (void)context;
  (void)source_op;
  (void)source_value;
  *out_low_value = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_UNREACHABLE("unused test materializer");
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "unused test materializer invoked");
}

extern "C" iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_f32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  (void)context;
  (void)source_op;
  (void)source_value;
  *out_low_value = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_UNREACHABLE("unused test materializer");
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "unused test materializer invoked");
}

extern "C" iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  (void)context;
  (void)source_op;
  (void)source_value;
  *out_low_value = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_UNREACHABLE("unused test materializer");
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "unused test materializer invoked");
}

extern "C" iree_status_t loom_amdgpu_lookup_or_materialize_native_i1_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  (void)context;
  (void)source_op;
  (void)source_value;
  *out_low_value = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_UNREACHABLE("unused test materializer");
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "unused test materializer invoked");
}

namespace {

using ModulePtr = ::loom::testing::ModulePtr;

const loom_low_descriptor_t kDescriptor = {};

const loom_target_config_t kTargetConfig = {
    /*.name=*/IREE_SV("amdgpu-test-config"),
    /*.contract_set_key=*/{},
    /*.contract_feature_bits=*/0,
};

const loom_target_export_plan_t kTargetExportPlan = {
    /*.name=*/IREE_SV("amdgpu-test-export"),
};

const loom_target_bundle_t kTargetBundle = {
    /*.name=*/IREE_SV("amdgpu-test-target"),
    /*.snapshot=*/nullptr,
    /*.export_plan=*/&kTargetExportPlan,
    /*.config=*/&kTargetConfig,
};

struct DescriptorResolverState {
  std::vector<iree_string_view_t> hidden_keys;
};

static bool ContainsStringView(const std::vector<iree_string_view_t>& values,
                               iree_string_view_t needle) {
  for (iree_string_view_t value : values) {
    if (iree_string_view_equal(value, needle)) {
      return true;
    }
  }
  return false;
}

iree_status_t ResolveDescriptorRef(
    void* user_data, const loom_low_lower_rule_match_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor) {
  (void)context;
  *out_descriptor = nullptr;
  auto* state = static_cast<DescriptorResolverState*>(user_data);
  if (descriptor_ref == LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE ||
      descriptor_ref >= rule_set->descriptor_ref_count) {
    return iree_ok_status();
  }
  const iree_string_view_t key = rule_set->descriptor_refs[descriptor_ref].key;
  if (ContainsStringView(state->hidden_keys, key)) {
    return iree_ok_status();
  }
  *out_descriptor = &kDescriptor;
  return iree_ok_status();
}

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

std::string JoinKeys(const std::vector<iree_string_view_t>& values) {
  std::string result;
  for (iree_string_view_t value : values) {
    if (!result.empty()) {
      result.append(", ");
    }
    result.append(ToString(value));
  }
  return result;
}

static loom_op_t* FindFirstOp(loom_region_t* region, loom_op_kind_t kind) {
  loom_block_t* block = nullptr;
  loom_region_for_each_block(region, block) {
    loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (op->kind == kind) {
        return op;
      }
      loom_region_t** regions = loom_op_regions(op);
      for (uint16_t i = 0; i < op->region_count; ++i) {
        loom_op_t* nested_op = FindFirstOp(regions[i], kind);
        if (nested_op != nullptr) {
          return nested_op;
        }
      }
    }
  }
  return nullptr;
}

class AmdgpuArithmeticLowerRulesTest : public ::testing::Test {
 protected:
  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  ModulePtr Parse(iree_string_view_t source) {
    const loom_text_parse_options_t parse_options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(source, IREE_SV("amdgpu_rules_test.loom"),
                                  &context_, &block_pool_, &parse_options,
                                  &module));
    IREE_ASSERT(module != nullptr);
    return ModulePtr(module);
  }

  void ExpectSelectedDescriptorKeys(
      iree_string_view_t source, loom_op_kind_t op_kind,
      std::vector<iree_string_view_t> hidden_descriptor_keys,
      std::vector<iree_string_view_t> expected_descriptor_keys) {
    ModulePtr module = Parse(source);
    loom_op_t* function_op = FindFirstOp(module->body, LOOM_OP_FUNC_DEF);
    ASSERT_NE(function_op, nullptr);
    loom_func_like_t function = loom_func_like_cast(module.get(), function_op);
    ASSERT_TRUE(loom_func_like_isa(function));
    loom_op_t* op = FindFirstOp(module->body, op_kind);
    ASSERT_NE(op, nullptr);

    DescriptorResolverState resolver_state = {
        /*.hidden_keys=*/std::move(hidden_descriptor_keys),
    };
    const loom_low_lower_rule_match_context_t match_context = {
        /*.module=*/module.get(),
        /*.function=*/function,
        /*.bundle=*/&kTargetBundle,
        /*.descriptor_set=*/nullptr,
        /*.feature_bits=*/0,
        /*.map_value=*/{},
        /*.can_materialize=*/{},
        /*.descriptor_ref=*/
        {
            /*.fn=*/ResolveDescriptorRef,
            /*.user_data=*/&resolver_state,
        },
        /*.fact_table=*/nullptr,
    };

    loom_low_lower_rule_selection_t selection = {};
    IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
        &match_context, &loom_amdgpu_arithmetic_lower_rule_set, op,
        &selection));
    ASSERT_NE(selection.rule, nullptr);

    std::vector<iree_string_view_t> actual_descriptor_keys;
    for (uint16_t i = 0; i < selection.rule->emit_count; ++i) {
      const uint16_t emit_index = selection.rule->emit_start + i;
      ASSERT_LT(emit_index, loom_amdgpu_arithmetic_lower_rule_set.emit_count);
      const loom_low_lower_emit_t* emit =
          &loom_amdgpu_arithmetic_lower_rule_set.emits[emit_index];
      if (emit->descriptor_ref == LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
        continue;
      }
      ASSERT_LT(emit->descriptor_ref,
                loom_amdgpu_arithmetic_lower_rule_set.descriptor_ref_count);
      const loom_low_descriptor_t* resolved_descriptor = nullptr;
      IREE_ASSERT_OK(
          ResolveDescriptorRef(&resolver_state, &match_context,
                               &loom_amdgpu_arithmetic_lower_rule_set,
                               emit->descriptor_ref, &resolved_descriptor));
      ASSERT_NE(resolved_descriptor, nullptr)
          << "selected rule emits hidden descriptor "
          << ToString(loom_amdgpu_arithmetic_lower_rule_set
                          .descriptor_refs[emit->descriptor_ref]
                          .key);
      actual_descriptor_keys.push_back(
          loom_amdgpu_arithmetic_lower_rule_set
              .descriptor_refs[emit->descriptor_ref]
              .key);
    }

    ASSERT_EQ(actual_descriptor_keys.size(), expected_descriptor_keys.size())
        << "expected: [" << JoinKeys(expected_descriptor_keys) << "], actual: ["
        << JoinKeys(actual_descriptor_keys) << "]";
    for (size_t i = 0; i < expected_descriptor_keys.size(); ++i) {
      EXPECT_TRUE(iree_string_view_equal(actual_descriptor_keys[i],
                                         expected_descriptor_keys[i]))
          << "descriptor " << i << ": expected "
          << ToString(expected_descriptor_keys[i]) << ", got "
          << ToString(actual_descriptor_keys[i]);
    }
  }

  void ExpectNoSelection(
      iree_string_view_t source, loom_op_kind_t op_kind,
      std::vector<iree_string_view_t> hidden_descriptor_keys) {
    ModulePtr module = Parse(source);
    loom_op_t* function_op = FindFirstOp(module->body, LOOM_OP_FUNC_DEF);
    ASSERT_NE(function_op, nullptr);
    loom_func_like_t function = loom_func_like_cast(module.get(), function_op);
    ASSERT_TRUE(loom_func_like_isa(function));
    loom_op_t* op = FindFirstOp(module->body, op_kind);
    ASSERT_NE(op, nullptr);

    DescriptorResolverState resolver_state = {
        /*.hidden_keys=*/std::move(hidden_descriptor_keys),
    };
    const loom_low_lower_rule_match_context_t match_context = {
        /*.module=*/module.get(),
        /*.function=*/function,
        /*.bundle=*/&kTargetBundle,
        /*.descriptor_set=*/nullptr,
        /*.feature_bits=*/0,
        /*.map_value=*/{},
        /*.can_materialize=*/{},
        /*.descriptor_ref=*/
        {
            /*.fn=*/ResolveDescriptorRef,
            /*.user_data=*/&resolver_state,
        },
        /*.fact_table=*/nullptr,
    };

    loom_low_lower_rule_selection_t selection = {};
    IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
        &match_context, &loom_amdgpu_arithmetic_lower_rule_set, op,
        &selection));
    EXPECT_TRUE(selection.has_source_op_span);
    EXPECT_EQ(selection.rule, nullptr);
  }

  loom_context_t context_;
  iree_arena_block_pool_t block_pool_;
};

TEST_F(AmdgpuArithmeticLowerRulesTest, SelectsNativeUnsignedBitfieldExtract) {
  ExpectSelectedDescriptorKeys(
      IREE_SV(R"(
func.def @extractu(%source: vector<1xi32>) -> (vector<1xi32>) {
  %bits = vector.bitfield.extractu %source {offset = 4, width = 4} : vector<1xi32> -> vector<1xi32>
  func.return %bits : vector<1xi32>
}
)"),
      LOOM_OP_VECTOR_BITFIELD_EXTRACTU, {},
      {IREE_SV("amdgpu.v_bfe_u32.offset_width_inline")});
}

TEST_F(AmdgpuArithmeticLowerRulesTest, SelectsUnsignedBitfieldExtractFallback) {
  ExpectSelectedDescriptorKeys(
      IREE_SV(R"(
func.def @extractu(%source: vector<1xi32>) -> (vector<1xi32>) {
  %bits = vector.bitfield.extractu %source {offset = 4, width = 4} : vector<1xi32> -> vector<1xi32>
  func.return %bits : vector<1xi32>
}
)"),
      LOOM_OP_VECTOR_BITFIELD_EXTRACTU,
      {IREE_SV("amdgpu.v_bfe_u32.offset_width_inline")},
      {
          IREE_SV("amdgpu.v_lshrrev_b32.src0_inline"),
          IREE_SV("amdgpu.v_and_b32.src0_inline"),
      });
}

TEST_F(AmdgpuArithmeticLowerRulesTest,
       RejectsUnsignedBitfieldExtractWhenFallbackDescriptorsMissing) {
  ExpectNoSelection(IREE_SV(R"(
func.def @extractu(%source: vector<1xi32>) -> (vector<1xi32>) {
  %bits = vector.bitfield.extractu %source {offset = 4, width = 4} : vector<1xi32> -> vector<1xi32>
  func.return %bits : vector<1xi32>
}
)"),
                    LOOM_OP_VECTOR_BITFIELD_EXTRACTU,
                    {
                        IREE_SV("amdgpu.v_bfe_u32.offset_width_inline"),
                        IREE_SV("amdgpu.v_lshrrev_b32.src0_inline"),
                    });
}

TEST_F(AmdgpuArithmeticLowerRulesTest, SelectsNativeSignedBitfieldExtract) {
  ExpectSelectedDescriptorKeys(
      IREE_SV(R"(
func.def @extracts(%source: vector<1xi32>) -> (vector<1xi32>) {
  %bits = vector.bitfield.extracts %source {offset = 4, width = 4} : vector<1xi32> -> vector<1xi32>
  func.return %bits : vector<1xi32>
}
)"),
      LOOM_OP_VECTOR_BITFIELD_EXTRACTS, {},
      {IREE_SV("amdgpu.v_bfe_i32.offset_width_inline")});
}

TEST_F(AmdgpuArithmeticLowerRulesTest, SelectsSignedBitfieldExtractFallback) {
  ExpectSelectedDescriptorKeys(
      IREE_SV(R"(
func.def @extracts(%source: vector<1xi32>) -> (vector<1xi32>) {
  %bits = vector.bitfield.extracts %source {offset = 4, width = 4} : vector<1xi32> -> vector<1xi32>
  func.return %bits : vector<1xi32>
}
)"),
      LOOM_OP_VECTOR_BITFIELD_EXTRACTS,
      {IREE_SV("amdgpu.v_bfe_i32.offset_width_inline")},
      {
          IREE_SV("amdgpu.v_lshlrev_b32.src0_inline"),
          IREE_SV("amdgpu.v_ashrrev_i32.src0_inline"),
      });
}

TEST_F(AmdgpuArithmeticLowerRulesTest,
       RejectsSignedBitfieldExtractWhenFallbackDescriptorsMissing) {
  ExpectNoSelection(IREE_SV(R"(
func.def @extracts(%source: vector<1xi32>) -> (vector<1xi32>) {
  %bits = vector.bitfield.extracts %source {offset = 4, width = 4} : vector<1xi32> -> vector<1xi32>
  func.return %bits : vector<1xi32>
}
)"),
                    LOOM_OP_VECTOR_BITFIELD_EXTRACTS,
                    {
                        IREE_SV("amdgpu.v_bfe_i32.offset_width_inline"),
                        IREE_SV("amdgpu.v_ashrrev_i32.src0_inline"),
                    });
}

TEST_F(AmdgpuArithmeticLowerRulesTest, SelectsNativeBitfieldInsert) {
  ExpectSelectedDescriptorKeys(IREE_SV(R"(
func.def @insert(%field: vector<1xi32>, %base: vector<1xi32>) -> (vector<1xi32>) {
  %bits = vector.bitfield.insert %field into %base {offset = 8, width = 4} : vector<1xi32>, vector<1xi32>
  func.return %bits : vector<1xi32>
}
)"),
                               LOOM_OP_VECTOR_BITFIELD_INSERT, {},
                               {
                                   IREE_SV("amdgpu.v_lshlrev_b32.src0_inline"),
                                   IREE_SV("amdgpu.v_bfi_b32.src0_lit"),
                               });
}

TEST_F(AmdgpuArithmeticLowerRulesTest, SelectsBitfieldInsertFallback) {
  ExpectSelectedDescriptorKeys(IREE_SV(R"(
func.def @insert(%field: vector<1xi32>, %base: vector<1xi32>) -> (vector<1xi32>) {
  %bits = vector.bitfield.insert %field into %base {offset = 8, width = 4} : vector<1xi32>, vector<1xi32>
  func.return %bits : vector<1xi32>
}
)"),
                               LOOM_OP_VECTOR_BITFIELD_INSERT,
                               {IREE_SV("amdgpu.v_bfi_b32.src0_lit")},
                               {
                                   IREE_SV("amdgpu.v_and_b32.src0_inline"),
                                   IREE_SV("amdgpu.v_lshlrev_b32.src0_inline"),
                                   IREE_SV("amdgpu.v_and_b32.lit"),
                                   IREE_SV("amdgpu.v_or_b32"),
                               });
}

TEST_F(AmdgpuArithmeticLowerRulesTest,
       RejectsBitfieldInsertWhenFallbackDescriptorsMissing) {
  ExpectNoSelection(IREE_SV(R"(
func.def @insert(%field: vector<1xi32>, %base: vector<1xi32>) -> (vector<1xi32>) {
  %bits = vector.bitfield.insert %field into %base {offset = 8, width = 4} : vector<1xi32>, vector<1xi32>
  func.return %bits : vector<1xi32>
}
)"),
                    LOOM_OP_VECTOR_BITFIELD_INSERT,
                    {
                        IREE_SV("amdgpu.v_bfi_b32.src0_lit"),
                        IREE_SV("amdgpu.v_lshlrev_b32.src0_inline"),
                    });
}

}  // namespace
