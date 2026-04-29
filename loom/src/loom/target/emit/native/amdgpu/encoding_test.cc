// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/encoding.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/encoding.h"
#include "loom/target/arch/amdgpu/low_registry.h"

namespace loom {
namespace {

uint32_t ReadU32LE(const uint8_t* data) {
  return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
         ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

bool IsSop1SMovB32(uint32_t word) {
  return (word & UINT32_C(0xFF80FF00)) == UINT32_C(0xBE800000);
}

uint32_t ReadVop3pOpSelHi(const uint8_t* data) {
  const uint32_t word0 = ReadU32LE(data);
  const uint32_t word1 = ReadU32LE(data + 4);
  return ((word1 >> 27) & 0x3u) | (((word0 >> 14) & 0x1u) << 2);
}

uint32_t ReadVop3pNeg(const uint8_t* data) {
  return (ReadU32LE(data + 4) >> 29) & 0x7u;
}

std::string AmdgpuVgprRegisterType(uint32_t unit_count) {
  if (unit_count == 1) {
    return "reg<amdgpu.vgpr>";
  }
  return "reg<amdgpu.vgpr x" + std::to_string(unit_count) + ">";
}

class AmdgpuEncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_amdgpu_low_descriptor_registry_initialize(&target_registry_);
    IREE_ASSERT_OK(loom_target_low_descriptor_set_select_for_bundle(
        &target_registry_.registry, &loom_amdgpu_low_target_bundle_gfx11_core,
        LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        &gfx11_descriptor_set_));
  }

  void TearDown() override {
    ResetModule();
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void ResetModule() {
    if (module_ != nullptr) {
      loom_module_free(module_);
      module_ = nullptr;
    }
  }

  loom_module_t* ParseSource(const std::string& source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;

    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("amdgpu_encoding_test.loom"), &context_,
                        &block_pool_, &parse_options, &module));
    EXPECT_NE(module, nullptr);
    return module;
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

  loom_op_t* FindFirstLowFunction(loom_module_t* module) {
    loom_block_t* block = loom_module_block(module);
    loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (loom_low_func_def_isa(op)) {
        return op;
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

  void BuildShiftedCopyTables(iree_arena_allocator_t* arena,
                              loom_low_emission_frame_t* out_frame) {
    const char* body =
        "low.func.def target(@gfx_target) @gfx_kernel(%source: "
        "reg<amdgpu.sgpr x3>, %tail : reg<amdgpu.sgpr>, %value : "
        "reg<amdgpu.vgpr>, %vaddr : reg<amdgpu.vgpr>, %soffset : "
        "reg<amdgpu.sgpr>) {\n"
        "  %shifted = low.copy %source : reg<amdgpu.sgpr x3> -> "
        "reg<amdgpu.sgpr x3>\n"
        "  %resource = low.concat(%shifted, %tail) : "
        "(reg<amdgpu.sgpr x3>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x4>\n"
        "  low.op<amdgpu.buffer_store_dword>(%value, %resource, %vaddr, "
        "%soffset) {offset = 0} : (reg<amdgpu.vgpr>, "
        "reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
        "  low.return\n"
        "}\n";
    std::string source =
        "target.profile @gfx_target preset(\"amdgpu-gfx11\")\n";
    source += body;
    ResetModule();
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

    loom_value_id_t source_value = FindValueIdByName("source");
    loom_value_id_t shifted_value = FindValueIdByName("shifted");
    ASSERT_NE(source_value, LOOM_VALUE_ID_INVALID);
    ASSERT_NE(shifted_value, LOOM_VALUE_ID_INVALID);
    const loom_low_allocation_fixed_value_t fixed_values[] = {
        {
            .value_id = source_value,
            .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
            .location_base = 0,
            .location_count = 3,
        },
        {
            .value_id = shifted_value,
            .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
            .location_base = 1,
            .location_count = 3,
        },
    };
    loom_op_t* low_function = FindFirstLowFunction(module_);
    ASSERT_NE(low_function, nullptr);
    loom_low_emission_frame_options_t frame_options = {
        .descriptor_registry = &target_registry_.registry,
        .allocation_fixed_values = fixed_values,
        .allocation_fixed_value_count = IREE_ARRAYSIZE(fixed_values),
    };
    IREE_ASSERT_OK(loom_low_emission_frame_build(
        module_, low_function, &frame_options, arena, out_frame));
  }

  void BuildTablesForPreset(const char* preset_key, const char* body,
                            iree_arena_allocator_t* arena,
                            loom_low_emission_frame_t* out_frame) {
    std::string source = "target.profile @gfx_target preset(\"";
    source += preset_key;
    source += "\")\n";
    source += body;
    ResetModule();
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

    loom_op_t* low_function = FindFirstLowFunction(module_);
    ASSERT_NE(low_function, nullptr);
    loom_low_emission_frame_options_t frame_options = {
        .descriptor_registry = &target_registry_.registry,
    };
    IREE_ASSERT_OK(loom_low_emission_frame_build(
        module_, low_function, &frame_options, arena, out_frame));
  }

  void BuildGfx11Tables(const char* body, iree_arena_allocator_t* arena,
                        loom_low_emission_frame_t* out_frame) {
    BuildTablesForPreset("amdgpu-gfx11", body, arena, out_frame);
  }

  void AddDirectSymbol(const char* name, loom_symbol_ref_t* out_ref) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(
        module_, iree_make_cstring_view(name), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    *out_ref = (loom_symbol_ref_t){
        .module_id = 0,
        .symbol_id = symbol_id,
    };
  }

  void BuildDirectRegisterType(uint16_t reg_class_id, uint32_t unit_count,
                               loom_type_t* out_type) {
    IREE_ASSERT_OK(loom_low_build_register_type(
        module_, DirectDescriptorSet(), reg_class_id, unit_count, out_type));
  }

  struct DirectRegisterArg {
    // Descriptor-set-local register class ID for the argument.
    uint16_t reg_class_id;
    // Number of contiguous allocation units in the argument type.
    uint32_t unit_count;
  };

  void BeginDirectGfx11Function(const DirectRegisterArg* args,
                                iree_host_size_t arg_count) {
    ResetModule();
    direct_nodes_.clear();
    direct_scheduled_node_indices_.clear();
    direct_blocks_.clear();
    direct_assignments_.clear();
    direct_liveness_value_ids_.clear();
    direct_assignment_indices_by_value_ordinal_.clear();
    direct_function_ = nullptr;
    direct_target_ = {};
    IREE_ASSERT_OK(loom_module_allocate(
        &context_, IREE_SV("amdgpu_encoding_direct_test"), &block_pool_,
        /*hints=*/nullptr, iree_allocator_system(), &module_));

    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &direct_builder_);
    loom_symbol_ref_t target_ref = loom_symbol_ref_null();
    AddDirectSymbol("gfx_target", &target_ref);
    loom_string_id_t preset_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("amdgpu-gfx11"),
                                             &preset_id));
    loom_op_t* profile_op = nullptr;
    IREE_ASSERT_OK(loom_target_profile_build(
        &direct_builder_, target_ref, preset_id, loom_named_attr_slice_empty(),
        /*location=*/0, &profile_op));

    std::vector<loom_type_t> arg_types;
    arg_types.resize(arg_count);
    for (iree_host_size_t i = 0; i < arg_count; ++i) {
      IREE_ASSERT_OK(loom_low_build_register_type(
          module_, DirectDescriptorSet(), args[i].reg_class_id,
          args[i].unit_count, &arg_types[i]));
    }

    loom_symbol_ref_t function_ref = loom_symbol_ref_null();
    AddDirectSymbol("gfx_kernel", &function_ref);
    IREE_ASSERT_OK(loom_low_func_def_build(
        &direct_builder_, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0,
        /*purity=*/0, /*allocation=*/0, /*schedule=*/0, target_ref, /*abi=*/0,
        loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
        loom_named_attr_slice_empty(), function_ref, arg_types.data(),
        arg_types.size(),
        /*result_types=*/nullptr, /*result_count=*/0, /*tied_results=*/nullptr,
        /*tied_result_count=*/0, /*predicates=*/nullptr,
        /*predicates_count=*/0, /*location=*/0, &direct_function_));
    (void)loom_builder_enter_region(&direct_builder_, direct_function_,
                                    loom_low_func_def_body(direct_function_));
    IREE_ASSERT_OK(loom_low_resolve_function_target(
        module_, direct_function_, &target_registry_.registry,
        iree_diagnostic_emitter_t{}, &direct_target_));
    ASSERT_EQ(direct_target_.descriptor_set, DirectDescriptorSet());
  }

  const loom_low_descriptor_set_t* DirectDescriptorSet() const {
    return gfx11_descriptor_set_;
  }

  void AddDirectAssignment(loom_value_id_t value_id, uint16_t reg_class_id,
                           uint32_t location_base, uint32_t unit_count) {
    ASSERT_LE(direct_assignments_.size(), (iree_host_size_t)UINT32_MAX);
    const uint32_t assignment_index = (uint32_t)direct_assignments_.size();
    direct_assignments_.push_back(loom_low_allocation_assignment_t{
        .value_id = value_id,
        .value_class = {},
        .descriptor_reg_class_id = reg_class_id,
        .start_point = 0,
        .end_point = UINT32_MAX,
        .unit_count = unit_count,
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        .location_base = location_base,
        .location_count = unit_count,
    });
    direct_liveness_value_ids_.push_back(value_id);
    direct_assignment_indices_by_value_ordinal_.push_back(assignment_index);
  }

  void AddDirectDescriptorPacket(loom_op_t* op, uint64_t descriptor_id) {
    const loom_low_descriptor_set_t* descriptor_set = DirectDescriptorSet();
    const uint32_t descriptor_ordinal =
        loom_low_descriptor_set_lookup_descriptor_by_id(descriptor_set,
                                                        descriptor_id);
    ASSERT_NE(descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
    const loom_low_descriptor_t* descriptor =
        loom_low_descriptor_set_descriptor_at(descriptor_set,
                                              descriptor_ordinal);
    ASSERT_NE(descriptor, nullptr);
    const uint32_t node_index = (uint32_t)direct_nodes_.size();
    direct_nodes_.push_back(loom_low_schedule_node_t{
        .op = op,
        .block = op->parent_block,
        .block_index = 0,
        .source_ordinal = node_index,
        .scheduled_ordinal = node_index,
        .kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR,
        .traits = 0,
        .descriptor_ordinal = descriptor_ordinal,
        .descriptor_id = descriptor_id,
        .schedule_class_id = descriptor->schedule_class_id,
        .schedule_class_name = iree_string_view_empty(),
        .latency_cycles = 0,
        .latency_kind = LOOM_LOW_LATENCY_KIND_UNKNOWN,
        .model_quality = LOOM_LOW_MODEL_QUALITY_UNKNOWN,
        .issue_use_count = 0,
        .hazard_count = 0,
        .effect_count = descriptor->effect_count,
    });
    direct_scheduled_node_indices_.push_back(node_index);
  }

  void AddDirectTerminatorPacket(loom_op_t* op) {
    const uint32_t node_index = (uint32_t)direct_nodes_.size();
    direct_nodes_.push_back(loom_low_schedule_node_t{
        .op = op,
        .block = op->parent_block,
        .block_index = 0,
        .source_ordinal = node_index,
        .scheduled_ordinal = node_index,
        .kind = LOOM_LOW_SCHEDULE_NODE_TERMINATOR,
        .traits = 0,
        .descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
        .descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE,
        .schedule_class_id = 0,
        .schedule_class_name = iree_string_view_empty(),
        .latency_cycles = 0,
        .latency_kind = LOOM_LOW_LATENCY_KIND_UNKNOWN,
        .model_quality = LOOM_LOW_MODEL_QUALITY_UNKNOWN,
        .issue_use_count = 0,
        .hazard_count = 0,
        .effect_count = 0,
    });
    direct_scheduled_node_indices_.push_back(node_index);
  }

  void FinishDirectTables(loom_low_schedule_table_t* out_schedule,
                          loom_low_allocation_table_t* out_allocation) {
    ASSERT_NE(direct_function_, nullptr);
    ASSERT_LE(direct_nodes_.size(), (iree_host_size_t)UINT32_MAX);
    ASSERT_LE(direct_scheduled_node_indices_.size(),
              (iree_host_size_t)UINT32_MAX);
    direct_blocks_.push_back(loom_low_schedule_block_t{
        .block =
            loom_region_entry_block(loom_low_func_def_body(direct_function_)),
        .node_start = 0,
        .node_count = (uint32_t)direct_nodes_.size(),
        .scheduled_node_start = 0,
        .scheduled_node_count = (uint32_t)direct_scheduled_node_indices_.size(),
    });
    *out_schedule = loom_low_schedule_table_t{
        .module = module_,
        .function_op = direct_function_,
        .target = direct_target_,
        .liveness = {},
        .blocks = direct_blocks_.data(),
        .block_count = direct_blocks_.size(),
        .nodes = direct_nodes_.data(),
        .node_count = direct_nodes_.size(),
        .dependencies = nullptr,
        .dependency_count = 0,
        .scheduled_node_indices = direct_scheduled_node_indices_.data(),
        .scheduled_node_count = direct_scheduled_node_indices_.size(),
        .pressure_steps = nullptr,
        .pressure_step_count = 0,
        .candidate_decisions = nullptr,
        .candidate_decision_count = 0,
        .resource_uses = nullptr,
        .resource_use_count = 0,
        .effect_uses = nullptr,
        .effect_use_count = 0,
        .hazard_uses = nullptr,
        .hazard_use_count = 0,
        .hazard_gaps = nullptr,
        .hazard_gap_count = 0,
        .model_summaries = nullptr,
        .model_summary_count = 0,
        .resource_summaries = nullptr,
        .resource_summary_count = 0,
    };
    *out_allocation = loom_low_allocation_table_t{
        .module = module_,
        .function_op = direct_function_,
        .target = direct_target_,
        .liveness =
            {
                .value_ids = direct_liveness_value_ids_.data(),
                .value_count = direct_liveness_value_ids_.size(),
            },
        .allocation_mode = 0,
        .assignments = direct_assignments_.data(),
        .assignment_count = direct_assignments_.size(),
        .assignment_indices_by_value_ordinal =
            direct_assignment_indices_by_value_ordinal_.data(),
        .spill_plans = nullptr,
        .spill_plan_count = 0,
        .remarks = nullptr,
        .remark_count = 0,
        .copy_decisions = nullptr,
        .copy_decision_count = 0,
        .spill_count = 0,
        .coalesced_copy_count = 0,
        .materialized_copy_count = 0,
    };
  }

  void BuildDirectI64Attr(const char* name, int64_t value,
                          loom_named_attr_t* out_attr) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(
        module_, iree_make_cstring_view(name), &name_id));
    *out_attr = loom_named_attr_t{
        .name_id = name_id,
        .reserved = 0,
        .value = loom_attr_i64(value),
    };
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t target_registry_ = {};
  const loom_low_descriptor_set_t* gfx11_descriptor_set_ = nullptr;
  loom_builder_t direct_builder_ = {};
  loom_op_t* direct_function_ = nullptr;
  loom_low_resolved_target_t direct_target_ = {};
  std::vector<loom_low_schedule_node_t> direct_nodes_;
  std::vector<uint32_t> direct_scheduled_node_indices_;
  std::vector<loom_low_schedule_block_t> direct_blocks_;
  std::vector<loom_low_allocation_assignment_t> direct_assignments_;
  std::vector<loom_value_id_t> direct_liveness_value_ids_;
  std::vector<uint32_t> direct_assignment_indices_by_value_ordinal_;
};

TEST_F(AmdgpuEncodingTest, DirectlyEncodesReturnPacket) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  BeginDirectGfx11Function(/*args=*/nullptr, /*arg_count=*/0);

  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_low_return_build(&direct_builder_, /*values=*/nullptr,
                                       /*values_count=*/0, /*location=*/0,
                                       &return_op));
  AddDirectTerminatorPacket(return_op);

  loom_low_schedule_table_t schedule = {};
  loom_low_allocation_table_t allocation = {};
  FinishDirectTables(&schedule, &allocation);
  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(&schedule, &allocation,
                                                       &text, &arena));

  ASSERT_EQ(text.data_length, 4u);
  EXPECT_EQ(ReadU32LE(text.data), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesStructuralBranchOffsets) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildGfx11Tables(
      "low.func.def target(@gfx_target) @gfx_kernel() {\n"
      "  low.br ^loop\n"
      "^loop:\n"
      "  low.br ^loop\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_EQ(text.data_length, 8u);
  EXPECT_EQ(ReadU32LE(text.data + 0), UINT32_C(0xBFA00000));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(0xBFA0FFFF));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, DirectlyEncodesSop1MovePacket) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  BeginDirectGfx11Function(/*args=*/nullptr, /*arg_count=*/0);

  loom_type_t sgpr = {};
  BuildDirectRegisterType(LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr);
  loom_named_attr_t imm32 = {};
  BuildDirectI64Attr("imm32", 7, &imm32);
  loom_op_t* move_op = nullptr;
  IREE_ASSERT_OK(loom_low_build_descriptor_op(
      &direct_builder_, DirectDescriptorSet(),
      LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B32, /*operands=*/nullptr,
      /*operand_count=*/0, loom_make_named_attr_slice(&imm32, 1), &sgpr,
      /*result_count=*/1, /*tied_results=*/nullptr,
      /*tied_result_count=*/0, /*location=*/0, &move_op));
  AddDirectAssignment(loom_op_results(move_op)[0],
                      LOOM_AMDGPU_REG_CLASS_ID_SGPR,
                      /*location_base=*/0, /*unit_count=*/1);
  AddDirectDescriptorPacket(move_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B32);

  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_low_return_build(&direct_builder_, /*values=*/nullptr,
                                       /*values_count=*/0, /*location=*/0,
                                       &return_op));
  AddDirectTerminatorPacket(return_op);

  loom_low_schedule_table_t schedule = {};
  loom_low_allocation_table_t allocation = {};
  FinishDirectTables(&schedule, &allocation);
  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(&schedule, &allocation,
                                                       &text, &arena));

  ASSERT_EQ(text.data_length, 8u);
  EXPECT_TRUE(IsSop1SMovB32(ReadU32LE(text.data + 0)));
  EXPECT_EQ(ReadU32LE(text.data + 0) & UINT32_C(0xFF), UINT32_C(0x87));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, DirectlyEncodesValuAddPacket) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  const DirectRegisterArg args[] = {
      {LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1},
      {LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1},
  };
  BeginDirectGfx11Function(args, IREE_ARRAYSIZE(args));

  loom_type_t vgpr = {};
  BuildDirectRegisterType(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr);
  const loom_value_id_t operands[] = {
      loom_region_entry_arg_id(loom_low_func_def_body(direct_function_), 0),
      loom_region_entry_arg_id(loom_low_func_def_body(direct_function_), 1),
  };
  loom_op_t* add_op = nullptr;
  IREE_ASSERT_OK(loom_low_build_descriptor_op(
      &direct_builder_, DirectDescriptorSet(),
      LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &vgpr, /*result_count=*/1,
      /*tied_results=*/nullptr, /*tied_result_count=*/0, /*location=*/0,
      &add_op));
  AddDirectAssignment(operands[0], LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                      /*location_base=*/0, /*unit_count=*/1);
  AddDirectAssignment(operands[1], LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                      /*location_base=*/1, /*unit_count=*/1);
  AddDirectAssignment(loom_op_results(add_op)[0], LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                      /*location_base=*/0,
                      /*unit_count=*/1);
  AddDirectDescriptorPacket(add_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32);

  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_low_return_build(&direct_builder_, /*values=*/nullptr,
                                       /*values_count=*/0, /*location=*/0,
                                       &return_op));
  AddDirectTerminatorPacket(return_op);

  loom_low_schedule_table_t schedule = {};
  loom_low_allocation_table_t allocation = {};
  FinishDirectTables(&schedule, &allocation);
  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(&schedule, &allocation,
                                                       &text, &arena));

  ASSERT_EQ(text.data_length, 8u);
  EXPECT_EQ(ReadU32LE(text.data + 0), UINT32_C(0x4A000300));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, DirectlyEncodesMubufStorePacket) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  const DirectRegisterArg args[] = {
      {LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1},
      {LOOM_AMDGPU_REG_CLASS_ID_SGPR, 4},
      {LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1},
      {LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1},
  };
  BeginDirectGfx11Function(args, IREE_ARRAYSIZE(args));

  const loom_value_id_t operands[] = {
      loom_region_entry_arg_id(loom_low_func_def_body(direct_function_), 0),
      loom_region_entry_arg_id(loom_low_func_def_body(direct_function_), 1),
      loom_region_entry_arg_id(loom_low_func_def_body(direct_function_), 2),
      loom_region_entry_arg_id(loom_low_func_def_body(direct_function_), 3),
  };
  loom_named_attr_t offset = {};
  BuildDirectI64Attr("offset", 8, &offset);
  loom_op_t* store_op = nullptr;
  IREE_ASSERT_OK(loom_low_build_descriptor_op(
      &direct_builder_, DirectDescriptorSet(),
      LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_STORE_DWORD, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(&offset, 1),
      /*result_types=*/nullptr, /*result_count=*/0, /*tied_results=*/nullptr,
      /*tied_result_count=*/0, /*location=*/0, &store_op));
  AddDirectAssignment(operands[0], LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                      /*location_base=*/0, /*unit_count=*/1);
  AddDirectAssignment(operands[1], LOOM_AMDGPU_REG_CLASS_ID_SGPR,
                      /*location_base=*/0, /*unit_count=*/4);
  AddDirectAssignment(operands[2], LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                      /*location_base=*/1, /*unit_count=*/1);
  AddDirectAssignment(operands[3], LOOM_AMDGPU_REG_CLASS_ID_SGPR,
                      /*location_base=*/4, /*unit_count=*/1);
  AddDirectDescriptorPacket(store_op,
                            LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_STORE_DWORD);

  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_low_return_build(&direct_builder_, /*values=*/nullptr,
                                       /*values_count=*/0, /*location=*/0,
                                       &return_op));
  AddDirectTerminatorPacket(return_op);

  loom_low_schedule_table_t schedule = {};
  loom_low_allocation_table_t allocation = {};
  FinishDirectTables(&schedule, &allocation);
  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(&schedule, &allocation,
                                                       &text, &arena));

  ASSERT_EQ(text.data_length, 12u);
  EXPECT_EQ(ReadU32LE(text.data + 0), UINT32_C(0xE0680008));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(0x04400001));
  EXPECT_EQ(ReadU32LE(text.data + 8), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, DirectlyEncodesDsStorePacket) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  const DirectRegisterArg args[] = {
      {LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1},
      {LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1},
  };
  BeginDirectGfx11Function(args, IREE_ARRAYSIZE(args));

  const loom_value_id_t operands[] = {
      loom_region_entry_arg_id(loom_low_func_def_body(direct_function_), 0),
      loom_region_entry_arg_id(loom_low_func_def_body(direct_function_), 1),
  };
  loom_named_attr_t offset = {};
  BuildDirectI64Attr("offset", 4, &offset);
  loom_op_t* store_op = nullptr;
  IREE_ASSERT_OK(loom_low_build_descriptor_op(
      &direct_builder_, DirectDescriptorSet(),
      LOOM_AMDGPU_DESCRIPTOR_ID_DS_WRITE_B32, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(&offset, 1),
      /*result_types=*/nullptr, /*result_count=*/0, /*tied_results=*/nullptr,
      /*tied_result_count=*/0, /*location=*/0, &store_op));
  AddDirectAssignment(operands[0], LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                      /*location_base=*/0, /*unit_count=*/1);
  AddDirectAssignment(operands[1], LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                      /*location_base=*/1, /*unit_count=*/1);
  AddDirectDescriptorPacket(store_op, LOOM_AMDGPU_DESCRIPTOR_ID_DS_WRITE_B32);

  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_low_return_build(&direct_builder_, /*values=*/nullptr,
                                       /*values_count=*/0, /*location=*/0,
                                       &return_op));
  AddDirectTerminatorPacket(return_op);

  loom_low_schedule_table_t schedule = {};
  loom_low_allocation_table_t allocation = {};
  FinishDirectTables(&schedule, &allocation);
  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(&schedule, &allocation,
                                                       &text, &arena));

  ASSERT_EQ(text.data_length, 12u);
  EXPECT_EQ(ReadU32LE(text.data + 0), UINT32_C(0xD8340004));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(0x00000100));
  EXPECT_EQ(ReadU32LE(text.data + 8), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesInlineScalarConstantAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildGfx11Tables(
      "low.func.def target(@gfx_target) @gfx_kernel(%resource: "
      "reg<amdgpu.sgpr x4>) {\n"
      "  %c0 = low.op<amdgpu.s_mov_b32>() {imm32 = 7} : () -> "
      "reg<amdgpu.sgpr>\n"
      "  %loaded = low.op<amdgpu.s_buffer_load_dword>(%resource, %c0) "
      "{offset = 0} : (reg<amdgpu.sgpr x4>, reg<amdgpu.sgpr>) -> "
      "reg<amdgpu.sgpr>\n"
      "  low.return\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 8u);
  EXPECT_TRUE(IsSop1SMovB32(ReadU32LE(text.data + 0)));
  EXPECT_EQ(ReadU32LE(text.data + 0) & UINT32_C(0xFF), UINT32_C(0x87));
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesLiteralScalarConstantAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildGfx11Tables(
      "low.func.def target(@gfx_target) @gfx_kernel(%resource: "
      "reg<amdgpu.sgpr x4>) {\n"
      "  %c0 = low.op<amdgpu.s_mov_b32>() {imm32 = 305419896} : () -> "
      "reg<amdgpu.sgpr>\n"
      "  %loaded = low.op<amdgpu.s_buffer_load_dword>(%resource, %c0) "
      "{offset = 0} : (reg<amdgpu.sgpr x4>, reg<amdgpu.sgpr>) -> "
      "reg<amdgpu.sgpr>\n"
      "  low.return\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 12u);
  EXPECT_TRUE(IsSop1SMovB32(ReadU32LE(text.data + 0)));
  EXPECT_EQ(ReadU32LE(text.data + 0) & UINT32_C(0xFF), UINT32_C(0xFF));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(0x12345678));
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesScalarMoveToM0AndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildGfx11Tables(
      "low.func.def target(@gfx_target) @gfx_kernel(%offset: "
      "reg<amdgpu.sgpr>, %value : reg<amdgpu.vgpr>) {\n"
      "  %m0 = low.op<amdgpu.s_mov_b32_m0>(%offset) : (reg<amdgpu.sgpr>) "
      "-> reg<amdgpu.m0>\n"
      "  low.op<amdgpu.ds_write_addtid_b32>(%value, %m0) {offset = 0} : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.m0>)\n"
      "  low.return\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 8u);
  const uint32_t move_word = ReadU32LE(text.data);
  EXPECT_TRUE(IsSop1SMovB32(move_word));
  EXPECT_EQ((move_word >> 16) & UINT32_C(0x7F), UINT32_C(125));
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesLiteralVectorConstantAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildGfx11Tables(
      "low.func.def target(@gfx_target) @gfx_kernel(%resource: "
      "reg<amdgpu.sgpr x4>, %vaddr : reg<amdgpu.vgpr>, %soffset : "
      "reg<amdgpu.sgpr>) {\n"
      "  %v0 = low.const<amdgpu.v_mov_b32> {imm32 = 42} : "
      "reg<amdgpu.vgpr>\n"
      "  low.op<amdgpu.buffer_store_dword>(%v0, %resource, %vaddr, "
      "%soffset) {offset = 0} : (reg<amdgpu.vgpr>, "
      "reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.return\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 12u);
  EXPECT_EQ(ReadU32LE(text.data + 0) & UINT32_C(0xFE01FFFF),
            UINT32_C(0x7E0002FF));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(42));
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesTiedBufferAtomicReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildGfx11Tables(
      "low.func.def target(@gfx_target) @gfx_kernel(%value: "
      "reg<amdgpu.vgpr>, %resource: reg<amdgpu.sgpr x4>, %vaddr: "
      "reg<amdgpu.vgpr>, %soffset: reg<amdgpu.sgpr>) {\n"
      "  %old = low.op<amdgpu.buffer_atomic_add_u32_rtn>(%value, "
      "%resource, %vaddr, %soffset) {offset = 0} : (reg<amdgpu.vgpr>, "
      "reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>) -> "
      "%value as reg<amdgpu.vgpr>\n"
      "  low.return\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_EQ(text.data_length, 12u);
  EXPECT_NE(ReadU32LE(text.data), UINT32_C(0));
  EXPECT_NE(ReadU32LE(text.data + 4), UINT32_C(0));
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, PacksGeneratedVectorRegisterMove) {
  const loom_amdgpu_encoding_table_t* table =
      loom_amdgpu_encoding_table_for_descriptor_set_id(
          loom_low_descriptor_stable_id_from_key(IREE_SV("amdgpu.gfx11.core")));
  ASSERT_NE(table, nullptr);

  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(
      loom_amdgpu_encoding_pack_v_mov_b32_vgpr(table, 7, 3, &packet));
  ASSERT_EQ(packet.word_count, 1u);
  const uint32_t word = packet.words[0];
  EXPECT_EQ(word & UINT32_C(0x1FF), UINT32_C(0x103));
  EXPECT_EQ((word >> 9) & UINT32_C(0xFF), table->v_mov_b32_opcode);
  EXPECT_EQ((word >> 17) & UINT32_C(0xFF), UINT32_C(7));
}

TEST_F(AmdgpuEncodingTest, EncodesLiveInAsNonEmittingPacket) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildGfx11Tables(
      "low.func.def target(@gfx_target) @gfx_kernel() {\n"
      "  %kernarg = low.live_in<amdgpu.kernarg_segment_ptr> : "
      "reg<amdgpu.sgpr x2>\n"
      "  low.return\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_EQ(text.data_length, 4u);
  EXPECT_EQ(ReadU32LE(text.data), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesGenericSoppCacheControl) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildGfx11Tables(
      "low.func.def target(@gfx_target) @gfx_kernel() {\n"
      "  low.op<amdgpu.s_icache_inv>() : ()\n"
      "  low.return\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 8u);
  ASSERT_EQ(text.data_length % 4, 0u);
  bool saw_icache_invalidate = false;
  for (iree_host_size_t i = 0; i + 4 <= text.data_length; i += 4) {
    saw_icache_invalidate |= ReadU32LE(text.data + i) == UINT32_C(0xBFBC0000);
  }
  EXPECT_TRUE(saw_icache_invalidate);
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesGfx12SmemPrefetchPackets) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildTablesForPreset(
      "amdgpu-gfx12",
      "low.func.def target(@gfx_target) @gfx_kernel(%base: "
      "reg<amdgpu.sgpr x2>, %resource : reg<amdgpu.sgpr x4>, "
      "%soffset : reg<amdgpu.sgpr>) {\n"
      "  low.op<amdgpu.s_prefetch_data>(%base, %soffset) {offset = 64, "
      "count = 2} : (reg<amdgpu.sgpr x2>, reg<amdgpu.sgpr>)\n"
      "  low.op<amdgpu.s_buffer_prefetch_data>(%resource, %soffset) "
      "{offset = 128, count = 1} : (reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.sgpr>)\n"
      "  low.op<amdgpu.s_prefetch_inst_pc_rel>(%soffset) {offset = 0, "
      "count = 1} : (reg<amdgpu.sgpr>)\n"
      "  low.return\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 16u);
  ASSERT_EQ(text.data_length % 4, 0u);
  EXPECT_NE(ReadU32LE(text.data), UINT32_C(0));
  EXPECT_NE(ReadU32LE(text.data + 4), UINT32_C(0));
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, SequencesOverlappingCopyBeforeClobber) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildShiftedCopyTables(&arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 16u);
  EXPECT_EQ(ReadU32LE(text.data + 0), UINT32_C(0xBE830002));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(0xBE820001));
  EXPECT_EQ(ReadU32LE(text.data + 8), UINT32_C(0xBE810000));
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesCoalescedConcatWithoutRegisterCopies) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildGfx11Tables(
      "low.func.def target(@gfx_target) @gfx_kernel(%r0: reg<amdgpu.sgpr>, "
      "%r1 : reg<amdgpu.sgpr>, %r2 : reg<amdgpu.sgpr>, %r3 : "
      "reg<amdgpu.sgpr>, %value : reg<amdgpu.vgpr>, %vaddr : "
      "reg<amdgpu.vgpr>) {\n"
      "  %resource = low.concat(%r0, %r1, %r2, %r3) : (reg<amdgpu.sgpr>, "
      "reg<amdgpu.sgpr>, reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> "
      "reg<amdgpu.sgpr x4>\n"
      "  %sum0 = low.op<amdgpu.s_add_u32>(%r0, %r1) : "
      "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  %sum1 = low.op<amdgpu.s_add_u32>(%r2, %r3) : "
      "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  %soffset = low.op<amdgpu.s_add_u32>(%sum0, %sum1) : "
      "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  low.op<amdgpu.buffer_store_dword>(%value, %resource, %vaddr, "
      "%soffset) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.return\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 8u);
  ASSERT_EQ(text.data_length % 4, 0u);
  bool saw_concat_copy = false;
  for (iree_host_size_t i = 0; i < text.data_length; i += 4) {
    saw_concat_copy |= IsSop1SMovB32(ReadU32LE(text.data + i));
  }
  EXPECT_FALSE(saw_concat_copy);
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesInitialGfx11Allowlist) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildGfx11Tables(
      "low.func.def target(@gfx_target) @gfx_kernel(%s0: "
      "reg<amdgpu.sgpr>, %s1 : reg<amdgpu.sgpr>, %v0 : "
      "reg<amdgpu.vgpr>, %v1 : reg<amdgpu.vgpr>, %resource : "
      "reg<amdgpu.sgpr x4>, %vaddr : reg<amdgpu.vgpr>) {\n"
      "  %s_sum = low.op<amdgpu.s_add_u32>(%s0, %s1) : "
      "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  %v_sum = low.op<amdgpu.v_add_u32>(%v0, %v1) : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
      "  %v_product = low.op<amdgpu.v_mul_lo_u32>(%v_sum, %v1) : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
      "  low.op<amdgpu.buffer_store_dword>(%v_product, %resource, %vaddr, "
      "%s_sum) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.op<amdgpu.s_waitcnt>() {vmcnt = 0, lgkmcnt = 0} : ()\n"
      "  low.op<amdgpu.s_wait_idle>() : ()\n"
      "  low.return\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 36u);
  EXPECT_EQ(ReadU32LE(text.data + 0), UINT32_C(0x80000100));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(0x4A000300));
  EXPECT_EQ(ReadU32LE(text.data + 8), UINT32_C(0xD72C0000));
  EXPECT_EQ(ReadU32LE(text.data + 12), UINT32_C(0x00020300));
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 12), UINT32_C(0xBF890007));
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 8), UINT32_C(0xBF8A0000));
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest,
       EncodesScalarAndVectorBasicsForCurrentAmdgpuFamilies) {
  struct Case {
    // Target preset used to select the low descriptor set.
    const char* preset_key;
    // Expected little-endian SOPP instruction word for `s_endpgm 0`.
    uint32_t expected_return_word;
  };
  const Case cases[] = {
      {"amdgpu-gfx950", UINT32_C(0xBF810000)},
      {"amdgpu-gfx11", UINT32_C(0xBFB00000)},
      {"amdgpu-gfx12", UINT32_C(0xBFB00000)},
      {"amdgpu-gfx1250", UINT32_C(0xBFB00000)},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.preset_key);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_low_emission_frame_t frame = {};
    BuildTablesForPreset(
        test_case.preset_key,
        "low.func.def target(@gfx_target) @gfx_kernel(%s0: "
        "reg<amdgpu.sgpr>, %v0 : "
        "reg<amdgpu.vgpr>, %v1 : reg<amdgpu.vgpr>, %resource : "
        "reg<amdgpu.sgpr x4>, %soffset : reg<amdgpu.sgpr>, %vaddr : "
        "reg<amdgpu.vgpr>) {\n"
        "  %loaded = low.op<amdgpu.s_buffer_load_dword>(%resource, "
        "%soffset) {offset = 0} : (reg<amdgpu.sgpr x4>, "
        "reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
        "  %s_sum = low.op<amdgpu.s_add_u32>(%s0, %loaded) : "
        "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
        "  %v_sum = low.op<amdgpu.v_add_u32>(%v0, %v1) : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  %converted = low.op<amdgpu.v_cvt_f32_i32>(%s_sum) : "
        "(reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr>\n"
        "  %v_min = low.op<amdgpu.v_min_f32>(%v_sum, %converted) : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  %v_max = low.op<amdgpu.v_max_f32>(%v_min, %converted) : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  %v_mix = low.op<amdgpu.v_add_u32>(%v_max, %converted) : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  %dot_s = low.op<amdgpu.v_dot4_i32_i8>(%v0, %v1, %v_mix) : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> "
        "reg<amdgpu.vgpr>\n"
        "  %dot_u = low.op<amdgpu.v_dot4_u32_u8>(%v0, %v1, %dot_s) : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> "
        "reg<amdgpu.vgpr>\n"
        "  low.op<amdgpu.buffer_store_dword>(%dot_u, %resource, %vaddr, "
        "%s_sum) {offset = 0} : (reg<amdgpu.vgpr>, "
        "reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
        "  low.return\n"
        "}\n",
        &arena, &frame);

    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
        &frame.schedule, &frame.allocation, &text, &arena));

    ASSERT_GT(text.data_length, 4u);
    EXPECT_EQ(text.data_length % 4, 0u);
    EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4),
              test_case.expected_return_word);
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuEncodingTest, EncodesGfx11PackedDotSourceModifiers) {
  struct Case {
    const char* low_op_name;
    uint32_t expected_neg;
  };
  const Case cases[] = {
      {"amdgpu.v_dot8_i32_i4", 3},
      {"amdgpu.v_dot8_i32_iu4.s4u4", 1},
      {"amdgpu.v_dot8_i32_iu4.u4s4", 2},
      {"amdgpu.v_dot8_u32_u4", 0},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.low_op_name);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_low_emission_frame_t frame = {};
    std::string body =
        "low.func.def target(@gfx_target) @gfx_kernel(%lhs: "
        "reg<amdgpu.vgpr>, %rhs : reg<amdgpu.vgpr>, %acc : "
        "reg<amdgpu.vgpr>, %resource : reg<amdgpu.sgpr x4>, %vaddr : "
        "reg<amdgpu.vgpr>, %soffset : reg<amdgpu.sgpr>) {\n"
        "  %dot = low.op<";
    body += test_case.low_op_name;
    body +=
        ">(%lhs, %rhs, %acc) : (reg<amdgpu.vgpr>, "
        "reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  low.op<amdgpu.buffer_store_dword>(%dot, %resource, %vaddr, "
        "%soffset) {offset = 0} : (reg<amdgpu.vgpr>, "
        "reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
        "  low.return\n"
        "}\n";
    BuildGfx11Tables(body.c_str(), &arena, &frame);

    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
        &frame.schedule, &frame.allocation, &text, &arena));

    ASSERT_GE(text.data_length, 20u);
    EXPECT_EQ(ReadVop3pOpSelHi(text.data), 7u);
    EXPECT_EQ(ReadVop3pNeg(text.data), test_case.expected_neg);
    EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4),
              UINT32_C(0xBFB00000));
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuEncodingTest, EncodesGfx11MubufStoreAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildGfx11Tables(
      "low.func.def target(@gfx_target) @gfx_kernel(%value: "
      "reg<amdgpu.vgpr>, %resource : reg<amdgpu.sgpr x4>, %vaddr : "
      "reg<amdgpu.vgpr>, %soffset : reg<amdgpu.sgpr>) {\n"
      "  low.op<amdgpu.buffer_store_dword>(%value, %resource, %vaddr, "
      "%soffset) {offset = 8} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.return\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_EQ(text.data_length, 12u);
  EXPECT_EQ(ReadU32LE(text.data + 0), UINT32_C(0xE0680008));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(0x04400001));
  EXPECT_EQ(ReadU32LE(text.data + 8), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesGfx11MubufLoadAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildGfx11Tables(
      "low.func.def target(@gfx_target) @gfx_kernel(%resource: "
      "reg<amdgpu.sgpr x4>, %vaddr : reg<amdgpu.vgpr>, %soffset : "
      "reg<amdgpu.sgpr>) {\n"
      "  %loaded = low.op<amdgpu.buffer_load_dword>(%resource, %vaddr, "
      "%soffset) {offset = 12} : (reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, "
      "reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr>\n"
      "  low.return\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_EQ(text.data_length, 12u);
  EXPECT_EQ(ReadU32LE(text.data + 0), UINT32_C(0xE050000C));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(0x04400000));
  EXPECT_EQ(ReadU32LE(text.data + 8), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesGfx11MubufOffZeroLoadStoreAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildGfx11Tables(
      "low.func.def target(@gfx_target) @gfx_kernel(%resource: "
      "reg<amdgpu.sgpr x4>) {\n"
      "  %loaded = low.op<amdgpu.buffer_load_dword_off_zero>(%resource) "
      "{offset = 12} : (reg<amdgpu.sgpr x4>) -> reg<amdgpu.vgpr>\n"
      "  low.op<amdgpu.buffer_store_dword_off_zero>(%loaded, %resource) "
      "{offset = 16} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>)\n"
      "  low.return\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_EQ(text.data_length, 20u);
  EXPECT_EQ(ReadU32LE(text.data + 0), UINT32_C(0xE050000C));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(0x80000000));
  EXPECT_EQ(ReadU32LE(text.data + 8), UINT32_C(0xE0680010));
  EXPECT_EQ(ReadU32LE(text.data + 12), UINT32_C(0x80000000));
  EXPECT_EQ(ReadU32LE(text.data + 16), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesGfx11MubufB128LoadStoreAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildGfx11Tables(
      "low.func.def target(@gfx_target) @gfx_kernel(%resource: "
      "reg<amdgpu.sgpr x4>, %vaddr : reg<amdgpu.vgpr>, %soffset : "
      "reg<amdgpu.sgpr>) {\n"
      "  %loaded = low.op<amdgpu.buffer_load_b128>(%resource, %vaddr, "
      "%soffset) {offset = 16} : (reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, "
      "reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr x4>\n"
      "  low.op<amdgpu.buffer_store_b128>(%loaded, %resource, %vaddr, "
      "%soffset) {offset = 32} : (reg<amdgpu.vgpr x4>, "
      "reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.return\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_EQ(text.data_length, 20u);
  EXPECT_EQ(ReadU32LE(text.data + 0), UINT32_C(0xE05C0010));
  EXPECT_EQ(ReadU32LE(text.data + 4), UINT32_C(0x04400400));
  EXPECT_EQ(ReadU32LE(text.data + 8), UINT32_C(0xE0740020));
  EXPECT_EQ(ReadU32LE(text.data + 12), UINT32_C(0x04400400));
  EXPECT_EQ(ReadU32LE(text.data + 16), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesBufferB128ForCurrentAmdgpuFamilies) {
  struct Case {
    // Target preset used to select the low descriptor set.
    const char* preset_key;
    // Descriptor key for a 128-bit buffer load on this descriptor set.
    const char* load_key;
    // Descriptor key for a 128-bit buffer store on this descriptor set.
    const char* store_key;
    // Expected little-endian SOPP instruction word for `s_endpgm 0`.
    uint32_t expected_return_word;
  };
  const Case cases[] = {
      {"amdgpu-gfx950", "amdgpu.buffer_load_dwordx4",
       "amdgpu.buffer_store_dwordx4", UINT32_C(0xBF810000)},
      {"amdgpu-gfx11", "amdgpu.buffer_load_b128", "amdgpu.buffer_store_b128",
       UINT32_C(0xBFB00000)},
      {"amdgpu-gfx12", "amdgpu.buffer_load_b128", "amdgpu.buffer_store_b128",
       UINT32_C(0xBFB00000)},
      {"amdgpu-gfx1250", "amdgpu.buffer_load_b128", "amdgpu.buffer_store_b128",
       UINT32_C(0xBFB00000)},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.preset_key);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_low_emission_frame_t frame = {};
    std::string body =
        "low.func.def target(@gfx_target) @gfx_kernel(%resource: "
        "reg<amdgpu.sgpr x4>, %vaddr : reg<amdgpu.vgpr>, %soffset : "
        "reg<amdgpu.sgpr>) {\n"
        "  %loaded = low.op<";
    body += test_case.load_key;
    body +=
        ">(%resource, %vaddr, %soffset) {offset = 16} : "
        "(reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>) -> "
        "reg<amdgpu.vgpr x4>\n"
        "  low.op<";
    body += test_case.store_key;
    body +=
        ">(%loaded, %resource, %vaddr, %soffset) {offset = 32} : "
        "(reg<amdgpu.vgpr x4>, reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, "
        "reg<amdgpu.sgpr>)\n"
        "  low.return\n"
        "}\n";
    BuildTablesForPreset(test_case.preset_key, body.c_str(), &arena, &frame);

    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
        &frame.schedule, &frame.allocation, &text, &arena));

    ASSERT_GE(text.data_length, 20u);
    EXPECT_EQ(text.data_length % 4, 0u);
    EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4),
              test_case.expected_return_word);
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuEncodingTest, EncodesGlobalPointerB128ForCurrentAmdgpuFamilies) {
  struct Case {
    // Target preset used to select the low descriptor set.
    const char* preset_key;
    // Whether the target descriptor exposes m0 as an architectural input.
    bool uses_m0;
    // Expected little-endian SOPP instruction word for `s_endpgm 0`.
    uint32_t expected_return_word;
  };
  const Case cases[] = {
      {"amdgpu-gfx950", true, UINT32_C(0xBF810000)},
      {"amdgpu-gfx11", false, UINT32_C(0xBFB00000)},
      {"amdgpu-gfx12", false, UINT32_C(0xBFB00000)},
      {"amdgpu-gfx1250", false, UINT32_C(0xBFB00000)},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.preset_key);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_low_emission_frame_t frame = {};
    std::string body =
        "low.func.def target(@gfx_target) @gfx_kernel(%addr: "
        "reg<amdgpu.vgpr x2>";
    if (test_case.uses_m0) {
      body += ", %m0 : reg<amdgpu.m0>";
    }
    body +=
        ") {\n"
        "  %loaded = low.op<amdgpu.global_load_b128>(%addr";
    if (test_case.uses_m0) {
      body += ", %m0";
    }
    body += ") {offset = -16} : (reg<amdgpu.vgpr x2>";
    if (test_case.uses_m0) {
      body += ", reg<amdgpu.m0>";
    }
    body +=
        ") -> reg<amdgpu.vgpr x4>\n"
        "  low.op<amdgpu.global_store_b128>(%addr, %loaded";
    if (test_case.uses_m0) {
      body += ", %m0";
    }
    body += ") {offset = 32} : (reg<amdgpu.vgpr x2>, reg<amdgpu.vgpr x4>";
    if (test_case.uses_m0) {
      body += ", reg<amdgpu.m0>";
    }
    body +=
        ")\n"
        "  low.return\n"
        "}\n";
    BuildTablesForPreset(test_case.preset_key, body.c_str(), &arena, &frame);

    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
        &frame.schedule, &frame.allocation, &text, &arena));

    ASSERT_GT(text.data_length, 4u);
    EXPECT_EQ(text.data_length % 4, 0u);
    EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4),
              test_case.expected_return_word);
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuEncodingTest, EncodesGlobalSaddrB128ForCurrentAmdgpuFamilies) {
  struct Case {
    // Target preset used to select the low descriptor set.
    const char* preset_key;
    // Whether the target descriptor exposes m0 as an architectural input.
    bool uses_m0;
    // Expected little-endian SOPP instruction word for `s_endpgm 0`.
    uint32_t expected_return_word;
  };
  const Case cases[] = {
      {"amdgpu-gfx950", true, UINT32_C(0xBF810000)},
      {"amdgpu-gfx11", false, UINT32_C(0xBFB00000)},
      {"amdgpu-gfx12", false, UINT32_C(0xBFB00000)},
      {"amdgpu-gfx1250", false, UINT32_C(0xBFB00000)},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.preset_key);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_low_emission_frame_t frame = {};
    std::string body =
        "low.func.def target(@gfx_target) @gfx_kernel(%addr: "
        "reg<amdgpu.vgpr>, %saddr : reg<amdgpu.sgpr x2>";
    if (test_case.uses_m0) {
      body += ", %m0 : reg<amdgpu.m0>";
    }
    body +=
        ") {\n"
        "  %loaded = low.op<amdgpu.global_load_b128_saddr>(%addr, %saddr";
    if (test_case.uses_m0) {
      body += ", %m0";
    }
    body += ") {offset = 4} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x2>";
    if (test_case.uses_m0) {
      body += ", reg<amdgpu.m0>";
    }
    body +=
        ") -> reg<amdgpu.vgpr x4>\n"
        "  low.op<amdgpu.global_store_b128_saddr>(%addr, %loaded, %saddr";
    if (test_case.uses_m0) {
      body += ", %m0";
    }
    body +=
        ") {offset = 8} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>, "
        "reg<amdgpu.sgpr x2>";
    if (test_case.uses_m0) {
      body += ", reg<amdgpu.m0>";
    }
    body +=
        ")\n"
        "  low.return\n"
        "}\n";
    BuildTablesForPreset(test_case.preset_key, body.c_str(), &arena, &frame);

    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
        &frame.schedule, &frame.allocation, &text, &arena));

    ASSERT_GT(text.data_length, 4u);
    EXPECT_EQ(text.data_length % 4, 0u);
    EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4),
              test_case.expected_return_word);
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuEncodingTest, EncodesGfx11MubufB64LoadStoreAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildGfx11Tables(
      "low.func.def target(@gfx_target) @gfx_kernel(%resource: "
      "reg<amdgpu.sgpr x4>, %vaddr : reg<amdgpu.vgpr>, %soffset : "
      "reg<amdgpu.sgpr>) {\n"
      "  %loaded = low.op<amdgpu.buffer_load_b64>(%resource, %vaddr, "
      "%soffset) {offset = 8} : (reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, "
      "reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr x2>\n"
      "  low.op<amdgpu.buffer_store_b64>(%loaded, %resource, %vaddr, "
      "%soffset) {offset = 16} : (reg<amdgpu.vgpr x2>, "
      "reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.return\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 12u);
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesDsMemoryForCurrentAmdgpuFamilies) {
  struct Case {
    // Target preset used to select the low descriptor set.
    const char* preset_key;
    // Expected little-endian SOPP instruction word for `s_endpgm 0`.
    uint32_t expected_return_word;
  };
  const Case cases[] = {
      {"amdgpu-gfx950", UINT32_C(0xBF810000)},
      {"amdgpu-gfx11", UINT32_C(0xBFB00000)},
      {"amdgpu-gfx12", UINT32_C(0xBFB00000)},
      {"amdgpu-gfx1250", UINT32_C(0xBFB00000)},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.preset_key);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_low_emission_frame_t frame = {};
    BuildTablesForPreset(
        test_case.preset_key,
        "low.func.def target(@gfx_target) @gfx_kernel(%addr: "
        "reg<amdgpu.vgpr>) {\n"
        "  %loaded32 = low.op<amdgpu.ds_read_b32>(%addr) {offset = 4} : "
        "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  low.op<amdgpu.ds_write_b32>(%addr, %loaded32) {offset = 4} : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>)\n"
        "  %loaded64 = low.op<amdgpu.ds_read_b64>(%addr) {offset = 8} : "
        "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x2>\n"
        "  low.op<amdgpu.ds_write_b64>(%addr, %loaded64) {offset = 8} : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr x2>)\n"
        "  %loaded96 = low.op<amdgpu.ds_read_b96>(%addr) {offset = 12} : "
        "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x3>\n"
        "  low.op<amdgpu.ds_write_b96>(%addr, %loaded96) {offset = 12} : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr x3>)\n"
        "  %loaded128 = low.op<amdgpu.ds_read_b128>(%addr) {offset = 16} : "
        "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x4>\n"
        "  low.op<amdgpu.ds_write_b128>(%addr, %loaded128) {offset = 16} : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>)\n"
        "  low.return\n"
        "}\n",
        &arena, &frame);

    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
        &frame.schedule, &frame.allocation, &text, &arena));

    ASSERT_GT(text.data_length, 4u);
    EXPECT_EQ(text.data_length % 4, 0u);
    EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4),
              test_case.expected_return_word);
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuEncodingTest, EncodesGfx11DsMemoryBarrierAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildGfx11Tables(
      "low.func.def target(@gfx_target) @gfx_kernel(%addr: "
      "reg<amdgpu.vgpr>, %value64 : reg<amdgpu.vgpr x2>, %value96 : "
      "reg<amdgpu.vgpr x3>, %value128 : reg<amdgpu.vgpr x4>) {\n"
      "  low.op<amdgpu.ds_write_b64>(%addr, %value64) {offset = 8} : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr x2>)\n"
      "  low.op<amdgpu.ds_write_b96>(%addr, %value96) {offset = 12} : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr x3>)\n"
      "  low.op<amdgpu.ds_write_b128>(%addr, %value128) {offset = 16} : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>)\n"
      "  low.op<amdgpu.s_barrier>() : ()\n"
      "  %loaded64 = low.op<amdgpu.ds_read_b64>(%addr) {offset = 8} : "
      "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x2>\n"
      "  %loaded96 = low.op<amdgpu.ds_read_b96>(%addr) {offset = 12} : "
      "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x3>\n"
      "  %loaded = low.op<amdgpu.ds_read_b128>(%addr) {offset = 16} : "
      "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x4>\n"
      "  low.return\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_GE(text.data_length, 16u);
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesGfx11Ds2AddrMemoryAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildGfx11Tables(
      "low.func.def target(@gfx_target) @gfx_kernel(%addr: "
      "reg<amdgpu.vgpr>, %value32a : reg<amdgpu.vgpr>, %value32b : "
      "reg<amdgpu.vgpr>, %value64a : reg<amdgpu.vgpr x2>, "
      "%value64b : reg<amdgpu.vgpr x2>) {\n"
      "  %loaded32 = low.op<amdgpu.ds_read2_b32>(%addr) {offset0 = 1, "
      "offset1 = 2} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x2>\n"
      "  low.op<amdgpu.ds_write2_b32>(%addr, %value32a, %value32b) "
      "{offset0 = 3, offset1 = 4} : (reg<amdgpu.vgpr>, "
      "reg<amdgpu.vgpr>, reg<amdgpu.vgpr>)\n"
      "  %loaded64st64 = low.op<amdgpu.ds_read2st64_b64>(%addr) "
      "{offset0 = 5, offset1 = 6} : (reg<amdgpu.vgpr>) -> "
      "reg<amdgpu.vgpr x4>\n"
      "  low.op<amdgpu.ds_write2st64_b64>(%addr, %value64a, %value64b) "
      "{offset0 = 7, offset1 = 8} : (reg<amdgpu.vgpr>, "
      "reg<amdgpu.vgpr x2>, reg<amdgpu.vgpr x2>)\n"
      "  low.return\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_GT(text.data_length, 4u);
  EXPECT_EQ(text.data_length % 4, 0u);
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesDsAddtidMemoryForCurrentAmdgpuFamilies) {
  struct Case {
    // Target preset used to select the low descriptor set.
    const char* preset_key;
    // Expected little-endian SOPP instruction word for `s_endpgm 0`.
    uint32_t expected_return_word;
  };
  const Case cases[] = {
      {"amdgpu-gfx950", UINT32_C(0xBF810000)},
      {"amdgpu-gfx11", UINT32_C(0xBFB00000)},
      {"amdgpu-gfx12", UINT32_C(0xBFB00000)},
      {"amdgpu-gfx1250", UINT32_C(0xBFB00000)},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.preset_key);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_low_emission_frame_t frame = {};
    BuildTablesForPreset(
        test_case.preset_key,
        "low.func.def target(@gfx_target) @gfx_kernel(%m0: reg<amdgpu.m0>, "
        "%value : reg<amdgpu.vgpr>) {\n"
        "  %loaded = low.op<amdgpu.ds_read_addtid_b32>(%m0) {offset = 16} : "
        "(reg<amdgpu.m0>) -> reg<amdgpu.vgpr>\n"
        "  low.op<amdgpu.ds_write_addtid_b32>(%value, %m0) {offset = 20} : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.m0>)\n"
        "  low.return\n"
        "}\n",
        &arena, &frame);

    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
        &frame.schedule, &frame.allocation, &text, &arena));

    ASSERT_GT(text.data_length, 4u);
    EXPECT_EQ(text.data_length % 4, 0u);
    EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4),
              test_case.expected_return_word);
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuEncodingTest, EncodesDsCrosslaneForCurrentAmdgpuFamilies) {
  struct Case {
    // Target preset used to select the low descriptor set.
    const char* preset_key;
    // Expected little-endian SOPP instruction word for `s_endpgm 0`.
    uint32_t expected_return_word;
  };
  const Case cases[] = {
      {"amdgpu-gfx950", UINT32_C(0xBF810000)},
      {"amdgpu-gfx11", UINT32_C(0xBFB00000)},
      {"amdgpu-gfx12", UINT32_C(0xBFB00000)},
      {"amdgpu-gfx1250", UINT32_C(0xBFB00000)},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.preset_key);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_low_emission_frame_t frame = {};
    BuildTablesForPreset(
        test_case.preset_key,
        "low.func.def target(@gfx_target) @gfx_kernel(%addr: "
        "reg<amdgpu.vgpr>) {\n"
        "  %swizzled = low.op<amdgpu.ds_swizzle_b32>(%addr) {offset = "
        "32} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  %permuted = low.op<amdgpu.ds_permute_b32>(%addr, %swizzled) "
        "{offset = 4} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> "
        "reg<amdgpu.vgpr>\n"
        "  %bpermuted = low.op<amdgpu.ds_bpermute_b32>(%addr, "
        "%permuted) {offset = 8} : (reg<amdgpu.vgpr>, "
        "reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  low.op<amdgpu.ds_write_b32>(%addr, %bpermuted) {offset = "
        "12} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>)\n"
        "  low.return\n"
        "}\n",
        &arena, &frame);

    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
        &frame.schedule, &frame.allocation, &text, &arena));

    ASSERT_GT(text.data_length, 4u);
    EXPECT_EQ(text.data_length % 4, 0u);
    EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4),
              test_case.expected_return_word);
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuEncodingTest, EncodesGfx12DsCrosslaneFetchInvalidAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildTablesForPreset(
      "amdgpu-gfx12",
      "low.func.def target(@gfx_target) @gfx_kernel(%addr: "
      "reg<amdgpu.vgpr>, %value : reg<amdgpu.vgpr>) {\n"
      "  %bpermuted = low.op<amdgpu.ds_bpermute_fi_b32>(%addr, "
      "%value) {offset = 16} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) "
      "-> reg<amdgpu.vgpr>\n"
      "  low.op<amdgpu.ds_write_b32>(%addr, %bpermuted) {offset = 20} "
      ": (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>)\n"
      "  low.return\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_GT(text.data_length, 4u);
  EXPECT_EQ(text.data_length % 4, 0u);
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBFB00000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesGfx950DsTransposeReadsAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildTablesForPreset(
      "amdgpu-gfx950",
      "low.func.def target(@gfx_target) @gfx_kernel(%addr: "
      "reg<amdgpu.vgpr>) {\n"
      "  %loaded_b4 = low.op<amdgpu.ds_read_b64_tr_b4>(%addr) "
      "{offset = 0} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x2>\n"
      "  %loaded_b6 = low.op<amdgpu.ds_read_b96_tr_b6>(%addr) "
      "{offset = 16} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x3>\n"
      "  %loaded_b8 = low.op<amdgpu.ds_read_b64_tr_b8>(%addr) "
      "{offset = 32} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x2>\n"
      "  %loaded_b16 = low.op<amdgpu.ds_read_b64_tr_b16>(%addr) "
      "{offset = 48} : (reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x2>\n"
      "  low.op<amdgpu.ds_write_b64>(%addr, %loaded_b4) {offset = 64} "
      ": (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x2>)\n"
      "  low.op<amdgpu.ds_write_b96>(%addr, %loaded_b6) {offset = 80} "
      ": (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x3>)\n"
      "  low.op<amdgpu.ds_write_b64>(%addr, %loaded_b8) {offset = 96} "
      ": (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x2>)\n"
      "  low.op<amdgpu.ds_write_b64>(%addr, %loaded_b16) {offset = "
      "112} : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr x2>)\n"
      "  low.return\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_GT(text.data_length, 4u);
  EXPECT_EQ(text.data_length % 4, 0u);
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBF810000));
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuEncodingTest, EncodesReturnForCurrentAmdgpuFamilies) {
  struct Case {
    // Target preset used to select the low descriptor set.
    const char* preset_key;
    // Expected little-endian SOPP instruction word for `s_endpgm 0`.
    uint32_t expected_return_word;
  };
  const Case cases[] = {
      {"amdgpu-gfx950", UINT32_C(0xBF810000)},
      {"amdgpu-gfx11", UINT32_C(0xBFB00000)},
      {"amdgpu-gfx12", UINT32_C(0xBFB00000)},
      {"amdgpu-gfx1250", UINT32_C(0xBFB00000)},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.preset_key);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_low_emission_frame_t frame = {};
    BuildTablesForPreset(test_case.preset_key,
                         "low.func.def target(@gfx_target) @gfx_kernel() {\n"
                         "  low.return\n"
                         "}\n",
                         &arena, &frame);

    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
        &frame.schedule, &frame.allocation, &text, &arena));

    ASSERT_EQ(text.data_length, 4u);
    EXPECT_EQ(ReadU32LE(text.data), test_case.expected_return_word);
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuEncodingTest, EncodesRdnaWmmaPacketAndReturn) {
  struct Case {
    // Target preset used to select the low descriptor set.
    const char* preset_key;
    // Number of VGPRs used by each f16 matrix input fragment.
    uint32_t operand_unit_count;
    // Expected little-endian SOPP instruction word for `s_endpgm 0`.
    uint32_t expected_return_word;
  };
  const Case cases[] = {
      {"amdgpu-gfx11", 8, UINT32_C(0xBFB00000)},
      {"amdgpu-gfx12", 4, UINT32_C(0xBFB00000)},
      {"amdgpu-gfx1250", 4, UINT32_C(0xBFB00000)},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.preset_key);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_low_emission_frame_t frame = {};
    std::string operand_type =
        AmdgpuVgprRegisterType(test_case.operand_unit_count);
    std::string body = "low.func.def target(@gfx_target) @gfx_kernel(%a: ";
    body += operand_type;
    body += ", %b : ";
    body += operand_type;
    body +=
        ", %acc : reg<amdgpu.vgpr x8>, %resource : "
        "reg<amdgpu.sgpr x4>, %vaddr : reg<amdgpu.vgpr>, %soffset : "
        "reg<amdgpu.sgpr>) {\n"
        "  %out = low.op<amdgpu.v_wmma_f32_16x16x16_f16>(%a, %b, %acc) : (";
    body += operand_type;
    body += ", ";
    body += operand_type;
    body +=
        ", reg<amdgpu.vgpr x8>) -> %acc as reg<amdgpu.vgpr x8>\n"
        "  %out_low = low.slice %out[0] : reg<amdgpu.vgpr x8> -> "
        "reg<amdgpu.vgpr x4>\n"
        "  low.op<amdgpu.buffer_store_b128>(%out_low, %resource, %vaddr, "
        "%soffset) {offset = 0} : (reg<amdgpu.vgpr x4>, "
        "reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
        "  low.return\n"
        "}\n";
    BuildTablesForPreset(test_case.preset_key, body.c_str(), &arena, &frame);

    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
        &frame.schedule, &frame.allocation, &text, &arena));

    ASSERT_GT(text.data_length, 12u);
    EXPECT_EQ(text.data_length % 4, 0u);
    EXPECT_NE(ReadU32LE(text.data), UINT32_C(0));
    EXPECT_NE(ReadU32LE(text.data + 4), UINT32_C(0));
    EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4),
              test_case.expected_return_word);
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuEncodingTest, EncodesRdnaIntegerWmmaPacketsAndReturn) {
  struct Case {
    // Target preset used to select the low descriptor set.
    const char* preset_key;
    // Number of VGPRs used by each IU8 matrix input fragment.
    uint32_t iu8_operand_unit_count;
    // Number of VGPRs used by each IU4 matrix input fragment.
    uint32_t iu4_operand_unit_count;
    // Expected little-endian SOPP instruction word for `s_endpgm 0`.
    uint32_t expected_return_word;
  };
  const Case cases[] = {
      {"amdgpu-gfx11", 4, 2, UINT32_C(0xBFB00000)},
      {"amdgpu-gfx12", 2, 1, UINT32_C(0xBFB00000)},
      {"amdgpu-gfx1250", 2, 1, UINT32_C(0xBFB00000)},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.preset_key);
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_low_emission_frame_t frame = {};
    std::string iu8_operand_type =
        AmdgpuVgprRegisterType(test_case.iu8_operand_unit_count);
    std::string iu4_operand_type =
        AmdgpuVgprRegisterType(test_case.iu4_operand_unit_count);
    std::string body = "low.func.def target(@gfx_target) @gfx_kernel(%a8: ";
    body += iu8_operand_type;
    body += ", %b8 : ";
    body += iu8_operand_type;
    body += ", %a4 : ";
    body += iu4_operand_type;
    body += ", %b4 : ";
    body += iu4_operand_type;
    body +=
        ", %acc : reg<amdgpu.vgpr x8>, %resource : "
        "reg<amdgpu.sgpr x4>, %vaddr : reg<amdgpu.vgpr>, %soffset : "
        "reg<amdgpu.sgpr>) {\n"
        "  %out_i8 = low.op<amdgpu.v_wmma_i32_16x16x16_iu8>(%a8, "
        "%b8, %acc) : (";
    body += iu8_operand_type;
    body += ", ";
    body += iu8_operand_type;
    body +=
        ", reg<amdgpu.vgpr x8>) -> %acc as reg<amdgpu.vgpr x8>\n"
        "  %out_i4 = low.op<amdgpu.v_wmma_i32_16x16x16_iu4>(%a4, "
        "%b4, %out_i8) : (";
    body += iu4_operand_type;
    body += ", ";
    body += iu4_operand_type;
    body +=
        ", reg<amdgpu.vgpr x8>) -> %out_i8 as reg<amdgpu.vgpr x8>\n"
        "  %out_low = low.slice %out_i4[0] : reg<amdgpu.vgpr x8> -> "
        "reg<amdgpu.vgpr x4>\n"
        "  low.op<amdgpu.buffer_store_b128>(%out_low, %resource, "
        "%vaddr, %soffset) {offset = 0} : (reg<amdgpu.vgpr x4>, "
        "reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
        "  low.return\n"
        "}\n";
    BuildTablesForPreset(test_case.preset_key, body.c_str(), &arena, &frame);

    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
        &frame.schedule, &frame.allocation, &text, &arena));

    ASSERT_GT(text.data_length, 16u);
    EXPECT_EQ(text.data_length % 4, 0u);
    EXPECT_NE(ReadU32LE(text.data), UINT32_C(0));
    EXPECT_NE(ReadU32LE(text.data + 4), UINT32_C(0));
    EXPECT_NE(ReadU32LE(text.data + 8), UINT32_C(0));
    EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4),
              test_case.expected_return_word);
    iree_arena_deinitialize(&arena);
  }
}

TEST_F(AmdgpuEncodingTest, EncodesGfx950MfmaPacketAndReturn) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_emission_frame_t frame = {};
  BuildTablesForPreset(
      "amdgpu-gfx950",
      "low.func.def target(@gfx_target) @gfx_kernel(%a: "
      "reg<amdgpu.vgpr x2>, %b : reg<amdgpu.vgpr x2>, %acc : "
      "reg<amdgpu.vgpr x4>, %vaddr : reg<amdgpu.vgpr>) {\n"
      "  %out = low.op<amdgpu.v_mfma_f32_16x16x16_f16>(%a, %b, %acc) : "
      "(reg<amdgpu.vgpr x2>, reg<amdgpu.vgpr x2>, reg<amdgpu.vgpr x4>) "
      "-> %acc as reg<amdgpu.vgpr x4>\n"
      "  low.op<amdgpu.ds_write_b128>(%vaddr, %out) {offset = 0} : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr x4>)\n"
      "  low.return\n"
      "}\n",
      &arena, &frame);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream(
      &frame.schedule, &frame.allocation, &text, &arena));

  ASSERT_GT(text.data_length, 12u);
  EXPECT_EQ(text.data_length % 4, 0u);
  EXPECT_NE(ReadU32LE(text.data), UINT32_C(0));
  EXPECT_NE(ReadU32LE(text.data + 4), UINT32_C(0));
  EXPECT_EQ(ReadU32LE(text.data + text.data_length - 4), UINT32_C(0xBF810000));
  iree_arena_deinitialize(&arena);
}

}  // namespace
}  // namespace loom
