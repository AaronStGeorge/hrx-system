// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/occupancy.h"

#include <sstream>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/testing/module_ptr.h"
#include "loom/verify/verify.h"

namespace {

using ModulePtr = ::loom::testing::ModulePtr;

std::string ToString(const iree_string_builder_t& builder) {
  if (iree_string_builder_size(&builder) == 0) {
    return std::string();
  }
  return std::string(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
}

std::string TargetPreamble(const char* target_symbol, const char* preset_key) {
  std::string source = "target.profile @";
  source += target_symbol;
  source += " preset(\"";
  source += preset_key;
  source += "\")\n\n";
  return source;
}

std::string RegisterPressureFunction(const char* target_symbol,
                                     const char* preset_key,
                                     const char* function_symbol,
                                     const char* register_class,
                                     int value_count) {
  std::ostringstream source;
  source << TargetPreamble(target_symbol, preset_key);
  source << "low.func.def target(@" << target_symbol << ") @" << function_symbol
         << "(";
  for (int i = 0; i < value_count; ++i) {
    if (i > 0) {
      source << ", ";
    }
    source << "%v" << i << " : reg<" << register_class << ">";
  }
  source << ") -> (";
  for (int i = 0; i < value_count; ++i) {
    if (i > 0) {
      source << ", ";
    }
    source << "reg<" << register_class << ">";
  }
  source << ") {\n  low.return ";
  for (int i = 0; i < value_count; ++i) {
    if (i > 0) {
      source << ", ";
    }
    source << "%v" << i;
  }
  source << " : ";
  for (int i = 0; i < value_count; ++i) {
    if (i > 0) {
      source << ", ";
    }
    source << "reg<" << register_class << ">";
  }
  source << "\n}\n";
  return source.str();
}

struct DiagnosticCapture {
  iree_host_size_t count = 0;
  bool saw_vgpr = false;
};

iree_status_t CaptureOccupancyDiagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  auto* capture = static_cast<DiagnosticCapture*>(user_data);
  if (!emission || !emission->error || emission->param_count != 9) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "malformed occupancy diagnostic emission");
  }
  EXPECT_EQ(emission->error->domain, LOOM_ERROR_DOMAIN_BACKEND);
  EXPECT_EQ(emission->error->code, 10);
  EXPECT_EQ(emission->params[4].kind, LOOM_PARAM_STRING);
  EXPECT_EQ(emission->params[7].kind, LOOM_PARAM_U32);
  EXPECT_EQ(emission->params[8].kind, LOOM_PARAM_STRING);
  if (iree_string_view_equal(emission->params[4].string,
                             IREE_SV("amdgpu.vgpr"))) {
    capture->saw_vgpr = true;
    EXPECT_EQ(emission->params[7].u32, 93u);
    EXPECT_TRUE(iree_string_view_equal(emission->params[8].string,
                                       IREE_SV("amdgpu.vgpr")));
  }
  ++capture->count;
  return iree_ok_status();
}

class AmdgpuOccupancyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    block_pool_initialized_ = true;

    loom_context_initialize(iree_allocator_system(), &context_);
    context_initialized_ = true;
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    loom_amdgpu_low_descriptor_registry_initialize(&low_registry_);
  }

  void TearDown() override {
    if (context_initialized_) {
      loom_context_deinitialize(&context_);
    }
    if (block_pool_initialized_) {
      iree_arena_block_pool_deinitialize(&block_pool_);
    }
  }

  iree_status_t ParseAndVerify(iree_string_view_t source,
                               ModulePtr* out_module) {
    out_module->reset();
    loom_text_parse_options_t parse_options = {};
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_.registry, &parse_options.low_asm_environment);

    loom_module_t* raw_module = nullptr;
    IREE_RETURN_IF_ERROR(
        loom_text_parse(source, IREE_SV("amdgpu_occupancy_test.loom"),
                        &context_, &block_pool_, &parse_options, &raw_module));
    if (!raw_module) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "test source did not parse");
    }
    ModulePtr module(raw_module);

    loom_verify_options_t verify_options = {};
    loom_verify_result_t verify_result = {};
    IREE_RETURN_IF_ERROR(
        loom_verify_module(module.get(), &verify_options, &verify_result));
    if (verify_result.error_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "test source failed generic verification");
    }

    loom_low_verify_options_t low_verify_options = {
        .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
        .descriptor_registry = &low_registry_.registry,
        .max_errors = 100,
    };
    loom_low_verify_result_t low_verify_result = {};
    IREE_RETURN_IF_ERROR(loom_low_verify_module(
        module.get(), &low_verify_options, &low_verify_result));
    if (low_verify_result.error_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "test source failed low verification");
    }

    *out_module = std::move(module);
    return iree_ok_status();
  }

  const loom_op_t* FirstLowFunction(const loom_module_t* module) {
    loom_block_t* module_block = loom_module_block((loom_module_t*)module);
    const loom_op_t* op = nullptr;
    loom_block_for_each_op(module_block, op) {
      if (loom_low_func_def_isa(op)) {
        return op;
      }
    }
    return nullptr;
  }

  iree_status_t AllocateAndBuildOccupancy(
      loom_module_t* module, const loom_op_t* low_function,
      const loom_low_allocation_budget_t* budgets,
      iree_host_size_t budget_count, iree_arena_allocator_t* arena,
      loom_low_allocation_table_t* out_allocation,
      loom_amdgpu_occupancy_table_t* out_occupancy,
      const loom_amdgpu_occupancy_options_t* occupancy_options = nullptr) {
    loom_low_allocation_options_t allocation_options = {
        .descriptor_registry = &low_registry_.registry,
        .budgets = budgets,
        .budget_count = budget_count,
    };
    IREE_RETURN_IF_ERROR(loom_low_allocate_function(
        module, low_function, &allocation_options, arena, out_allocation));
    return loom_amdgpu_occupancy_build(out_allocation, occupancy_options, arena,
                                       out_occupancy);
  }

  const loom_amdgpu_occupancy_register_class_t* FindClass(
      const loom_amdgpu_occupancy_table_t& occupancy,
      iree_string_view_t register_class) {
    for (iree_host_size_t i = 0; i < occupancy.register_class_count; ++i) {
      if (iree_string_view_equal(occupancy.register_classes[i].register_class,
                                 register_class)) {
        return &occupancy.register_classes[i];
      }
    }
    return nullptr;
  }

  iree_arena_block_pool_t block_pool_ = {};
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t low_registry_ = {};
  bool block_pool_initialized_ = false;
  bool context_initialized_ = false;
};

TEST_F(AmdgpuOccupancyTest, Gfx11VgprPressureReportsNextCliff) {
  std::string source = RegisterPressureFunction(
      "gfx11_target", "amdgpu-gfx11", "gfx11_pressure", "amdgpu.vgpr", 65);
  ModulePtr module;
  IREE_ASSERT_OK(ParseAndVerify(
      iree_make_string_view(source.data(), source.size()), &module));
  const loom_op_t* low_function = FirstLowFunction(module.get());
  ASSERT_NE(low_function, nullptr);

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_allocation_table_t allocation = {};
  loom_amdgpu_occupancy_table_t occupancy = {};
  IREE_ASSERT_OK(AllocateAndBuildOccupancy(module.get(), low_function, nullptr,
                                           0, &arena, &allocation, &occupancy));

  EXPECT_EQ(occupancy.wave_size, 64u);
  EXPECT_EQ(occupancy.max_waves_per_simd, 16u);
  EXPECT_EQ(occupancy.flat_workgroup_size, 0u);
  EXPECT_EQ(occupancy.waves_per_workgroup, 0u);
  EXPECT_EQ(occupancy.resident_waves_per_simd, 15u);
  EXPECT_EQ(occupancy.occupancy_percent, 93u);
  ASSERT_NE(occupancy.limiting_register_class_index,
            LOOM_AMDGPU_OCCUPANCY_CLASS_NONE);

  const loom_amdgpu_occupancy_register_class_t* vgpr =
      FindClass(occupancy, IREE_SV("amdgpu.vgpr"));
  ASSERT_NE(vgpr, nullptr);
  EXPECT_EQ(vgpr->allocated_units, 65u);
  EXPECT_EQ(vgpr->rounded_units, 68u);
  EXPECT_EQ(vgpr->pool_units, 1024u);
  EXPECT_EQ(vgpr->wave_limit, 15u);
  EXPECT_EQ(vgpr->next_cliff_units, 69u);
  EXPECT_EQ(vgpr->units_until_next_cliff, 4u);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_amdgpu_occupancy_format_json(&occupancy, &builder));
  std::string json = ToString(builder);
  EXPECT_NE(json.find("\"format\":\"loom.amdgpu.occupancy.v0\""),
            std::string::npos);
  EXPECT_NE(json.find("\"descriptor_set\":\"amdgpu.gfx11.core\""),
            std::string::npos);
  EXPECT_NE(json.find("\"limiting_resource\":\"amdgpu.vgpr\""),
            std::string::npos);
  EXPECT_NE(json.find("\"next_cliff_units\":69"), std::string::npos);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuOccupancyTest, SpillsReportScratchPressure) {
  std::string source = RegisterPressureFunction(
      "gfx11_target", "amdgpu-gfx11", "gfx11_spill", "amdgpu.vgpr", 3);
  ModulePtr module;
  IREE_ASSERT_OK(ParseAndVerify(
      iree_make_string_view(source.data(), source.size()), &module));
  const loom_op_t* low_function = FirstLowFunction(module.get());
  ASSERT_NE(low_function, nullptr);

  const loom_low_allocation_budget_t budgets[] = {
      {
          .register_class = IREE_SV("amdgpu.vgpr"),
          .max_units = 2,
      },
  };
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_allocation_table_t allocation = {};
  loom_amdgpu_occupancy_table_t occupancy = {};
  IREE_ASSERT_OK(AllocateAndBuildOccupancy(module.get(), low_function, budgets,
                                           IREE_ARRAYSIZE(budgets), &arena,
                                           &allocation, &occupancy));

  EXPECT_EQ(occupancy.spill_count, 1u);
  EXPECT_EQ(occupancy.scratch_spill_bytes, 4u);
  EXPECT_EQ(occupancy.spill_store_count, 1u);
  EXPECT_EQ(occupancy.spill_reload_count, 1u);

  const loom_amdgpu_occupancy_register_class_t* vgpr =
      FindClass(occupancy, IREE_SV("amdgpu.vgpr"));
  ASSERT_NE(vgpr, nullptr);
  EXPECT_EQ(vgpr->allocated_units, 2u);
  EXPECT_EQ(vgpr->spill_count, 1u);
  EXPECT_EQ(vgpr->spill_bytes, 4u);
  EXPECT_EQ(vgpr->spill_store_count, 1u);
  EXPECT_EQ(vgpr->spill_reload_count, 1u);
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuOccupancyTest, EmitsStructuredOccupancyRemarks) {
  std::string source = RegisterPressureFunction(
      "gfx11_target", "amdgpu-gfx11", "gfx11_pressure", "amdgpu.vgpr", 65);
  ModulePtr module;
  IREE_ASSERT_OK(ParseAndVerify(
      iree_make_string_view(source.data(), source.size()), &module));
  const loom_op_t* low_function = FirstLowFunction(module.get());
  ASSERT_NE(low_function, nullptr);

  DiagnosticCapture capture;
  const loom_amdgpu_occupancy_options_t occupancy_options = {
      .emitter = {.fn = CaptureOccupancyDiagnostic, .user_data = &capture},
      .diagnostic_flags = LOOM_AMDGPU_OCCUPANCY_DIAGNOSTIC_SUMMARY,
  };
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_low_allocation_table_t allocation = {};
  loom_amdgpu_occupancy_table_t occupancy = {};
  IREE_ASSERT_OK(AllocateAndBuildOccupancy(module.get(), low_function, nullptr,
                                           0, &arena, &allocation, &occupancy,
                                           &occupancy_options));

  EXPECT_EQ(capture.count, 2u);
  EXPECT_TRUE(capture.saw_vgpr);
  iree_arena_deinitialize(&arena);
}

TEST_F(AmdgpuOccupancyTest, AllCurrentDescriptorSetsSelectModels) {
  struct Case {
    const char* target_symbol;
    const char* preset_key;
    const char* function_symbol;
    const char* register_class;
    const char* target_cpu;
    iree_host_size_t expected_class_count;
  };
  const Case cases[] = {
      {"gfx950_target", "amdgpu-gfx950", "gfx950_pressure", "amdgpu.agpr",
       "gfx950", 3},
      {"gfx11_target", "amdgpu-gfx11", "gfx11_pressure", "amdgpu.vgpr",
       "gfx1100", 2},
      {"gfx12_target", "amdgpu-gfx12", "gfx12_pressure", "amdgpu.vgpr",
       "gfx1200", 2},
      {"gfx1250_target", "amdgpu-gfx1250", "gfx1250_pressure", "amdgpu.vgpr",
       "gfx1250", 2},
  };

  for (const Case& test_case : cases) {
    std::string source = RegisterPressureFunction(
        test_case.target_symbol, test_case.preset_key,
        test_case.function_symbol, test_case.register_class, 1);
    ModulePtr module;
    IREE_ASSERT_OK(ParseAndVerify(
        iree_make_string_view(source.data(), source.size()), &module));
    const loom_op_t* low_function = FirstLowFunction(module.get());
    ASSERT_NE(low_function, nullptr);

    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_low_allocation_table_t allocation = {};
    loom_amdgpu_occupancy_table_t occupancy = {};
    IREE_ASSERT_OK(AllocateAndBuildOccupancy(module.get(), low_function,
                                             nullptr, 0, &arena, &allocation,
                                             &occupancy));

    EXPECT_TRUE(iree_string_view_equal(
        occupancy.target_cpu, iree_make_cstring_view(test_case.target_cpu)))
        << test_case.preset_key;
    EXPECT_EQ(occupancy.register_class_count, test_case.expected_class_count)
        << test_case.preset_key;
    const loom_amdgpu_occupancy_register_class_t* register_class =
        FindClass(occupancy, iree_make_cstring_view(test_case.register_class));
    ASSERT_NE(register_class, nullptr) << test_case.preset_key;
    EXPECT_EQ(register_class->allocated_units, 1u) << test_case.preset_key;
    iree_arena_deinitialize(&arena);
  }
}

}  // namespace
