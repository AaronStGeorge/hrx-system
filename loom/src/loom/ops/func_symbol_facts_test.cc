// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/func_symbol_facts.h"

#include <memory>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/target/facts.h"
#include "loom/target/preset_registry.h"
#include "loom/target/types.h"

namespace loom {
namespace {

struct ModuleDeleter {
  void operator()(loom_module_t* module) const { loom_module_free(module); }
};
using ModulePtr = std::unique_ptr<loom_module_t, ModuleDeleter>;

static const loom_target_snapshot_t kPresetSnapshot = {
    .name = IREE_SVL("test.profile"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_WASM,
    .target_triple = IREE_SVL("wasm32-unknown-unknown"),
    .data_layout = IREE_SVL(""),
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY,
    .target_cpu = IREE_SVL("generic"),
    .target_features = IREE_SVL("+simd128"),
    .default_pointer_bitwidth = 32,
    .index_bitwidth = 32,
    .offset_bitwidth = 32,
    .memory_spaces =
        {
            .generic = 0,
            .global = 0,
            .workgroup = UINT32_MAX,
            .constant = 0,
            .private_memory = UINT32_MAX,
            .host = UINT32_MAX,
            .descriptor = UINT32_MAX,
        },
};

static const loom_target_export_plan_t kPresetExportPlan = {
    .name = IREE_SVL("test.profile"),
    .export_symbol = IREE_SVL("must_not_escape"),
    .abi_kind = LOOM_TARGET_ABI_WASM_FUNCTION,
    .linkage = LOOM_TARGET_LINKAGE_DEFAULT,
    .hal_kernel =
        {
            .binding_alignment = 0,
            .required_workgroup_size = {.x = 0, .y = 0, .z = 0},
            .flat_workgroup_size_min = 0,
            .flat_workgroup_size_max = 0,
            .buffer_resource_flags = 0,
        },
};

static const loom_target_config_t kPresetConfig = {
    .name = IREE_SVL("test.profile"),
    .contract_set_key = IREE_SVL("wasm.core.simd128"),
    .contract_feature_bits = 1,
};

static const loom_target_bundle_t kPresetBundle = {
    .name = IREE_SVL("test.profile"),
    .snapshot = &kPresetSnapshot,
    .export_plan = &kPresetExportPlan,
    .config = &kPresetConfig,
};

static const loom_target_bundle_t* const kPresetBundles[] = {
    &kPresetBundle,
};

static const loom_target_preset_registry_t kPresetRegistry = {
    .target_bundles = kPresetBundles,
    .target_bundle_count = IREE_ARRAYSIZE(kPresetBundles),
};

class FuncSymbolFactsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_op_registry_initialize_context(iree_allocator_system(),
                                                       &context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    resources_[0] =
        loom_target_profile_preset_registry_resource(&kPresetRegistry);
    const loom_symbol_fact_table_options_t options = {
        .resources = loom_make_symbol_fact_resource_list(resources_, 1),
    };
    loom_symbol_fact_table_initialize_with_options(&fact_table_, &options,
                                                   &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("func_symbol_facts_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_symbol_id_t FindSymbol(const loom_module_t* module,
                              iree_string_view_t name) {
    loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return symbol_id;
  }

  const loom_func_symbol_facts_t* LookupFunc(const loom_module_t* module,
                                             iree_string_view_t name) {
    const loom_symbol_facts_base_t* base_facts = nullptr;
    IREE_CHECK_OK(loom_symbol_fact_table_lookup(
        &fact_table_, module, FindSymbol(module, name), &base_facts));
    const loom_func_symbol_facts_t* facts =
        loom_func_symbol_facts_cast(base_facts);
    IREE_ASSERT(facts != nullptr);
    return facts;
  }

  // Block pool shared by parser, module allocation, and analysis storage.
  iree_arena_block_pool_t block_pool_;

  // Context with production dialects registered.
  loom_context_t context_;

  // Arena for symbol fact table storage and fact payloads.
  iree_arena_allocator_t analysis_arena_;

  // Dense symbol fact table under test.
  loom_symbol_fact_table_t fact_table_;

  // Resource storage borrowed by the symbol fact table.
  loom_symbol_fact_resource_t resources_[1];
};

TEST_F(FuncSymbolFactsTest, SourceFuncFactsRemainTargetIndependent) {
  ModulePtr module = ParseModule(R"(
func.def public device @semantic() {
  func.return
}
)");

  const loom_func_symbol_facts_t* facts =
      LookupFunc(module.get(), IREE_SV("semantic"));
  EXPECT_EQ(facts->base.domain, &loom_func_symbol_fact_domain);
  EXPECT_EQ(facts->base.symbol_kind, LOOM_SYMBOL_FUNC_DEF);
  EXPECT_TRUE(facts->has_body);
  EXPECT_EQ(facts->visibility, 1);
  EXPECT_EQ(facts->calling_convention, 2);
  EXPECT_EQ(facts->purity, 0);
  EXPECT_EQ(facts->argument_count, 0);
  EXPECT_EQ(facts->result_count, 0);
  EXPECT_FALSE(loom_symbol_ref_is_valid(facts->target_symbol));
  EXPECT_EQ(facts->target_profile, nullptr);
  EXPECT_EQ(facts->target_bundle, nullptr);
}

TEST_F(FuncSymbolFactsTest, DeclarationFactsCarryImportContract) {
  ModulePtr module = ParseModule(R"(
func.decl import("env", "do.work") @do_work(%arg0: i32) -> (i32)
)");

  const loom_func_symbol_facts_t* facts =
      LookupFunc(module.get(), IREE_SV("do_work"));
  EXPECT_FALSE(facts->has_body);
  EXPECT_TRUE(facts->imports);
  EXPECT_TRUE(iree_string_view_equal(facts->import_module, IREE_SV("env")));
  EXPECT_TRUE(iree_string_view_equal(facts->import_symbol, IREE_SV("do.work")));
  EXPECT_EQ(facts->argument_count, 1);
  EXPECT_EQ(facts->result_count, 1);
}

TEST_F(FuncSymbolFactsTest, ImportSymbolDefaultsToDeclarationName) {
  ModulePtr module = ParseModule(R"(
func.decl import("env") @do_work(%arg0: i32) -> (i32)
)");

  const loom_func_symbol_facts_t* facts =
      LookupFunc(module.get(), IREE_SV("do_work"));
  EXPECT_TRUE(facts->imports);
  EXPECT_TRUE(iree_string_view_equal(facts->import_module, IREE_SV("env")));
  EXPECT_TRUE(iree_string_view_equal(facts->import_symbol, IREE_SV("do_work")));
}

TEST_F(FuncSymbolFactsTest, LowFuncResolvesTargetProfile) {
  ModulePtr module = ParseModule(R"(
target.profile @wasm preset("test.profile")

low.func.def target(@wasm) @kernel() {
  low.return
}
)");

  const loom_func_symbol_facts_t* facts =
      LookupFunc(module.get(), IREE_SV("kernel"));
  EXPECT_TRUE(iree_string_view_equal(facts->name, IREE_SV("kernel")));
  EXPECT_TRUE(loom_symbol_ref_is_valid(facts->target_symbol));
  ASSERT_NE(facts->target_profile, nullptr);
  ASSERT_NE(facts->target_bundle, nullptr);
  EXPECT_EQ(facts->target_bundle->snapshot, &facts->target_profile->snapshot);
  EXPECT_EQ(facts->export_plan.abi_kind, LOOM_TARGET_ABI_WASM_FUNCTION);
  EXPECT_TRUE(iree_string_view_is_empty(facts->export_plan.export_symbol));
}

TEST_F(FuncSymbolFactsTest, FuncOwnedContractOverridesPreset) {
  ModulePtr module = ParseModule(R"(
target.profile @wasm preset("test.profile")

low.func.def target(@wasm) abi(hal_kernel, {
  hal_binding_alignment = 16,
  hal_workgroup_size_x = 8,
  hal_workgroup_size_y = 4,
  hal_workgroup_size_z = 2,
  hal_flat_workgroup_size_min = 32,
  hal_flat_workgroup_size_max = 64,
  hal_buffer_resource_flags = 7
}) export("dispatch", {linkage = "dso_local", ordinal = 5}) @kernel() {
  low.return
}
)");

  const loom_func_symbol_facts_t* facts =
      LookupFunc(module.get(), IREE_SV("kernel"));
  ASSERT_NE(facts->target_profile, nullptr);
  ASSERT_NE(facts->target_bundle, nullptr);
  EXPECT_TRUE(facts->exports);
  EXPECT_EQ(facts->export_plan.abi_kind, LOOM_TARGET_ABI_HAL_KERNEL);
  EXPECT_EQ(facts->export_plan.linkage, LOOM_TARGET_LINKAGE_DSO_LOCAL);
  EXPECT_TRUE(
      iree_string_view_equal(facts->export_plan.name, IREE_SV("kernel")));
  EXPECT_TRUE(iree_string_view_equal(facts->export_plan.export_symbol,
                                     IREE_SV("dispatch")));
  EXPECT_EQ(facts->export_plan.hal_kernel.binding_alignment, 16u);
  EXPECT_EQ(facts->export_plan.hal_kernel.required_workgroup_size.x, 8u);
  EXPECT_EQ(facts->export_plan.hal_kernel.required_workgroup_size.y, 4u);
  EXPECT_EQ(facts->export_plan.hal_kernel.required_workgroup_size.z, 2u);
  EXPECT_EQ(facts->export_plan.hal_kernel.flat_workgroup_size_min, 32u);
  EXPECT_EQ(facts->export_plan.hal_kernel.flat_workgroup_size_max, 64u);
  EXPECT_EQ(facts->export_plan.hal_kernel.buffer_resource_flags, 7u);
  EXPECT_TRUE(facts->has_export_ordinal);
  EXPECT_EQ(facts->export_ordinal, 5u);
}

TEST_F(FuncSymbolFactsTest, ContractRequiresTargetProfile) {
  ModulePtr module = ParseModule(R"(
func.def abi(wasm_function) @semantic() {
  func.return
}
)");

  const loom_symbol_facts_base_t* base_facts = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_symbol_fact_table_lookup(
          &fact_table_, module.get(),
          FindSymbol(module.get(), IREE_SV("semantic")), &base_facts));
  EXPECT_EQ(base_facts, nullptr);
}

TEST_F(FuncSymbolFactsTest, TargetMustResolveToTargetProfileFacts) {
  ModulePtr module = ParseModule(R"(
target.profile @profile preset("test.profile")
target.artifact @not_profile target(@profile)

low.func.def target(@not_profile) @kernel() {
  low.return
}
)");

  const loom_symbol_facts_base_t* base_facts = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_symbol_fact_table_lookup(&fact_table_, module.get(),
                                    FindSymbol(module.get(), IREE_SV("kernel")),
                                    &base_facts));
  EXPECT_EQ(base_facts, nullptr);
}

}  // namespace
}  // namespace loom
