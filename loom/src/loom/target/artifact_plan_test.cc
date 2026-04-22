// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/artifact_plan.h"

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
#include "loom/util/call_graph.h"

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
    .export_symbol = IREE_SVL(""),
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

class ArtifactPlanTest : public ::testing::Test {
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
                                  IREE_SV("artifact_plan_test.loom"), &context_,
                                  &block_pool_, &options, &module));
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

  loom_symbol_ref_t SymbolRef(const loom_module_t* module,
                              iree_string_view_t name) {
    return (loom_symbol_ref_t){
        .module_id = 0,
        .symbol_id = FindSymbol(module, name),
    };
  }

  // Block pool shared by parser, module allocation, and analysis storage.
  iree_arena_block_pool_t block_pool_;

  // Context with production dialects registered.
  loom_context_t context_;

  // Arena for symbol facts, call graph, and artifact plan storage.
  iree_arena_allocator_t analysis_arena_;

  // Dense symbol fact table under test.
  loom_symbol_fact_table_t fact_table_;

  // Resource storage borrowed by the symbol fact table.
  loom_symbol_fact_resource_t resources_[1];
};

TEST_F(ArtifactPlanTest, CollectsExportedEntryAndPrivateClosure) {
  ModulePtr module = ParseModule(R"(
target.profile @wasm preset("test.profile")
target.artifact @module target(@wasm)

func.def target(@wasm) abi(wasm_function) export("entry", {artifact = @module}) @entry() {
  func.call @helper() : ()
  func.return
}

func.def target(@wasm) abi(wasm_function) @helper() {
  func.return
}

func.def target(@wasm) abi(wasm_function) @unused() {
  func.return
}
)");

  loom_call_graph_t call_graph;
  IREE_ASSERT_OK(
      loom_call_graph_build(module.get(), &analysis_arena_, &call_graph));

  loom_target_artifact_plan_t plan;
  IREE_ASSERT_OK(loom_target_artifact_plan_build(
      module.get(), SymbolRef(module.get(), IREE_SV("module")), &fact_table_,
      &call_graph, &analysis_arena_, &plan));

  EXPECT_EQ(plan.entry_count, 1u);
  EXPECT_EQ(plan.entry_symbol_ids[0],
            FindSymbol(module.get(), IREE_SV("entry")));
  EXPECT_EQ(plan.func_count, 2u);
  EXPECT_EQ(plan.func_symbol_ids[0],
            FindSymbol(module.get(), IREE_SV("entry")));
  EXPECT_EQ(plan.func_symbol_ids[1],
            FindSymbol(module.get(), IREE_SV("helper")));
}

TEST_F(ArtifactPlanTest, RejectsClosureCrossingIntoAnotherArtifact) {
  ModulePtr module = ParseModule(R"(
target.profile @wasm preset("test.profile")
target.artifact @module target(@wasm)
target.artifact @other_module target(@wasm)

func.def target(@wasm) abi(wasm_function) export("entry", {artifact = @module}) @entry() {
  func.call @other() : ()
  func.return
}

func.def target(@wasm) abi(wasm_function) export("other", {artifact = @other_module}) @other() {
  func.return
}
)");

  loom_call_graph_t call_graph;
  IREE_ASSERT_OK(
      loom_call_graph_build(module.get(), &analysis_arena_, &call_graph));

  loom_target_artifact_plan_t plan;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_artifact_plan_build(
          module.get(), SymbolRef(module.get(), IREE_SV("module")),
          &fact_table_, &call_graph, &analysis_arena_, &plan));
}

TEST_F(ArtifactPlanTest, AllowsExternalDeclarationCalls) {
  ModulePtr module = ParseModule(R"(
target.profile @wasm preset("test.profile")
target.artifact @module target(@wasm)

func.def target(@wasm) abi(wasm_function) export("entry", {artifact = @module}) @entry() {
  func.call @external() : ()
  func.return
}

func.decl import("env") @external()
)");

  loom_call_graph_t call_graph;
  IREE_ASSERT_OK(
      loom_call_graph_build(module.get(), &analysis_arena_, &call_graph));

  loom_target_artifact_plan_t plan;
  IREE_ASSERT_OK(loom_target_artifact_plan_build(
      module.get(), SymbolRef(module.get(), IREE_SV("module")), &fact_table_,
      &call_graph, &analysis_arena_, &plan));

  EXPECT_EQ(plan.entry_count, 1u);
  EXPECT_EQ(plan.func_count, 1u);
  EXPECT_EQ(plan.func_symbol_ids[0],
            FindSymbol(module.get(), IREE_SV("entry")));
}

TEST_F(ArtifactPlanTest, RejectsNonFunctionCallees) {
  ModulePtr module = ParseModule(R"(
target.profile @wasm preset("test.profile")
target.artifact @module target(@wasm)

func.def target(@wasm) abi(wasm_function) export("entry", {artifact = @module}) @entry() {
  func.call @wasm() : ()
  func.return
}
)");

  loom_call_graph_t call_graph;
  IREE_ASSERT_OK(
      loom_call_graph_build(module.get(), &analysis_arena_, &call_graph));

  loom_target_artifact_plan_t plan;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_artifact_plan_build(
          module.get(), SymbolRef(module.get(), IREE_SV("module")),
          &fact_table_, &call_graph, &analysis_arena_, &plan));
}

}  // namespace
}  // namespace loom
