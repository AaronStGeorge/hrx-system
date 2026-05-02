// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/artifact_plan.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/target/facts.h"
#include "loom/ops/target/ops.h"
#include "loom/target/preset_registry.h"
#include "loom/target/types.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"
#include "loom/util/call_graph.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

static const loom_target_snapshot_t kPresetSnapshot = {
    .name = IREE_SVL("test.profile"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
    .target_triple = IREE_SVL("test-low-unknown"),
    .data_layout = IREE_SVL(""),
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .target_cpu = IREE_SVL("generic"),
    .target_features = IREE_SVL("+test"),
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
    .abi_kind = LOOM_TARGET_ABI_OBJECT_FUNCTION,
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
    .contract_set_key = IREE_SVL("test.low.core"),
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
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
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

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
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

  iree_status_t BuildArtifactPlan(const loom_module_t* module,
                                  iree_string_view_t artifact_name,
                                  testing::DiagnosticEmissionCapture* capture,
                                  bool* out_valid,
                                  loom_target_artifact_plan_t* out_plan) {
    loom_call_graph_t call_graph;
    IREE_RETURN_IF_ERROR(
        loom_call_graph_build(module, &analysis_arena_, &call_graph));
    iree_diagnostic_emitter_t diagnostic_emitter =
        capture ? capture->emitter() : iree_diagnostic_emitter_t{};
    return loom_target_artifact_plan_build(
        module, SymbolRef(module, artifact_name), &fact_table_, &call_graph,
        diagnostic_emitter, &analysis_arena_, out_valid, out_plan);
  }

  void ExpectArtifactPlanError(const loom_module_t* module, uint16_t code) {
    testing::DiagnosticEmissionCapture capture;
    loom_target_artifact_plan_t plan;
    bool valid = true;
    IREE_ASSERT_OK(
        BuildArtifactPlan(module, IREE_SV("module"), &capture, &valid, &plan));
    EXPECT_FALSE(valid);
    ASSERT_EQ(capture.emissions.size(), 1u);
    EXPECT_EQ(capture.emissions[0].error,
              loom_error_def_lookup(LOOM_ERROR_DOMAIN_TARGET, code));
  }

  // Block pool shared by parser, module allocation, and analysis storage.
  iree_arena_block_pool_t block_pool_;

  // Context with the target and function dialects registered.
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
target.profile @test_target preset("test.profile")
target.artifact @module target(@test_target)

func.def target(@test_target) abi(object_function) export("entry", {artifact = @module}) @entry() {
  func.call @helper() : ()
  func.return
}

func.def target(@test_target) abi(object_function) @helper() {
  func.return
}

func.def target(@test_target) abi(object_function) @unused() {
  func.return
}
)");

  loom_target_artifact_plan_t plan;
  bool valid = false;
  IREE_ASSERT_OK(BuildArtifactPlan(module.get(), IREE_SV("module"), nullptr,
                                   &valid, &plan));
  ASSERT_TRUE(valid);

  EXPECT_EQ(plan.entry_count, 1u);
  EXPECT_EQ(plan.entry_symbol_ids[0],
            FindSymbol(module.get(), IREE_SV("entry")));
  EXPECT_EQ(plan.func_count, 2u);
  EXPECT_EQ(plan.func_symbol_ids[0],
            FindSymbol(module.get(), IREE_SV("entry")));
  EXPECT_EQ(plan.func_symbol_ids[1],
            FindSymbol(module.get(), IREE_SV("helper")));
}

TEST_F(ArtifactPlanTest, OrdersEntriesByDenseExportOrdinal) {
  ModulePtr module = ParseModule(R"(
target.profile @test_target preset("test.profile")
target.artifact @module target(@test_target)

func.def target(@test_target) abi(object_function) export("second", {artifact = @module, ordinal = 1}) @second() {
  func.return
}

func.def target(@test_target) abi(object_function) export("first", {artifact = @module, ordinal = 0}) @first() {
  func.return
}
)");

  loom_target_artifact_plan_t plan;
  bool valid = false;
  IREE_ASSERT_OK(BuildArtifactPlan(module.get(), IREE_SV("module"), nullptr,
                                   &valid, &plan));
  ASSERT_TRUE(valid);

  ASSERT_EQ(plan.entry_count, 2u);
  EXPECT_EQ(plan.entry_symbol_ids[0],
            FindSymbol(module.get(), IREE_SV("first")));
  EXPECT_EQ(plan.entry_symbol_ids[1],
            FindSymbol(module.get(), IREE_SV("second")));
}

TEST_F(ArtifactPlanTest, RejectsMixedExportOrdinalPolicy) {
  ModulePtr module = ParseModule(R"(
target.profile @test_target preset("test.profile")
target.artifact @module target(@test_target)

func.def target(@test_target) abi(object_function) export("first", {artifact = @module, ordinal = 0}) @first() {
  func.return
}

func.def target(@test_target) abi(object_function) export("second", {artifact = @module}) @second() {
  func.return
}
)");

  ExpectArtifactPlanError(module.get(), 34);
}

TEST_F(ArtifactPlanTest, RejectsEntryTargetMismatchingArtifactTarget) {
  ModulePtr module = ParseModule(R"(
target.profile @test_target preset("test.profile")
target.profile @other preset("test.profile")
target.artifact @module target(@test_target)

func.def target(@other) abi(object_function) export("entry", {artifact = @module}) @entry() {
  func.return
}
)");

  ExpectArtifactPlanError(module.get(), 33);
}

TEST_F(ArtifactPlanTest, RejectsClosureCrossingIntoAnotherArtifact) {
  ModulePtr module = ParseModule(R"(
target.profile @test_target preset("test.profile")
target.artifact @module target(@test_target)
target.artifact @other_module target(@test_target)

func.def target(@test_target) abi(object_function) export("entry", {artifact = @module}) @entry() {
  func.call @other() : ()
  func.return
}

func.def target(@test_target) abi(object_function) export("other", {artifact = @other_module}) @other() {
  func.return
}
)");

  ExpectArtifactPlanError(module.get(), 31);
}

TEST_F(ArtifactPlanTest, AllowsExternalDeclarationCalls) {
  ModulePtr module = ParseModule(R"(
target.profile @test_target preset("test.profile")
target.artifact @module target(@test_target)

func.def target(@test_target) abi(object_function) export("entry", {artifact = @module}) @entry() {
  func.call @external() : ()
  func.return
}

func.decl import("env") @external()
)");

  loom_target_artifact_plan_t plan;
  bool valid = false;
  IREE_ASSERT_OK(BuildArtifactPlan(module.get(), IREE_SV("module"), nullptr,
                                   &valid, &plan));
  ASSERT_TRUE(valid);

  EXPECT_EQ(plan.entry_count, 1u);
  EXPECT_EQ(plan.func_count, 1u);
  EXPECT_EQ(plan.func_symbol_ids[0],
            FindSymbol(module.get(), IREE_SV("entry")));
}

TEST_F(ArtifactPlanTest, RejectsNonFunctionCallees) {
  ModulePtr module = ParseModule(R"(
target.profile @test_target preset("test.profile")
target.artifact @module target(@test_target)

func.def target(@test_target) abi(object_function) export("entry", {artifact = @module}) @entry() {
  func.call @test_target() : ()
  func.return
}
)");

  ExpectArtifactPlanError(module.get(), 30);
}

}  // namespace
}  // namespace loom
