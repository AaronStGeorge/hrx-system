// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/artifact_manifest_collect.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbol_dependencies.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/target/artifact_manifest.h"
#include "loom/testing/module_ptr.h"
#include "loom/util/call_graph.h"
#include "loom/util/stream.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class ArtifactManifestCollectTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_GLOBAL, loom_global_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_symbol_fact_table_initialize(&fact_table_, &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
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

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("artifact_manifest_collect.loom"),
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

  loom_symbol_ref_t SymbolRef(const loom_module_t* module,
                              iree_string_view_t name) {
    return (loom_symbol_ref_t){
        .module_id = 0,
        .symbol_id = FindSymbol(module, name),
    };
  }

  iree_status_t BuildPlanAndDependencies(
      const loom_module_t* module, iree_string_view_t artifact_name,
      loom_target_artifact_plan_t* out_plan,
      loom_symbol_dependency_table_t* out_dependencies) {
    loom_call_graph_t call_graph;
    IREE_RETURN_IF_ERROR(
        loom_call_graph_build(module, &analysis_arena_, &call_graph));
    bool valid = false;
    IREE_RETURN_IF_ERROR(loom_target_artifact_plan_build(
        module, SymbolRef(module, artifact_name), &fact_table_, &call_graph,
        iree_diagnostic_emitter_t{}, &analysis_arena_, &valid, out_plan));
    if (!valid) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "artifact plan unexpectedly invalid");
    }
    return loom_symbol_dependency_table_build(module, &analysis_arena_,
                                              out_dependencies);
  }

  iree_string_view_t CollectAndFormat(
      const loom_module_t* module,
      const loom_target_artifact_manifest_collect_options_t* collect_options,
      loom_target_artifact_manifest_mode_t format_mode,
      iree_string_builder_t* builder) {
    loom_target_artifact_plan_t plan;
    loom_symbol_dependency_table_t dependencies;
    IREE_CHECK_OK(BuildPlanAndDependencies(module, IREE_SV("module"), &plan,
                                           &dependencies));
    loom_target_artifact_manifest_t manifest;
    IREE_CHECK_OK(loom_target_artifact_manifest_collect_from_plan(
        module, &plan, &fact_table_, &dependencies, collect_options,
        &analysis_arena_, &manifest));

    iree_string_builder_initialize(iree_allocator_system(), builder);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(builder, &stream);
    const loom_target_artifact_manifest_format_options_t format_options = {
        .mode = format_mode,
    };
    IREE_CHECK_OK(loom_target_artifact_manifest_format_json(
        &manifest, &format_options, &stream));
    return iree_string_builder_view(builder);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  iree_arena_allocator_t analysis_arena_;
  loom_symbol_fact_table_t fact_table_;
};

TEST_F(ArtifactManifestCollectTest, CollectsSummaryFunctionsAndGlobals) {
  ModulePtr module = ParseModule(R"(
target.generic<reference> @test_target {
  artifact_format = elf,
  subgroup_size = 32
}
target.artifact @module target(@test_target) {artifact_format = elf}

global.constant @weights : i32 = 1
global.variable @scratch : i32

func.def target(@test_target) abi(object_function) export("run", {artifact = @module, ordinal = 0}) @entry(%input: i32) {
  func.call @helper() : ()
  func.return
}

func.def target(@test_target) abi(object_function) export("aux", {artifact = @module, ordinal = 1}) @aux() {
  func.return
}

func.def target(@test_target) abi(object_function) @helper() {
  %value = global.load @weights : i32
  global.store %value, @scratch : i32
  func.return
}

func.def target(@test_target) abi(object_function) @unused() {
  func.return
}
)");

  loom_target_artifact_manifest_collect_options_t options;
  loom_target_artifact_manifest_collect_options_initialize(&options);
  options.mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY;
  options.flags =
      LOOM_TARGET_ARTIFACT_MANIFEST_COLLECT_FLAG_ARTIFACT_BYTE_LENGTH;
  options.artifact_byte_length = 0;

  iree_string_builder_t builder;
  iree_string_view_t output =
      CollectAndFormat(module.get(), &options,
                       LOOM_TARGET_ARTIFACT_MANIFEST_MODE_SUMMARY, &builder);
  EXPECT_TRUE(iree_string_view_equal(
      output, IREE_SV("{\"kind\":\"loom.artifact_manifest\","
                      "\"schema_version\":1,"
                      "\"mode\":\"summary\","
                      "\"artifact\":{\"format\":\"elf\","
                      "\"name\":\"module\","
                      "\"byte_length\":0},"
                      "\"targets\":[{\"name\":\"test_target\"}],"
                      "\"functions\":[{\"name\":\"run\","
                      "\"source\":\"entry\","
                      "\"targets\":[\"test_target\"],"
                      "\"interface\":{\"parameter_count\":1},"
                      "\"execution\":{\"subgroup_size\":32}},"
                      "{\"name\":\"aux\","
                      "\"targets\":[\"test_target\"],"
                      "\"interface\":{\"parameter_count\":0},"
                      "\"execution\":{\"subgroup_size\":32}}],"
                      "\"globals\":[{\"name\":\"weights\","
                      "\"targets\":[\"test_target\"]},"
                      "{\"name\":\"scratch\","
                      "\"targets\":[\"test_target\"]}]}")));
  iree_string_builder_deinitialize(&builder);
}

TEST_F(ArtifactManifestCollectTest, CollectsDetailsParameters) {
  ModulePtr module = ParseModule(R"(
target.generic<reference> @test_target {artifact_format = spirv_binary}
target.artifact @module target(@test_target) {artifact_format = spirv_binary}

func.def target(@test_target) abi(object_function) export("entry", {artifact = @module}) @entry(%lhs: i32, %rhs: f32) {
  func.return
}
)");

  loom_target_artifact_manifest_collect_options_t options;
  loom_target_artifact_manifest_collect_options_initialize(&options);
  options.mode = LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS;

  iree_string_builder_t builder;
  iree_string_view_t output =
      CollectAndFormat(module.get(), &options,
                       LOOM_TARGET_ARTIFACT_MANIFEST_MODE_DETAILS, &builder);
  EXPECT_TRUE(iree_string_view_equal(
      output, IREE_SV("{\"kind\":\"loom.artifact_manifest\","
                      "\"schema_version\":1,"
                      "\"mode\":\"details\","
                      "\"artifact\":{\"format\":\"spirv-binary\","
                      "\"name\":\"module\"},"
                      "\"targets\":[{\"name\":\"test_target\"}],"
                      "\"functions\":[{\"name\":\"entry\","
                      "\"targets\":[\"test_target\"],"
                      "\"interface\":{\"parameter_count\":2,"
                      "\"parameters\":[{\"name\":\"lhs\","
                      "\"kind\":\"value\","
                      "\"index\":0},"
                      "{\"name\":\"rhs\","
                      "\"kind\":\"value\","
                      "\"index\":1}]}}]}")));
  iree_string_builder_deinitialize(&builder);
}

TEST_F(ArtifactManifestCollectTest, NoneModeLeavesManifestEmpty) {
  ModulePtr module = ParseModule(R"(
target.generic<reference> @test_target
target.artifact @module target(@test_target)

func.def target(@test_target) abi(object_function) export("entry", {artifact = @module}) @entry() {
  func.return
}
)");
  loom_target_artifact_plan_t plan;
  loom_symbol_dependency_table_t dependencies;
  IREE_ASSERT_OK(BuildPlanAndDependencies(module.get(), IREE_SV("module"),
                                          &plan, &dependencies));

  loom_target_artifact_manifest_collect_options_t options;
  loom_target_artifact_manifest_collect_options_initialize(&options);
  loom_target_artifact_manifest_t manifest;
  IREE_ASSERT_OK(loom_target_artifact_manifest_collect_from_plan(
      module.get(), &plan, &fact_table_, &dependencies, &options,
      &analysis_arena_, &manifest));
  EXPECT_TRUE(iree_string_view_is_empty(manifest.artifact.format));
  EXPECT_EQ(manifest.function_count, 0u);
  EXPECT_EQ(manifest.global_count, 0u);
}

}  // namespace
}  // namespace loom
