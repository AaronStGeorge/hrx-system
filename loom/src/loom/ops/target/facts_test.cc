// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/target/facts.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/target/ops.h"
#include "loom/testing/module_ptr.h"

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
    .export_symbol = IREE_SVL("kernel"),
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

class TargetProfileFactsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
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
                                  IREE_SV("target_profile_facts_test.loom"),
                                  &context_, &block_pool_, &options, &module));
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

  std::string PrintModule(const loom_module_t* module) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_CHECK_OK(loom_text_print_module_to_builder(module, &builder,
                                                    LOOM_TEXT_PRINT_DEFAULT));
    std::string result(builder.buffer, builder.size);
    iree_string_builder_deinitialize(&builder);
    return result;
  }

  loom_symbol_id_t FindSymbol(const loom_module_t* module,
                              iree_string_view_t name) {
    loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return symbol_id;
  }

  const loom_target_profile_symbol_facts_t* LookupProfile(
      const loom_module_t* module, iree_string_view_t name) {
    const loom_symbol_facts_base_t* base_facts = nullptr;
    IREE_CHECK_OK(loom_symbol_fact_table_lookup(
        &fact_table_, module, FindSymbol(module, name), &base_facts));
    const loom_target_profile_symbol_facts_t* profile_facts =
        loom_target_profile_symbol_facts_cast(base_facts);
    IREE_ASSERT(profile_facts != nullptr);
    return profile_facts;
  }

  const loom_target_artifact_symbol_facts_t* LookupArtifact(
      const loom_module_t* module, iree_string_view_t name) {
    const loom_symbol_facts_base_t* base_facts = nullptr;
    IREE_CHECK_OK(loom_symbol_fact_table_lookup(
        &fact_table_, module, FindSymbol(module, name), &base_facts));
    const loom_target_artifact_symbol_facts_t* artifact_facts =
        loom_target_artifact_symbol_facts_cast(base_facts);
    IREE_ASSERT(artifact_facts != nullptr);
    return artifact_facts;
  }

  // Block pool shared by parser, module allocation, and analysis storage.
  iree_arena_block_pool_t block_pool_;

  // Context with the target dialect registered.
  loom_context_t context_;

  // Arena for symbol fact table storage and fact payloads.
  iree_arena_allocator_t analysis_arena_;

  // Dense symbol fact table under test.
  loom_symbol_fact_table_t fact_table_;

  // Resource storage borrowed by the symbol fact table.
  loom_symbol_fact_resource_t resources_[1];
};

TEST_F(TargetProfileFactsTest, ResolvesPresetIntoDenseBundleFacts) {
  ModulePtr module = ParseModule(R"(
target.profile @test_target preset("test.profile")
)");

  const loom_target_profile_symbol_facts_t* facts =
      LookupProfile(module.get(), IREE_SV("test_target"));
  EXPECT_EQ(facts->base.domain, &loom_target_profile_symbol_fact_domain);
  EXPECT_EQ(facts->base.symbol_kind, LOOM_SYMBOL_RECORD);
  EXPECT_EQ(facts->preset_bundle, &kPresetBundle);
  EXPECT_TRUE(
      iree_string_view_equal(facts->preset_key, IREE_SV("test.profile")));
  EXPECT_TRUE(iree_string_view_equal(facts->name, IREE_SV("test_target")));
  EXPECT_EQ(facts->bundle.snapshot, &facts->snapshot);
  EXPECT_EQ(facts->bundle.export_plan, &facts->export_plan);
  EXPECT_EQ(facts->bundle.config, &facts->config);
  EXPECT_EQ(facts->snapshot.codegen_format,
            LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE);
  EXPECT_EQ(facts->snapshot.default_pointer_bitwidth, 32u);
  EXPECT_TRUE(iree_string_view_equal(facts->snapshot.target_features,
                                     IREE_SV("+test")));
  EXPECT_TRUE(iree_string_view_is_empty(facts->export_plan.export_symbol));
  EXPECT_EQ(facts->export_plan.abi_kind, LOOM_TARGET_ABI_OBJECT_FUNCTION);
  EXPECT_TRUE(iree_string_view_equal(facts->config.contract_set_key,
                                     IREE_SV("test.low.core")));
}

TEST_F(TargetProfileFactsTest, PrintsCompactProfileSyntax) {
  ModulePtr module = ParseModule(R"(
target.profile @test_target preset("test.profile") {index_bitwidth = 64}
)");

  std::string printed = PrintModule(module.get());
  EXPECT_NE(printed.find("target.profile @test_target preset(\"test.profile\") "
                         "{index_bitwidth = 64}"),
            std::string::npos);
}

TEST_F(TargetProfileFactsTest, ArtifactFactsDefaultFromTargetProfile) {
  ModulePtr module = ParseModule(R"(
target.profile @test_target preset("test.profile")
target.artifact @module target(@test_target)
)");

  const loom_target_profile_symbol_facts_t* profile =
      LookupProfile(module.get(), IREE_SV("test_target"));
  const loom_target_artifact_symbol_facts_t* artifact =
      LookupArtifact(module.get(), IREE_SV("module"));
  EXPECT_EQ(artifact->base.domain, &loom_target_artifact_symbol_fact_domain);
  EXPECT_EQ(artifact->base.symbol_kind, LOOM_SYMBOL_RECORD);
  EXPECT_TRUE(iree_string_view_equal(artifact->name, IREE_SV("module")));
  EXPECT_EQ(artifact->target_profile, profile);
  EXPECT_EQ(artifact->format, LOOM_TARGET_ARTIFACT_FORMAT_ELF);
  EXPECT_EQ(artifact->abi_kind, LOOM_TARGET_ARTIFACT_ABI_KIND_OBJECT_FILE);
}

TEST_F(TargetProfileFactsTest, ArtifactFactsOverridePackagingAbi) {
  ModulePtr module = ParseModule(R"(
target.profile @gfx preset("test.profile")
target.artifact @hsaco target(@gfx) {
  abi = hal_executable,
  artifact_format = elf
}
)");

  const loom_target_artifact_symbol_facts_t* artifact =
      LookupArtifact(module.get(), IREE_SV("hsaco"));
  EXPECT_EQ(artifact->format, LOOM_TARGET_ARTIFACT_FORMAT_ELF);
  EXPECT_EQ(artifact->abi_kind, LOOM_TARGET_ARTIFACT_ABI_KIND_HAL_EXECUTABLE);
}

TEST_F(TargetProfileFactsTest, SparseOverridesReplacePresetFields) {
  ModulePtr module = ParseModule(R"(
target.profile @gfx preset("test.profile") {
  abi = hal_kernel,
  artifact_format = elf,
  codegen_format = low_native,
  contract_feature_bits = 7,
  contract_set_key = "amdgpu.gfx11",
  hal_binding_alignment = 16,
  index_bitwidth = 64,
  memory_space_descriptor = 9,
  subgroup_size = 32,
  target_cpu = "gfx1100",
  target_features = "+wavefrontsize32"
}
)");

  const loom_target_profile_symbol_facts_t* facts =
      LookupProfile(module.get(), IREE_SV("gfx"));
  EXPECT_EQ(facts->snapshot.codegen_format,
            LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE);
  EXPECT_EQ(facts->snapshot.artifact_format, LOOM_TARGET_ARTIFACT_FORMAT_ELF);
  EXPECT_TRUE(
      iree_string_view_equal(facts->snapshot.target_cpu, IREE_SV("gfx1100")));
  EXPECT_TRUE(iree_string_view_equal(facts->snapshot.target_features,
                                     IREE_SV("+wavefrontsize32")));
  EXPECT_EQ(facts->snapshot.index_bitwidth, 64u);
  EXPECT_EQ(facts->snapshot.subgroup_size, 32u);
  EXPECT_EQ(facts->snapshot.memory_spaces.descriptor, 9u);
  EXPECT_EQ(facts->export_plan.abi_kind, LOOM_TARGET_ABI_HAL_KERNEL);
  EXPECT_EQ(facts->export_plan.hal_kernel.binding_alignment, 16u);
  EXPECT_EQ(facts->export_plan.hal_kernel.required_workgroup_size.x, 0u);
  EXPECT_EQ(facts->export_plan.hal_kernel.required_workgroup_size.y, 0u);
  EXPECT_EQ(facts->export_plan.hal_kernel.required_workgroup_size.z, 0u);
  EXPECT_EQ(facts->export_plan.hal_kernel.flat_workgroup_size_min, 0u);
  EXPECT_EQ(facts->export_plan.hal_kernel.flat_workgroup_size_max, 0u);
  EXPECT_TRUE(iree_string_view_equal(facts->config.contract_set_key,
                                     IREE_SV("amdgpu.gfx11")));
  EXPECT_EQ(facts->config.contract_feature_bits, 7u);
}

TEST_F(TargetProfileFactsTest, MissingPresetRegistryFailsLoudly) {
  loom_symbol_fact_table_initialize(&fact_table_, &analysis_arena_);
  ModulePtr module = ParseModule(R"(
target.profile @missing preset("test.profile")
)");

  const loom_symbol_facts_base_t* facts = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_symbol_fact_table_lookup(
          &fact_table_, module.get(),
          FindSymbol(module.get(), IREE_SV("missing")), &facts));
  EXPECT_EQ(facts, nullptr);
}

TEST_F(TargetProfileFactsTest, UnknownPresetFailsLoudly) {
  ModulePtr module = ParseModule(R"(
target.profile @missing preset("missing.profile")
)");

  const loom_symbol_facts_base_t* facts = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_symbol_fact_table_lookup(
          &fact_table_, module.get(),
          FindSymbol(module.get(), IREE_SV("missing")), &facts));
  EXPECT_EQ(facts, nullptr);
}

TEST_F(TargetProfileFactsTest, UnknownOverrideFailsLoudly) {
  ModulePtr module = ParseModule(R"(
target.profile @bad preset("test.profile") {source = "must_not_exist"}
)");

  const loom_symbol_facts_base_t* facts = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_symbol_fact_table_lookup(
                            &fact_table_, module.get(),
                            FindSymbol(module.get(), IREE_SV("bad")), &facts));
  EXPECT_EQ(facts, nullptr);
}

TEST_F(TargetProfileFactsTest, ArtifactTargetMustResolveToProfileFacts) {
  ModulePtr module = ParseModule(R"(
target.profile @profile preset("test.profile")
target.artifact @not_profile target(@profile)
target.artifact @bad target(@not_profile)
)");

  const loom_symbol_facts_base_t* facts = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_symbol_fact_table_lookup(
                            &fact_table_, module.get(),
                            FindSymbol(module.get(), IREE_SV("bad")), &facts));
  EXPECT_EQ(facts, nullptr);
}

TEST_F(TargetProfileFactsTest, ArtifactAbiMustMatchFormat) {
  ModulePtr module = ParseModule(R"(
target.profile @test_target preset("test.profile")
target.artifact @bad target(@test_target) {
  abi = hal_executable,
  artifact_format = vm_bytecode
}
)");

  const loom_symbol_facts_base_t* facts = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_symbol_fact_table_lookup(
                            &fact_table_, module.get(),
                            FindSymbol(module.get(), IREE_SV("bad")), &facts));
  EXPECT_EQ(facts, nullptr);
}

}  // namespace
}  // namespace loom
