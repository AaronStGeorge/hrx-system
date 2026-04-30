// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/requirements.h"
#include "loom/target/low_descriptor_registry_core_test.h"
#include "loom/target/low_descriptor_registry_manifest.h"
#include "loom/target/low_descriptor_registry_verify.h"

namespace loom {
namespace {

TEST(LowDescriptorRegistryTest, RegistryVerifiesSelectedTargetPackages) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&registry);

  EXPECT_NE(registry.registry.descriptor_set_providers, nullptr);
  EXPECT_EQ(registry.registry.descriptor_set_provider_count, 1u);
  IREE_EXPECT_OK(loom_target_low_descriptor_registry_verify(
      &registry, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
}

TEST(LowDescriptorRegistryTest, AppendToTablesBuildsBorrowedSubset) {
  loom_target_low_descriptor_registry_t source_registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&source_registry);

  loom_low_descriptor_set_provider_t descriptor_set_providers[8] = {};
  const loom_target_bundle_t* target_bundles[8] = {};
  ASSERT_LE(source_registry.descriptor_set_provider_count,
            IREE_ARRAYSIZE(descriptor_set_providers));
  ASSERT_LE(source_registry.target_bundle_count,
            IREE_ARRAYSIZE(target_bundles));

  iree_host_size_t descriptor_set_provider_count = 0;
  iree_host_size_t target_bundle_count = 0;
  IREE_ASSERT_OK(loom_target_low_descriptor_registry_append_to_tables(
      &source_registry, descriptor_set_providers,
      IREE_ARRAYSIZE(descriptor_set_providers), &descriptor_set_provider_count,
      target_bundles, IREE_ARRAYSIZE(target_bundles), &target_bundle_count));

  EXPECT_EQ(descriptor_set_provider_count,
            source_registry.descriptor_set_provider_count);
  EXPECT_EQ(target_bundle_count, source_registry.target_bundle_count);
  for (iree_host_size_t i = 0; i < descriptor_set_provider_count; ++i) {
    EXPECT_EQ(descriptor_set_providers[i],
              source_registry.descriptor_set_providers[i]);
  }
  for (iree_host_size_t i = 0; i < target_bundle_count; ++i) {
    EXPECT_EQ(target_bundles[i], source_registry.target_bundles[i]);
  }

  loom_target_low_descriptor_registry_t joined_registry = {};
  loom_target_low_descriptor_registry_initialize_from_tables(
      &joined_registry, descriptor_set_providers, descriptor_set_provider_count,
      target_bundles, target_bundle_count);
  IREE_EXPECT_OK(loom_target_low_descriptor_registry_verify(
      &joined_registry, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
}

TEST(LowDescriptorRegistryTest, AppendToTablesRejectsOverflow) {
  loom_target_low_descriptor_registry_t source_registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&source_registry);
  ASSERT_GT(source_registry.descriptor_set_provider_count, 0u);
  ASSERT_GT(source_registry.target_bundle_count, 0u);

  loom_low_descriptor_set_provider_t descriptor_set_providers[8] = {};
  const loom_target_bundle_t* target_bundles[8] = {};
  iree_host_size_t descriptor_set_provider_count =
      IREE_ARRAYSIZE(descriptor_set_providers);
  iree_host_size_t target_bundle_count = IREE_ARRAYSIZE(target_bundles);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_target_low_descriptor_registry_append_to_tables(
          &source_registry, descriptor_set_providers,
          IREE_ARRAYSIZE(descriptor_set_providers),
          &descriptor_set_provider_count, target_bundles,
          IREE_ARRAYSIZE(target_bundles), &target_bundle_count));
  EXPECT_EQ(descriptor_set_provider_count,
            IREE_ARRAYSIZE(descriptor_set_providers));
  EXPECT_EQ(target_bundle_count, IREE_ARRAYSIZE(target_bundles));
}

TEST(LowDescriptorRegistryTest, RegistrySatisfiesTargetLowFoundation) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&registry);

  IREE_ASSERT_OK(loom_low_descriptor_registry_verify_requirements(
      &registry.registry,
      LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
}

TEST(LowDescriptorRegistryTest, LooksUpRepresentativeDescriptorSets) {
  const loom_low_descriptor_set_t* descriptor_set = nullptr;
  IREE_ASSERT_OK(loom_target_core_test_low_descriptor_set_lookup(
      IREE_SV("test.low.core"), &descriptor_set));
  ASSERT_NE(descriptor_set, nullptr);

  iree_string_view_t set_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset, &set_key));
  EXPECT_TRUE(iree_string_view_equal(set_key, IREE_SV("test.low.core")));
}

TEST(LowDescriptorRegistryTest, LooksUpRepresentativeDescriptors) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&registry);

  struct ExpectedDescriptor {
    // Descriptor-set key owning the representative descriptor.
    const char* set_key;
    // Descriptor key that must remain addressable through the selected set.
    const char* descriptor_key;
  };
  const ExpectedDescriptor expected_descriptors[] = {
      {"test.low.core", "test.add.phys"},
      {"test.low.core", "test.load.v4i32"},
      {"test.low.core", "test.store.v4i32"},
      {"test.low.core", "test.call.i32"},
  };
  EXPECT_EQ(
      loom_low_descriptor_registry_descriptor_set_count(&registry.registry),
      1u);

  for (const ExpectedDescriptor& expected : expected_descriptors) {
    const loom_low_descriptor_set_t* descriptor_set = nullptr;
    IREE_ASSERT_OK(loom_target_core_test_low_descriptor_set_lookup(
        iree_make_cstring_view(expected.set_key), &descriptor_set));
    ASSERT_NE(descriptor_set, nullptr) << expected.set_key;

    uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
    IREE_ASSERT_OK(loom_low_descriptor_set_lookup_descriptor(
        descriptor_set, iree_make_cstring_view(expected.descriptor_key),
        &descriptor_ordinal))
        << expected.set_key << " :: " << expected.descriptor_key;
    EXPECT_NE(descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE)
        << expected.set_key << " :: " << expected.descriptor_key;
  }
}

TEST(LowDescriptorRegistryTest, TargetBundlesSelectFoundationDescriptorSets) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&registry);

  struct ExpectedBundle {
    // Target-low preset bundle key.
    const char* bundle_key;
    // Descriptor-set key selected by the bundle config.
    const char* descriptor_set_key;
    // Snapshot codegen format required by the bundle.
    loom_target_codegen_format_t codegen_format;
    // Snapshot artifact format required by the bundle.
    loom_target_artifact_format_t artifact_format;
    // Export ABI kind required by the bundle.
    loom_target_abi_kind_t abi_kind;
  };
  const ExpectedBundle expected_bundles[] = {
      {"test-low", "test.low.core", LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
       LOOM_TARGET_ARTIFACT_FORMAT_ELF, LOOM_TARGET_ABI_OBJECT_FUNCTION},
  };

  ASSERT_EQ(registry.target_bundle_count, IREE_ARRAYSIZE(expected_bundles));
  for (const ExpectedBundle& expected : expected_bundles) {
    const loom_target_bundle_t* bundle = nullptr;
    IREE_ASSERT_OK(loom_target_low_descriptor_registry_lookup_bundle(
        &registry, iree_make_cstring_view(expected.bundle_key), &bundle));
    ASSERT_NE(bundle, nullptr) << expected.bundle_key;
    ASSERT_NE(bundle->snapshot, nullptr) << expected.bundle_key;
    ASSERT_NE(bundle->export_plan, nullptr) << expected.bundle_key;
    ASSERT_NE(bundle->config, nullptr) << expected.bundle_key;
    EXPECT_EQ(bundle->snapshot->codegen_format, expected.codegen_format)
        << expected.bundle_key;
    EXPECT_EQ(bundle->snapshot->artifact_format, expected.artifact_format)
        << expected.bundle_key;
    EXPECT_EQ(bundle->export_plan->abi_kind, expected.abi_kind)
        << expected.bundle_key;
    EXPECT_TRUE(iree_string_view_equal(
        bundle->config->contract_set_key,
        iree_make_cstring_view(expected.descriptor_set_key)))
        << expected.bundle_key;

    const loom_low_descriptor_set_t* descriptor_set = nullptr;
    IREE_ASSERT_OK(loom_target_low_descriptor_set_select_for_bundle(
        &registry.registry, bundle, &descriptor_set))
        << expected.bundle_key;
    ASSERT_NE(descriptor_set, nullptr) << expected.bundle_key;

    iree_string_view_t descriptor_set_key = iree_string_view_empty();
    IREE_ASSERT_OK(loom_low_descriptor_set_string(
        descriptor_set, descriptor_set->key_string_offset,
        &descriptor_set_key));
    EXPECT_TRUE(iree_string_view_equal(
        descriptor_set_key,
        iree_make_cstring_view(expected.descriptor_set_key)))
        << expected.bundle_key;
  }
}

TEST(LowDescriptorRegistryTest, FormatsRegistryManifestJson) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&registry);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_target_low_descriptor_registry_format_manifest_json(
      &registry, &builder));
  EXPECT_NE(iree_string_builder_size(&builder), 0u);
  iree_string_builder_deinitialize(&builder);
}

TEST(LowDescriptorRegistryTest, FormatManifestRejectsMissingBuilder) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&registry);

  iree_status_t status =
      loom_target_low_descriptor_registry_format_manifest_json(&registry,
                                                               nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorRegistryTest, MissingKeyReturnsNullDescriptorSet) {
  const loom_low_descriptor_set_t* descriptor_set = nullptr;
  IREE_ASSERT_OK(loom_target_core_test_low_descriptor_set_lookup(
      IREE_SV("target.missing"), &descriptor_set));
  EXPECT_EQ(descriptor_set, nullptr);
}

TEST(LowDescriptorRegistryTest, MissingBundleKeyFailsLoudly) {
  const loom_target_bundle_t* bundle = nullptr;
  iree_status_t status =
      loom_target_core_test_low_bundle_lookup(IREE_SV(""), &bundle);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(bundle, nullptr);

  status = loom_target_core_test_low_bundle_lookup(IREE_SV("target.missing"),
                                                   &bundle);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  EXPECT_EQ(bundle, nullptr);
}

TEST(LowDescriptorRegistryTest, RegistryRejectsMissingBundleTable) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&registry);
  registry.target_bundles = nullptr;

  iree_status_t status = loom_target_low_descriptor_registry_verify(
      &registry, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  const loom_target_bundle_t* bundle = nullptr;
  status = loom_target_low_descriptor_registry_lookup_bundle(
      &registry, IREE_SV("test-low"), &bundle);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(bundle, nullptr);
}

TEST(LowDescriptorRegistryTest, RegistryRejectsMismatchedDescriptorView) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&registry);
  registry.registry.descriptor_set_provider_count = 0;

  iree_status_t status = loom_target_low_descriptor_registry_verify(
      &registry, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorRegistryTest, RegistryRejectsDuplicateBundleKeys) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&registry);

  const loom_target_bundle_t* bundle = nullptr;
  IREE_ASSERT_OK(loom_target_low_descriptor_registry_lookup_bundle(
      &registry, IREE_SV("test-low"), &bundle));
  ASSERT_NE(bundle, nullptr);

  const loom_target_bundle_t* duplicate_bundles[] = {bundle, bundle};
  registry.target_bundles = duplicate_bundles;
  registry.target_bundle_count = IREE_ARRAYSIZE(duplicate_bundles);

  iree_status_t status = loom_target_low_descriptor_registry_verify(
      &registry, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorRegistryTest, BundleSelectionRequiresExplicitLinkedSet) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&registry);

  const loom_target_config_t empty_config = {
      .name = IREE_SVL("empty"),
  };
  const loom_target_bundle_t empty_bundle = {
      .name = IREE_SVL("empty-bundle"),
      .config = &empty_config,
  };
  const loom_low_descriptor_set_t* descriptor_set = nullptr;
  iree_status_t status = loom_target_low_descriptor_set_select_for_bundle(
      &registry.registry, &empty_bundle, &descriptor_set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(descriptor_set, nullptr);

  const loom_target_config_t missing_config = {
      .name = IREE_SVL("missing"),
      .contract_set_key = IREE_SVL("target.missing"),
  };
  const loom_target_bundle_t missing_bundle = {
      .name = IREE_SVL("missing-bundle"),
      .config = &missing_config,
  };
  status = loom_target_low_descriptor_set_select_for_bundle(
      &registry.registry, &missing_bundle, &descriptor_set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  EXPECT_EQ(descriptor_set, nullptr);
}

TEST(LowDescriptorRegistryTest, RegistryRejectsBundleSelectingMissingSet) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&registry);

  const loom_target_bundle_t* source_bundle = nullptr;
  IREE_ASSERT_OK(loom_target_low_descriptor_registry_lookup_bundle(
      &registry, IREE_SV("test-low"), &source_bundle));
  ASSERT_NE(source_bundle, nullptr);

  const loom_target_config_t missing_config = {
      .name = IREE_SVL("missing"),
      .contract_set_key = IREE_SVL("target.missing"),
  };
  const loom_target_bundle_t missing_bundle = {
      .name = IREE_SVL("missing-bundle"),
      .snapshot = source_bundle->snapshot,
      .export_plan = source_bundle->export_plan,
      .config = &missing_config,
  };
  const loom_target_bundle_t* bundles[] = {&missing_bundle};
  registry.target_bundles = bundles;
  registry.target_bundle_count = IREE_ARRAYSIZE(bundles);

  iree_status_t status = loom_target_low_descriptor_registry_verify(
      &registry, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
}

}  // namespace
}  // namespace loom
