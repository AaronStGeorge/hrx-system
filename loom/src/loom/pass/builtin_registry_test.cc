// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/builtin_registry.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

static const iree_string_view_t kExpectedBuiltinPassKeys[] = {
    IREE_SVL("branch-fusion"),
    IREE_SVL("branch-sink"),
    IREE_SVL("canonicalize"),
    IREE_SVL("cfg-simplify"),
    IREE_SVL("cse"),
    IREE_SVL("dce"),
    IREE_SVL("kernel-async-legality"),
    IREE_SVL("licm"),
    IREE_SVL("loop-fusion"),
    IREE_SVL("low-materialize-allocation"),
    IREE_SVL("normalize-kernel-resources"),
    IREE_SVL("refine-boundaries"),
    IREE_SVL("scf-to-cfg"),
    IREE_SVL("source-to-low"),
    IREE_SVL("strip-hints"),
    IREE_SVL("symbol-dce"),
    IREE_SVL("vector-memory-footprint"),
    IREE_SVL("vector-to-scalar"),
};

static const loom_pass_descriptor_t* LookupBuiltinPass(
    iree_string_view_t name) {
  const loom_pass_descriptor_t* descriptor = nullptr;
  IREE_EXPECT_OK(loom_pass_registry_lookup(loom_pass_builtin_registry(), name,
                                           &descriptor));
  return descriptor;
}

TEST(PassBuiltinRegistryTest, BuiltinRegistryVerifies) {
  IREE_ASSERT_OK(loom_pass_registry_verify(loom_pass_builtin_registry()));
}

TEST(PassBuiltinRegistryTest, BuiltinRegistryHasExpectedPasses) {
  const loom_pass_registry_t* registry = loom_pass_builtin_registry();
  ASSERT_EQ(registry->descriptor_count,
            IREE_ARRAYSIZE(kExpectedBuiltinPassKeys));
  for (iree_host_size_t i = 0; i < registry->descriptor_count; ++i) {
    EXPECT_TRUE(iree_string_view_equal(registry->descriptors[i].key,
                                       kExpectedBuiltinPassKeys[i]))
        << "unexpected builtin pass registry entry " << i;
    EXPECT_TRUE(iree_string_view_equal(registry->descriptors[i].key,
                                       registry->descriptors[i].info()->name))
        << "builtin pass descriptor key does not match pass info";
  }
}

TEST(PassBuiltinRegistryTest, LookupKnownAndUnknownPasses) {
  const loom_pass_descriptor_t* descriptor = nullptr;
  IREE_ASSERT_OK(loom_pass_registry_lookup(
      loom_pass_builtin_registry(), IREE_SV("canonicalize"), &descriptor));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_TRUE(iree_string_view_equal(descriptor->key, IREE_SV("canonicalize")));
  ASSERT_NE(descriptor->info, nullptr);
  EXPECT_EQ(descriptor->info()->kind, LOOM_PASS_FUNCTION);
  EXPECT_NE(descriptor->function_run, nullptr);
  EXPECT_NE(descriptor->create, nullptr);

  descriptor = reinterpret_cast<const loom_pass_descriptor_t*>(0x1);
  IREE_ASSERT_OK(loom_pass_registry_lookup(loom_pass_builtin_registry(),
                                           IREE_SV("definitely-not-a-pass"),
                                           &descriptor));
  EXPECT_EQ(descriptor, nullptr);
}

TEST(PassBuiltinRegistryTest, ValidatesBuiltinOptionSchemas) {
  const loom_pass_descriptor_t* canonicalize =
      LookupBuiltinPass(IREE_SV("canonicalize"));
  ASSERT_NE(canonicalize, nullptr);
  IREE_ASSERT_OK(loom_pass_descriptor_validate_options(
      canonicalize, IREE_SV("max-iterations=4")));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_descriptor_validate_options(
                            canonicalize, IREE_SV("max-iterations=0")));

  const loom_pass_descriptor_t* allocation =
      LookupBuiltinPass(IREE_SV("low-materialize-allocation"));
  ASSERT_NE(allocation, nullptr);
  ASSERT_EQ(allocation->requirement_count, 1u);
  EXPECT_TRUE(
      iree_string_view_equal(allocation->requirement_defs[0].key,
                             IREE_SV("target.low-descriptor-registry")));
  IREE_ASSERT_OK(loom_pass_descriptor_validate_options(
      allocation, IREE_SV("diagnostics=spills")));
  IREE_ASSERT_OK(loom_pass_descriptor_validate_options(
      allocation, IREE_SV("budgets=vm.i32=2;vm.ref=1")));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_descriptor_validate_options(
                            allocation, IREE_SV("diagnostics=verbose")));

  const loom_pass_descriptor_t* source_to_low =
      LookupBuiltinPass(IREE_SV("source-to-low"));
  ASSERT_NE(source_to_low, nullptr);
  ASSERT_EQ(source_to_low->requirement_count, 2u);
  EXPECT_TRUE(
      iree_string_view_equal(source_to_low->requirement_defs[0].key,
                             IREE_SV("target.low-descriptor-registry")));
  EXPECT_TRUE(
      iree_string_view_equal(source_to_low->requirement_defs[1].key,
                             IREE_SV("target.low-lower-policy-registry")));
  IREE_ASSERT_OK(loom_pass_descriptor_validate_options(
      source_to_low, IREE_SV("max-errors=0")));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_descriptor_validate_options(
                            source_to_low, IREE_SV("max-errors=-1")));
}

}  // namespace
}  // namespace loom
