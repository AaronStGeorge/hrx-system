// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/records.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

loom_target_snapshot_t TestSnapshot() {
  loom_target_snapshot_t snapshot = {};
  snapshot.name = IREE_SVL("test-x86");
  snapshot.codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LLVMIR;
  snapshot.target_triple = IREE_SVL("x86_64-unknown-linux-gnu");
  snapshot.data_layout = IREE_SVL("e-m:e-p:64:64-i64:64-n8:16:32:64-S128");
  snapshot.artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;
  snapshot.target_cpu = IREE_SVL("x86-64-v3");
  snapshot.target_features = IREE_SVL("+avx2,+fma");
  snapshot.default_pointer_bitwidth = 64;
  snapshot.index_bitwidth = 64;
  snapshot.offset_bitwidth = 64;
  snapshot.memory_spaces.generic = 0;
  snapshot.memory_spaces.global = 0;
  snapshot.memory_spaces.workgroup = 0;
  snapshot.memory_spaces.constant = 0;
  snapshot.memory_spaces.private_memory = 0;
  snapshot.memory_spaces.host = 0;
  snapshot.memory_spaces.descriptor = UINT32_MAX;
  return snapshot;
}

loom_target_export_plan_t TestExportPlan() {
  loom_target_export_plan_t export_plan = {};
  export_plan.name = IREE_SVL("object-export");
  export_plan.source_symbol = IREE_SVL("@source");
  export_plan.export_symbol = IREE_SVL("exported");
  export_plan.abi_kind = LOOM_TARGET_ABI_OBJECT_FUNCTION;
  export_plan.linkage = LOOM_TARGET_LINKAGE_DSO_LOCAL;
  return export_plan;
}

loom_target_config_t TestConfig() {
  loom_target_config_t config = {};
  config.name = IREE_SVL("packed-dot");
  config.contract_set_key = IREE_SVL("x86.packed_dot.avx2");
  return config;
}

loom_target_bundle_t TestBundle(const loom_target_snapshot_t* snapshot,
                                const loom_target_export_plan_t* export_plan,
                                const loom_target_config_t* config) {
  loom_target_bundle_t bundle = {};
  bundle.name = IREE_SVL("test-bundle");
  bundle.snapshot = snapshot;
  bundle.export_plan = export_plan;
  bundle.config = config;
  return bundle;
}

TEST(TargetRecordsTest, FingerprintsAreStable) {
  loom_target_snapshot_t snapshot = TestSnapshot();
  uint64_t snapshot_fingerprint = 0;
  IREE_ASSERT_OK(
      loom_target_snapshot_fingerprint(&snapshot, &snapshot_fingerprint));
  EXPECT_EQ(snapshot_fingerprint, UINT64_C(14116158818610785230));

  loom_target_export_plan_t export_plan = TestExportPlan();
  uint64_t export_plan_fingerprint = 0;
  IREE_ASSERT_OK(loom_target_export_plan_fingerprint(&export_plan,
                                                     &export_plan_fingerprint));
  EXPECT_EQ(export_plan_fingerprint, UINT64_C(7170274714156440628));

  loom_target_config_t config = TestConfig();
  uint64_t config_fingerprint = 0;
  IREE_ASSERT_OK(loom_target_config_fingerprint(&config, &config_fingerprint));
  EXPECT_EQ(config_fingerprint, UINT64_C(11122253220825331316));

  loom_target_bundle_t bundle = TestBundle(&snapshot, &export_plan, &config);
  uint64_t bundle_fingerprint = 0;
  IREE_ASSERT_OK(loom_target_bundle_fingerprint(&bundle, &bundle_fingerprint));
  EXPECT_EQ(bundle_fingerprint, UINT64_C(4044221851011246781));
}

TEST(TargetRecordsTest, FingerprintsIncludeConfigPolicy) {
  loom_target_snapshot_t snapshot = TestSnapshot();
  loom_target_export_plan_t export_plan = TestExportPlan();
  loom_target_config_t config = TestConfig();
  loom_target_config_t other_config = config;
  other_config.contract_set_key = IREE_SVL("x86.packed_dot.avx512");

  uint64_t config_fingerprint = 0;
  IREE_ASSERT_OK(loom_target_config_fingerprint(&config, &config_fingerprint));
  uint64_t other_config_fingerprint = 0;
  IREE_ASSERT_OK(
      loom_target_config_fingerprint(&other_config, &other_config_fingerprint));
  EXPECT_NE(config_fingerprint, other_config_fingerprint);

  loom_target_bundle_t bundle = TestBundle(&snapshot, &export_plan, &config);
  loom_target_bundle_t other_bundle =
      TestBundle(&snapshot, &export_plan, &other_config);
  uint64_t bundle_fingerprint = 0;
  IREE_ASSERT_OK(loom_target_bundle_fingerprint(&bundle, &bundle_fingerprint));
  uint64_t other_bundle_fingerprint = 0;
  IREE_ASSERT_OK(
      loom_target_bundle_fingerprint(&other_bundle, &other_bundle_fingerprint));
  EXPECT_NE(bundle_fingerprint, other_bundle_fingerprint);
}

TEST(TargetRecordsTest, RejectsMissingInputs) {
  uint64_t fingerprint = 0;
  iree_status_t status = loom_target_snapshot_fingerprint(NULL, &fingerprint);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(fingerprint, 0u);
  iree_status_ignore(status);

  loom_target_snapshot_t snapshot = TestSnapshot();
  status = loom_target_snapshot_fingerprint(&snapshot, NULL);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);

  loom_target_export_plan_t export_plan = TestExportPlan();
  loom_target_config_t config = TestConfig();
  loom_target_bundle_t broken_bundle = TestBundle(NULL, &export_plan, &config);
  status = loom_target_bundle_fingerprint(&broken_bundle, &fingerprint);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(fingerprint, 0u);
  iree_status_ignore(status);
}

}  // namespace
}  // namespace loom
