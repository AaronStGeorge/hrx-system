// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/maintenance.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(MaintenanceTest, DisabledPolicyNeedsNoGroupOrWakeTarget) {
  loom_serve_maintenance_t maintenance;
  IREE_ASSERT_OK(
      loom_serve_maintenance_initialize(nullptr, 0, nullptr, &maintenance));
  bool progress = true;
  IREE_ASSERT_OK(
      loom_serve_maintenance_advance(&maintenance, false, &progress));
  EXPECT_FALSE(progress);
  EXPECT_FALSE(loom_serve_maintenance_is_pending(&maintenance));
  IREE_ASSERT_OK(loom_serve_maintenance_advance(&maintenance, true, &progress));
  EXPECT_FALSE(progress);
  EXPECT_FALSE(loom_serve_maintenance_is_pending(&maintenance));
  loom_serve_maintenance_deinitialize(&maintenance);
}

TEST(MaintenanceTest, InvalidConfigurationAcceptsNoAsynchronousWork) {
  const loom_serve_device_idle_policy_t policies[] = {
      {.workspace_timeout = -1},
      {.parameter_timeout = -1},
      {.workspace_timeout = 1},
      {.parameter_timeout = 1},
  };
  for (const auto policy : policies) {
    loom_serve_maintenance_t maintenance;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_maintenance_initialize(nullptr, 1, &policy, &maintenance));
    EXPECT_FALSE(loom_serve_maintenance_is_pending(&maintenance));
    loom_serve_maintenance_deinitialize(&maintenance);
  }
}

TEST(MaintenanceTest, OneTimerUsesTheShortestEnabledDevicePolicy) {
  loom_serve_host_t* host = nullptr;
  IREE_ASSERT_OK(
      loom_serve_host_create(1, nullptr, iree_allocator_system(), &host));
  const loom_serve_device_options_t options[] = {
      {
          .driver = loom_serve_host_task_driver(host),
          .uri = IREE_SV("task"),
          .name = IREE_SV("first"),
      },
      {
          .driver = loom_serve_host_task_driver(host),
          .uri = IREE_SV("task"),
          .name = IREE_SV("second"),
      }};
  loom_serve_device_group_t* group = nullptr;
  IREE_ASSERT_OK(loom_serve_device_group_allocate(
      host, IREE_ARRAYSIZE(options), options, iree_allocator_system(), &group));
  loom_serve_device_idle_policy_t policies[2] = {};
  loom_serve_maintenance_t maintenance;
  IREE_ASSERT_OK(
      loom_serve_maintenance_initialize(group, 2, policies, &maintenance));
  EXPECT_EQ(maintenance.timer.interval, 0);
  EXPECT_EQ(maintenance.timer.proactor, nullptr);
  loom_serve_maintenance_deinitialize(&maintenance);

  policies[1].workspace_timeout = iree_make_duration_ms(2500);
  IREE_ASSERT_OK(
      loom_serve_maintenance_initialize(group, 2, policies, &maintenance));
  EXPECT_EQ(maintenance.timer.interval, iree_make_duration_ms(1000));
  EXPECT_EQ(maintenance.policies, policies);
  EXPECT_FALSE(loom_serve_maintenance_is_pending(&maintenance));
  loom_serve_maintenance_deinitialize(&maintenance);

  policies[0].parameter_timeout = iree_make_duration_ms(100);
  IREE_ASSERT_OK(
      loom_serve_maintenance_initialize(group, 2, policies, &maintenance));
  EXPECT_EQ(maintenance.timer.interval, iree_make_duration_ms(100));
  loom_serve_maintenance_deinitialize(&maintenance);

  for (const auto count : {1u, 3u}) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_maintenance_initialize(
                              group, count, policies, &maintenance));
    loom_serve_maintenance_deinitialize(&maintenance);
  }
  policies[1].workspace_timeout = -1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_maintenance_initialize(group, 2, policies, &maintenance));
  loom_serve_maintenance_deinitialize(&maintenance);
  IREE_EXPECT_OK(loom_serve_device_group_free(group));
  loom_serve_host_release(host);
}

}  // namespace
