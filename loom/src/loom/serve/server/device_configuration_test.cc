// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/device_configuration.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class DeviceConfigurationTest : public ::testing::Test {
 protected:
  void TearDown() override {
    loom_serve_device_configuration_deinitialize(&configuration_);
  }
  iree_status_t Parse(const std::string& json) {
    loom_serve_device_configuration_deinitialize(&configuration_);
    return loom_serve_device_configuration_initialize(
        iree_make_string_view(json.data(), json.size()),
        iree_allocator_system(), &configuration_);
  }
  static std::string String(iree_string_view_t value) {
    return std::string(value.data, value.size);
  }
  // Owned decoded configuration, independent of the input document lifetime.
  loom_serve_device_configuration_t configuration_ = {};
};

TEST_F(DeviceConfigurationTest,
       OwnsNamedIndependentPoliciesWithoutOpeningDevices) {
  std::string json = R"({"devices":{
    "g\u0070u":{"uri":"amdgpu://0?queue_count=4","memory_bytes":8589934592,
      "weight_slab_bytes":2097152,"state_slab_bytes":65536,
      "workspace_idle_ms":2000,"parameter_idle_ms":60000},
    "npu":{"uri":"xdna://0","pool_backing":"fixed"}
  }})";
  IREE_ASSERT_OK(Parse(json));
  json.assign(json.size(), 'x');
  ASSERT_EQ(configuration_.count, 2u);
  const auto& gpu = configuration_.options[0];
  const auto& npu = configuration_.options[1];
  EXPECT_EQ(String(gpu.name), "gpu");
  EXPECT_EQ(String(gpu.uri), "amdgpu://0?queue_count=4");
  EXPECT_EQ(gpu.driver, nullptr);
  EXPECT_EQ(gpu.backing, LOOM_SERVE_DEVICE_BACKING_ELASTIC);
  EXPECT_EQ(gpu.memory_limit, UINT64_C(8589934592));
  EXPECT_EQ(gpu.weight_slab_size, 2097152u);
  EXPECT_EQ(gpu.state_slab_size, 65536u);
  EXPECT_EQ(configuration_.idle[0].workspace_timeout,
            iree_make_duration_ms(2000));
  EXPECT_EQ(configuration_.idle[0].parameter_timeout,
            iree_make_duration_ms(60000));
  EXPECT_EQ(String(npu.name), "npu");
  EXPECT_EQ(String(npu.uri), "xdna://0");
  EXPECT_EQ(npu.driver, nullptr);
  EXPECT_EQ(npu.backing, LOOM_SERVE_DEVICE_BACKING_FIXED);
  EXPECT_EQ(npu.memory_limit, 0u);
  EXPECT_EQ(configuration_.idle[1].workspace_timeout, 0);
  EXPECT_EQ(configuration_.idle[1].parameter_timeout, 0);
}

TEST_F(DeviceConfigurationTest, UriConvenienceUsesTheSameDefaults) {
  IREE_ASSERT_OK(Parse(R"({"devices":{"device":{"uri":"task"}}})"));
  loom_serve_device_configuration_t convenience = {};
  IREE_ASSERT_OK(loom_serve_device_configuration_initialize_uri(
      IREE_SV("task"), iree_allocator_system(), &convenience));
  EXPECT_EQ(convenience.count, configuration_.count);
  const auto& actual = convenience.options[0];
  const auto& expected = configuration_.options[0];
  EXPECT_EQ(String(actual.name), String(expected.name));
  EXPECT_EQ(String(actual.uri), String(expected.uri));
  EXPECT_EQ(actual.backing, expected.backing);
  EXPECT_EQ(actual.weight_slab_size, UINT64_C(1) << 30);
  EXPECT_EQ(actual.state_slab_size, UINT64_C(64) << 20);
  EXPECT_EQ(actual.memory_limit, 0u);
  loom_serve_device_configuration_deinitialize(&convenience);
}

TEST_F(DeviceConfigurationTest,
       RejectsMalformedDeclarationsWithoutPartialOwners) {
  const char* invalid[] = {
      "",
      "[]",
      "{}",
      R"({"devices":{}})",
      R"({"devices":[]})",
      R"({"devices":{},"devices":{}})",
      R"({"devices":{},"unknown":0})",
      R"({"devices":{"gpu":{}}})",
      R"({"devices":{"gpu":[]}})",
      R"({"devices":{"":{"uri":"task"}}})",
      R"({"devices":{"bad key":{"uri":"task"}}})",
      R"({"devices":{"gpu":{"uri":"task"},"g\u0070u":{"uri":"task"}}})",
      R"({"devices":{"gpu":{"uri":""}}})",
      R"({"devices":{"gpu":{"uri":0}}})",
      R"({"devices":{"gpu":{"uri":"task\u0000"}}})",
      R"({"devices":{"gpu":{"uri":"task","uri":"xdna"}}})",
      R"({"devices":{"gpu":{"uri":"task","memory_bytes":"4096"}}})",
      R"({"devices":{"gpu":{"uri":"task","memory_bytes":-1}}})",
      R"({"devices":{"gpu":{"uri":"task","memory_bytes":1.5}}})",
      R"({"devices":{"gpu":{"uri":"task","memory_bytes":1e3}}})",
      R"({"devices":{"gpu":{"uri":"task","memory_bytes":1,"pool_backing":"fixed"}}})",
      R"({"devices":{"gpu":{"uri":"task","pool_backing":"automatic"}}})",
      R"({"devices":{"gpu":{"uri":"task","parameter_idle_ms":false}}})",
      R"({"devices":{"gpu":{"uri":"task","workspace_idle_ms":-1}}})",
      R"({"devices":{"gpu":{"uri":"task","unknown":1}}})",
      R"({"devices":{"gpu":{"uri":"task"}}} trailing)",
  };
  for (const char* json : invalid) {
    SCOPED_TRACE(json);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(json));
    EXPECT_EQ(configuration_.count, 0u);
    EXPECT_EQ(configuration_.strings, nullptr);
  }
}

TEST_F(DeviceConfigurationTest, RejectsOverflowAndAcceptsExplicitZero) {
  for (const char* fields : {R"("memory_bytes":18446744073709551616)",
                             R"("workspace_idle_ms":9223372036855)",
                             R"("parameter_idle_ms":9223372036855)"}) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        Parse(std::string(R"({"devices":{"gpu":{"uri":"task",)") + fields +
              "}}}"));
    EXPECT_EQ(configuration_.count, 0u);
    EXPECT_EQ(configuration_.strings, nullptr);
  }
  IREE_ASSERT_OK(Parse(R"({"devices":{"gpu":{"uri":"task",
    "weight_slab_bytes":0,"state_slab_bytes":0,"memory_bytes":0,
    "parameter_idle_ms":0,"workspace_idle_ms":0}}})"));
  EXPECT_EQ(configuration_.options[0].weight_slab_size, 0u);
  EXPECT_EQ(configuration_.options[0].state_slab_size, 0u);
}

}  // namespace
