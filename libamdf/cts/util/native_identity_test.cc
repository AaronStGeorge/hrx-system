// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/util/native_identity.h"

#include "gtest/gtest.h"

namespace amdf::cts {
namespace {

TEST(NativeIdentityTest, ParsesLinuxDevice) {
  amdf_endpoint_native_identity_t identity = {};
  ASSERT_TRUE(ParseNativeIdentity("linux_device:226:128", &identity));
  EXPECT_EQ(identity.type, AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE);
  EXPECT_EQ(identity.value.linux_device.major, 226u);
  EXPECT_EQ(identity.value.linux_device.minor, 128u);
}

TEST(NativeIdentityTest, ParsesWindowsAdapter) {
  amdf_endpoint_native_identity_t identity = {};
  ASSERT_TRUE(
      ParseNativeIdentity("windows_adapter:abcdef0123456789:3", &identity));
  EXPECT_EQ(identity.type, AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_WINDOWS_ADAPTER);
  EXPECT_EQ(identity.value.windows_adapter.luid, UINT64_C(0xabcdef0123456789));
  EXPECT_EQ(identity.value.windows_adapter.physical_adapter_index, 3u);
}

TEST(NativeIdentityTest, RejectsMalformedIdentity) {
  amdf_endpoint_native_identity_t identity = {};
  EXPECT_FALSE(ParseNativeIdentity("", &identity));
  EXPECT_FALSE(ParseNativeIdentity("linux_device:226", &identity));
  EXPECT_FALSE(ParseNativeIdentity("linux_device:226:128x", &identity));
  EXPECT_FALSE(ParseNativeIdentity("windows_adapter:0x1234:0", &identity));
  EXPECT_FALSE(ParseNativeIdentity("windows_adapter:1234:-1", &identity));
}

TEST(NativeIdentityTest, ComparesActiveNamespace) {
  amdf_endpoint_native_identity_t linux_a = {};
  amdf_endpoint_native_identity_t linux_b = {};
  amdf_endpoint_native_identity_t windows = {};
  ASSERT_TRUE(ParseNativeIdentity("linux_device:226:128", &linux_a));
  ASSERT_TRUE(ParseNativeIdentity("linux_device:226:128", &linux_b));
  ASSERT_TRUE(ParseNativeIdentity("windows_adapter:e2:0", &windows));
  EXPECT_TRUE(NativeIdentityIsEqual(linux_a, linux_b));
  ++linux_b.value.linux_device.minor;
  EXPECT_FALSE(NativeIdentityIsEqual(linux_a, linux_b));
  EXPECT_FALSE(NativeIdentityIsEqual(linux_a, windows));
  EXPECT_FALSE(NativeIdentityIsEqual({}, {}));
}

}  // namespace
}  // namespace amdf::cts
