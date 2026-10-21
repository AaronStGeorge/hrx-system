// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/util/native_identity.h"

#include <charconv>

namespace amdf::cts {
namespace {

template <typename First, typename Second>
bool ParseIdentityPair(std::string_view text, int first_base, First* first,
                       Second* second) {
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto first_result = std::from_chars(begin, end, *first, first_base);
  if (first_result.ec != std::errc{} || first_result.ptr == end ||
      *first_result.ptr != ':') {
    return false;
  }
  const auto second_result =
      std::from_chars(first_result.ptr + 1, end, *second);
  return second_result.ec == std::errc{} && second_result.ptr == end;
}

}  // namespace

bool ParseNativeIdentity(std::string_view text,
                         amdf_endpoint_native_identity_t* out_identity) {
  amdf_endpoint_native_identity_t identity = {};
  if (text.starts_with("linux_device:")) {
    identity.type = AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE;
    if (!ParseIdentityPair(text.substr(13), 10,
                           &identity.value.linux_device.major,
                           &identity.value.linux_device.minor)) {
      return false;
    }
  } else if (text.starts_with("windows_adapter:")) {
    identity.type = AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_WINDOWS_ADAPTER;
    if (!ParseIdentityPair(
            text.substr(16), 16, &identity.value.windows_adapter.luid,
            &identity.value.windows_adapter.physical_adapter_index)) {
      return false;
    }
  } else {
    return false;
  }
  *out_identity = identity;
  return true;
}

bool NativeIdentityIsEqual(const amdf_endpoint_native_identity_t& lhs,
                           const amdf_endpoint_native_identity_t& rhs) {
  if (lhs.type != rhs.type) {
    return false;
  }
  switch (lhs.type) {
    case AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE:
      return lhs.value.linux_device.major == rhs.value.linux_device.major &&
             lhs.value.linux_device.minor == rhs.value.linux_device.minor;
    case AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_WINDOWS_ADAPTER:
      return lhs.value.windows_adapter.luid == rhs.value.windows_adapter.luid &&
             lhs.value.windows_adapter.physical_adapter_index ==
                 rhs.value.windows_adapter.physical_adapter_index;
    default:
      return false;
  }
}

}  // namespace amdf::cts
