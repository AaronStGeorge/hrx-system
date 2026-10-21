// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_UTIL_NATIVE_IDENTITY_H_
#define AMDF_CTS_UTIL_NATIVE_IDENTITY_H_

#include <string_view>

#include "amdf/base.h"

namespace amdf::cts {

// Parses the native identity syntax recorded in CTS results, independently of
// endpoint-ID encodings and foreign runtimes' device ordinals.
bool ParseNativeIdentity(std::string_view text,
                         amdf_endpoint_native_identity_t* out_identity);

// Compares identities in their active native namespace.
bool NativeIdentityIsEqual(const amdf_endpoint_native_identity_t& lhs,
                           const amdf_endpoint_native_identity_t& rhs);

}  // namespace amdf::cts

#endif  // AMDF_CTS_UTIL_NATIVE_IDENTITY_H_
