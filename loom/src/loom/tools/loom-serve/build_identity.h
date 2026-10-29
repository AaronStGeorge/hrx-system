// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TOOLS_LOOM_SERVE_BUILD_IDENTITY_H_
#define LOOM_TOOLS_LOOM_SERVE_BUILD_IDENTITY_H_

#ifdef __cplusplus
extern "C" {
#endif

// Stable product identity embedded by the executable's final link.
typedef struct loom_serve_build_identity_t {
  // Loom package version, or "development" for an unstamped build.
  const char* version;
  // Source commit hash, or "unknown" for an unstamped build.
  const char* commit;
} loom_serve_build_identity_t;

// Returns the identity embedded by the final executable link.
loom_serve_build_identity_t loom_serve_build_identity(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_SERVE_BUILD_IDENTITY_H_
