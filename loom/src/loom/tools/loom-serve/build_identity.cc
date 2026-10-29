// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "build_identity.h"

#include <string.h>

// Bazel supplies these values to stamped final links. Native and ordinary
// unstamped builds retain an explicit, deterministic development identity.
#ifndef BUILD_EMBED_LABEL
#define BUILD_EMBED_LABEL "redacted"
#endif  // BUILD_EMBED_LABEL
#ifndef BUILD_SCM_REVISION
#define BUILD_SCM_REVISION "0"
#endif  // BUILD_SCM_REVISION

loom_serve_build_identity_t loom_serve_build_identity(void) {
  const char* version = BUILD_EMBED_LABEL;
  if (!version[0] || strcmp(version, "redacted") == 0) {
    version = "development";
  }
  const char* commit = BUILD_SCM_REVISION;
  if (!commit[0] || strcmp(commit, "0") == 0 ||
      strcmp(commit, "redacted") == 0) {
    commit = "unknown";
  }
  return {version, commit};
}
