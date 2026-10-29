// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "build_identity.h"

#include <string.h>

// Bazel supplies this value to stamped final links. Native and ordinary
// unstamped builds retain an explicit, deterministic development identity.
#ifndef BUILD_EMBED_LABEL
#define BUILD_EMBED_LABEL "redacted"
#endif  // BUILD_EMBED_LABEL

loom_serve_build_identity_t loom_serve_build_identity(void) {
  const char* label = BUILD_EMBED_LABEL;
  if (!label[0] || strcmp(label, "redacted") == 0) {
    return {"development", 11, "unknown", 7};
  }
  const char* separator = strchr(label, '@');
  if (!separator || !separator[1]) {
    return {label, strlen(label), "unknown", 7};
  }
  return {
      label,
      (size_t)(separator - label),
      separator + 1,
      strlen(separator + 1),
  };
}
