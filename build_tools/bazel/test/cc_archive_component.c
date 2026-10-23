// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "build_tools/bazel/test/cc_archive_fixture.h"

int archive_component_value(void) {
  return archive_native_value() + archive_stamp_value();
}
