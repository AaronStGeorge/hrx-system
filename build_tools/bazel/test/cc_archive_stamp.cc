// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Linkstamps are C++ compilations, even when the archive contains C objects.
extern "C" int archive_stamp_value(void) { return 22; }
