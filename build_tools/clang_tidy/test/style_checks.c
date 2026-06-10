// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

int iree_clang_tidy_style_direct_goto(int flag) {
  if (flag) goto cleanup;
  return 0;
cleanup:
  return 1;
}

#define IREE_CLANG_TIDY_STYLE_LOCAL_GOTO_MACRO goto macro_cleanup
int iree_clang_tidy_style_local_goto_macro(int flag) {
  if (flag) {
    IREE_CLANG_TIDY_STYLE_LOCAL_GOTO_MACRO;
  }
  return 0;
macro_cleanup:
  return 1;
}

#if defined(__GNUC__)
int iree_clang_tidy_style_computed_goto(int index) {
  static void* labels[] = {&&dispatch_zero, &&dispatch_one};
  goto* labels[index & 1];
dispatch_zero:
  return 0;
dispatch_one:
  return 1;
}
#endif  // defined(__GNUC__)
