// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

typedef struct iree_status_handle_t* iree_status_t;

iree_status_t iree_status_ignore(iree_status_t status);
iree_status_t iree_clang_tidy_status_assigned_source(void);
iree_status_t iree_clang_tidy_status_dropped_source(void);
iree_status_t iree_clang_tidy_status_ignored_source(void);
iree_status_t iree_clang_tidy_status_named_call_source(void);
iree_status_t iree_clang_tidy_status_returned_source(void);
iree_status_t iree_clang_tidy_status_void_cast_source(void);

iree_status_t iree_clang_tidy_status_returned(void) {
  return iree_clang_tidy_status_returned_source();
}

void iree_clang_tidy_status_assigned(void) {
  iree_status_t status = iree_clang_tidy_status_assigned_source();
  (void)status;
}

void iree_clang_tidy_status_ignored(void) {
  iree_status_ignore(iree_clang_tidy_status_ignored_source());
}

void iree_clang_tidy_status_dropped(void) {
  iree_clang_tidy_status_dropped_source();
}

void iree_clang_tidy_status_void_cast(void) {
  (void)iree_clang_tidy_status_void_cast_source();
}

void iree_clang_tidy_status_named_call_dropped(void) {
  iree_clang_tidy_status_named_call_source();
}
