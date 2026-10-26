// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_REPORTING_ROW_LIST_H_
#define LOOM_TARGET_REPORTING_ROW_LIST_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Linked storage block for homogeneous compile report detail rows.
// Payloads immediately follow this header; their type is determined by the
// report field containing the list.
typedef struct loom_target_compile_report_vec_t {
  // Next row block in allocation order, or NULL for the final block.
  struct loom_target_compile_report_vec_t* next;
  // Number of rows populated in this block.
  iree_host_size_t count;
  // Maximum number of rows that fit in this block.
  iree_host_size_t capacity;
} loom_target_compile_report_vec_t;

// Owned linked list of homogeneous compile report detail rows.
typedef struct loom_target_compile_report_row_list_t {
  // First row storage block, or NULL when empty.
  loom_target_compile_report_vec_t* head;
  // Last row storage block, or NULL when empty.
  loom_target_compile_report_vec_t* tail;
  // Total number of rows stored across all blocks.
  iree_host_size_t count;
} loom_target_compile_report_row_list_t;

// Returns mutable row storage for |vec|.
static inline void* loom_target_compile_report_vec_rows(
    loom_target_compile_report_vec_t* vec) {
  return (void*)(vec + 1);
}

// Returns immutable row storage for |vec|.
static inline const void* loom_target_compile_report_vec_const_rows(
    const loom_target_compile_report_vec_t* vec) {
  return (const void*)(vec + 1);
}

// Releases every storage block owned by |list| and resets it to empty.
void loom_target_compile_report_row_list_deinitialize(
    iree_allocator_t allocator, loom_target_compile_report_row_list_t* list);

// Appends one |row_size|-byte row to |list|.
iree_status_t loom_target_compile_report_row_list_append(
    loom_target_compile_report_row_list_t* list, iree_host_size_t row_size,
    iree_allocator_t allocator, const void* row);

// Appends every row in |source| to |target|.
iree_status_t loom_target_compile_report_row_list_append_all(
    loom_target_compile_report_row_list_t* target,
    const loom_target_compile_report_row_list_t* source,
    iree_host_size_t row_size, iree_allocator_t allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TARGET_REPORTING_ROW_LIST_H_
