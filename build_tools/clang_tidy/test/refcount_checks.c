// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef NULL
#define NULL ((void*)0)
#endif  // NULL

#ifndef IREE_LIKELY
#define IREE_LIKELY(x) (x)
#endif  // IREE_LIKELY

#ifndef IREE_ASSERT_ARGUMENT
#define IREE_ASSERT_ARGUMENT(x) ((void)(x))
#endif  // IREE_ASSERT_ARGUMENT

typedef int iree_atomic_ref_count_t;
typedef int iree_status_t;

typedef struct iree_clang_tidy_refcounted_t {
  iree_atomic_ref_count_t ref_count;
} iree_clang_tidy_refcounted_t;

typedef struct iree_clang_tidy_refcount_misused_counter_t {
  iree_atomic_ref_count_t pending_submissions;
} iree_clang_tidy_refcount_misused_counter_t;

typedef struct iree_clang_tidy_refcounted_with_counter_t {
  iree_atomic_ref_count_t ref_count;
  iree_atomic_ref_count_t queued_callbacks;
} iree_clang_tidy_refcounted_with_counter_t;

typedef struct iree_clang_tidy_not_refcount_anchored_t {
  int state;
  iree_atomic_ref_count_t side_counter;
} iree_clang_tidy_not_refcount_anchored_t;

void iree_atomic_ref_count_inc(iree_atomic_ref_count_t* ref_count);
int iree_atomic_ref_count_dec(iree_atomic_ref_count_t* ref_count);
iree_status_t iree_ok_status(void);

iree_status_t iree_clang_tidy_refcount_status_retain(
    iree_clang_tidy_refcounted_t* resource) {
  iree_atomic_ref_count_inc(&resource->ref_count);
  return iree_ok_status();
}

iree_status_t iree_clang_tidy_refcount_status_release(
    iree_clang_tidy_refcounted_t* resource) {
  if (iree_atomic_ref_count_dec(&resource->ref_count) == 1) {
    return iree_ok_status();
  }
  return iree_ok_status();
}

void iree_clang_tidy_refcount_void_retain(
    iree_clang_tidy_refcounted_t* resource) {
  iree_atomic_ref_count_inc(&resource->ref_count);
}

void iree_clang_tidy_refcount_void_release(
    iree_clang_tidy_refcounted_t* resource) {
  if (!resource) {
    return;
  }
  if (iree_atomic_ref_count_dec(&resource->ref_count) == 1) {
  }
}

void iree_clang_tidy_refcount_unguarded_release(
    iree_clang_tidy_refcounted_t* resource) {
  if (iree_atomic_ref_count_dec(&resource->ref_count) == 1) {
  }
}

void iree_clang_tidy_refcount_asserting_release(
    iree_clang_tidy_refcounted_t* resource) {
  IREE_ASSERT_ARGUMENT(resource);
  if (resource && iree_atomic_ref_count_dec(&resource->ref_count) == 1) {
  }
}

void iree_clang_tidy_refcount_ignored_dec(
    iree_clang_tidy_refcounted_t* resource) {
  (void)iree_atomic_ref_count_dec(&resource->ref_count);
}

void iree_clang_tidy_refcount_early_null_release(
    iree_clang_tidy_refcounted_t* resource) {
  if (!resource) {
    return;
  }
  if (iree_atomic_ref_count_dec(&resource->ref_count) == 1) {
  }
}

void iree_clang_tidy_refcount_inline_null_release(
    iree_clang_tidy_refcounted_t* resource) {
  if (resource && iree_atomic_ref_count_dec(&resource->ref_count) == 1) {
  }
}

void iree_clang_tidy_refcount_likely_null_release(
    iree_clang_tidy_refcounted_t* resource) {
  if (IREE_LIKELY(resource) &&
      iree_atomic_ref_count_dec(&resource->ref_count) == 1) {
  }
}

void iree_clang_tidy_refcount_local_counter_release(
    iree_clang_tidy_refcounted_t* resource) {
  iree_atomic_ref_count_t* ref_count = &resource->ref_count;
  if (iree_atomic_ref_count_dec(ref_count) == 1) {
  }
}

iree_status_t iree_clang_tidy_refcount_lookup_retain(
    iree_clang_tidy_refcounted_t* resource) {
  (void)resource;
  return iree_ok_status();
}

iree_status_t iree_clang_tidy_virtual_memory_release(void* memory) {
  (void)memory;
  return iree_ok_status();
}
