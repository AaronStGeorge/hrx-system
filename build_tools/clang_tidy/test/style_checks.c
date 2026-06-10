// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef NULL
#define NULL ((void*)0)
#endif  // NULL

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

typedef struct iree_clang_tidy_style_resource_t
    iree_clang_tidy_style_resource_t;
typedef struct iree_clang_tidy_style_pool_t iree_clang_tidy_style_pool_t;
typedef unsigned long iree_clang_tidy_style_handle_t;
typedef int iree_atomic_ref_count_t;
typedef int iree_status_t;

typedef struct iree_clang_tidy_style_refcounted_t {
  iree_atomic_ref_count_t ref_count;
} iree_clang_tidy_style_refcounted_t;

void iree_clang_tidy_style_resource_release(
    iree_clang_tidy_style_resource_t* resource);
void iree_clang_tidy_style_resource_deinitialize(
    iree_clang_tidy_style_resource_t* resource);
void iree_clang_tidy_style_resource_destroy(
    iree_clang_tidy_style_resource_t* resource);
void iree_clang_tidy_style_pool_release(iree_clang_tidy_style_pool_t* pool,
                                        void* entry);
void iree_clang_tidy_style_handle_release(
    iree_clang_tidy_style_handle_t handle);
void iree_clang_tidy_style_extra_cleanup(void);

typedef void (*iree_clang_tidy_style_release_fn_t)(
    iree_clang_tidy_style_resource_t* resource);

void iree_atomic_ref_count_inc(iree_atomic_ref_count_t* ref_count);
int iree_atomic_ref_count_dec(iree_atomic_ref_count_t* ref_count);
iree_status_t iree_ok_status(void);

iree_status_t iree_clang_tidy_style_refcount_status_retain(
    iree_clang_tidy_style_refcounted_t* resource) {
  iree_atomic_ref_count_inc(&resource->ref_count);
  return iree_ok_status();
}

iree_status_t iree_clang_tidy_style_refcount_status_release(
    iree_clang_tidy_style_refcounted_t* resource) {
  if (iree_atomic_ref_count_dec(&resource->ref_count) == 1) {
    return iree_ok_status();
  }
  return iree_ok_status();
}

void iree_clang_tidy_style_refcount_void_retain(
    iree_clang_tidy_style_refcounted_t* resource) {
  iree_atomic_ref_count_inc(&resource->ref_count);
}

void iree_clang_tidy_style_refcount_void_release(
    iree_clang_tidy_style_refcounted_t* resource) {
  (void)iree_atomic_ref_count_dec(&resource->ref_count);
}

iree_status_t iree_clang_tidy_style_refcount_lookup_retain(
    iree_clang_tidy_style_refcounted_t* resource) {
  (void)resource;
  return iree_ok_status();
}

iree_status_t iree_clang_tidy_style_virtual_memory_release(void* memory) {
  (void)memory;
  return iree_ok_status();
}

void iree_clang_tidy_style_guarded_release(
    iree_clang_tidy_style_resource_t* resource) {
  if (resource) iree_clang_tidy_style_resource_release(resource);
}

void iree_clang_tidy_style_guarded_release_with_clear(
    iree_clang_tidy_style_resource_t* resource) {
  if (resource != NULL) {
    iree_clang_tidy_style_resource_release(resource);
    resource = NULL;
  }
}

void iree_clang_tidy_style_null_guard_ignored(
    iree_clang_tidy_style_resource_t* resource) {
  if (resource == NULL) {
    iree_clang_tidy_style_resource_release(resource);
  }
}

void iree_clang_tidy_style_different_predicate_ignored(
    int owns_resource, iree_clang_tidy_style_resource_t* resource) {
  if (owns_resource) {
    iree_clang_tidy_style_resource_release(resource);
  }
}

void iree_clang_tidy_style_extra_behavior_ignored(
    iree_clang_tidy_style_resource_t* resource) {
  if (resource) {
    iree_clang_tidy_style_resource_release(resource);
    iree_clang_tidy_style_extra_cleanup();
  }
}

void iree_clang_tidy_style_deinitialize_ignored(
    iree_clang_tidy_style_resource_t* resource) {
  if (resource) {
    iree_clang_tidy_style_resource_deinitialize(resource);
  }
}

void iree_clang_tidy_style_destroy_ignored(
    iree_clang_tidy_style_resource_t* resource) {
  if (resource) {
    iree_clang_tidy_style_resource_destroy(resource);
  }
}

void iree_clang_tidy_style_multi_argument_release_ignored(
    iree_clang_tidy_style_pool_t* pool, void* entry) {
  if (pool) {
    iree_clang_tidy_style_pool_release(pool, entry);
  }
}

void iree_clang_tidy_style_non_pointer_release_ignored(
    iree_clang_tidy_style_handle_t handle) {
  if (handle) {
    iree_clang_tidy_style_handle_release(handle);
  }
}

void iree_clang_tidy_style_indirect_release_ignored(
    iree_clang_tidy_style_resource_t* resource,
    iree_clang_tidy_style_release_fn_t release_fn) {
  if (resource) {
    release_fn(resource);
  }
}
