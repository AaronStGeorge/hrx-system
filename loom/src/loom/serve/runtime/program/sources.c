// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/sources.h"

#include <string.h>

#include "iree/base/internal/path.h"
#include "loomc/iree.h"

static iree_string_view_t loom_serve_sources_copy_string(
    iree_string_view_t value, char** cursor) {
  if (!value.size) {
    return iree_string_view_empty();
  }
  memcpy(*cursor, value.data, value.size);
  iree_string_view_t copy = iree_make_string_view(*cursor, value.size);
  *cursor += value.size;
  return copy;
}

iree_status_t loom_serve_sources_clone(const loom_serve_sources_t* sources,
                                       iree_allocator_t host_allocator,
                                       loom_serve_sources_t** out_sources) {
  *out_sources = NULL;
  iree_host_size_t size = 0;
  bool representable = iree_host_size_checked_mul_add(
      sizeof(*sources), sources->count, sizeof(*sources->values), &size);
  const iree_string_view_t paths[] = {sources->base_directory, sources->prepare,
                                      sources->control};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(paths) && representable;
       ++i) {
    representable = iree_host_size_checked_add(size, paths[i].size, &size);
  }
  for (iree_host_size_t i = 0; i < sources->count && representable; ++i) {
    representable =
        iree_host_size_checked_add(size, sources->values[i].size, &size);
  }
  if (!representable) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "source namespace storage is not representable");
  }
  loom_serve_sources_t* copy = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, size, (void**)&copy));
  iree_string_view_t* values = (iree_string_view_t*)(copy + 1);
  char* cursor = (char*)(values + sources->count);
  copy->base_directory =
      loom_serve_sources_copy_string(sources->base_directory, &cursor);
  copy->prepare = loom_serve_sources_copy_string(sources->prepare, &cursor);
  copy->control = loom_serve_sources_copy_string(sources->control, &cursor);
  copy->count = sources->count;
  copy->values = sources->count ? values : NULL;
  for (iree_host_size_t i = 0; i < sources->count; ++i) {
    values[i] = loom_serve_sources_copy_string(sources->values[i], &cursor);
  }
  copy->package = sources->package;
  loom_serve_package_retain(copy->package);
  *out_sources = copy;
  return iree_ok_status();
}

void loom_serve_sources_free(loom_serve_sources_t* sources,
                             iree_allocator_t host_allocator) {
  if (sources) {
    loom_serve_package_release(sources->package);
    iree_allocator_free(host_allocator, sources);
  }
}

iree_status_t loom_serve_sources_open(const loom_serve_sources_t* sources,
                                      iree_string_view_t locator,
                                      iree_allocator_t host_allocator,
                                      loomc_source_t** out_source) {
  *out_source = NULL;
  if (sources->package && !loom_serve_path_is_absolute(locator)) {
    return loom_serve_package_open(sources->package, locator, out_source);
  }
  char* path = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_path_resolve(sources->base_directory, locator,
                                               host_allocator, &path));
  iree_status_t status = iree_status_from_loomc(loomc_source_create_from_path(
      loomc_make_cstring_view(path), NULL,
      loomc_allocator_from_iree(host_allocator), out_source));
  iree_allocator_free(host_allocator, path);
  return status;
}

bool loom_serve_path_is_absolute(iree_string_view_t path) {
#if defined(IREE_PLATFORM_WINDOWS)
  return (path.size >= 2 && ((path.data[0] == '\\' && path.data[1] == '\\') ||
                             (path.data[0] == '/' && path.data[1] == '/'))) ||
         (path.size >= 3 && path.data[1] == ':' &&
          ((path.data[0] >= 'A' && path.data[0] <= 'Z') ||
           (path.data[0] >= 'a' && path.data[0] <= 'z')) &&
          (path.data[2] == '/' || path.data[2] == '\\'));
#else
  return path.size && path.data[0] == '/';
#endif
}

iree_status_t loom_serve_path_resolve(iree_string_view_t base,
                                      iree_string_view_t path,
                                      iree_allocator_t host_allocator,
                                      char** out_path) {
  *out_path = NULL;
  if (loom_serve_path_is_absolute(path)) {
    return iree_file_path_join(iree_string_view_empty(), path, host_allocator,
                               out_path);
  }
#if defined(IREE_PLATFORM_WINDOWS)
  // A root-relative or drive-relative name depends on process drive state,
  // not the model document. It is neither a portable relative locator nor a
  // fully qualified local path.
  if ((path.size && (path.data[0] == '/' || path.data[0] == '\\')) ||
      (path.size >= 2 && path.data[1] == ':')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model paths must be relative or fully qualified");
  }
#endif
  if (iree_string_view_is_empty(base)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "relative model path '%.*s' requires a base directory", (int)path.size,
        path.data);
  }
  return iree_file_path_join(base, path, host_allocator, out_path);
}
