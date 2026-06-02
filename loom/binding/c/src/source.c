// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "source.h"

#include <string.h>

#include "iree/base/internal/atomics.h"

struct loomc_source_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used to release this object and copied storage.
  loomc_allocator_t allocator;
  // Source format.
  loomc_source_format_t format;
  // Copied diagnostic/cache identifier.
  loomc_string_view_t identifier;
  // Source bytes.
  loomc_byte_span_t contents;
  // Storage policy for contents.
  loomc_source_storage_t storage;
  // Callback used to release externally owned contents.
  loomc_source_release_fn_t release;
  // User data passed to release.
  void* release_user_data;
};

loomc_status_t loomc_source_copy_string(loomc_string_view_t source,
                                        loomc_allocator_t allocator,
                                        loomc_string_view_t* out_string) {
  if (out_string == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_string must not be NULL");
  }
  *out_string = loomc_string_view_empty();
  if (loomc_string_view_is_empty(source)) {
    return loomc_ok_status();
  }
  char* target = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc_uninitialized(
      allocator, source.size, (void**)&target));
  memcpy(target, source.data, source.size);
  *out_string = loomc_make_string_view(target, source.size);
  return loomc_ok_status();
}

static loomc_status_t loomc_source_validate_options(
    const loomc_source_options_t* options) {
  if (options == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source options must not be NULL");
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(LOOMC_STATUS_UNIMPLEMENTED,
                             "source option extensions are not supported");
  }
  if (options->format > LOOMC_SOURCE_FORMAT_BYTECODE) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source format is invalid");
  }
  if (options->storage > LOOMC_SOURCE_STORAGE_EXTERNAL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source storage policy is invalid");
  }
  if (options->contents.data == NULL && options->contents.data_length != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source contents have length but no data");
  }
  if (options->storage == LOOMC_SOURCE_STORAGE_EXTERNAL &&
      options->release == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "external source storage requires a release");
  }
  return loomc_ok_status();
}

static void loomc_source_destroy(loomc_source_t* source) {
  loomc_allocator_t allocator = source->allocator;
  loomc_allocator_free(allocator, (void*)source->identifier.data);
  if (source->storage == LOOMC_SOURCE_STORAGE_COPY) {
    loomc_allocator_free(allocator, (void*)source->contents.data);
  } else if (source->storage == LOOMC_SOURCE_STORAGE_EXTERNAL) {
    source->release(source->release_user_data, source->contents);
  }
  loomc_allocator_free(allocator, source);
}

loomc_status_t loomc_source_create(const loomc_source_options_t* options,
                                   loomc_allocator_t allocator,
                                   loomc_source_t** out_source) {
  if (out_source == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_source must not be NULL");
  }
  *out_source = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_source_validate_options(options));

  allocator = loomc_allocator_or_system(allocator);
  loomc_source_t* source = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*source), (void**)&source));
  iree_atomic_ref_count_init(&source->ref_count);
  source->allocator = allocator;
  source->format = options->format;
  source->storage = LOOMC_SOURCE_STORAGE_BORROWED;
  source->release = options->release;
  source->release_user_data = options->release_user_data;

  loomc_status_t status = loomc_source_copy_string(
      options->identifier, allocator, &source->identifier);
  if (loomc_status_is_ok(status)) {
    if (options->storage == LOOMC_SOURCE_STORAGE_COPY &&
        options->contents.data_length != 0) {
      uint8_t* contents = NULL;
      status = loomc_allocator_malloc_uninitialized(
          allocator, options->contents.data_length, (void**)&contents);
      if (loomc_status_is_ok(status)) {
        memcpy(contents, options->contents.data, options->contents.data_length);
        source->contents =
            loomc_make_byte_span(contents, options->contents.data_length);
        source->storage = LOOMC_SOURCE_STORAGE_COPY;
      }
    } else {
      source->contents = options->contents;
      source->storage = options->storage;
    }
  }

  if (loomc_status_is_ok(status)) {
    *out_source = source;
  } else {
    loomc_source_destroy(source);
  }
  return status;
}

loomc_status_t loomc_source_create_take_contents(loomc_source_format_t format,
                                                 loomc_string_view_t identifier,
                                                 loomc_byte_span_t contents,
                                                 loomc_allocator_t allocator,
                                                 loomc_source_t** out_source) {
  if (out_source == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_source must not be NULL");
  }
  if (contents.data == NULL && contents.data_length != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source contents have length but no data");
  }
  *out_source = NULL;
  allocator = loomc_allocator_or_system(allocator);

  loomc_source_t* source = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*source), (void**)&source));
  iree_atomic_ref_count_init(&source->ref_count);
  source->allocator = allocator;
  source->format = format;
  source->contents = loomc_byte_span_empty();
  source->storage = LOOMC_SOURCE_STORAGE_BORROWED;
  source->release = NULL;
  source->release_user_data = NULL;

  loomc_status_t status =
      loomc_source_copy_string(identifier, allocator, &source->identifier);
  if (loomc_status_is_ok(status)) {
    source->contents = contents;
    source->storage = LOOMC_SOURCE_STORAGE_COPY;
    *out_source = source;
  } else {
    loomc_source_destroy(source);
  }
  return status;
}

loomc_status_t loomc_source_take_contents(loomc_source_t* source,
                                          loomc_byte_span_t* out_contents) {
  if (out_contents == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_contents must not be NULL");
  }
  *out_contents = loomc_byte_span_empty();
  if (source == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source must not be NULL");
  }
  if (source->storage != LOOMC_SOURCE_STORAGE_COPY) {
    return loomc_make_status(
        LOOMC_STATUS_FAILED_PRECONDITION,
        "source contents are not allocator-owned and cannot be transferred");
  }

  *out_contents = source->contents;
  source->contents = loomc_byte_span_empty();
  source->storage = LOOMC_SOURCE_STORAGE_BORROWED;
  return loomc_ok_status();
}

void loomc_source_retain(loomc_source_t* source) {
  if (source == NULL) {
    return;
  }
  iree_atomic_ref_count_inc(&source->ref_count);
}

void loomc_source_release(loomc_source_t* source) {
  if (source == NULL) {
    return;
  }
  if (iree_atomic_ref_count_dec(&source->ref_count) == 1) {
    loomc_source_destroy(source);
  }
}

loomc_source_format_t loomc_source_format(const loomc_source_t* source) {
  return source ? source->format : LOOMC_SOURCE_FORMAT_UNKNOWN;
}

loomc_string_view_t loomc_source_identifier(const loomc_source_t* source) {
  return source ? source->identifier : loomc_string_view_empty();
}

loomc_byte_span_t loomc_source_contents(const loomc_source_t* source) {
  return source ? source->contents : loomc_byte_span_empty();
}
