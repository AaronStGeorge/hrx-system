// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_BASE_H_
#define LOOMC_BASE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/// @file
/// Base types, descriptors, spans, and host allocation helpers.
///
/// This header contains the small C ABI building blocks used by all other Loom
/// C API headers. The types here avoid exposing implementation dependencies and
/// are designed to be straightforward to mirror from foreign-function bindings.

#ifdef __cplusplus
extern "C" {
#endif

/// Exports a public Loom C API symbol from a shared library build.
#if defined(_WIN32)
#if defined(LOOMC_BUILDING_SHARED_LIBRARY)
#define LOOMC_API_EXPORT __declspec(dllexport)
#elif defined(LOOMC_USING_SHARED_LIBRARY)
#define LOOMC_API_EXPORT __declspec(dllimport)
#else
#define LOOMC_API_EXPORT
#endif
#else
#define LOOMC_API_EXPORT __attribute__((visibility("default")))
#endif

/// Marks a public callback function pointer type.
#define LOOMC_API_PTR

/// Host allocation and container size type.
///
/// Values have the same range as C `size_t`. Public ABI fields use this
/// spelling so bindings can mirror one Loom-specific type name rather than
/// repeating the C standard library type across the API.
typedef size_t loomc_host_size_t;

/// Largest representable `loomc_host_size_t` value.
#define LOOMC_HOST_SIZE_MAX ((loomc_host_size_t)(SIZE_MAX))

/// Opaque status handle.
///
/// Small status codes are encoded directly in the pointer value. Rich status
/// modes may allocate storage behind the handle and require
/// `loomc_status_free` or `loomc_status_consume_code`.
typedef struct loomc_status_handle_t* loomc_status_t;

/// Non-owning string view over bytes that need not be NUL-terminated.
///
/// @lifetime
/// A string view never owns `data`. The API receiving the view documents
/// whether it copies the bytes or how long the caller must keep the bytes
/// alive.
typedef struct loomc_string_view_t {
  /// First byte of the string, or `NULL` when the view is empty.
  const char* data;

  /// Number of bytes in `data`.
  loomc_host_size_t size;
} loomc_string_view_t;

/// Returns an empty string view.
///
/// @return A view with `data == NULL` and `size == 0`.
static inline loomc_string_view_t loomc_string_view_empty(void) {
  loomc_string_view_t value = {NULL, 0};
  return value;
}

/// Returns true when `value` contains no bytes.
///
/// @param value View to inspect.
/// @return True when `value.data == NULL` or `value.size == 0`.
static inline bool loomc_string_view_is_empty(loomc_string_view_t value) {
  return value.data == NULL || value.size == 0;
}

/// Returns a string view over an explicit byte count.
///
/// @param data First byte of the string. May be `NULL` when `size` is zero.
/// @param size Number of bytes in `data`.
/// @return A borrowed view over the provided byte range.
static inline loomc_string_view_t loomc_make_string_view(
    const char* data, loomc_host_size_t size) {
  loomc_string_view_t value = {data, size};
  return value;
}

/// Returns a string view over a NUL-terminated C string.
///
/// @param data NUL-terminated string, or `NULL` for an empty view.
/// @return A borrowed view over `data` excluding the trailing NUL byte.
static inline loomc_string_view_t loomc_make_cstring_view(const char* data) {
  loomc_string_view_t value = {data, data ? strlen(data) : 0};
  return value;
}

/// Returns true when two views contain identical bytes.
///
/// @param lhs First view to compare.
/// @param rhs Second view to compare.
/// @return True when the views have the same length and byte contents.
static inline bool loomc_string_view_equal(loomc_string_view_t lhs,
                                           loomc_string_view_t rhs) {
  if (lhs.size != rhs.size) {
    return false;
  }
  if (lhs.size == 0) {
    return true;
  }
  return memcmp(lhs.data, rhs.data, lhs.size) == 0;
}

/// Non-owning view over immutable bytes.
///
/// @lifetime
/// A byte span never owns `data`. The API receiving the span documents whether
/// it copies the bytes or how long the caller must keep the bytes alive.
typedef struct loomc_byte_span_t {
  /// First byte of the span, or `NULL` when the span is empty.
  const uint8_t* data;

  /// Number of bytes in `data`.
  loomc_host_size_t data_length;
} loomc_byte_span_t;

/// Returns an empty byte span.
///
/// @return A span with `data == NULL` and `data_length == 0`.
static inline loomc_byte_span_t loomc_byte_span_empty(void) {
  loomc_byte_span_t value = {NULL, 0};
  return value;
}

/// Returns true when `span` contains no bytes.
///
/// @param span Span to inspect.
/// @return True when `span.data == NULL` or `span.data_length == 0`.
static inline bool loomc_byte_span_is_empty(loomc_byte_span_t span) {
  return span.data == NULL || span.data_length == 0;
}

/// Returns a byte span over an explicit byte count.
///
/// @param data First byte of the span. May be `NULL` when `data_length` is
/// zero.
/// @param data_length Number of bytes in `data`.
/// @return A borrowed span over the provided byte range.
static inline loomc_byte_span_t loomc_make_byte_span(
    const void* data, loomc_host_size_t data_length) {
  loomc_byte_span_t value = {(const uint8_t*)data, data_length};
  return value;
}

/// Structure type values used by extensible public descriptors.
///
/// Extensible descriptors store one of these values in their `type` field and
/// store `sizeof(descriptor_type)` in `structure_size`. Callers zero-initialize
/// descriptors and set the fields they use.
typedef enum loomc_structure_type_e {
  /// No structure type was supplied.
  LOOMC_STRUCTURE_TYPE_NONE = 0,

  /// `loomc_source_options_t`.
  LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS = 1,

  /// `loomc_workspace_options_t`.
  LOOMC_STRUCTURE_TYPE_WORKSPACE_OPTIONS = 2,

  /// `loomc_context_options_t`.
  LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS = 3,

  /// `loomc_link_index_builder_options_t`.
  LOOMC_STRUCTURE_TYPE_LINK_INDEX_BUILDER_OPTIONS = 4,

  /// `loomc_linker_options_t`.
  LOOMC_STRUCTURE_TYPE_LINKER_OPTIONS = 5,

  /// `loomc_link_options_t`.
  LOOMC_STRUCTURE_TYPE_LINK_OPTIONS = 6,

  /// `loomc_module_serialize_options_t`.
  LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS = 7,

  /// `loomc_module_deserialize_options_t`.
  LOOMC_STRUCTURE_TYPE_MODULE_DESERIALIZE_OPTIONS = 8,

  /// `loomc_compiler_options_t`.
  LOOMC_STRUCTURE_TYPE_COMPILER_OPTIONS = 9,

  /// `loomc_compile_options_t`.
  LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS = 10,

  /// `loomc_pass_program_options_t`.
  LOOMC_STRUCTURE_TYPE_PASS_PROGRAM_OPTIONS = 11,

  /// `loomc_context_target_options_t`.
  LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS = 12,

  /// `loomc_target_pipeline_options_t`.
  LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS = 13,

  /// `loomc_spirv_emit_options_t`.
  LOOMC_STRUCTURE_TYPE_SPIRV_EMIT_OPTIONS = 14,
} loomc_structure_type_t;

/// Controls the behavior of a `loomc_allocator_ctl_fn_t` callback.
typedef enum loomc_allocator_command_e {
  /// Allocates memory without requiring zero initialization.
  LOOMC_ALLOCATOR_COMMAND_MALLOC = 0,

  /// Allocates memory and zeros it before returning.
  LOOMC_ALLOCATOR_COMMAND_CALLOC = 1,

  /// Resizes an existing allocation.
  LOOMC_ALLOCATOR_COMMAND_REALLOC = 2,

  /// Frees an existing allocation.
  LOOMC_ALLOCATOR_COMMAND_FREE = 3,
} loomc_allocator_command_t;

/// Allocation parameters passed to `loomc_allocator_ctl_fn_t` callbacks.
typedef struct loomc_allocator_alloc_params_t {
  /// Minimum allocation size in bytes.
  loomc_host_size_t byte_length;
} loomc_allocator_alloc_params_t;

/// Host allocation control callback.
///
/// @param self Callback-owned allocator state pointer.
/// @param command Allocation command to perform.
/// @param params Command-specific parameter structure.
/// @param inout_ptr Pointer to allocation storage. Allocation commands write
/// the allocated pointer. Free commands read the pointer to free.
/// @return OK on success or a non-OK status when the allocator cannot satisfy
/// the command.
///
/// @ownership
/// Memory returned by an allocator must be freed through the same allocator.
typedef loomc_status_t(LOOMC_API_PTR* loomc_allocator_ctl_fn_t)(
    void* self, loomc_allocator_command_t command, const void* params,
    void** inout_ptr);

/// Host allocator used for persistent Loom object storage.
///
/// A zero-initialized allocator value is accepted by APIs that take an
/// allocator and is treated as `loomc_allocator_system()`.
typedef struct loomc_allocator_t {
  /// Callback-owned state pointer.
  void* self;

  /// Callback implementing allocation commands.
  loomc_allocator_ctl_fn_t ctl;
} loomc_allocator_t;

/// Returns a process-global system allocator.
///
/// @return Allocator backed by the process heap.
LOOMC_API_EXPORT loomc_allocator_t loomc_allocator_system(void);

/// Returns `allocator` when it is valid, otherwise the system allocator.
///
/// @param allocator Allocator to normalize.
/// @return `allocator` when `allocator.ctl` is non-NULL, otherwise
/// `loomc_allocator_system()`.
LOOMC_API_EXPORT loomc_allocator_t
loomc_allocator_or_system(loomc_allocator_t allocator);

/// Allocates zero-initialized memory from `allocator`.
///
/// @param allocator Allocator used for the allocation, or a zero-initialized
/// value to use the system allocator.
/// @param byte_length Number of bytes to allocate.
/// @param out_ptr Receives the allocated memory on success.
/// @return OK on success or a non-OK status on allocation failure.
///
/// @ownership
/// The caller owns `out_ptr` on success and frees it with
/// `loomc_allocator_free` using the same allocator.
LOOMC_API_EXPORT loomc_status_t loomc_allocator_malloc(
    loomc_allocator_t allocator, loomc_host_size_t byte_length, void** out_ptr);

/// Allocates uninitialized memory from `allocator`.
///
/// @param allocator Allocator used for the allocation, or a zero-initialized
/// value to use the system allocator.
/// @param byte_length Number of bytes to allocate.
/// @param out_ptr Receives the allocated memory on success.
/// @return OK on success or a non-OK status on allocation failure.
///
/// @ownership
/// The caller owns `out_ptr` on success and frees it with
/// `loomc_allocator_free` using the same allocator.
LOOMC_API_EXPORT loomc_status_t loomc_allocator_malloc_uninitialized(
    loomc_allocator_t allocator, loomc_host_size_t byte_length, void** out_ptr);

/// Frees memory previously returned by `allocator`.
///
/// @param allocator Allocator that returned `ptr`, or a zero-initialized value
/// when `ptr` came from the system allocator.
/// @param ptr Allocation to free. Passing `NULL` is allowed.
LOOMC_API_EXPORT void loomc_allocator_free(loomc_allocator_t allocator,
                                           void* ptr);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_BASE_H_
