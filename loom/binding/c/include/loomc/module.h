// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_MODULE_H_
#define LOOMC_MODULE_H_

#include <stdio.h>

#include "loomc/context.h"
#include "loomc/result.h"
#include "loomc/source.h"

/// @file
/// Opaque in-memory Loom modules.
///
/// Modules are the typed handoff between parsing, linking, optimization, and
/// compilation. They let embedders compose API operations without printing and
/// reparsing `.loom` text on the hot path. Serialization is the boundary that
/// turns a module back into text or bytecode bytes for storage, display, cache
/// keys, or handoff to systems that consume sources. Deserialization is the
/// inverse boundary that turns text or bytecode bytes into an owned mutable
/// module handle.
///
/// @par Example
/// Serialize a linked module back to an immutable source handle:
///
/// @code{.c}
/// loomc_module_serialize_options_t options = {
///     .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
///     .structure_size = sizeof(loomc_module_serialize_options_t),
///     .format = LOOMC_SOURCE_FORMAT_BYTECODE,
///     .identifier = loomc_make_cstring_view("linked.loombc"),
/// };
/// loomc_source_t* serialized = NULL;
/// loomc_status_t status = loomc_module_serialize_to_source(
///     module, &options, loomc_allocator_system(), &serialized);
/// if (!loomc_status_is_ok(status)) {
///   return status;
/// }
///
/// // Reuse serialized as ordinary source input, cache it, or inspect its
/// bytes.
///
/// loomc_source_release(serialized);
/// @endcode
///
/// @par Example
/// Deserialize a source handle into a module and inspect the result:
///
/// @code{.c}
/// loomc_module_t* module = NULL;
/// loomc_result_t* result = NULL;
/// loomc_status_t status = loomc_module_deserialize_from_source(
///     context, serialized, NULL, loomc_allocator_system(), &module, &result);
/// if (!loomc_status_is_ok(status)) {
///   return status;
/// }
/// if (!loomc_result_succeeded(result)) {
///   // Diagnostics are available through loomc_result_diagnostic_at.
/// }
///
/// loomc_result_release(result);
/// loomc_module_release(module);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque parsed, linked, optimized, or lowered module.
///
/// @thread_safety
/// Retain/release operations are safe from multiple threads. Module contents
/// are not internally synchronized. Read-only operations such as serialization
/// may run concurrently with other read-only operations when the caller
/// guarantees that no mutating module operation is active. Mutating operations
/// such as compiling a module require exclusive access to that module handle.
typedef struct loomc_module_t loomc_module_t;

/// Three-dimensional unsigned extent used by function metadata.
typedef struct loomc_dimension3_t {
  /// Extent along the x dimension.
  uint32_t x;

  /// Extent along the y dimension.
  uint32_t y;

  /// Extent along the z dimension.
  uint32_t z;
} loomc_dimension3_t;

/// Public function category reported from a module query.
typedef enum loomc_module_function_kind_e {
  /// No specific function kind was requested or reported.
  LOOMC_MODULE_FUNCTION_KIND_UNKNOWN = 0,

  /// Source-level `func.def` function definition.
  LOOMC_MODULE_FUNCTION_KIND_FUNCTION = 1,

  /// Source-level `kernel.def` dispatchable kernel definition.
  LOOMC_MODULE_FUNCTION_KIND_KERNEL = 2,

  /// Target-bound function definition produced by lowering.
  LOOMC_MODULE_FUNCTION_KIND_TARGET_FUNCTION = 3,

  /// Target-bound dispatchable kernel entry produced by lowering.
  LOOMC_MODULE_FUNCTION_KIND_TARGET_KERNEL = 4,
} loomc_module_function_kind_t;

/// Returns true when a function kind is dispatchable kernel-shaped.
///
/// @param kind Function kind to inspect.
/// @return True for source-level and target-bound kernel function kinds.
static inline bool loomc_module_function_kind_is_kernel(
    loomc_module_function_kind_t kind) {
  return kind == LOOMC_MODULE_FUNCTION_KIND_KERNEL ||
         kind == LOOMC_MODULE_FUNCTION_KIND_TARGET_KERNEL;
}

/// Module function metadata flag bits.
typedef enum loomc_module_function_flag_bits_e {
  /// Function symbol is visible outside the module for linking.
  LOOMC_MODULE_FUNCTION_FLAG_PUBLIC = 1u << 0,

  /// `loomc_module_function_try_get_export_info` can return export metadata.
  LOOMC_MODULE_FUNCTION_FLAG_HAS_EXPORT_INFO = 1u << 1,
} loomc_module_function_flag_bits_t;

/// Bitmask of `loomc_module_function_flag_bits_t` values.
typedef uint32_t loomc_module_function_flags_t;

/// Function metadata view written into caller-provided storage.
///
/// This is the common identity record for every function kind. Function-kind
/// specific payloads, such as kernel launch metadata, are queried with typed
/// accessors instead of being mixed into this structure.
///
/// @lifetime
/// `symbol_name` borrows from the module that produced this view. The view and
/// `symbol_ordinal` remain valid until that module is released or mutated.
typedef struct loomc_module_function_t {
  /// Module symbol-table ordinal used by follow-up metadata queries.
  uint32_t symbol_ordinal;

  /// Loom module symbol name, without a leading `@`.
  loomc_string_view_t symbol_name;

  /// Function category.
  loomc_module_function_kind_t kind;

  /// Present metadata flags.
  loomc_module_function_flags_t flags;
} loomc_module_function_t;

/// Export metadata flag bits.
typedef enum loomc_module_function_export_flag_bits_e {
  /// `loomc_module_function_export_info_t::export_symbol` is present.
  LOOMC_MODULE_FUNCTION_EXPORT_FLAG_HAS_SYMBOL = 1u << 0,

  /// `loomc_module_function_export_info_t::export_ordinal` is present.
  LOOMC_MODULE_FUNCTION_EXPORT_FLAG_HAS_ORDINAL = 1u << 1,
} loomc_module_function_export_flag_bits_t;

/// Bitmask of `loomc_module_function_export_flag_bits_t` values.
typedef uint32_t loomc_module_function_export_flags_t;

/// Export metadata view written into caller-provided storage.
///
/// Export metadata is separate from `loomc_module_function_t` because not every
/// function participates in artifact export, and export contracts may grow
/// independently from function identity.
///
/// @lifetime
/// `export_symbol` borrows from the module passed to the query. The view
/// remains valid until that module is released or mutated.
typedef struct loomc_module_function_export_info_t {
  /// Present export metadata flags.
  loomc_module_function_export_flags_t flags;

  /// Optional artifact export symbol without a leading `@`.
  loomc_string_view_t export_symbol;

  /// Optional artifact export ordinal.
  uint32_t export_ordinal;
} loomc_module_function_export_info_t;

/// Kernel function metadata flag bits.
typedef enum loomc_module_kernel_function_flag_bits_e {
  /// `loomc_module_kernel_function_info_t::static_dispatch_workgroup_count` is
  /// present.
  LOOMC_MODULE_KERNEL_FUNCTION_FLAG_HAS_STATIC_DISPATCH_WORKGROUP_COUNT = 1u
                                                                          << 0,

  /// `loomc_module_kernel_function_info_t::static_workgroup_size` is present.
  LOOMC_MODULE_KERNEL_FUNCTION_FLAG_HAS_STATIC_WORKGROUP_SIZE = 1u << 1,
} loomc_module_kernel_function_flag_bits_t;

/// Bitmask of `loomc_module_kernel_function_flag_bits_t` values.
typedef uint32_t loomc_module_kernel_function_flags_t;

/// Kernel-specific metadata view written into caller-provided storage.
///
/// Probe this view only for functions whose kind is
/// `LOOMC_MODULE_FUNCTION_KIND_KERNEL` or
/// `LOOMC_MODULE_FUNCTION_KIND_TARGET_KERNEL`. Fields whose presence depends
/// on source metadata, lowering state, or analysis have corresponding `flags`
/// bits. When a flag is absent, the related field is a zero value and should
/// not be interpreted as a default chosen by Loom.
typedef struct loomc_module_kernel_function_info_t {
  /// Present kernel metadata flags.
  loomc_module_kernel_function_flags_t flags;

  /// Optional statically known dispatch workgroup count.
  loomc_dimension3_t static_dispatch_workgroup_count;

  /// Optional statically known workgroup size.
  loomc_dimension3_t static_workgroup_size;
} loomc_module_kernel_function_info_t;

/// Module function query options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_MODULE_FUNCTION_QUERY_OPTIONS`, set `structure_size`
/// to `sizeof(loomc_module_function_query_options_t)`, and fill the requested
/// fields.
typedef struct loomc_module_function_query_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_MODULE_FUNCTION_QUERY_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for future function query options.
  const void* next;

  /// Optional symbol selector. Empty enumerates all matching functions. Both
  /// `function` and `@function` spellings are accepted.
  loomc_string_view_t function_symbol;

  /// Optional kind filter. `LOOMC_MODULE_FUNCTION_KIND_UNKNOWN` accepts every
  /// supported function kind.
  loomc_module_function_kind_t kind;
} loomc_module_function_query_options_t;

/// Text presentation policy used when serializing `.loom` text.
typedef enum loomc_module_text_presentation_e {
  /// Prefer target-low assembly syntax when the module selects one descriptor
  /// set unambiguously; otherwise print canonical text.
  LOOMC_MODULE_TEXT_PRESENTATION_DEFAULT = 0,

  /// Force canonical text with descriptor-backed target-low operations printed
  /// as ordinary `low.op<...>` operations.
  LOOMC_MODULE_TEXT_PRESENTATION_GENERIC = 1,

  /// Force descriptor-backed target-low assembly syntax. Serialization fails
  /// when no descriptor set is provided or can be selected unambiguously.
  LOOMC_MODULE_TEXT_PRESENTATION_LOW_ASM = 2,
} loomc_module_text_presentation_t;

/// Module serialization options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS`, set `structure_size` to
/// `sizeof(loomc_module_serialize_options_t)`, and fill the requested fields.
typedef struct loomc_module_serialize_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS`
  /// when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for future serialization options.
  const void* next;

  /// Output source format. Unknown selects textual `.loom`.
  loomc_source_format_t format;

  /// Identifier to attach to a returned source. Empty uses a format-specific
  /// default. Path and `FILE*` serialization do not interpret this field.
  loomc_string_view_t identifier;

  /// Presentation policy for textual `.loom` output.
  loomc_module_text_presentation_t text_presentation;

  /// Target-low descriptor-set key used by low assembly presentation.
  ///
  /// Empty lets the serializer infer a single descriptor set from target-low
  /// functions in the module. Non-empty requests low assembly presentation
  /// unless `text_presentation` is `LOOMC_MODULE_TEXT_PRESENTATION_GENERIC`,
  /// which is rejected as contradictory.
  loomc_string_view_t low_asm_descriptor_set_key;
} loomc_module_serialize_options_t;

/// Module deserialization options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_MODULE_DESERIALIZE_OPTIONS`, set `structure_size` to
/// `sizeof(loomc_module_deserialize_options_t)`, and fill the requested fields.
typedef struct loomc_module_deserialize_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_MODULE_DESERIALIZE_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for future deserialization options.
  const void* next;

  /// Input source format. Unknown uses the source format when available, then
  /// falls back to bytecode magic detection and textual `.loom`.
  loomc_source_format_t format;

  /// Identifier used for diagnostics and module provenance. Empty uses the
  /// input source identifier, filesystem path, or a format-specific default.
  loomc_string_view_t identifier;

  /// Persistent module arena block size in bytes. Zero selects the default.
  loomc_host_size_t block_size;
} loomc_module_deserialize_options_t;

/// Retains an opaque module for another owner.
///
/// @param module Module to retain.
///
/// @thread_safety
/// Retain/release operations are intended to be safe from multiple threads.
LOOMC_API_EXPORT void loomc_module_retain(loomc_module_t* module);

/// Releases an opaque module from one owner.
///
/// @param module Module to release. Passing `NULL` is allowed.
///
/// @thread_safety
/// Retain/release operations are intended to be safe from multiple threads. The
/// module is destroyed when the final reference is released.
LOOMC_API_EXPORT void loomc_module_release(loomc_module_t* module);

/// Queries function metadata from a module.
///
/// @param module Module to inspect.
/// @param options Query options. `NULL` enumerates all supported function
/// kinds.
/// @param allocator Host allocator used for the returned result.
/// @param function_capacity Number of entries available in `out_functions`.
/// @param out_functions Caller-owned output storage. May be `NULL` only when
/// `function_capacity` is zero.
/// @param out_function_count Receives the total number of matching functions,
/// which may be larger than `function_capacity`.
/// @param out_result Receives a retained result for the query.
/// @return OK when the query ran to a result. Non-OK statuses represent API
/// misuse or infrastructure failures before a result could be produced.
///
/// @ownership
/// The caller owns `out_functions` storage. The caller always owns
/// `out_result` on an OK return and releases it with `loomc_result_release`.
///
/// @lifetime
/// Function string views borrow from `module`. Returned views and
/// `symbol_ordinal` identities remain valid until the module is released or
/// mutated. Follow-up metadata queries must use the same module that produced
/// the function views.
///
/// @thread_safety
/// Function queries are read-only with respect to `module`. Concurrent queries
/// of the same module are valid when the caller guarantees that no mutating
/// module operation is active.
///
/// @par Example
/// Find source kernels with a fully static launch grid:
///
/// @code{.c}
/// static loomc_status_t dispatch_static_kernels(loomc_module_t* module) {
///   loomc_allocator_t allocator = loomc_allocator_system();
///   loomc_module_function_query_options_t options = {
///       .type = LOOMC_STRUCTURE_TYPE_MODULE_FUNCTION_QUERY_OPTIONS,
///       .structure_size = sizeof(loomc_module_function_query_options_t),
///       .kind = LOOMC_MODULE_FUNCTION_KIND_KERNEL,
///   };
///
///   loomc_module_function_t* functions = NULL;
///   loomc_host_size_t function_count = 0;
///   loomc_result_t* result = NULL;
///   loomc_status_t status = loomc_module_query_functions(
///       module, &options, allocator, 0, NULL, &function_count, &result);
///
///   if (loomc_status_is_ok(status) && loomc_result_succeeded(result) &&
///       function_count != 0) {
///     loomc_result_release(result);
///     result = NULL;
///     status = loomc_allocator_malloc(
///         allocator, function_count * sizeof(*functions), (void**)&functions);
///   }
///   if (loomc_status_is_ok(status) && functions != NULL) {
///     status = loomc_module_query_functions(module, &options, allocator,
///                                           function_count, functions,
///                                           &function_count, &result);
///   }
///   if (loomc_status_is_ok(status) && result != NULL &&
///       loomc_result_succeeded(result)) {
///     for (loomc_host_size_t i = 0; i < function_count; ++i) {
///       loomc_module_kernel_function_flags_t static_grid_flag =
///           LOOMC_MODULE_KERNEL_FUNCTION_FLAG_HAS_STATIC_DISPATCH_WORKGROUP_COUNT;
///       loomc_module_kernel_function_info_t kernel_info;
///       bool has_static_grid =
///           loomc_module_function_try_get_kernel_info(module, &functions[i],
///                                                    &kernel_info) &&
///           (kernel_info.flags & static_grid_flag);
///       if (has_static_grid) {
///         dispatch(functions[i].symbol_name,
///                  kernel_info.static_dispatch_workgroup_count);
///       }
///     }
///   }
///
///   loomc_allocator_free(allocator, functions);
///   loomc_result_release(result);
///   return status;
/// }
/// @endcode
LOOMC_API_EXPORT loomc_status_t loomc_module_query_functions(
    const loomc_module_t* module,
    const loomc_module_function_query_options_t* options,
    loomc_allocator_t allocator, loomc_host_size_t function_capacity,
    loomc_module_function_t* out_functions,
    loomc_host_size_t* out_function_count, loomc_result_t** out_result);

/// Looks up one function by symbol name without allocating a status.
///
/// @param module Module to inspect.
/// @param symbol_name Function symbol name, with or without a leading `@`.
/// @param out_function Receives function metadata when the lookup succeeds.
/// @return True when `symbol_name` names a supported function.
LOOMC_API_EXPORT bool loomc_module_try_lookup_function(
    const loomc_module_t* module, loomc_string_view_t symbol_name,
    loomc_module_function_t* out_function);

/// Looks up one function by symbol name.
///
/// @param module Module to inspect.
/// @param symbol_name Function symbol name, with or without a leading `@`.
/// @param out_function Receives function metadata when the lookup succeeds.
/// @return OK when `symbol_name` names a supported function, NOT_FOUND when it
/// does not, or another non-OK status for API misuse.
LOOMC_API_EXPORT loomc_status_t loomc_module_lookup_function(
    const loomc_module_t* module, loomc_string_view_t symbol_name,
    loomc_module_function_t* out_function);

/// Tries to get export metadata for a function without allocating a status.
///
/// @param module Module that produced `function`.
/// @param function Function metadata from `module`.
/// @param out_info Receives export metadata when available.
/// @return True when `function` has export metadata.
LOOMC_API_EXPORT bool loomc_module_function_try_get_export_info(
    const loomc_module_t* module, const loomc_module_function_t* function,
    loomc_module_function_export_info_t* out_info);

/// Gets export metadata for a function.
///
/// @param module Module that produced `function`.
/// @param function Function metadata from `module`.
/// @param out_info Receives export metadata.
/// @return OK when export metadata is available, NOT_FOUND when `function` has
/// no export metadata, or another non-OK status for API misuse.
LOOMC_API_EXPORT loomc_status_t loomc_module_function_get_export_info(
    const loomc_module_t* module, const loomc_module_function_t* function,
    loomc_module_function_export_info_t* out_info);

/// Tries to get kernel metadata for a function without allocating a status.
///
/// @param module Module that produced `function`.
/// @param function Function metadata from `module`.
/// @param out_info Receives kernel metadata when `function` is kernel-shaped.
/// @return True when `function` is a kernel function.
LOOMC_API_EXPORT bool loomc_module_function_try_get_kernel_info(
    const loomc_module_t* module, const loomc_module_function_t* function,
    loomc_module_kernel_function_info_t* out_info);

/// Gets kernel metadata for a function.
///
/// @param module Module that produced `function`.
/// @param function Function metadata from `module`.
/// @param out_info Receives kernel metadata.
/// @return OK when `function` is kernel-shaped, NOT_FOUND when it is not, or
/// another non-OK status for API misuse.
LOOMC_API_EXPORT loomc_status_t loomc_module_function_get_kernel_info(
    const loomc_module_t* module, const loomc_module_function_t* function,
    loomc_module_kernel_function_info_t* out_info);

/// Serializes a module into an immutable source handle.
///
/// @param module Module to serialize.
/// @param options Serialization options. `NULL` selects textual `.loom`.
/// @param allocator Host allocator used for source-owned storage.
/// @param out_source Receives one retained source on success.
/// @return OK when serialization succeeded.
///
/// @ownership
/// The caller owns the returned source and releases it with
/// `loomc_source_release`. Serialized bytes are owned by that source and remain
/// valid until the source is released.
///
/// @thread_safety
/// Serialization is read-only with respect to `module`. Concurrent
/// serialization of the same module is valid when the caller guarantees that no
/// mutating operation is active.
LOOMC_API_EXPORT loomc_status_t loomc_module_serialize_to_source(
    const loomc_module_t* module,
    const loomc_module_serialize_options_t* options,
    loomc_allocator_t allocator, loomc_source_t** out_source);

/// Serializes a module to an open C `FILE*`.
///
/// @param module Module to serialize.
/// @param options Serialization options. `NULL` selects textual `.loom`.
/// @param file Open writable file handle, such as `stdout`.
/// @return OK when serialization succeeded.
///
/// @ownership
/// The caller retains ownership of `file`; this function does not close it.
LOOMC_API_EXPORT loomc_status_t loomc_module_serialize_to_file(
    const loomc_module_t* module,
    const loomc_module_serialize_options_t* options, FILE* file);

/// Serializes a module to a filesystem path.
///
/// @param module Module to serialize.
/// @param options Serialization options. `NULL` selects textual `.loom`.
/// @param path Output file path. The path is a borrowed byte view and does not
/// need to be NUL-terminated.
/// @param allocator Host allocator used for transient file path/stream storage.
/// @return OK when serialization succeeded.
LOOMC_API_EXPORT loomc_status_t loomc_module_serialize_to_path(
    const loomc_module_t* module,
    const loomc_module_serialize_options_t* options, loomc_string_view_t path,
    loomc_allocator_t allocator);

/// Deserializes an immutable source handle into an owned mutable module.
///
/// @param context Context used to resolve Loom dialect and bytecode metadata.
/// @param source Source bytes to deserialize.
/// @param options Deserialization options. `NULL` infers format from `source`.
/// @param allocator Host allocator used for module-owned storage.
/// @param out_module Receives a retained module when the result succeeds.
/// Receives `NULL` when the result fails.
/// @param out_result Receives a retained result for the operation.
/// @return OK when deserialization ran to a result. Non-OK statuses represent
/// API misuse or infrastructure failures before a result could be produced.
///
/// @ownership
/// The caller owns `out_module` when non-NULL and releases it with
/// `loomc_module_release`. The caller always owns `out_result` on an OK return
/// and releases it with `loomc_result_release`.
///
/// @lifetime
/// Returned modules and results do not borrow from `source`. Diagnostics retain
/// `source` when they reference it. Later module operations may mutate the
/// returned module in place.
LOOMC_API_EXPORT loomc_status_t loomc_module_deserialize_from_source(
    loomc_context_t* context, const loomc_source_t* source,
    const loomc_module_deserialize_options_t* options,
    loomc_allocator_t allocator, loomc_module_t** out_module,
    loomc_result_t** out_result);

/// Deserializes bytes from an open C `FILE*` into a module.
///
/// @param context Context used to resolve Loom dialect and bytecode metadata.
/// @param file Open readable file handle. Bytes are read from the current file
/// position to EOF.
/// @param options Deserialization options. `NULL` infers format from bytes.
/// @param allocator Host allocator used for module-owned storage and temporary
/// input bytes.
/// @param out_module Receives a retained module when the result succeeds.
/// Receives `NULL` when the result fails.
/// @param out_result Receives a retained result for the operation.
/// @return OK when deserialization ran to a result. Non-OK statuses represent
/// API misuse, file read failures, or infrastructure failures before a result
/// could be produced.
///
/// @ownership
/// The caller retains ownership of `file`; this function does not close it.
LOOMC_API_EXPORT loomc_status_t loomc_module_deserialize_from_file(
    loomc_context_t* context, FILE* file,
    const loomc_module_deserialize_options_t* options,
    loomc_allocator_t allocator, loomc_module_t** out_module,
    loomc_result_t** out_result);

/// Deserializes bytes from a filesystem path into a module.
///
/// @param context Context used to resolve Loom dialect and bytecode metadata.
/// @param path Input file path. The path is a borrowed byte view and does not
/// need to be NUL-terminated.
/// @param options Deserialization options. `NULL` infers format from bytes.
/// @param allocator Host allocator used for module-owned storage and temporary
/// input bytes.
/// @param out_module Receives a retained module when the result succeeds.
/// Receives `NULL` when the result fails.
/// @param out_result Receives a retained result for the operation.
/// @return OK when deserialization ran to a result. Non-OK statuses represent
/// API misuse, file read failures, or infrastructure failures before a result
/// could be produced.
LOOMC_API_EXPORT loomc_status_t loomc_module_deserialize_from_path(
    loomc_context_t* context, loomc_string_view_t path,
    const loomc_module_deserialize_options_t* options,
    loomc_allocator_t allocator, loomc_module_t** out_module,
    loomc_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_MODULE_H_
