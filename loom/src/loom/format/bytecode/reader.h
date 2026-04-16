// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bytecode metadata reader and validator.
//
// This is the front door for untrusted .loombc input before full IR body
// materialization exists. It validates file/module/section structure and all
// lightweight metadata tables that the linker and lazy function reader need:
// strings, sources, encodings, types, ops, locations, and symbols.

#ifndef LOOM_FORMAT_BYTECODE_READER_H_
#define LOOM_FORMAT_BYTECODE_READER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/diagnostic.h"
#include "loom/format/bytecode/format.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Options controlling bytecode reads.
typedef struct loom_bytecode_read_options_t {
  // Sink for structured malformed-bytecode diagnostics.
  loom_diagnostic_sink_t diagnostic_sink;

  // Runs the module verifier after successful materialization. Verification
  // diagnostics are emitted to diagnostic_sink and counted in the read result.
  bool verify_module;

  // Maximum verifier errors to emit when verify_module is set. Zero means
  // unlimited, matching loom_verify_options_t.
  uint32_t verify_max_errors;
} loom_bytecode_read_options_t;

// Summary of one decoded module's lightweight metadata.
typedef struct loom_bytecode_module_metadata_summary_t {
  uint64_t value_count;     // Allocation-summary SSA value count.
  uint64_t region_count;    // Allocation-summary region count.
  uint64_t block_count;     // Allocation-summary block count.
  uint64_t op_count;        // Allocation-summary operation count.
  uint64_t string_count;    // STRINGS table entry count.
  uint64_t source_count;    // SOURCES table entry count.
  uint64_t type_count;      // TYPES table entry count.
  uint64_t encoding_count;  // ENCODINGS instance table entry count.
  uint64_t op_name_count;   // OPS table entry count.
  uint64_t location_count;  // LOCATIONS table entry count, or 0 when omitted.
  uint64_t symbol_count;    // SYMBOLS table entry count.
} loom_bytecode_module_metadata_summary_t;

// Result of reading bytecode metadata.
typedef struct loom_bytecode_read_result_t {
  uint32_t error_count;    // Number of error diagnostics emitted.
  uint32_t warning_count;  // Number of warning diagnostics emitted.
  uint16_t module_count;   // Number of module directory entries decoded.
  // File-level source-location mode from the header.
  loom_bytecode_location_mode_t location_mode;
  // Summary for the first module. Multi-module validation is supported, but
  // this V1 result keeps only the first summary until archive-level APIs need
  // per-module output.
  loom_bytecode_module_metadata_summary_t first_module;
} loom_bytecode_read_result_t;

// Validates .loombc file structure and metadata sections.
//
// Malformed bytecode is reported through |options->diagnostic_sink| and in
// |out_result->error_count|. The function returns a non-OK status only for
// infrastructure failures such as OOM or a sink callback failure.
//
// |context| must be finalized and contain the dialect/encoding registrations
// used to resolve OPS and ENCODINGS metadata. |block_pool| provides transient
// arena storage for decoded metadata and is reset before return.
iree_status_t loom_bytecode_read_metadata(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    const loom_bytecode_read_options_t* options,
    loom_bytecode_read_result_t* out_result);

// Reads a single-module .loombc file and materializes its IR into |out_module|.
//
// Malformed bytecode follows the same diagnostic contract as
// loom_bytecode_read_metadata: diagnostics are emitted and counted in
// |out_result| while the function returns OK unless infrastructure fails. When
// malformed-bytecode diagnostics are emitted, |out_module| is NULL.
//
// |host_allocator| owns the returned module object. All IR storage inside that
// module is arena-owned by the module and released by loom_module_free.
iree_status_t loom_bytecode_read_module(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    const loom_bytecode_read_options_t* options,
    loom_bytecode_read_result_t* out_result, loom_module_t** out_module,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_H_
