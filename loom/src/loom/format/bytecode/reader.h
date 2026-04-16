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

#ifdef __cplusplus
extern "C" {
#endif

// Options controlling bytecode reads.
typedef struct loom_bytecode_read_options_t {
  // Sink for structured malformed-bytecode diagnostics.
  loom_diagnostic_sink_t diagnostic_sink;
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

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_H_
