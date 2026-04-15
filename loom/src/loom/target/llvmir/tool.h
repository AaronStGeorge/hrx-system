// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Installed LLVM toolchain runner.
//
// Loom emits LLVM IR and bitcode without linking LLVM. This layer is the narrow
// boundary that invokes an installed LLVM binary distribution when tests or JIT
// compilation need LLVM's verifier, assembler, disassembler, or object writer.

#ifndef LOOM_TARGET_LLVMIR_TOOL_H_
#define LOOM_TARGET_LLVMIR_TOOL_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_llvmir_tool_kind_e {
  LOOM_LLVMIR_TOOL_LLVM_AS = 0,
  LOOM_LLVMIR_TOOL_LLVM_DIS = 1,
  LOOM_LLVMIR_TOOL_OPT = 2,
  LOOM_LLVMIR_TOOL_LLC = 3,
} loom_llvmir_tool_kind_t;

typedef struct loom_llvmir_toolchain_t {
  // Directory containing LLVM tool executables, or empty to search PATH.
  iree_string_view_t root_path;
} loom_llvmir_toolchain_t;

typedef struct loom_llvmir_tool_output_t {
  // Captured bytes allocated from the caller allocator, or NULL when empty.
  char* data;
  // Number of bytes in |data|, excluding the trailing NUL terminator.
  iree_host_size_t length;
} loom_llvmir_tool_output_t;

typedef struct loom_llvmir_tool_result_t {
  // Process exit code. Values are platform exit codes, including signal-derived
  // POSIX codes when a process does not exit normally.
  int exit_code;
  // Captured stdout bytes.
  loom_llvmir_tool_output_t stdout_text;
  // Captured stderr bytes.
  loom_llvmir_tool_output_t stderr_text;
} loom_llvmir_tool_result_t;

// Initializes |out_toolchain| from LOOM_LLVMIR_TOOLCHAIN_ROOT, falling back to
// PATH lookup when the environment variable is absent.
void loom_llvmir_toolchain_initialize_from_environment(
    loom_llvmir_toolchain_t* out_toolchain);

// Returns the executable basename for |tool_kind|, without a toolchain root.
iree_string_view_t loom_llvmir_tool_name(loom_llvmir_tool_kind_t tool_kind);

// Invokes an LLVM tool with argv-style arguments and captures stdout/stderr.
//
// This function does not use shell command strings. A nonzero child exit code
// is represented in |out_result| and still returns OK: launch/capture failures
// are status failures, tool diagnostics are child process results.
iree_status_t loom_llvmir_tool_run(const loom_llvmir_toolchain_t* toolchain,
                                   loom_llvmir_tool_kind_t tool_kind,
                                   const iree_string_view_t* arguments,
                                   iree_host_size_t argument_count,
                                   iree_allocator_t allocator,
                                   loom_llvmir_tool_result_t* out_result);

bool loom_llvmir_tool_result_succeeded(const loom_llvmir_tool_result_t* result);

void loom_llvmir_tool_result_deinitialize(loom_llvmir_tool_result_t* result,
                                          iree_allocator_t allocator);

void loom_llvmir_tool_output_deinitialize(loom_llvmir_tool_output_t* output,
                                          iree_allocator_t allocator);

// Runs `<tool> --version` and returns captured stdout in |out_version_text|.
iree_status_t loom_llvmir_tool_query_version(
    const loom_llvmir_toolchain_t* toolchain, loom_llvmir_tool_kind_t tool_kind,
    iree_allocator_t allocator, loom_llvmir_tool_output_t* out_version_text);

// Runs `llvm-as <input_path> -o <output_path>`.
iree_status_t loom_llvmir_tool_assemble_text_file(
    const loom_llvmir_toolchain_t* toolchain, iree_string_view_t input_path,
    iree_string_view_t output_path, iree_allocator_t allocator);

// Runs `llvm-dis <input_path> -o -` and returns textual LLVM IR.
iree_status_t loom_llvmir_tool_disassemble_bitcode_file(
    const loom_llvmir_toolchain_t* toolchain, iree_string_view_t input_path,
    iree_allocator_t allocator, loom_llvmir_tool_output_t* out_text);

// Writes |bitcode| to a temporary file, runs `llvm-dis <temp> -o -`, and
// returns textual LLVM IR.
iree_status_t loom_llvmir_tool_disassemble_bitcode(
    const loom_llvmir_toolchain_t* toolchain, iree_const_byte_span_t bitcode,
    iree_allocator_t allocator, loom_llvmir_tool_output_t* out_text);

// Runs `opt -passes=verify <input_path> -disable-output`.
iree_status_t loom_llvmir_tool_verify_bitcode_file(
    const loom_llvmir_toolchain_t* toolchain, iree_string_view_t input_path,
    iree_allocator_t allocator);

// Runs `llc <input_path> -filetype=obj -o <output_path>`.
//
// |extra_arguments| are appended after the standard arguments and may include
// target-specific flags such as `-mcpu=gfx1100`.
iree_status_t loom_llvmir_tool_compile_object_file(
    const loom_llvmir_toolchain_t* toolchain, iree_string_view_t input_path,
    iree_string_view_t output_path, const iree_string_view_t* extra_arguments,
    iree_host_size_t extra_argument_count, iree_allocator_t allocator);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_TOOL_H_
