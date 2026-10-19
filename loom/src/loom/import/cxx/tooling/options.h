// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_TOOLING_OPTIONS_H_
#define LOOM_IMPORT_CXX_TOOLING_OPTIONS_H_

#include "iree/base/internal/arena.h"
#include "loom/tooling/input/format.h"
#include "loomc/import/cxx.h"

#ifdef __cplusplus
extern "C" {
#endif

// C/C++ filename and language selection metadata, independent of the importer.
extern const loom_input_format_t loom_cxx_input_format;

// Parses the provider option spelling (std, triple, data-model, I,
// isystem, D, root, fp-model, and builtin-includes; quoted whitespace and
// escapes are supported). All arrays and any joined include paths borrow from
// |arena|; scalar string views borrow from either |input_options| or |arena|.
// |source_path| is the physical main-source path used to resolve relative
// include directories.
iree_status_t loom_cxx_input_parse_options(
    iree_string_view_t source_path, iree_string_view_t input_options,
    iree_arena_allocator_t* arena, iree_allocator_t host_allocator,
    loomc_cxx_import_options_t* out_options);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_IMPORT_CXX_TOOLING_OPTIONS_H_
