// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TOOLING_INPUT_FORMAT_H_
#define LOOM_TOOLING_INPUT_FORMAT_H_

#include "loom/tooling/io/source_path.h"

#ifdef __cplusplus
extern "C" {
#endif

// Shared source selection and per-provider options for multi-input tools.
typedef struct loom_input_options_t {
  // Explicit source format, or empty to select from each filename.
  iree_string_view_t format;
  // Borrowed entries spelled "format:options", at most one per format.
  iree_string_view_list_t provider_options;
  // Display remapping for diagnostics and retained module source names.
  loom_tooling_source_path_options_t source_path_options;
} loom_input_options_t;

// Resolves the option string for |provider|. Malformed entries and duplicate
// entries for this provider fail. Entries for other providers are independent.
iree_status_t loom_input_options_for_provider(iree_string_view_list_t entries,
                                              iree_string_view_t provider,
                                              iree_string_view_t* out_options);

// Source format metadata independent of the native or LoomC import route.
typedef struct loom_input_format_t {
  // Explicit input-format name, such as "loom" or "cxx".
  iree_string_view_t name;
  // Recognized filename suffixes including the leading dot.
  iree_string_view_list_t suffixes;
} loom_input_format_t;

// Optional formats supplied in addition to builtin Loom text and bytecode.
typedef struct loom_input_format_list_t {
  // Immutable format descriptors borrowed during selection.
  const loom_input_format_t* const* values;
  // Number of descriptors in values.
  iree_host_size_t count;
} loom_input_format_list_t;

extern const loom_input_format_t loom_input_text_format;
extern const loom_input_format_t loom_input_bytecode_format;

// Selects an explicit format or filename suffix. Unknown extensions fail;
// extensionless input defaults to Loom text. The optional ordinal identifies
// the selected format in [text, bytecode, ...formats] order.
iree_status_t loom_input_format_select(loom_input_format_list_t formats,
                                       iree_string_view_t name,
                                       iree_string_view_t path,
                                       const loom_input_format_t** out_format,
                                       iree_host_size_t* out_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_INPUT_FORMAT_H_
