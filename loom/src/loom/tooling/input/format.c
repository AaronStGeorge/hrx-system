// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/input/format.h"

#include "iree/base/internal/path.h"

iree_status_t loom_input_options_for_provider(iree_string_view_list_t entries,
                                              iree_string_view_t provider,
                                              iree_string_view_t* out_options) {
  *out_options = iree_string_view_empty();
  bool matched = false;
  for (iree_host_size_t i = 0; i < entries.count; ++i) {
    iree_string_view_t name, options;
    if (iree_string_view_split(entries.values[i], ':', &name, &options) < 1) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "input options require 'format:options'");
    }
    if (iree_string_view_equal(name, provider)) {
      if (matched) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate input options for '%.*s'",
                                (int)provider.size, provider.data);
      }
      matched = true;
      *out_options = options;
    }
  }
  return iree_ok_status();
}

static const iree_string_view_t loom_input_text_suffixes[] = {
    IREE_SVL(".loom"),
    IREE_SVL(".loom-test"),
};

const loom_input_format_t loom_input_text_format = {
    .name = IREE_SVL("loom"),
    .suffixes = {IREE_ARRAYSIZE(loom_input_text_suffixes),
                 loom_input_text_suffixes},
};

static const iree_string_view_t loom_input_bytecode_suffixes[] = {
    IREE_SVL(".loombc"),
};

const loom_input_format_t loom_input_bytecode_format = {
    .name = IREE_SVL("loombc"),
    .suffixes = {IREE_ARRAYSIZE(loom_input_bytecode_suffixes),
                 loom_input_bytecode_suffixes},
};

iree_status_t loom_input_format_select(loom_input_format_list_t formats,
                                       iree_string_view_t format,
                                       iree_string_view_t path,
                                       const loom_input_format_t** out_format,
                                       iree_host_size_t* out_ordinal) {
  *out_format = NULL;
  const loom_input_format_t* const builtin_formats[] = {
      &loom_input_text_format, &loom_input_bytecode_format};
  const iree_host_size_t builtin_count = IREE_ARRAYSIZE(builtin_formats);
  for (iree_host_size_t i = 0; i < builtin_count + formats.count; ++i) {
    const loom_input_format_t* candidate =
        i < builtin_count ? builtin_formats[i]
                          : formats.values[i - builtin_count];
    if (!iree_string_view_is_empty(format)) {
      if (iree_string_view_equal(format, candidate->name)) {
        *out_format = candidate;
        if (out_ordinal) {
          *out_ordinal = i;
        }
        return iree_ok_status();
      }
    } else {
      for (iree_host_size_t j = 0; j < candidate->suffixes.count; ++j) {
        if (iree_string_view_ends_with(path, candidate->suffixes.values[j])) {
          *out_format = candidate;
          if (out_ordinal) {
            *out_ordinal = i;
          }
          return iree_ok_status();
        }
      }
    }
  }
  if (!iree_string_view_is_empty(format)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "input format '%.*s' is not linked into this tool",
                            (int)format.size, format.data);
  }
  if (!iree_string_view_is_empty(iree_file_path_extension(path))) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "no linked input provider accepts '%.*s'",
                            (int)path.size, path.data);
  }
  *out_format = &loom_input_text_format;
  if (out_ordinal) {
    *out_ordinal = 0;
  }
  return iree_ok_status();
}
