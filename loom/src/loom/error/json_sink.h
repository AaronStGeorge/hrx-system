// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// JSON diagnostic sink for loom diagnostics.
//
// Emits one JSON object per diagnostic (JSONL format — one object per line)
// to a caller-provided loom_output_stream_t. Every diagnostic is
// structured: all error identity fields, typed parameters, rendered message,
// and fix hints are included. The message is rendered from the error def's
// template.
//
// Output example:
//
//   {"severity":"error","domain":"TYPE","code":1,"emitter":"verifier",
//    "origin":{"filename":"model.loom","start_line":42,"start_column":15,
//              "end_line":42,"end_column":17,"start_byte":128,
//              "end_byte":130},
//    "source_location":{"filename":"model.loom","start_line":42,
//                       "start_column":15,"end_line":42,"end_column":17,
//                       "start_byte":128,"end_byte":130},
//    "highlights":[{"start_byte":128,"end_byte":130}],
//    "message":"'rhs' type f32 does not match 'lhs' type i32",
//    "fix_hint":"Ensure 'rhs' and 'lhs' have the same type",
//    "params":{"field_a":"lhs","type_a":"i32","field_b":"rhs","type_b":"f32"}}
//
// Usage:
//
//   iree_string_builder_t json_output;
//   iree_string_builder_initialize(allocator, &json_output);
//   loom_output_stream_t stream;
//   loom_output_stream_for_builder(&json_output, &stream);
//   loom_json_sink_options_t json_options = {
//       .stream = &stream,
//       .type_formatter = {loom_type_format_minimal, NULL},
//   };
//   loom_diagnostic_sink_t sink = {
//       .fn = loom_diagnostic_json_sink,
//       .user_data = &json_options,
//   };
//   // ... pass sink to parser/verifier ...
//   // json_output now contains one line per diagnostic.

#ifndef LOOM_ERROR_JSON_SINK_H_
#define LOOM_ERROR_JSON_SINK_H_

#include "iree/base/api.h"
#include "loom/error/diagnostic.h"
#include "loom/error/renderer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Options for the JSON sink. Passed as user_data to
// loom_diagnostic_json_sink.
typedef struct loom_json_sink_options_t {
  // Output stream. One JSON object is written per diagnostic,
  // terminated by a newline.
  loom_output_stream_t* stream;

  // Type formatter for rendering TYPE params. When .fn is NULL,
  // TYPE params emit "<type>".
  loom_type_formatter_t type_formatter;
} loom_json_sink_options_t;

// Diagnostic sink callback that emits JSONL output. Pass a
// loom_json_sink_options_t* as user_data.
iree_status_t loom_diagnostic_json_sink(void* user_data,
                                        const loom_diagnostic_t* diagnostic);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ERROR_JSON_SINK_H_
