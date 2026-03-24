// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/json.h"

//===----------------------------------------------------------------------===//
// JSON-escaping stream adapter
//===----------------------------------------------------------------------===//

// Flushes a literal run [run_start, cursor) to the inner stream.
static inline iree_status_t loom_json_flush_run(loom_output_stream_t* inner,
                                                const char* run_start,
                                                const char* cursor) {
  if (cursor > run_start) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        inner, iree_make_string_view(run_start,
                                     (iree_host_size_t)(cursor - run_start))));
  }
  return iree_ok_status();
}

// Write callback that JSON-escapes all incoming text per RFC 8259.
//
// Walks the input byte by byte, accumulating literal runs and flushing
// escape sequences. Multi-byte UTF-8 passes through unchanged except
// for U+2028 and U+2029 which are escaped for JavaScript/HTML safety.
static iree_status_t loom_json_escape_write(void* user_data,
                                            iree_string_view_t text) {
  loom_output_stream_t* inner = ((loom_json_escape_stream_t*)user_data)->inner;
  const char* run_start = text.data;
  const char* end = text.data + text.size;
  for (const char* cursor = text.data; cursor < end; ++cursor) {
    const char* escape = NULL;
    switch (*cursor) {
      case '"':
        escape = "\\\"";
        break;
      case '\\':
        escape = "\\\\";
        break;
      case '\b':
        escape = "\\b";
        break;
      case '\f':
        escape = "\\f";
        break;
      case '\n':
        escape = "\\n";
        break;
      case '\r':
        escape = "\\r";
        break;
      case '\t':
        escape = "\\t";
        break;
      default:
        if ((unsigned char)*cursor < 0x20) {
          // ASCII control character — emit as \uNNNN.
          IREE_RETURN_IF_ERROR(loom_json_flush_run(inner, run_start, cursor));
          IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
              inner, "\\u%04x", (unsigned)(unsigned char)*cursor));
          run_start = cursor + 1;
        } else if ((unsigned char)*cursor == 0xE2 && cursor + 2 < end &&
                   (unsigned char)cursor[1] == 0x80 &&
                   ((unsigned char)cursor[2] == 0xA8 ||
                    (unsigned char)cursor[2] == 0xA9)) {
          // U+2028 LINE SEPARATOR (E2 80 A8) or
          // U+2029 PARAGRAPH SEPARATOR (E2 80 A9).
          // These are valid JSON per RFC 8259 but break JavaScript string
          // literals and HTML <script> blocks. Escape them for safety.
          IREE_RETURN_IF_ERROR(loom_json_flush_run(inner, run_start, cursor));
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
              inner,
              ((unsigned char)cursor[2] == 0xA8) ? "\\u2028" : "\\u2029"));
          cursor += 2;  // Skip the two continuation bytes.
          run_start = cursor + 1;
        }
        // All other bytes (printable ASCII, multi-byte UTF-8) pass through.
        continue;
    }
    // Flush the literal run before this escaped character.
    IREE_RETURN_IF_ERROR(loom_json_flush_run(inner, run_start, cursor));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(inner, escape));
    run_start = cursor + 1;
  }
  // Flush any trailing literal run.
  IREE_RETURN_IF_ERROR(loom_json_flush_run(inner, run_start, end));
  return iree_ok_status();
}

void loom_json_escape_stream_init(loom_output_stream_t* inner,
                                  loom_json_escape_stream_t* escape_data,
                                  loom_output_stream_t* out_stream) {
  escape_data->inner = inner;
  out_stream->write = loom_json_escape_write;
  out_stream->user_data = escape_data;
  out_stream->offset = 0;
}

//===----------------------------------------------------------------------===//
// Convenience: quoted + escaped string writes
//===----------------------------------------------------------------------===//

iree_status_t loom_json_write_escaped_string(loom_output_stream_t* stream,
                                             iree_string_view_t value) {
  loom_json_escape_stream_t escape_data;
  loom_output_stream_t escape_stream;
  loom_json_escape_stream_init(stream, &escape_data, &escape_stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(&escape_stream, value));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  return iree_ok_status();
}
