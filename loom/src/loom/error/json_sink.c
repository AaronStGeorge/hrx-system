// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/json_sink.h"

#include <inttypes.h>

#include "loom/error/renderer.h"
#include "loom/util/json.h"

//===----------------------------------------------------------------------===//
// Param value rendering for JSON
//===----------------------------------------------------------------------===//

// Renders a single param value as a JSON value (string or number).
static iree_status_t loom_json_render_param_value(
    const loom_diagnostic_param_t* param, loom_type_formatter_t type_formatter,
    loom_output_stream_t* stream) {
  switch (param->kind) {
    case LOOM_PARAM_STRING:
      return loom_json_write_escaped_string(stream, param->string);
    case LOOM_PARAM_I64:
      return loom_output_stream_write_format(stream, "%" PRId64, param->i64);
    case LOOM_PARAM_U32:
      return loom_output_stream_write_format(stream, "%" PRIu32, param->u32);
    case LOOM_PARAM_BOOL:
      return loom_output_stream_write_cstring(
          stream, param->boolean ? "true" : "false");
    case LOOM_PARAM_TYPE: {
      // Render the type through the JSON-escaping adapter directly
      // into the output stream. Zero allocations.
      loom_json_escape_stream_t escape_data;
      loom_output_stream_t escape_stream;
      loom_json_escape_stream_init(stream, &escape_data, &escape_stream);
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
      if (type_formatter.fn) {
        IREE_RETURN_IF_ERROR(type_formatter.fn(
            param->type, type_formatter.user_data, &escape_stream));
      } else {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(&escape_stream, "<type>"));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
      return iree_ok_status();
    }
    default:
      return loom_json_write_escaped_cstring(stream, "<?>");
  }
}

//===----------------------------------------------------------------------===//
// JSON sink implementation
//===----------------------------------------------------------------------===//

iree_status_t loom_diagnostic_json_sink(void* user_data,
                                        const loom_diagnostic_t* diagnostic) {
  loom_json_sink_options_t* options = (loom_json_sink_options_t*)user_data;
  loom_output_stream_t* stream = options->stream;

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));

  // Severity (always present).
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "\"severity\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, loom_diagnostic_severity_name(diagnostic->severity)));

  const loom_error_def_t* error = diagnostic->error;

  // Domain.
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"domain\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, loom_error_domain_name(error->domain)));

  // Code.
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"code\":%" PRIu16, error->code));

  // Emitter.
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"emitter\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, loom_emitter_name(diagnostic->emitter)));

  // Message: rendered from the error def's template and params, streamed
  // through the JSON-escaping adapter directly to the output. Zero allocs.
  {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"message\":\""));
    loom_json_escape_stream_t escape_data;
    loom_output_stream_t escape_stream;
    loom_json_escape_stream_init(stream, &escape_data, &escape_stream);
    IREE_RETURN_IF_ERROR(loom_diagnostic_render_message(
        error, diagnostic->params, diagnostic->param_count,
        options->type_formatter, &escape_stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  }

  // Fix hint (when present).
  if (error->fix_hint_template) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"fix_hint\":\""));
    loom_json_escape_stream_t escape_data;
    loom_output_stream_t escape_stream;
    loom_json_escape_stream_init(stream, &escape_data, &escape_stream);
    IREE_RETURN_IF_ERROR(loom_diagnostic_render_fix_hint(
        error, diagnostic->params, diagnostic->param_count,
        options->type_formatter, &escape_stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  }

  // Params object (when present). Clamp to the schema length to avoid
  // OOB reads on param_defs if a diagnostic carries more runtime params
  // than the error definition declares.
  iree_host_size_t emit_param_count =
      (diagnostic->params && error->param_defs)
          ? iree_min(diagnostic->param_count, error->param_count)
          : 0;
  if (emit_param_count > 0) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"params\":{"));
    for (iree_host_size_t i = 0; i < emit_param_count; ++i) {
      if (i > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
      }
      // Validate that the runtime param kind matches the schema.
      if (diagnostic->params[i].kind != error->param_defs[i].kind) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "diagnostic param %zu kind mismatch: runtime %d vs schema %d",
            (size_t)i, (int)diagnostic->params[i].kind,
            (int)error->param_defs[i].kind);
      }
      // Param name.
      IREE_RETURN_IF_ERROR(
          loom_json_write_escaped_cstring(stream, error->param_defs[i].name));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ":"));
      // Param value.
      IREE_RETURN_IF_ERROR(loom_json_render_param_value(
          &diagnostic->params[i], options->type_formatter, stream));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  }

  // Close object and newline.
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}\n"));

  return iree_ok_status();
}
