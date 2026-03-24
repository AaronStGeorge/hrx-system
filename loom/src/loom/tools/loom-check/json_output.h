// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// JSON output for loom-check --json mode.
//
// Writes a structured JSON object for each processed file, containing
// per-case outcomes, the file's default mode, and a summary of counts.
// All string values are properly escaped via loom/util/json.h. Output
// is written directly to a loom_output_stream_t — no intermediate
// buffer, no file I/O (the caller decides where the stream goes).
//
// Output schema:
//
//   {
//     "file": "<filename>",
//     "default_mode": "roundtrip",
//     "cases": [
//       {
//         "index": 1,
//         "mode": "roundtrip",
//         "xfail": false,
//         "raw_outcome": "pass",
//         "final_outcome": "pass",
//         "detail": ""
//       }
//     ],
//     "summary": {
//       "total": 1,
//       "passed": 1,
//       "failed": 0,
//       "skipped": 0
//     }
//   }

#ifndef LOOM_TOOLS_LOOM_CHECK_JSON_OUTPUT_H_
#define LOOM_TOOLS_LOOM_CHECK_JSON_OUTPUT_H_

#include "iree/base/api.h"
#include "loom/tools/loom-check/check.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Writes the JSON output for a processed file. The caller provides all
// results and the stream to write to. Does not do file I/O — the
// caller can direct the stream to stdout, a string builder, or wherever.
//
// |filename| appears as the "file" field. |file| provides case metadata
// (mode, xfail, etc.). |results| is a parallel array of per-case
// outcomes. The pass/fail/skip counts are provided directly by the
// caller (rather than recomputed) to stay consistent with the text
// output path.
iree_status_t loom_check_json_write_file_result(
    iree_string_view_t filename, const loom_check_file_t* file,
    const loom_check_result_t* results, iree_host_size_t pass_count,
    iree_host_size_t fail_count, iree_host_size_t skip_count,
    loom_output_stream_t* stream);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_JSON_OUTPUT_H_
