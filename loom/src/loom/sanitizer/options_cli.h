// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Text option parsing for sanitizer check sets.

#ifndef LOOM_SANITIZER_OPTIONS_CLI_H_
#define LOOM_SANITIZER_OPTIONS_CLI_H_

#include "iree/base/api.h"
#include "loom/sanitizer/options.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parses a sanitizer check set such as "access", "value|operation", "all",
// or "none". |diagnostic_name| is included in parse failures and should name
// the option being parsed, such as "--sanitizer" or "pass option 'checks'".
iree_status_t loom_sanitizer_checks_parse(iree_string_view_t value,
                                          iree_string_view_t diagnostic_name,
                                          loom_sanitizer_checks_t* out_checks);

// Parses a sanitizer option check set with no behavior flags. |checks == 0|
// returns disabled sanitizer options.
iree_status_t loom_sanitizer_options_parse_checks(
    iree_string_view_t value, iree_string_view_t diagnostic_name,
    loom_sanitizer_options_t* out_options);

// Formats a sanitizer check set in canonical pass-pipeline spelling.
iree_status_t loom_sanitizer_checks_format(loom_sanitizer_checks_t checks,
                                           iree_string_view_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SANITIZER_OPTIONS_CLI_H_
