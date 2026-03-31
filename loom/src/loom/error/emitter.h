// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lightweight structured diagnostic emission callback.
//
// This is the narrow interface between producers that know "which op
// failed with which structured error" and the subsystem that owns the
// final diagnostic sink. Unlike loom_diagnostic_sink_t, this callback
// operates before a loom_diagnostic_t has been materialized.

#ifndef LOOM_ERROR_EMITTER_H_
#define LOOM_ERROR_EMITTER_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_op_t loom_op_t;
typedef struct loom_error_def_t loom_error_def_t;
typedef struct loom_diagnostic_param_t loom_diagnostic_param_t;

// Structured diagnostic emission callback used by verifier hooks and
// other subsystems that want their caller to build the final diagnostic.
typedef iree_status_t (*iree_diagnostic_emitter_fn_t)(
    void* user_data, const loom_op_t* op, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count);

// Callback object pairing the emission function with caller-owned state.
typedef struct iree_diagnostic_emitter_t {
  iree_diagnostic_emitter_fn_t fn;
  void* user_data;
} iree_diagnostic_emitter_t;

// Emits a structured diagnostic through |emitter|. A NULL callback is
// treated as a no-op so optional emitters do not require special cases.
static inline iree_status_t iree_diagnostic_emit(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  if (emitter.fn) {
    return emitter.fn(emitter.user_data, op, error, params, param_count);
  }
  return iree_ok_status();
}

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ERROR_EMITTER_H_
