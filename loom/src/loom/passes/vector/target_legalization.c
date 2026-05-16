// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/passes/vector/target_legalization.h"

#include "loom/ops/vector/ops.h"
#include "loom/passes/vector/to_scalar.h"

static iree_status_t loom_vector_legalize_reduce_axes(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_reduce_axes_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static const loom_target_legalizer_entry_t kVectorLegalizerEntries[] = {
    {
        .root_kind = LOOM_OP_VECTOR_REDUCE_AXES,
        .legalize = loom_vector_legalize_reduce_axes,
    },
};

static const loom_target_legalizer_provider_t kVectorLegalizerProvider = {
    .name = IREE_SVL("vector"),
    .entries = kVectorLegalizerEntries,
    .entry_count = IREE_ARRAYSIZE(kVectorLegalizerEntries),
};

const loom_target_legalizer_provider_t* loom_vector_target_legalizer_provider(
    void) {
  return &kVectorLegalizerProvider;
}
