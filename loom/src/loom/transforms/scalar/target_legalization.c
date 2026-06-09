// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scalar/target_legalization.h"

#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/rewrite/rewriter.h"

static iree_status_t loom_scalar_legalize_fmai(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_scalar_fmai_result(op));

  // Fused no-wrap facts do not prove that the decomposed multiply and add can
  // each carry independent no-wrap flags, so the reference split is
  // conservative.
  loom_op_t* product_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_muli_build(
      &rewriter->builder, 0, loom_scalar_fmai_a(op), loom_scalar_fmai_b(op),
      result_type, op->location, &product_op));
  loom_op_t* add_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_addi_build(
      &rewriter->builder, 0, loom_scalar_fmai_c(op),
      loom_scalar_muli_result(product_op), result_type, op->location, &add_op));

  loom_value_id_t replacement = loom_scalar_addi_result(add_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static const loom_target_legalizer_entry_t kScalarLegalizerEntries[] = {
    {
        .root_kind = LOOM_OP_SCALAR_FMAI,
        .legalize = loom_scalar_legalize_fmai,
    },
};

static const loom_target_legalizer_provider_t kScalarLegalizerProvider = {
    .name = IREE_SVL("scalar"),
    .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE,
    .entries = kScalarLegalizerEntries,
    .entry_count = IREE_ARRAYSIZE(kScalarLegalizerEntries),
};

const loom_target_legalizer_provider_t* loom_scalar_target_legalizer_provider(
    void) {
  return &kScalarLegalizerProvider;
}
