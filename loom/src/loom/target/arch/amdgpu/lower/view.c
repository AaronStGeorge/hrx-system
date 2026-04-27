// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/lower/internal.h"

iree_status_t loom_amdgpu_select_view_plan(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_low_lower_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = loom_low_lower_plan_empty();
  switch (source_op->kind) {
    case LOOM_OP_VIEW_SUBVIEW:
      if (loom_amdgpu_value_is_byte_addressable_view(
              context, loom_view_subview_result(source_op))) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, NULL);
      }
      return iree_ok_status();
    default:
      return iree_ok_status();
  }
}

iree_status_t loom_amdgpu_lower_view_op(loom_low_lower_context_t* context,
                                        const loom_op_t* source_op) {
  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  switch (source_op->kind) {
    case LOOM_OP_VIEW_SUBVIEW:
      source = loom_view_subview_source(source_op);
      result = loom_view_subview_result(source_op);
      break;
    default:
      IREE_ASSERT_UNREACHABLE();
      return iree_ok_status();
  }

  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source, &low_source));
  return loom_low_lower_bind_value(context, result, low_source);
}
