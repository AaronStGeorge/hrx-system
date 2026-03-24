// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/transforms/rewriter.h"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Checks if the given value is defined by a scalar.constant op with the
// specified integer value.
static bool loom_scalar_is_constant_i64(loom_rewriter_t* rewriter,
                                        loom_value_id_t value_id,
                                        int64_t expected_value) {
  loom_value_t* value = loom_module_value(rewriter->module, value_id);
  if (loom_value_is_block_arg(value)) return false;

  loom_op_t* def = loom_value_def_op(value);
  if (!def || !loom_scalar_constant_isa(def)) return false;

  int64_t actual_value = loom_attr_as_i64(loom_op_attrs(def)[0]);
  return actual_value == expected_value;
}

//===----------------------------------------------------------------------===//
// scalar.fmai canonicalization: fmai(a, b, c) = a*b + c
//===----------------------------------------------------------------------===//

// Pattern: fmai(x, 0, c) -> c or fmai(0, x, c) -> c.
// When either multiplicand is zero, the product is zero, so the result
// is just the addend c.
static iree_status_t loom_scalar_fmai_try_fold_zero_multiplier(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  loom_value_id_t a = loom_scalar_fmai_a(op);
  loom_value_id_t b = loom_scalar_fmai_b(op);
  loom_value_id_t c = loom_scalar_fmai_c(op);

  if (!loom_scalar_is_constant_i64(rewriter, a, 0) &&
      !loom_scalar_is_constant_i64(rewriter, b, 0)) {
    *out_changed = false;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &c, 1));
  *out_changed = true;
  return iree_ok_status();
}

// Pattern: fmai(a, b, 0) -> muli(a, b).
// When the addend is zero, this is just a multiply.
static iree_status_t loom_scalar_fmai_try_fold_zero_addend(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  loom_value_id_t a = loom_scalar_fmai_a(op);
  loom_value_id_t b = loom_scalar_fmai_b(op);
  loom_value_id_t c = loom_scalar_fmai_c(op);
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_scalar_fmai_result(op));
  loom_location_id_t location = op->location;

  if (!loom_scalar_is_constant_i64(rewriter, c, 0)) {
    *out_changed = false;
    return iree_ok_status();
  }

  loom_op_t* mul = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_muli_build(&rewriter->builder,
                                              /*instance_flags=*/0, a, b,
                                              result_type, location, &mul));
  loom_value_id_t result = loom_scalar_muli_result(mul);
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &result, 1));
  *out_changed = true;
  return iree_ok_status();
}

// Pattern: fmai(a, 1, c) -> addi(a, c) or fmai(1, b, c) -> addi(b, c).
// When either multiplicand is one, the product is the other operand,
// so this reduces to an add.
static iree_status_t loom_scalar_fmai_try_fold_identity_multiplier(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  loom_value_id_t a = loom_scalar_fmai_a(op);
  loom_value_id_t b = loom_scalar_fmai_b(op);
  loom_value_id_t c = loom_scalar_fmai_c(op);
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_scalar_fmai_result(op));
  loom_location_id_t location = op->location;

  if (loom_scalar_is_constant_i64(rewriter, a, 1)) {
    loom_op_t* add = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_addi_build(&rewriter->builder,
                                                /*instance_flags=*/0, b, c,
                                                result_type, location, &add));
    loom_value_id_t result = loom_scalar_addi_result(add);
    IREE_RETURN_IF_ERROR(
        loom_rewriter_replace_all_uses_and_erase(rewriter, op, &result, 1));
    *out_changed = true;
    return iree_ok_status();
  }

  if (loom_scalar_is_constant_i64(rewriter, b, 1)) {
    loom_op_t* add = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_addi_build(&rewriter->builder,
                                                /*instance_flags=*/0, a, c,
                                                result_type, location, &add));
    loom_value_id_t result = loom_scalar_addi_result(add);
    IREE_RETURN_IF_ERROR(
        loom_rewriter_replace_all_uses_and_erase(rewriter, op, &result, 1));
    *out_changed = true;
    return iree_ok_status();
  }

  *out_changed = false;
  return iree_ok_status();
}

iree_status_t loom_scalar_fmai_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  // Try patterns in order of simplicity (most beneficial first).
  bool changed = false;
  IREE_RETURN_IF_ERROR(
      loom_scalar_fmai_try_fold_zero_multiplier(op, rewriter, &changed));
  if (!changed) {
    IREE_RETURN_IF_ERROR(
        loom_scalar_fmai_try_fold_zero_addend(op, rewriter, &changed));
  }
  if (!changed) {
    IREE_RETURN_IF_ERROR(
        loom_scalar_fmai_try_fold_identity_multiplier(op, rewriter, &changed));
  }
  if (changed) {
    rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  }
  return iree_ok_status();
}
