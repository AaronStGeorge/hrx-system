// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_defs.h"
#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"

iree_status_t loom_scalar_constant_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_scalar_constant_result(op));
  if (!loom_type_is_scalar(result_type)) {
    return iree_ok_status();
  }

  loom_attribute_t value = loom_scalar_constant_value(op);
  loom_attr_kind_t expected_kind = LOOM_ATTR_ANY;
  if (loom_attr_matches_scalar_type(value, loom_type_element_type(result_type),
                                    &expected_kind)) {
    return iree_ok_status();
  }

  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("value")),
      loom_param_u32(value.kind),
      loom_param_u32(expected_kind),
  };
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = &loom_err_type_005,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}
