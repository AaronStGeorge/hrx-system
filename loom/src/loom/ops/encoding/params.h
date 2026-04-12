// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Helpers for viewing encoding.define static and dynamic parameters together.

#ifndef LOOM_OPS_ENCODING_PARAMS_H_
#define LOOM_OPS_ENCODING_PARAMS_H_

#include "loom/ir/module.h"
#include "loom/ops/encoding/ops.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_encoding_define_param_view_t {
  // Static encoding table entry referenced by the `spec` attribute.
  const loom_encoding_t* spec;

  // Static family parameters from `spec`, sorted by parameter name.
  loom_named_attr_slice_t static_attrs;

  // Dynamic parameter operand values from encoding.define.
  loom_value_slice_t dynamic_values;

  // Dynamic parameter names, sorted by name and mapped to dynamic_values
  // ordinals with i64 attributes.
  loom_named_attr_slice_t dynamic_names;
} loom_encoding_define_param_view_t;

static inline loom_encoding_define_param_view_t loom_encoding_define_param_view(
    const loom_module_t* module, const loom_op_t* op) {
  const loom_encoding_t* spec =
      loom_module_encoding(module, loom_encoding_define_spec(op));
  loom_encoding_define_param_view_t view = {
      .spec = spec,
      .dynamic_values = loom_encoding_define_params(op),
      .dynamic_names = loom_encoding_define_param_names(op),
  };
  if (spec) {
    view.static_attrs = loom_encoding_attrs(spec);
  }
  return view;
}

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_ENCODING_PARAMS_H_
