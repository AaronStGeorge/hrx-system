// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_defs.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/util/fact_table.h"

static iree_status_t loom_buffer_emit(iree_diagnostic_emitter_t emitter,
                                      const loom_op_t* op,
                                      const loom_error_def_t* error,
                                      const loom_diagnostic_param_t* params,
                                      iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_buffer_verify_strided_layout_rank(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_type_t result_type,
    loom_value_id_t layout_id) {
  loom_value_facts_t static_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {
      0};
  loom_value_fact_address_layout_t layout = {0};
  if (!loom_encoding_query_value_address_layout(
          module, layout_id, static_strides, IREE_ARRAYSIZE(static_strides),
          &layout)) {
    return iree_ok_status();
  }
  if (layout.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) {
    return iree_ok_status();
  }

  uint8_t result_rank = loom_type_rank(result_type);
  if (layout.rank == result_rank) return iree_ok_status();

  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("result type")),
      loom_param_i64(result_rank),
      loom_param_string(IREE_SV("layout stride list")),
      loom_param_i64(layout.rank),
  };
  return loom_buffer_emit(emitter, op, &loom_err_shape_001, params,
                          IREE_ARRAYSIZE(params));
}

iree_status_t loom_buffer_view_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_buffer_view_result(op));
  if (!loom_type_is_view(result_type)) {
    return iree_ok_status();
  }

  if (!loom_type_has_encoding(result_type)) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("result type layout")),
        loom_param_string(IREE_SV("view result type")),
    };
    return loom_buffer_emit(emitter, op, &loom_err_encoding_001, params,
                            IREE_ARRAYSIZE(params));
  }

  if (!loom_type_has_ssa_encoding(result_type)) return iree_ok_status();

  loom_value_id_t layout_id = loom_type_encoding_value_id(result_type);
  return loom_buffer_verify_strided_layout_rank(module, op, emitter,
                                                result_type, layout_id);
}
