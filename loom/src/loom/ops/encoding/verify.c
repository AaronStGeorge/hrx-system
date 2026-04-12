// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_defs.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/params.h"

static iree_status_t loom_encoding_emit(iree_diagnostic_emitter_t emitter,
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

static iree_status_t loom_encoding_define_emit_duplicate_static_dynamic_param(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_string_id_t name_id) {
  iree_string_view_t param_name = module->strings.entries[name_id];
  loom_diagnostic_param_t params[] = {
      loom_param_string(param_name),
  };
  return loom_encoding_emit(emitter, op, &loom_err_encoding_006, params,
                            IREE_ARRAYSIZE(params));
}

iree_status_t loom_encoding_define_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  loom_encoding_define_param_view_t params =
      loom_encoding_define_param_view(module, op);
  if (!params.spec) return iree_ok_status();
  if (params.dynamic_names.count == 0) return iree_ok_status();

  for (iree_host_size_t static_index = 0;
       static_index < params.static_attrs.count; ++static_index) {
    loom_string_id_t static_name_id =
        params.static_attrs.entries[static_index].name_id;
    for (iree_host_size_t dynamic_index = 0;
         dynamic_index < params.dynamic_names.count; ++dynamic_index) {
      if (params.dynamic_names.entries[dynamic_index].name_id ==
          static_name_id) {
        return loom_encoding_define_emit_duplicate_static_dynamic_param(
            module, op, emitter, static_name_id);
      }
    }
  }

  return iree_ok_status();
}
