// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/symbol_dce.h"

#include "loom/analysis/symbol_dependencies.h"
#include "loom/analysis/symbol_liveness.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

enum {
  LOOM_SYMBOL_DCE_STAT_SYMBOLS_ELIMINATED = 0,
  LOOM_SYMBOL_DCE_STAT_FUNCTIONS_ELIMINATED = 1,
};

static const loom_pass_statistic_def_t kSymbolDCEStatistics[] = {
    {IREE_SVL("symbols-eliminated"),
     IREE_SVL("Number of unreachable symbol definitions removed.")},
    {IREE_SVL("functions-eliminated"),
     IREE_SVL("Number of unreachable private function-like symbols removed.")},
};

static const loom_pass_info_t loom_symbol_dce_pass_info_storage = {
    .name = IREE_SVL("symbol-dce"),
    .description = IREE_SVL("Remove unreachable private symbol definitions."),
    .kind = LOOM_PASS_MODULE,
    .statistic_defs = kSymbolDCEStatistics,
    .statistic_count = IREE_ARRAYSIZE(kSymbolDCEStatistics),
};

const loom_pass_info_t* loom_symbol_dce_pass_info(void) {
  return &loom_symbol_dce_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Reachability
//===----------------------------------------------------------------------===//

typedef struct loom_symbol_dce_state_t {
  // Active pass instance for scratch allocation and statistics.
  loom_pass_t* pass;
  // Module being rewritten.
  loom_module_t* module;
  // Rebuilt module symbol dependency table.
  loom_symbol_dependency_table_t dependencies;
  // Computed live symbol set.
  loom_symbol_liveness_t liveness;
} loom_symbol_dce_state_t;

static bool loom_symbol_dce_symbol_is_erasable(const loom_module_t* module,
                                               const loom_symbol_t* symbol) {
  if (!symbol || !symbol->defining_op) {
    return false;
  }
  if (iree_any_bit_set(symbol->flags, LOOM_SYMBOL_FLAG_PUBLIC)) {
    return false;
  }

  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
    loom_func_like_t function =
        loom_func_like_cast(module, symbol->defining_op);
    if (!loom_func_like_isa(function)) {
      return true;
    }
    if (loom_func_like_visibility(function) != 0) {
      return false;
    }
    if (loom_func_like_export_symbol(function) != LOOM_STRING_ID_INVALID ||
        loom_func_like_export_attrs(function).count > 0 ||
        loom_symbol_ref_is_valid(loom_func_like_artifact(function))) {
      return false;
    }
    return true;
  }

  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_CONFIG)) {
    return true;
  }

  return false;
}

static bool loom_symbol_dce_symbol_is_root(void* user_data,
                                           const loom_module_t* module,
                                           loom_symbol_id_t symbol_id,
                                           const loom_symbol_t* symbol) {
  (void)user_data;
  (void)symbol_id;
  return !loom_symbol_dce_symbol_is_erasable(module, symbol);
}

static iree_status_t loom_symbol_dce_compute_live_symbols(
    loom_symbol_dce_state_t* state) {
  IREE_RETURN_IF_ERROR(loom_symbol_dependency_table_build(
      state->module, state->pass->arena, &state->dependencies));
  loom_symbol_liveness_options_t options = {
      // Encodings are module-table records that serialize with the module.
      // Until there is encoding-table DCE, their symbol refs are roots.
      .flags = LOOM_SYMBOL_LIVENESS_INCLUDE_MODULE_EDGES,
      .root_query = loom_symbol_dce_symbol_is_root,
  };
  return loom_symbol_liveness_compute(state->module, &state->dependencies,
                                      &options, state->pass->arena,
                                      &state->liveness);
}

//===----------------------------------------------------------------------===//
// Erasure
//===----------------------------------------------------------------------===//

typedef struct loom_symbol_dce_erasure_t {
  // Defining op that should be erased from the module body.
  loom_op_t* op;
  // True when |op| defines a function-like symbol.
  bool is_function_like;
} loom_symbol_dce_erasure_t;

static iree_status_t loom_symbol_dce_collect_erasures(
    loom_symbol_dce_state_t* state, loom_symbol_dce_erasure_t** out_erasures,
    iree_host_size_t* out_erasure_count) {
  *out_erasures = NULL;
  *out_erasure_count = 0;
  if (state->module->symbols.count == 0) return iree_ok_status();

  loom_symbol_dce_erasure_t* erasures = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->pass->arena, state->module->symbols.count, sizeof(*erasures),
      (void**)&erasures));

  const loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(state->module, symbol) {
    uint16_t symbol_id = (uint16_t)(symbol - state->module->symbols.entries);
    if (loom_symbol_liveness_is_live(&state->liveness, symbol_id)) continue;
    if (!loom_symbol_dce_symbol_is_erasable(state->module, symbol)) continue;

    const bool is_function_like =
        loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE);
    erasures[(*out_erasure_count)++] = (loom_symbol_dce_erasure_t){
        .op = symbol->defining_op,
        .is_function_like = is_function_like,
    };
  }

  *out_erasures = erasures;
  return iree_ok_status();
}

static iree_status_t loom_symbol_dce_erase_unreachable_symbols(
    loom_symbol_dce_state_t* state) {
  loom_symbol_dce_erasure_t* erasures = NULL;
  iree_host_size_t erasure_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_symbol_dce_collect_erasures(state, &erasures, &erasure_count));

  for (iree_host_size_t i = 0; i < erasure_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_op_erase(state->module, erasures[i].op));
    loom_pass_mark_changed(state->pass);
    loom_pass_statistic_add(state->pass,
                            LOOM_SYMBOL_DCE_STAT_SYMBOLS_ELIMINATED, 1);
    if (erasures[i].is_function_like) {
      loom_pass_statistic_add(state->pass,
                              LOOM_SYMBOL_DCE_STAT_FUNCTIONS_ELIMINATED, 1);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_symbol_dce_run(loom_pass_t* pass, loom_module_t* module) {
  loom_symbol_dce_state_t state = {
      .pass = pass,
      .module = module,
  };
  IREE_RETURN_IF_ERROR(loom_symbol_dce_compute_live_symbols(&state));
  IREE_RETURN_IF_ERROR(loom_symbol_dce_erase_unreachable_symbols(&state));
  return loom_module_compact_symbols(module, pass->arena, NULL);
}
