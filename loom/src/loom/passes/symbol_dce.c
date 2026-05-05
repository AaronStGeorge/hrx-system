// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/passes/symbol_dce.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
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
    .description = IREE_SVL("Remove unreachable private function symbols."),
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

typedef struct loom_symbol_dce_worklist_t {
  // Symbol ids still waiting for defining-op traversal.
  uint16_t* entries;
  // Number of queued symbol ids.
  iree_host_size_t count;
  // Capacity of entries.
  iree_host_size_t capacity;
} loom_symbol_dce_worklist_t;

typedef struct loom_symbol_dce_state_t {
  // Active pass instance for scratch allocation and statistics.
  loom_pass_t* pass;
  // Module being rewritten.
  loom_module_t* module;
  // One byte per module symbol: non-zero means reachable from a retained root.
  uint8_t* live_symbols;
  // Pending reachable symbols whose defining op still needs to be scanned.
  loom_symbol_dce_worklist_t worklist;
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

  return false;
}

static iree_status_t loom_symbol_dce_worklist_initialize(
    iree_arena_allocator_t* arena, iree_host_size_t initial_capacity,
    loom_symbol_dce_worklist_t* worklist) {
  worklist->count = 0;
  worklist->capacity = iree_max(initial_capacity, (iree_host_size_t)16);
  return iree_arena_allocate_array(arena, worklist->capacity,
                                   sizeof(*worklist->entries),
                                   (void**)&worklist->entries);
}

static iree_status_t loom_symbol_dce_mark_symbol_id(
    loom_symbol_dce_state_t* state, uint16_t symbol_id) {
  if (symbol_id >= state->module->symbols.count) return iree_ok_status();
  if (state->live_symbols[symbol_id]) return iree_ok_status();
  if (state->worklist.count >= state->worklist.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->pass->arena, state->worklist.count, state->worklist.count + 1,
        sizeof(*state->worklist.entries), &state->worklist.capacity,
        (void**)&state->worklist.entries));
  }
  state->live_symbols[symbol_id] = 1;
  state->worklist.entries[state->worklist.count++] = symbol_id;
  return iree_ok_status();
}

static iree_status_t loom_symbol_dce_mark_symbol_ref(
    loom_symbol_dce_state_t* state, loom_symbol_ref_t ref) {
  if (!loom_symbol_ref_is_valid(ref)) return iree_ok_status();
  if (ref.module_id != 0) return iree_ok_status();
  return loom_symbol_dce_mark_symbol_id(state, ref.symbol_id);
}

static iree_status_t loom_symbol_dce_mark_attr_refs(
    loom_symbol_dce_state_t* state, const loom_attribute_t* attr) {
  if (!attr) return iree_ok_status();
  if (attr->kind == LOOM_ATTR_SYMBOL) {
    return loom_symbol_dce_mark_symbol_ref(state, attr->symbol);
  }
  if (attr->kind != LOOM_ATTR_DICT || attr->count == 0) {
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < attr->count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_symbol_dce_mark_attr_refs(state, &attr->dict_entries[i].value));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_dce_mark_named_attr_refs(
    loom_symbol_dce_state_t* state, const loom_named_attr_t* attrs,
    iree_host_size_t attr_count) {
  for (iree_host_size_t i = 0; i < attr_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_symbol_dce_mark_attr_refs(state, &attrs[i].value));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_dce_mark_op_attr_refs(
    loom_symbol_dce_state_t* state, const loom_op_t* op) {
  const loom_attribute_t* attrs = loom_op_const_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_symbol_dce_mark_attr_refs(state, &attrs[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_dce_mark_region_refs(
    loom_symbol_dce_state_t* state, const loom_region_t* region) {
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(loom_symbol_dce_mark_op_attr_refs(state, op));
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (regions[i]) {
          IREE_RETURN_IF_ERROR(
              loom_symbol_dce_mark_region_refs(state, regions[i]));
        }
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_dce_mark_module_encoding_refs(
    loom_symbol_dce_state_t* state) {
  for (iree_host_size_t i = 0; i < state->module->encodings.count; ++i) {
    const loom_encoding_t* encoding = &state->module->encodings.entries[i];
    IREE_RETURN_IF_ERROR(loom_symbol_dce_mark_named_attr_refs(
        state, encoding->attributes, encoding->attribute_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_dce_traverse_symbol(
    loom_symbol_dce_state_t* state, uint16_t symbol_id) {
  if (symbol_id >= state->module->symbols.count) return iree_ok_status();
  const loom_symbol_t* symbol = &state->module->symbols.entries[symbol_id];
  const loom_op_t* op = symbol->defining_op;
  if (!op) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_symbol_dce_mark_op_attr_refs(state, op));
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (regions[i]) {
      IREE_RETURN_IF_ERROR(loom_symbol_dce_mark_region_refs(state, regions[i]));
    }
  }

  return iree_ok_status();
}

static iree_status_t loom_symbol_dce_seed_roots(
    loom_symbol_dce_state_t* state) {
  const loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(state->module, symbol) {
    if (!symbol->defining_op) continue;
    if (loom_symbol_dce_symbol_is_erasable(state->module, symbol)) continue;
    uint16_t symbol_id = (uint16_t)(symbol - state->module->symbols.entries);
    IREE_RETURN_IF_ERROR(loom_symbol_dce_mark_symbol_id(state, symbol_id));
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_dce_compute_live_symbols(
    loom_symbol_dce_state_t* state) {
  IREE_RETURN_IF_ERROR(loom_symbol_dce_seed_roots(state));
  // Encodings are module-table records that serialize with the module. Until
  // there is encoding-table DCE, their symbol refs are part of the root set.
  IREE_RETURN_IF_ERROR(loom_symbol_dce_mark_module_encoding_refs(state));
  while (state->worklist.count > 0) {
    uint16_t symbol_id = state->worklist.entries[--state->worklist.count];
    IREE_RETURN_IF_ERROR(loom_symbol_dce_traverse_symbol(state, symbol_id));
  }
  return iree_ok_status();
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
    if (state->live_symbols[symbol_id]) continue;
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
  if (module->symbols.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        pass->arena, module->symbols.count, sizeof(*state.live_symbols),
        (void**)&state.live_symbols));
    memset(state.live_symbols, 0,
           module->symbols.count * sizeof(*state.live_symbols));
  }
  IREE_RETURN_IF_ERROR(loom_symbol_dce_worklist_initialize(
      pass->arena, module->symbols.count, &state.worklist));
  IREE_RETURN_IF_ERROR(loom_symbol_dce_compute_live_symbols(&state));
  IREE_RETURN_IF_ERROR(loom_symbol_dce_erase_unreachable_symbols(&state));
  return loom_module_compact_symbols(module, pass->arena, NULL);
}
