// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/source_selection.h"

#include <string.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/target/facts.h"
#include "loom/target/function_contract.h"

static iree_status_t loom_low_source_selection_lookup_func_facts(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    loom_symbol_id_t symbol_id,
    const loom_func_symbol_facts_t** out_func_facts) {
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup(fact_table, module,
                                                     symbol_id, &base_facts));
  *out_func_facts = loom_func_symbol_facts_cast(base_facts);
  return iree_ok_status();
}

static iree_status_t loom_low_source_selection_try_func(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    loom_symbol_fact_table_t* fact_table, loom_symbol_id_t symbol_id,
    bool* out_compatible, loom_low_source_selection_t* out_selection) {
  *out_compatible = false;
  const loom_func_symbol_facts_t* func_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_low_source_selection_lookup_func_facts(
      module, fact_table, symbol_id, &func_facts));
  if (!func_facts || !func_facts->has_body) {
    return iree_ok_status();
  }
  if (!loom_symbol_ref_is_valid(func_facts->target_symbol)) {
    return iree_ok_status();
  }
  bool contract_valid = false;
  IREE_RETURN_IF_ERROR(loom_target_function_contract_resolve(
      module, fact_table, func_facts, options->diagnostic_emitter,
      &contract_valid, &out_selection->target_bundle_storage));
  if (!contract_valid) {
    return iree_ok_status();
  }
  out_selection->target_bundle = &out_selection->target_bundle_storage.bundle;
  const loom_low_lower_policy_t* policy =
      loom_low_lower_policy_registry_lookup_for_bundle(
          options->policy_registry, out_selection->target_bundle);
  if (policy == NULL) {
    return iree_ok_status();
  }

  out_selection->func = loom_func_like_cast(module, func_facts->func_op);
  out_selection->func_facts = func_facts;
  out_selection->target_ref = func_facts->target_symbol;
  out_selection->policy = policy;
  *out_compatible = true;
  return iree_ok_status();
}

static void loom_low_source_selection_assign(
    const loom_low_source_selection_t* source,
    loom_low_source_selection_t* out_selection) {
  *out_selection = *source;
  loom_target_bundle_storage_rebind(&out_selection->target_bundle_storage);
  out_selection->target_bundle = &out_selection->target_bundle_storage.bundle;
}

iree_status_t loom_low_select_source_funcs(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_source_selection_list_t* out_selection_list) {
  *out_selection_list = (loom_low_source_selection_list_t){0};
  loom_symbol_fact_table_t fact_table = {0};
  loom_symbol_fact_table_initialize(&fact_table, arena);
  if (module->symbols.count == 0) {
    return iree_ok_status();
  }

  loom_low_source_selection_t* selections = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*selections), (void**)&selections));
  iree_host_size_t selection_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    bool compatible = false;
    loom_low_source_selection_t candidate = {0};
    IREE_RETURN_IF_ERROR(loom_low_source_selection_try_func(
        module, options, &fact_table, (loom_symbol_id_t)i, &compatible,
        &candidate));
    if (!compatible) {
      continue;
    }
    loom_low_source_selection_assign(&candidate, &selections[selection_count]);
    ++selection_count;
  }

  out_selection_list->values = selections;
  out_selection_list->count = selection_count;
  return iree_ok_status();
}
