// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/func_like_facts.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/facts.h"

static iree_status_t loom_func_like_string_from_id(
    const loom_module_t* module, loom_string_id_t string_id,
    iree_string_view_t field_name, iree_string_view_t* out_string) {
  *out_string = iree_string_view_empty();
  if (string_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function-like %.*s string id %u is invalid",
                            (int)field_name.size, field_name.data,
                            (uint32_t)string_id);
  }
  *out_string = module->strings.entries[string_id];
  return iree_ok_status();
}

static iree_status_t loom_func_like_resolve_target_profile(
    loom_symbol_fact_context_t* context, loom_symbol_ref_t target_symbol,
    const loom_target_profile_symbol_facts_t** out_target_profile) {
  *out_target_profile = NULL;
  if (!loom_symbol_ref_is_valid(target_symbol)) return iree_ok_status();

  const loom_symbol_facts_base_t* target_base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_context_lookup_ref(
      context, target_symbol, &target_base_facts));
  const loom_target_profile_symbol_facts_t* target_profile =
      loom_target_profile_symbol_facts_cast(target_base_facts);
  if (!target_profile) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "function-like target must resolve to target.profile symbol facts");
  }
  *out_target_profile = target_profile;
  return iree_ok_status();
}

static iree_status_t loom_func_like_symbol_fact_compute(
    const loom_symbol_fact_domain_t* domain,
    loom_symbol_fact_context_t* context, const loom_module_t* module,
    loom_symbol_id_t symbol_id, const loom_symbol_t* symbol,
    const loom_symbol_facts_base_t** out_facts) {
  IREE_ASSERT_ARGUMENT(domain);
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(out_facts);
  *out_facts = NULL;

  loom_func_like_t function = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(function)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "function-like symbol fact domain attached to a non-FuncLike op");
  }

  loom_func_like_symbol_facts_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_context_allocate(
      context, sizeof(*facts), (void**)&facts));
  memset(facts, 0, sizeof(*facts));

  facts->base.domain = domain;
  facts->base.symbol_kind = symbol->kind;
  facts->function_op = function.op;
  facts->symbol = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = symbol_id,
  };
  IREE_RETURN_IF_ERROR(loom_func_like_string_from_id(
      module, symbol->name_id, IREE_SV("symbol"), &facts->name));
  facts->visibility = loom_func_like_visibility(function);
  facts->calling_convention = loom_func_like_cc(function);
  facts->purity = loom_func_like_purity(function);
  facts->has_body = loom_func_like_body(function) != NULL;
  facts->argument_ids =
      loom_func_like_arg_ids(function, &facts->argument_count);
  facts->result_ids = loom_op_const_results(function.op);
  facts->result_count = function.op->result_count;
  facts->target_symbol = loom_func_like_target(function);
  IREE_RETURN_IF_ERROR(loom_func_like_resolve_target_profile(
      context, facts->target_symbol, &facts->target_profile));

  if (facts->target_profile) {
    facts->target_bundle = &facts->target_profile->bundle;
    facts->export_plan = facts->target_profile->export_plan;
    facts->export_plan.name = facts->name;
    facts->export_plan.source_symbol = iree_string_view_empty();
    facts->export_plan.export_symbol = iree_string_view_empty();
  }

  *out_facts = &facts->base;
  return iree_ok_status();
}

const loom_symbol_fact_domain_t loom_func_like_symbol_fact_domain = {
    .compute = loom_func_like_symbol_fact_compute,
};

const loom_func_like_symbol_facts_t* loom_func_like_symbol_facts_cast(
    const loom_symbol_facts_base_t* facts) {
  if (!facts || facts->domain != &loom_func_like_symbol_fact_domain) {
    return NULL;
  }
  return (const loom_func_like_symbol_facts_t*)facts;
}
