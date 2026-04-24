// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/source_selection.h"

#include <inttypes.h>
#include <string.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/target/facts.h"
#include "loom/target/function_contract.h"
#include "loom/target/preset_registry.h"

static iree_string_view_t loom_low_source_selection_kind(
    const loom_low_source_selection_options_t* options) {
  if (!options || iree_string_view_is_empty(options->lowering_kind)) {
    return IREE_SV("source-to-low");
  }
  return options->lowering_kind;
}

static iree_status_t loom_low_source_selection_initialize_fact_table(
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_arena_allocator_t* arena, loom_symbol_fact_table_t* out_fact_table) {
  if (descriptor_registry == NULL ||
      descriptor_registry->target_bundle_count == 0 ||
      descriptor_registry->target_bundles == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "source-to-low func selection requires target profile presets");
  }
  loom_target_preset_registry_t* preset_registry = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, sizeof(*preset_registry),
                                           (void**)&preset_registry));
  *preset_registry = (loom_target_preset_registry_t){
      .target_bundles = descriptor_registry->target_bundles,
      .target_bundle_count = descriptor_registry->target_bundle_count,
  };
  loom_symbol_fact_resource_t* resource = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*resource), (void**)&resource));
  *resource = loom_target_profile_preset_registry_resource(preset_registry);
  const loom_symbol_fact_table_options_t fact_options = {
      .resources = loom_make_symbol_fact_resource_list(resource, 1),
  };
  loom_symbol_fact_table_initialize_with_options(out_fact_table, &fact_options,
                                                 arena);
  return iree_ok_status();
}

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
    bool require_compatible, bool* out_compatible,
    loom_low_source_selection_t* out_selection) {
  *out_compatible = false;
  const loom_func_symbol_facts_t* func_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_low_source_selection_lookup_func_facts(
      module, fact_table, symbol_id, &func_facts));
  if (!func_facts || !func_facts->has_body) {
    if (!require_compatible) {
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol is not a func with a body");
  }
  if (!loom_symbol_ref_is_valid(func_facts->target_symbol)) {
    if (!require_compatible) {
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source func @%.*s must declare a target profile",
                            (int)func_facts->name.size, func_facts->name.data);
  }
  IREE_RETURN_IF_ERROR(loom_target_function_contract_resolve(
      module, fact_table, func_facts, &out_selection->target_bundle_storage));
  out_selection->target_bundle = &out_selection->target_bundle_storage.bundle;
  const loom_low_lower_policy_t* policy = NULL;
  iree_status_t status = loom_low_lower_policy_registry_lookup_for_bundle(
      options->policy_registry, out_selection->target_bundle, &policy);
  if (!iree_status_is_ok(status)) {
    if (!require_compatible &&
        iree_status_code(status) == IREE_STATUS_NOT_FOUND) {
      iree_status_free(status);
      return iree_ok_status();
    }
    return status;
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

static iree_status_t loom_low_source_selection_select_single(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    loom_symbol_fact_table_t* fact_table,
    loom_low_source_selection_t* out_selection) {
  iree_host_size_t compatible_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    if (i > UINT16_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "symbol index exceeds source selection range");
    }
    bool compatible = false;
    loom_low_source_selection_t candidate = {0};
    IREE_RETURN_IF_ERROR(loom_low_source_selection_try_func(
        module, options, fact_table, (loom_symbol_id_t)i,
        /*require_compatible=*/false, &compatible, &candidate));
    if (!compatible) {
      continue;
    }
    ++compatible_count;
    if (compatible_count == 1) {
      loom_low_source_selection_assign(&candidate, out_selection);
    }
  }

  iree_string_view_t lowering_kind = loom_low_source_selection_kind(options);
  if (compatible_count == 0) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "module contains no %.*s-compatible func with a target profile",
        (int)lowering_kind.size, lowering_kind.data);
  }
  if (compatible_count > 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module contains %" PRIhsz
                            " %.*s-compatible funcs; expected exactly one",
                            compatible_count, (int)lowering_kind.size,
                            lowering_kind.data);
  }
  return iree_ok_status();
}

iree_status_t loom_low_select_source_funcs(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_source_selection_list_t* out_selection_list) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_selection_list);
  *out_selection_list = (loom_low_source_selection_list_t){0};
  if (options->descriptor_registry == NULL ||
      options->policy_registry == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "source-to-low func selection requires descriptor and policy "
        "registries");
  }
  IREE_RETURN_IF_ERROR(
      loom_low_lower_policy_registry_verify(options->policy_registry));

  loom_symbol_fact_table_t fact_table = {0};
  IREE_RETURN_IF_ERROR(loom_low_source_selection_initialize_fact_table(
      options->descriptor_registry, arena, &fact_table));
  if (module->symbols.count == 0) {
    return iree_ok_status();
  }

  loom_low_source_selection_t* selections = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*selections), (void**)&selections));
  iree_host_size_t selection_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    if (i > UINT16_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "symbol index exceeds source selection range");
    }
    bool compatible = false;
    loom_low_source_selection_t candidate = {0};
    IREE_RETURN_IF_ERROR(loom_low_source_selection_try_func(
        module, options, &fact_table, (loom_symbol_id_t)i,
        /*require_compatible=*/false, &compatible, &candidate));
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

iree_status_t loom_low_select_source_func(
    const loom_module_t* module,
    const loom_low_source_selection_options_t* options,
    iree_arena_allocator_t* arena, loom_low_source_selection_t* out_selection) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_selection);
  memset(out_selection, 0, sizeof(*out_selection));
  if (options->descriptor_registry == NULL ||
      options->policy_registry == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "source-to-low func selection requires descriptor and policy "
        "registries");
  }
  IREE_RETURN_IF_ERROR(
      loom_low_lower_policy_registry_verify(options->policy_registry));

  loom_symbol_fact_table_t fact_table = {0};
  IREE_RETURN_IF_ERROR(loom_low_source_selection_initialize_fact_table(
      options->descriptor_registry, arena, &fact_table));
  return loom_low_source_selection_select_single(module, options, &fact_table,
                                                 out_selection);
}
