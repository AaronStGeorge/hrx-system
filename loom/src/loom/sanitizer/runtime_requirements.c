// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/sanitizer/runtime_requirements.h"

#include <string.h>

#include "loom/analysis/symbol_liveness.h"
#include "loom/ir/module.h"
#include "loom/ir/symbol_map.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/util/walk.h"

typedef struct loom_sanitizer_runtime_requirements_state_t {
  // Requirements discovered from executable operations and options.
  loom_sanitizer_runtime_requirements_t requirements;
  // All requirements that can be discovered under the reporting policy.
  loom_sanitizer_runtime_requirements_t complete_requirements;
  // Reporting policy applied to authored sanitizer operations.
  loom_sanitizer_reporting_mode_t reporting_mode;
} loom_sanitizer_runtime_requirements_state_t;

loom_sanitizer_runtime_requirements_t
loom_sanitizer_runtime_requirements_from_options(
    const loom_sanitizer_options_t* options) {
  if (options == NULL || !loom_sanitizer_options_is_enabled(options)) {
    return LOOM_SANITIZER_RUNTIME_REQUIREMENT_NONE;
  }

  loom_sanitizer_runtime_requirements_t requirements =
      options->reporting_mode == LOOM_SANITIZER_REPORTING_MODE_TRAP
          ? LOOM_SANITIZER_RUNTIME_REQUIREMENT_NONE
          : LOOM_SANITIZER_RUNTIME_REQUIREMENT_FEEDBACK;
  if (iree_any_bit_set(options->checks, LOOM_SANITIZER_CHECK_ACCESS)) {
    requirements |= LOOM_SANITIZER_RUNTIME_REQUIREMENT_ACCESS_SHADOW;
  }
  if (iree_any_bit_set(options->checks, LOOM_SANITIZER_CHECK_RACE)) {
    requirements |= LOOM_SANITIZER_RUNTIME_REQUIREMENT_RACE_SHADOW;
  }
  return requirements;
}

static void loom_sanitizer_runtime_requirements_add_feedback(
    loom_sanitizer_runtime_requirements_state_t* state) {
  if (state->reporting_mode != LOOM_SANITIZER_REPORTING_MODE_TRAP) {
    state->requirements |= LOOM_SANITIZER_RUNTIME_REQUIREMENT_FEEDBACK;
  }
}

static iree_status_t loom_sanitizer_runtime_requirements_visit(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  loom_sanitizer_runtime_requirements_state_t* state =
      (loom_sanitizer_runtime_requirements_state_t*)user_data;

  switch (op->kind) {
    case LOOM_OP_KERNEL_ASSERT:
    case LOOM_OP_SANITIZER_ASSERT_VALUE:
    case LOOM_OP_SANITIZER_ASSERT_OP:
    case LOOM_OP_SANITIZER_ASSERT_LAYOUT:
      loom_sanitizer_runtime_requirements_add_feedback(state);
      break;
    case LOOM_OP_SANITIZER_ASSERT_ACCESS:
    case LOOM_OP_SANITIZER_ASSERT_ACCESSES:
      state->requirements |= LOOM_SANITIZER_RUNTIME_REQUIREMENT_ACCESS_SHADOW;
      loom_sanitizer_runtime_requirements_add_feedback(state);
      break;
    case LOOM_OP_SANITIZER_RACE_ACCESS:
    case LOOM_OP_SANITIZER_RACE_FRAGMENT_ACCESS:
      state->requirements |= LOOM_SANITIZER_RUNTIME_REQUIREMENT_RACE_SHADOW;
      loom_sanitizer_runtime_requirements_add_feedback(state);
      break;
    case LOOM_OP_SANITIZER_RACE_SYNC:
      state->requirements |= LOOM_SANITIZER_RUNTIME_REQUIREMENT_RACE_SHADOW;
      break;
    default:
      break;
  }

  *out_result = state->requirements == state->complete_requirements
                    ? LOOM_WALK_ABORT
                    : LOOM_WALK_CONTINUE;
  return iree_ok_status();
}

// Template applicability is not available before live-device target selection.
// Incoming family references already index provider membership; each reachable
// family expands once, even when many applications demand it.
static iree_status_t loom_sanitizer_runtime_requirements_visit_family(
    void* user_data, loom_symbol_liveness_contributor_context_t* context,
    const loom_template_demand_t* demand) {
  uint8_t* expanded_families = user_data;
  const loom_symbol_id_t family = demand->family_symbol_id;
  if (expanded_families[family]) {
    return iree_ok_status();
  }
  expanded_families[family] = 1;
  const loom_symbol_reference_symbol_occurrences_t references =
      loom_symbol_reference_table_symbol(context->references, family);
  iree_status_t status = iree_ok_status();
  for (loom_symbol_reference_occurrence_id_t occurrence_id =
           references.first_incoming_occurrence_id;
       iree_status_is_ok(status) &&
       occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID;) {
    const loom_symbol_reference_occurrence_t* occurrence =
        loom_symbol_reference_table_occurrence(context->references,
                                               occurrence_id);
    occurrence_id = occurrence->next_incoming_occurrence_id;
    if (occurrence->source_symbol_id == LOOM_SYMBOL_ID_INVALID) {
      continue;
    }
    const loom_symbol_t* symbol =
        &context->module->symbols.entries[occurrence->source_symbol_id];
    if (!loom_symbol_implements(symbol,
                                LOOM_SYMBOL_INTERFACE_TEMPLATE_PROVIDER)) {
      continue;
    }
    const loom_func_like_t provider =
        loom_func_like_const_cast(context->module, symbol->defining_op);
    const loom_symbol_ref_t provider_family =
        loom_func_like_template_family(provider);
    if (provider_family.module_id == 0 && provider_family.symbol_id == family) {
      status = loom_symbol_liveness_mark_symbol_id(
          context, occurrence->source_symbol_id);
    }
  }
  return status;
}

static iree_status_t loom_sanitizer_runtime_requirements_resolve_roots(
    const loom_module_t* module, iree_string_view_list_t roots,
    iree_arena_allocator_t* arena, loom_symbol_id_t** out_root_ids) {
  // Build the name index once instead of scanning the symbol table per root.
  loom_symbol_map_t names = {0};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (symbol->name_id == LOOM_STRING_ID_INVALID) {
      continue;
    }
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    status = loom_symbol_map_find_or_insert(&names, arena, symbol->name_id,
                                            (loom_symbol_id_t)i, &symbol_id);
  }
  loom_symbol_id_t* root_ids = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(arena, roots.count, sizeof(*root_ids),
                                       (void**)&root_ids);
  }
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < roots.count;
       ++i) {
    iree_string_view_t name = iree_string_view_trim(roots.values[i]);
    if (iree_string_view_starts_with_char(name, '@')) {
      name = iree_string_view_remove_prefix(name, 1);
    }
    root_ids[i] =
        loom_symbol_map_find(&names, loom_module_lookup_string(module, name));
    if (root_ids[i] == LOOM_SYMBOL_ID_INVALID) {
      status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                "runtime requirement root '@%.*s' not found",
                                (int)name.size, name.data);
    }
  }
  *out_root_ids = root_ids;
  return status;
}

static iree_status_t loom_sanitizer_runtime_requirements_select(
    const loom_module_t* module, const loom_symbol_id_t* root_ids,
    iree_host_size_t root_count, iree_arena_allocator_t* arena,
    const uint8_t** out_live_symbols) {
  loom_symbol_reference_table_t references = {0};
  iree_status_t status =
      loom_symbol_reference_table_build(module, arena, &references);
  uint8_t* expanded_families = NULL;
  if (iree_status_is_ok(status) && references.template_demands.count != 0) {
    status = iree_arena_allocate_array(arena, module->symbols.count, 1,
                                       (void**)&expanded_families);
    if (iree_status_is_ok(status)) {
      memset(expanded_families, 0, module->symbols.count);
    }
  }
  const loom_symbol_liveness_contributor_t contributor = {
      .visit_template_demand = loom_sanitizer_runtime_requirements_visit_family,
      .user_data = expanded_families,
  };
  const loom_symbol_liveness_options_t liveness_options = {
      .root_symbol_ids = {.values = root_ids, .count = root_count},
      .contributors = &contributor,
      .contributor_count = expanded_families != NULL ? 1 : 0,
  };
  if (iree_status_is_ok(status)) {
    loom_symbol_liveness_t liveness = {0};
    status = loom_symbol_liveness_compute(module, &references,
                                          &liveness_options, arena, &liveness);
    if (iree_status_is_ok(status)) {
      *out_live_symbols = liveness.live_symbols;
    }
  }
  return status;
}

iree_status_t loom_sanitizer_runtime_requirements_query(
    const loom_module_t* module, iree_string_view_list_t roots,
    const loom_sanitizer_options_t* options,
    loom_sanitizer_runtime_requirements_t* out_requirements) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(out_requirements);
  *out_requirements = LOOM_SANITIZER_RUNTIME_REQUIREMENT_NONE;

  const loom_sanitizer_reporting_mode_t reporting_mode =
      options != NULL ? options->reporting_mode
                      : LOOM_SANITIZER_REPORTING_MODE_DEFAULT;
  const loom_sanitizer_runtime_requirements_t option_requirements =
      loom_sanitizer_runtime_requirements_from_options(options);
  loom_sanitizer_runtime_requirements_state_t state = {
      .requirements = option_requirements,
      .complete_requirements =
          LOOM_SANITIZER_RUNTIME_REQUIREMENT_ACCESS_SHADOW |
          LOOM_SANITIZER_RUNTIME_REQUIREMENT_RACE_SHADOW |
          (reporting_mode == LOOM_SANITIZER_REPORTING_MODE_TRAP
               ? LOOM_SANITIZER_RUNTIME_REQUIREMENT_NONE
               : LOOM_SANITIZER_RUNTIME_REQUIREMENT_FEEDBACK),
      .reporting_mode = reporting_mode,
  };
  iree_arena_allocator_t arena;
  iree_arena_initialize(module->arena.block_pool, &arena);
  loom_symbol_id_t* root_ids = NULL;
  loom_sanitizer_runtime_requirements_t* symbol_requirements = NULL;
  iree_status_t status = iree_ok_status();
  if (roots.count != 0) {
    status = loom_sanitizer_runtime_requirements_resolve_roots(
        module, roots, &arena, &root_ids);
    if (iree_status_is_ok(status) &&
        option_requirements != state.complete_requirements) {
      status = iree_arena_allocate_array(&arena, module->symbols.count,
                                         sizeof(*symbol_requirements),
                                         (void**)&symbol_requirements);
    }
    if (iree_status_is_ok(status) && symbol_requirements != NULL) {
      memset(symbol_requirements, 0,
             module->symbols.count * sizeof(*symbol_requirements));
    }
  }
  loom_sanitizer_runtime_requirements_t requirements = option_requirements;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < module->symbols.count &&
       option_requirements != state.complete_requirements &&
       (roots.count != 0 || requirements != state.complete_requirements);
       ++i) {
    loom_func_like_t function = loom_func_like_const_cast(
        module, module->symbols.entries[i].defining_op);
    if (!loom_func_like_isa(function)) {
      continue;
    }

    state.requirements = roots.count != 0 ? option_requirements : requirements;
    loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
    status = loom_walk_function(
        module, function, LOOM_WALK_PRE_ORDER,
        (loom_walk_callback_t){loom_sanitizer_runtime_requirements_visit,
                               &state},
        &walk_result);
    requirements |= state.requirements;
    if (symbol_requirements != NULL) {
      symbol_requirements[i] = state.requirements;
    }
  }
  // Most modules have no authored services beyond the requested options. Only
  // build dependency closure when selection can change the answer. Retain the
  // per-symbol facts above so the selected bodies never need another walk.
  if (iree_status_is_ok(status) && roots.count != 0 &&
      requirements != option_requirements) {
    const uint8_t* live_symbols = NULL;
    status = loom_sanitizer_runtime_requirements_select(
        module, root_ids, roots.count, &arena, &live_symbols);
    requirements = option_requirements;
    for (iree_host_size_t i = 0;
         iree_status_is_ok(status) && i < module->symbols.count; ++i) {
      if (live_symbols[i]) {
        requirements |= symbol_requirements[i];
      }
    }
  }
  if (iree_status_is_ok(status)) {
    *out_requirements = requirements;
  }
  iree_arena_deinitialize(&arena);
  return status;
}
