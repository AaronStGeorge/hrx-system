// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/linker.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/symbol_map.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"

typedef struct loom_link_module_map_t {
  // Source module whose symbols are covered by target_symbols.
  const loom_module_t* source_module;
  // Source symbol index to target symbol reference table.
  loom_symbol_ref_t* target_symbols;
  // Number of entries in target_symbols.
  iree_host_size_t target_symbol_count;
} loom_link_module_map_t;

typedef struct loom_link_contract_source_t {
  // Source module containing the declaration contract.
  const loom_module_t* source_module;
  // Source symbol table index for the declaration contract.
  uint16_t source_symbol_id;
  // Next declaration contract for the same linked target symbol.
  struct loom_link_contract_source_t* next;
} loom_link_contract_source_t;

typedef struct loom_link_materialization_t {
  // Source module containing the chosen symbol-defining op.
  const loom_module_t* source_module;
  // Source symbol table index for the chosen symbol-defining op.
  uint16_t source_symbol_id;
  // True when the chosen materialization is only a declaration.
  bool is_declaration;
  // Declaration contracts linked into the chosen materialization.
  loom_link_contract_source_t* contract_sources;
} loom_link_materialization_t;

typedef struct loom_link_func_contract_attr_t {
  // Source declaration attribute index.
  uint8_t source_attr_index;
  // Target materialization attribute index.
  uint8_t target_attr_index;
  // Diagnostic field name for conflicts.
  iree_string_view_t field_name;
} loom_link_func_contract_attr_t;

typedef struct loom_link_state_t {
  // Linked output module being constructed.
  loom_module_t* target_module;
  // Scratch arena for remap tables and linker bookkeeping.
  iree_arena_allocator_t* scratch_arena;
  // Hash map from target-module string IDs to target symbol IDs.
  loom_symbol_map_t target_symbol_lookup;
  // Per-source module symbol remap tables.
  loom_link_module_map_t* module_maps;
  // Number of entries in module_maps.
  iree_host_size_t module_map_count;
  // Lazily initialized source-module to target-module remap tables.
  loom_ir_remap_t* module_remaps;
  // Chosen source materialization for each target symbol.
  loom_link_materialization_t* materializations;
  // Number of entries allocated in materializations.
  iree_host_size_t materialization_count;
  // Number of entries available in materializations.
  iree_host_size_t materialization_capacity;
  // Target-symbol liveness table for root-selective links.
  uint8_t* live_target_symbols;
  // Target-symbol dependency scan table for root-selective links.
  uint8_t* scanned_target_symbols;
  // Target symbols already cloned into the output module.
  uint8_t* cloned_target_symbols;
} loom_link_state_t;

static bool loom_link_symbol_is_declaration(const loom_symbol_t* symbol) {
  return symbol->kind == LOOM_SYMBOL_FUNC_DECL;
}

static bool loom_link_is_selective(const loom_link_state_t* state) {
  return state->live_target_symbols != NULL;
}

static iree_string_view_t loom_link_target_symbol_name(
    const loom_module_t* target_module, loom_symbol_ref_t target_ref) {
  const loom_symbol_t* symbol =
      &target_module->symbols.entries[target_ref.symbol_id];
  return target_module->strings.entries[symbol->name_id];
}

static iree_status_t loom_link_ensure_materialization_capacity(
    loom_link_state_t* state, iree_host_size_t required_count) {
  if (required_count <= state->materialization_capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity = state->materialization_capacity == 0
                                      ? 16
                                      : state->materialization_capacity;
  while (new_capacity < required_count) {
    new_capacity *= 2;
  }
  loom_link_materialization_t* materializations = state->materializations;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      state->scratch_arena, state->materialization_count, new_capacity,
      sizeof(*materializations), &state->materialization_capacity,
      (void**)&materializations));
  memset(materializations + state->materialization_count, 0,
         (new_capacity - state->materialization_count) *
             sizeof(*materializations));
  state->materializations = materializations;
  return iree_ok_status();
}

static iree_status_t loom_link_get_or_add_target_symbol(
    loom_link_state_t* state, const loom_module_t* source_module,
    uint16_t source_symbol_id, loom_symbol_ref_t* out_target_ref) {
  const loom_symbol_t* source_symbol =
      &source_module->symbols.entries[source_symbol_id];
  if (source_symbol->name_id >= source_module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source symbol %u name id %u is out of range",
                            (unsigned)source_symbol_id,
                            (unsigned)source_symbol->name_id);
  }

  loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      state->target_module,
      source_module->strings.entries[source_symbol->name_id], &target_name_id));

  uint16_t target_symbol_id =
      loom_symbol_map_find(&state->target_symbol_lookup, target_name_id);
  if (target_symbol_id == LOOM_SYMBOL_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(
        state->target_module, target_name_id, &target_symbol_id));
    IREE_RETURN_IF_ERROR(loom_symbol_map_insert(
        &state->target_symbol_lookup, state->scratch_arena, target_name_id,
        target_symbol_id));
    IREE_RETURN_IF_ERROR(loom_link_ensure_materialization_capacity(
        state, (iree_host_size_t)target_symbol_id + 1));
    if (target_symbol_id >= state->materialization_count) {
      state->materialization_count = (iree_host_size_t)target_symbol_id + 1;
    }
  }

  *out_target_ref =
      (loom_symbol_ref_t){.module_id = 0, .symbol_id = target_symbol_id};
  return iree_ok_status();
}

static iree_status_t loom_link_source_symbol_name(
    const loom_module_t* source_module, uint16_t source_symbol_id,
    iree_string_view_t* out_name) {
  const loom_symbol_t* source_symbol =
      &source_module->symbols.entries[source_symbol_id];
  if (source_symbol->name_id >= source_module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source symbol %u name id %u is out of range",
                            (unsigned)source_symbol_id,
                            (unsigned)source_symbol->name_id);
  }
  *out_name = source_module->strings.entries[source_symbol->name_id];
  return iree_ok_status();
}

static iree_status_t loom_link_record_materialization(
    loom_link_state_t* state, const loom_module_t* source_module,
    uint16_t source_symbol_id, loom_symbol_ref_t target_ref) {
  const loom_symbol_t* source_symbol =
      &source_module->symbols.entries[source_symbol_id];
  if (!source_symbol->defining_op) {
    return iree_ok_status();
  }

  loom_link_materialization_t* materialization =
      &state->materializations[target_ref.symbol_id];
  const bool new_is_declaration =
      loom_link_symbol_is_declaration(source_symbol);
  if (new_is_declaration) {
    loom_link_contract_source_t* contract_source = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(state->scratch_arena,
                                             sizeof(*contract_source),
                                             (void**)&contract_source));
    *contract_source = (loom_link_contract_source_t){
        .source_module = source_module,
        .source_symbol_id = source_symbol_id,
        .next = materialization->contract_sources,
    };
    materialization->contract_sources = contract_source;
  }
  if (!materialization->source_module) {
    loom_link_contract_source_t* contract_sources =
        materialization->contract_sources;
    *materialization = (loom_link_materialization_t){
        .source_module = source_module,
        .source_symbol_id = source_symbol_id,
        .is_declaration = new_is_declaration,
        .contract_sources = contract_sources,
    };
    return iree_ok_status();
  }

  if (materialization->is_declaration) {
    if (!new_is_declaration) {
      loom_link_contract_source_t* contract_sources =
          materialization->contract_sources;
      *materialization = (loom_link_materialization_t){
          .source_module = source_module,
          .source_symbol_id = source_symbol_id,
          .is_declaration = false,
          .contract_sources = contract_sources,
      };
    }
    return iree_ok_status();
  }

  if (new_is_declaration) {
    return iree_ok_status();
  }

  iree_string_view_t name =
      loom_link_target_symbol_name(state->target_module, target_ref);
  return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                          "duplicate concrete symbol definition '@%.*s'",
                          (int)name.size, name.data);
}

static iree_status_t loom_link_map_module_symbols(
    loom_link_state_t* state, loom_link_module_map_t* module_map,
    bool materialize_all_symbols) {
  const loom_module_t* source_module = module_map->source_module;
  module_map->target_symbol_count = source_module->symbols.count;
  if (module_map->target_symbol_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, module_map->target_symbol_count,
      sizeof(*module_map->target_symbols),
      (void**)&module_map->target_symbols));
  for (iree_host_size_t i = 0; i < module_map->target_symbol_count; ++i) {
    module_map->target_symbols[i] = loom_symbol_ref_null();
  }
  if (!materialize_all_symbols) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < module_map->target_symbol_count; ++i) {
    loom_symbol_ref_t target_ref = loom_symbol_ref_null();
    IREE_RETURN_IF_ERROR(loom_link_get_or_add_target_symbol(
        state, source_module, (uint16_t)i, &target_ref));
    module_map->target_symbols[i] = target_ref;
    IREE_RETURN_IF_ERROR(loom_link_record_materialization(
        state, source_module, (uint16_t)i, target_ref));
  }
  return iree_ok_status();
}

static const loom_link_module_map_t* loom_link_find_module_map(
    const loom_link_state_t* state, const loom_module_t* source_module) {
  for (iree_host_size_t i = 0; i < state->module_map_count; ++i) {
    if (state->module_maps[i].source_module == source_module) {
      return &state->module_maps[i];
    }
  }
  return NULL;
}

static bool loom_link_find_module_map_index(const loom_link_state_t* state,
                                            const loom_module_t* source_module,
                                            iree_host_size_t* out_index) {
  for (iree_host_size_t i = 0; i < state->module_map_count; ++i) {
    if (state->module_maps[i].source_module == source_module) {
      *out_index = i;
      return true;
    }
  }
  return false;
}

static iree_status_t loom_link_map_source_symbol(
    loom_link_state_t* state, const loom_module_t* source_module,
    uint16_t source_symbol_id, loom_symbol_ref_t* out_target_ref) {
  iree_host_size_t module_index = 0;
  if (!loom_link_find_module_map_index(state, source_module, &module_index)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source module is not part of this link");
  }
  loom_link_module_map_t* module_map = &state->module_maps[module_index];
  if (source_symbol_id >= module_map->target_symbol_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source symbol ref {module=0, symbol=%u} is out of range",
        (unsigned)source_symbol_id);
  }
  loom_symbol_ref_t target_ref = module_map->target_symbols[source_symbol_id];
  if (!loom_symbol_ref_is_valid(target_ref)) {
    IREE_RETURN_IF_ERROR(loom_link_get_or_add_target_symbol(
        state, source_module, source_symbol_id, &target_ref));
    module_map->target_symbols[source_symbol_id] = target_ref;
  }
  *out_target_ref = target_ref;
  return iree_ok_status();
}

static iree_status_t loom_link_remap_symbol(void* user_data,
                                            const loom_module_t* source_module,
                                            loom_module_t* target_module,
                                            loom_symbol_ref_t source_ref,
                                            loom_symbol_ref_t* out_target_ref) {
  loom_link_state_t* state = (loom_link_state_t*)user_data;
  if (target_module != state->target_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "link symbol remap target module mismatch");
  }
  if (source_ref.module_id != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source symbol ref {module=%u, symbol=%u} is not "
                            "module-local",
                            (unsigned)source_ref.module_id,
                            (unsigned)source_ref.symbol_id);
  }
  if (loom_link_is_selective(state)) {
    const loom_link_module_map_t* module_map =
        loom_link_find_module_map(state, source_module);
    if (!module_map) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source module is not part of this link");
    }
    if (source_ref.symbol_id >= module_map->target_symbol_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source symbol ref {module=0, symbol=%u} is out of range",
          (unsigned)source_ref.symbol_id);
    }
    loom_symbol_ref_t target_ref =
        module_map->target_symbols[source_ref.symbol_id];
    if (!loom_symbol_ref_is_valid(target_ref)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "selective link missed reachable source symbol ref {module=0, "
          "symbol=%u}",
          (unsigned)source_ref.symbol_id);
    }
    *out_target_ref = target_ref;
    return iree_ok_status();
  }
  return loom_link_map_source_symbol(state, source_module, source_ref.symbol_id,
                                     out_target_ref);
}

static iree_status_t loom_link_get_module_remap(
    loom_link_state_t* state, const loom_module_t* source_module,
    loom_ir_remap_t** out_remap) {
  iree_host_size_t module_index = 0;
  if (!loom_link_find_module_map_index(state, source_module, &module_index)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source module is not part of this link");
  }
  loom_ir_remap_t* remap = &state->module_remaps[module_index];
  if (!remap->source_module) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
        source_module, state->target_module, state->scratch_arena,
        &(loom_ir_remap_options_t){
            .remap_symbol = loom_link_remap_symbol,
            .remap_symbol_user_data = state,
        },
        remap));
  }
  *out_remap = remap;
  return iree_ok_status();
}

static bool loom_link_op_symbol_ref(const loom_module_t* module,
                                    const loom_op_t* op,
                                    loom_symbol_ref_t* out_ref) {
  *out_ref = loom_symbol_ref_null();
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || !vtable->symbol_def || !vtable->attr_descriptors) {
    return false;
  }
  uint8_t symbol_attr_index = vtable->symbol_def->name_attr_index;
  if (symbol_attr_index >= vtable->attribute_count ||
      symbol_attr_index >= op->attribute_count) {
    return false;
  }
  const loom_attr_descriptor_t* descriptor =
      &vtable->attr_descriptors[symbol_attr_index];
  if (descriptor->attr_kind != LOOM_ATTR_SYMBOL) {
    return false;
  }
  *out_ref = loom_attr_as_symbol(loom_op_const_attrs(op)[symbol_attr_index]);
  return loom_symbol_ref_is_valid(*out_ref) && out_ref->module_id == 0 &&
         out_ref->symbol_id < module->symbols.count;
}

static iree_status_t loom_link_mark_target_symbol_live(
    loom_link_state_t* state, loom_symbol_ref_t target_ref) {
  if (!loom_link_is_selective(state)) {
    return iree_ok_status();
  }
  if (target_ref.module_id != 0 ||
      target_ref.symbol_id >= state->materialization_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target symbol ref {module=%u, symbol=%u} is out "
                            "of range",
                            (unsigned)target_ref.module_id,
                            (unsigned)target_ref.symbol_id);
  }
  state->live_target_symbols[target_ref.symbol_id] = 1;
  return iree_ok_status();
}

static iree_status_t loom_link_mark_source_symbol_live(
    loom_link_state_t* state, const loom_module_t* source_module,
    loom_symbol_ref_t source_ref) {
  if (!loom_symbol_ref_is_valid(source_ref) || source_ref.module_id != 0) {
    return iree_ok_status();
  }
  loom_symbol_ref_t target_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_link_map_source_symbol(
      state, source_module, source_ref.symbol_id, &target_ref));
  return loom_link_mark_target_symbol_live(state, target_ref);
}

static iree_status_t loom_link_mark_attr_symbol_refs_live(
    loom_link_state_t* state, const loom_module_t* source_module,
    const loom_attribute_t* attr, uint8_t dict_depth);

static iree_status_t loom_link_mark_encoding_symbol_refs_live(
    loom_link_state_t* state, const loom_module_t* source_module,
    uint16_t encoding_id, uint8_t encoding_depth) {
  if (encoding_id == 0) {
    return iree_ok_status();
  }
  if (encoding_depth >= LOOM_ATTR_DICT_MAX_NESTING_DEPTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "encoding nesting exceeds max depth %u",
                            (unsigned)LOOM_ATTR_DICT_MAX_NESTING_DEPTH);
  }
  const loom_encoding_t* encoding =
      loom_module_encoding(source_module, encoding_id);
  if (!encoding) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source encoding id %u out of range (source module has %" PRIhsz
        " encodings)",
        (unsigned)encoding_id, source_module->encodings.count);
  }
  for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_link_mark_attr_symbol_refs_live(
        state, source_module, &encoding->attributes[i].value,
        (uint8_t)(encoding_depth + 1)));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_mark_type_symbol_refs_live(
    loom_link_state_t* state, const loom_module_t* source_module,
    loom_type_t type) {
  if (!loom_type_has_static_encoding(type)) {
    return iree_ok_status();
  }
  return loom_link_mark_encoding_symbol_refs_live(state, source_module,
                                                  type.encoding_id, 0);
}

static iree_status_t loom_link_mark_attr_symbol_refs_live(
    loom_link_state_t* state, const loom_module_t* source_module,
    const loom_attribute_t* attr, uint8_t dict_depth) {
  if (!attr) {
    return iree_ok_status();
  }
  switch ((loom_attr_kind_t)attr->kind) {
    case LOOM_ATTR_SYMBOL:
      return loom_link_mark_source_symbol_live(state, source_module,
                                               loom_attr_as_symbol(*attr));
    case LOOM_ATTR_TYPE:
      if (attr->type_id >= source_module->types.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "type attribute references source type id %u but source module has "
            "%" PRIhsz " types",
            (unsigned)attr->type_id, source_module->types.count);
      }
      return loom_link_mark_type_symbol_refs_live(
          state, source_module, source_module->types.entries[attr->type_id]);
    case LOOM_ATTR_ENCODING:
      return loom_link_mark_encoding_symbol_refs_live(
          state, source_module, loom_attr_as_encoding_id(*attr), dict_depth);
    case LOOM_ATTR_DICT:
      if (dict_depth >= LOOM_ATTR_DICT_MAX_NESTING_DEPTH) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "dict attribute nesting exceeds max depth %u",
                                (unsigned)LOOM_ATTR_DICT_MAX_NESTING_DEPTH);
      }
      if (attr->count > 0 && !attr->dict_entries) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty dict attribute has a NULL entry pointer");
      }
      for (uint16_t i = 0; i < attr->count; ++i) {
        IREE_RETURN_IF_ERROR(loom_link_mark_attr_symbol_refs_live(
            state, source_module, &attr->dict_entries[i].value,
            (uint8_t)(dict_depth + 1)));
      }
      return iree_ok_status();
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_link_mark_op_symbol_refs_live(
    loom_link_state_t* state, const loom_module_t* source_module,
    const loom_op_t* op) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_link_mark_type_symbol_refs_live(
        state, source_module,
        loom_module_value_type(source_module, operands[i])));
  }

  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_link_mark_type_symbol_refs_live(
        state, source_module,
        loom_module_value_type(source_module, results[i])));
  }

  const loom_attribute_t* attrs = loom_op_const_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_link_mark_attr_symbol_refs_live(
        state, source_module, &attrs[i], 0));
  }

  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    const loom_region_t* region = regions[i];
    if (!region) {
      continue;
    }
    const loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
        IREE_RETURN_IF_ERROR(loom_link_mark_type_symbol_refs_live(
            state, source_module,
            loom_module_value_type(source_module,
                                   loom_block_arg_id(block, arg_index))));
      }
      const loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        IREE_RETURN_IF_ERROR(
            loom_link_mark_op_symbol_refs_live(state, source_module, child_op));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_link_record_target_symbol_materializations(
    loom_link_state_t* state, uint16_t target_symbol_id) {
  iree_string_view_t target_name = loom_link_target_symbol_name(
      state->target_module,
      (loom_symbol_ref_t){.module_id = 0, .symbol_id = target_symbol_id});
  for (iree_host_size_t module_index = 0;
       module_index < state->module_map_count; ++module_index) {
    const loom_link_module_map_t* module_map =
        &state->module_maps[module_index];
    for (iree_host_size_t source_symbol_id = 0;
         source_symbol_id < module_map->target_symbol_count;
         ++source_symbol_id) {
      iree_string_view_t source_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_link_source_symbol_name(
          module_map->source_module, (uint16_t)source_symbol_id, &source_name));
      if (!iree_string_view_equal(source_name, target_name)) {
        continue;
      }
      loom_symbol_ref_t target_ref = loom_symbol_ref_null();
      IREE_RETURN_IF_ERROR(
          loom_link_map_source_symbol(state, module_map->source_module,
                                      (uint16_t)source_symbol_id, &target_ref));
      IREE_RETURN_IF_ERROR(loom_link_record_materialization(
          state, module_map->source_module, (uint16_t)source_symbol_id,
          target_ref));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_link_mark_materialization_dependencies_live(
    loom_link_state_t* state, uint16_t target_symbol_id) {
  const loom_link_materialization_t* materialization =
      &state->materializations[target_symbol_id];
  if (materialization->source_module) {
    const loom_symbol_t* source_symbol =
        &materialization->source_module->symbols
             .entries[materialization->source_symbol_id];
    if (source_symbol->defining_op) {
      IREE_RETURN_IF_ERROR(loom_link_mark_op_symbol_refs_live(
          state, materialization->source_module, source_symbol->defining_op));
    }
  }
  for (const loom_link_contract_source_t* contract_source =
           materialization->contract_sources;
       contract_source; contract_source = contract_source->next) {
    const loom_symbol_t* source_symbol =
        &contract_source->source_module->symbols
             .entries[contract_source->source_symbol_id];
    if (source_symbol->defining_op) {
      IREE_RETURN_IF_ERROR(loom_link_mark_op_symbol_refs_live(
          state, contract_source->source_module, source_symbol->defining_op));
    }
  }
  return iree_ok_status();
}

static bool loom_link_is_contract_source(
    const loom_link_materialization_t* materialization,
    const loom_module_t* source_module, uint16_t source_symbol_id) {
  for (const loom_link_contract_source_t* contract_source =
           materialization->contract_sources;
       contract_source; contract_source = contract_source->next) {
    if (contract_source->source_module == source_module &&
        contract_source->source_symbol_id == source_symbol_id) {
      return true;
    }
  }
  return false;
}

static bool loom_link_should_clone_symbol_op(
    const loom_link_state_t* state, const loom_module_t* source_module,
    loom_symbol_ref_t source_ref, const loom_module_t** out_clone_module,
    uint16_t* out_clone_symbol_id, loom_symbol_ref_t* out_target_ref) {
  const loom_link_module_map_t* module_map =
      loom_link_find_module_map(state, source_module);
  if (!module_map || source_ref.symbol_id >= module_map->target_symbol_count) {
    return false;
  }
  loom_symbol_ref_t target_ref =
      module_map->target_symbols[source_ref.symbol_id];
  if (target_ref.symbol_id >= state->materialization_count) {
    return false;
  }
  if (loom_link_is_selective(state) &&
      !state->live_target_symbols[target_ref.symbol_id]) {
    return false;
  }
  if (state->cloned_target_symbols[target_ref.symbol_id]) {
    return false;
  }
  const loom_link_materialization_t* materialization =
      &state->materializations[target_ref.symbol_id];
  if (!materialization->source_module) {
    return false;
  }

  const bool selected_here =
      materialization->source_module == source_module &&
      materialization->source_symbol_id == source_ref.symbol_id;
  const bool declaration_anchor =
      loom_link_is_selective(state) &&
      loom_link_is_contract_source(materialization, source_module,
                                   source_ref.symbol_id);
  if (!selected_here && !declaration_anchor) {
    return false;
  }

  *out_clone_module = materialization->source_module;
  *out_clone_symbol_id = materialization->source_symbol_id;
  *out_target_ref = target_ref;
  return true;
}

static iree_status_t loom_link_incompatible_contract_status(
    const loom_link_state_t* state, loom_symbol_ref_t target_ref,
    iree_string_view_t field_name) {
  iree_string_view_t symbol_name =
      loom_link_target_symbol_name(state->target_module, target_ref);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "linked declaration for '@%.*s' has incompatible "
                          "function contract field '%.*s'",
                          (int)symbol_name.size, symbol_name.data,
                          (int)field_name.size, field_name.data);
}

static iree_status_t loom_link_signature_count_status(
    const loom_link_state_t* state, loom_symbol_ref_t target_ref,
    iree_string_view_t field_name, iree_host_size_t source_count,
    iree_host_size_t target_count) {
  iree_string_view_t symbol_name =
      loom_link_target_symbol_name(state->target_module, target_ref);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "linked declaration for '@%.*s' has %" PRIhsz
                          " %.*s but selected "
                          "symbol has %" PRIhsz,
                          (int)symbol_name.size, symbol_name.data, source_count,
                          (int)field_name.size, field_name.data, target_count);
}

static iree_status_t loom_link_signature_type_status(
    const loom_link_state_t* state, loom_symbol_ref_t target_ref,
    iree_string_view_t field_name, iree_host_size_t ordinal) {
  iree_string_view_t symbol_name =
      loom_link_target_symbol_name(state->target_module, target_ref);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "linked declaration for '@%.*s' has incompatible "
                          "%.*s type at index %" PRIhsz,
                          (int)symbol_name.size, symbol_name.data,
                          (int)field_name.size, field_name.data, ordinal);
}

static iree_status_t loom_link_map_func_signature_values(
    loom_ir_remap_t* remap, loom_func_like_t source_func,
    loom_func_like_t target_func, const loom_link_state_t* state,
    loom_symbol_ref_t target_ref) {
  uint16_t source_arg_count = 0;
  const loom_value_id_t* source_args =
      loom_func_like_arg_ids(source_func, &source_arg_count);
  uint16_t target_arg_count = 0;
  const loom_value_id_t* target_args =
      loom_func_like_arg_ids(target_func, &target_arg_count);
  if (source_arg_count != target_arg_count) {
    return loom_link_signature_count_status(state, target_ref, IREE_SV("args"),
                                            source_arg_count, target_arg_count);
  }
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(remap, source_args, target_args,
                                                source_arg_count));

  const uint16_t source_result_count = source_func.op->result_count;
  const uint16_t target_result_count = target_func.op->result_count;
  if (source_result_count != target_result_count) {
    return loom_link_signature_count_status(
        state, target_ref, IREE_SV("results"), source_result_count,
        target_result_count);
  }
  return loom_ir_remap_map_values(remap, loom_op_const_results(source_func.op),
                                  loom_op_const_results(target_func.op),
                                  source_result_count);
}

static iree_status_t loom_link_check_func_signature_types(
    loom_ir_remap_t* remap, const loom_module_t* source_module,
    loom_func_like_t source_func, loom_func_like_t target_func,
    const loom_link_state_t* state, loom_symbol_ref_t target_ref) {
  uint16_t source_arg_count = 0;
  const loom_value_id_t* source_args =
      loom_func_like_arg_ids(source_func, &source_arg_count);
  uint16_t target_arg_count = 0;
  const loom_value_id_t* target_args =
      loom_func_like_arg_ids(target_func, &target_arg_count);
  if (source_arg_count != target_arg_count) {
    return loom_link_signature_count_status(state, target_ref, IREE_SV("args"),
                                            source_arg_count, target_arg_count);
  }
  for (uint16_t i = 0; i < source_arg_count; ++i) {
    loom_type_t source_type =
        loom_module_value_type(source_module, source_args[i]);
    loom_type_t remapped_type = {0};
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_type(remap, source_type, &remapped_type));
    loom_type_t target_type =
        loom_module_value_type(state->target_module, target_args[i]);
    if (!loom_type_equal(remapped_type, target_type)) {
      return loom_link_signature_type_status(state, target_ref, IREE_SV("arg"),
                                             i);
    }
  }

  const loom_value_id_t* source_results = loom_op_const_results(source_func.op);
  const loom_value_id_t* target_results = loom_op_const_results(target_func.op);
  if (source_func.op->result_count != target_func.op->result_count) {
    return loom_link_signature_count_status(
        state, target_ref, IREE_SV("results"), source_func.op->result_count,
        target_func.op->result_count);
  }
  for (uint16_t i = 0; i < source_func.op->result_count; ++i) {
    loom_type_t source_type =
        loom_module_value_type(source_module, source_results[i]);
    loom_type_t remapped_type = {0};
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_type(remap, source_type, &remapped_type));
    loom_type_t target_type =
        loom_module_value_type(state->target_module, target_results[i]);
    if (!loom_type_equal(remapped_type, target_type)) {
      return loom_link_signature_type_status(state, target_ref,
                                             IREE_SV("result"), i);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_link_check_func_tied_results(
    const loom_link_state_t* state, loom_symbol_ref_t target_ref,
    loom_func_like_t source_func, loom_func_like_t target_func) {
  if (source_func.op->tied_result_count != target_func.op->tied_result_count) {
    return loom_link_signature_count_status(
        state, target_ref, IREE_SV("tied results"),
        source_func.op->tied_result_count, target_func.op->tied_result_count);
  }
  if (source_func.op->tied_result_count == 0) {
    return iree_ok_status();
  }
  const iree_host_size_t byte_count =
      (iree_host_size_t)source_func.op->tied_result_count *
      sizeof(loom_tied_result_t);
  if (memcmp(loom_op_tied_results(source_func.op),
             loom_op_tied_results(target_func.op), byte_count) != 0) {
    return loom_link_incompatible_contract_status(state, target_ref,
                                                  IREE_SV("tied_results"));
  }
  return iree_ok_status();
}

static bool loom_link_func_attr_present(loom_func_like_t func,
                                        uint8_t attr_index) {
  if (attr_index == LOOM_ATTR_INDEX_NONE) {
    return false;
  }
  return !loom_attr_is_absent(loom_op_attrs(func.op)[attr_index]);
}

static iree_status_t loom_link_merge_func_attr(
    loom_ir_remap_t* remap, const loom_link_state_t* state,
    loom_symbol_ref_t target_ref, loom_func_like_t source_func,
    uint8_t source_attr_index, loom_func_like_t target_func,
    uint8_t target_attr_index, iree_string_view_t field_name) {
  if (!loom_link_func_attr_present(source_func, source_attr_index)) {
    return iree_ok_status();
  }
  if (target_attr_index == LOOM_ATTR_INDEX_NONE) {
    return loom_link_incompatible_contract_status(state, target_ref,
                                                  field_name);
  }

  loom_attribute_t source_attr =
      loom_op_attrs(source_func.op)[source_attr_index];
  loom_attribute_t remapped_attr = {0};
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_attribute(remap, source_attr, &remapped_attr));

  loom_attribute_t* target_attr =
      &loom_op_attrs(target_func.op)[target_attr_index];
  if (loom_attr_is_absent(*target_attr)) {
    *target_attr = remapped_attr;
    return iree_ok_status();
  }
  if (!loom_attribute_equal(target_attr, &remapped_attr)) {
    return loom_link_incompatible_contract_status(state, target_ref,
                                                  field_name);
  }
  return iree_ok_status();
}

static iree_status_t loom_link_merge_func_contract(loom_link_state_t* state,
                                                   loom_symbol_ref_t target_ref,
                                                   loom_op_t* target_op) {
  if (target_ref.symbol_id >= state->materialization_count) {
    return iree_ok_status();
  }
  const loom_link_materialization_t* materialization =
      &state->materializations[target_ref.symbol_id];
  if (!materialization->contract_sources) {
    return iree_ok_status();
  }

  loom_func_like_t target_func =
      loom_func_like_cast(state->target_module, target_op);
  if (!loom_func_like_isa(target_func)) {
    return loom_link_incompatible_contract_status(state, target_ref,
                                                  IREE_SV("func_like"));
  }

  for (const loom_link_contract_source_t* source =
           materialization->contract_sources;
       source; source = source->next) {
    const loom_symbol_t* source_symbol =
        &source->source_module->symbols.entries[source->source_symbol_id];
    loom_func_like_t source_func = loom_func_like_cast(
        source->source_module, (loom_op_t*)source_symbol->defining_op);
    if (!loom_func_like_isa(source_func)) {
      return loom_link_incompatible_contract_status(state, target_ref,
                                                    IREE_SV("declaration"));
    }

    loom_ir_remap_t contract_remap = {0};
    IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
        source->source_module, state->target_module, state->scratch_arena,
        &(loom_ir_remap_options_t){
            .remap_symbol = loom_link_remap_symbol,
            .remap_symbol_user_data = state,
        },
        &contract_remap));
    IREE_RETURN_IF_ERROR(loom_link_map_func_signature_values(
        &contract_remap, source_func, target_func, state, target_ref));
    IREE_RETURN_IF_ERROR(loom_link_check_func_signature_types(
        &contract_remap, source->source_module, source_func, target_func, state,
        target_ref));
    IREE_RETURN_IF_ERROR(loom_link_check_func_tied_results(
        state, target_ref, source_func, target_func));

    const loom_link_func_contract_attr_t attrs[] = {
        {
            .source_attr_index = source_func.vtable->visibility_attr_index,
            .target_attr_index = target_func.vtable->visibility_attr_index,
            .field_name = IREE_SV("visibility"),
        },
        {
            .source_attr_index = source_func.vtable->import_module_attr_index,
            .target_attr_index = target_func.vtable->import_module_attr_index,
            .field_name = IREE_SV("import_module"),
        },
        {
            .source_attr_index = source_func.vtable->import_symbol_attr_index,
            .target_attr_index = target_func.vtable->import_symbol_attr_index,
            .field_name = IREE_SV("import_symbol"),
        },
        {
            .source_attr_index = source_func.vtable->cc_attr_index,
            .target_attr_index = target_func.vtable->cc_attr_index,
            .field_name = IREE_SV("cc"),
        },
        {
            .source_attr_index = source_func.vtable->purity_attr_index,
            .target_attr_index = target_func.vtable->purity_attr_index,
            .field_name = IREE_SV("purity"),
        },
        {
            .source_attr_index = source_func.vtable->target_attr_index,
            .target_attr_index = target_func.vtable->target_attr_index,
            .field_name = IREE_SV("target"),
        },
        {
            .source_attr_index = source_func.vtable->abi_attr_index,
            .target_attr_index = target_func.vtable->abi_attr_index,
            .field_name = IREE_SV("abi"),
        },
        {
            .source_attr_index = source_func.vtable->abi_attrs_attr_index,
            .target_attr_index = target_func.vtable->abi_attrs_attr_index,
            .field_name = IREE_SV("abi_attrs"),
        },
        {
            .source_attr_index = source_func.vtable->export_symbol_attr_index,
            .target_attr_index = target_func.vtable->export_symbol_attr_index,
            .field_name = IREE_SV("export_symbol"),
        },
        {
            .source_attr_index = source_func.vtable->export_attrs_attr_index,
            .target_attr_index = target_func.vtable->export_attrs_attr_index,
            .field_name = IREE_SV("export_attrs"),
        },
        {
            .source_attr_index = source_func.vtable->predicates_attr_index,
            .target_attr_index = target_func.vtable->predicates_attr_index,
            .field_name = IREE_SV("predicates"),
        },
    };
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(attrs); ++i) {
      IREE_RETURN_IF_ERROR(loom_link_merge_func_attr(
          &contract_remap, state, target_ref, source_func,
          attrs[i].source_attr_index, target_func, attrs[i].target_attr_index,
          attrs[i].field_name));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_link_clone_module_body(
    loom_link_state_t* state, const loom_module_t* source_module) {
  loom_builder_t builder;
  loom_builder_initialize(state->target_module, &state->target_module->arena,
                          loom_module_block(state->target_module), &builder);

  const loom_block_t* source_block =
      loom_region_const_entry_block(source_module->body);
  for (const loom_op_t* source_op = source_block->first_op; source_op;
       source_op = source_op->next_op) {
    loom_symbol_ref_t source_ref = loom_symbol_ref_null();
    const bool has_symbol_ref =
        loom_link_op_symbol_ref(source_module, source_op, &source_ref);
    const loom_module_t* clone_module = source_module;
    const loom_op_t* clone_op = source_op;
    loom_symbol_ref_t target_ref = loom_symbol_ref_null();
    if (has_symbol_ref) {
      uint16_t clone_symbol_id = LOOM_SYMBOL_ID_INVALID;
      if (!loom_link_should_clone_symbol_op(state, source_module, source_ref,
                                            &clone_module, &clone_symbol_id,
                                            &target_ref)) {
        continue;
      }
      const loom_symbol_t* clone_symbol =
          &clone_module->symbols.entries[clone_symbol_id];
      clone_op = clone_symbol->defining_op;
    } else if (loom_link_is_selective(state)) {
      continue;
    }

    loom_ir_remap_t* remap = NULL;
    IREE_RETURN_IF_ERROR(
        loom_link_get_module_remap(state, clone_module, &remap));
    loom_op_t* cloned_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_ir_clone_op(&builder, clone_op, remap, &cloned_op));
    if (has_symbol_ref) {
      state->cloned_target_symbols[target_ref.symbol_id] = 1;
      IREE_RETURN_IF_ERROR(
          loom_link_merge_func_contract(state, target_ref, cloned_op));
    }
  }
  return iree_ok_status();
}

static iree_string_view_t loom_link_normalize_root_name(
    iree_string_view_t root_name) {
  if (iree_string_view_starts_with_char(root_name, '@')) {
    root_name = iree_string_view_remove_prefix(root_name, 1);
  }
  return root_name;
}

static iree_status_t loom_link_find_root_target_symbol(
    loom_link_state_t* state, iree_string_view_t root_name,
    loom_symbol_ref_t* out_target_ref) {
  root_name = loom_link_normalize_root_name(root_name);
  if (iree_string_view_is_empty(root_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "root symbol name must not be empty");
  }

  for (iree_host_size_t module_index = 0;
       module_index < state->module_map_count; ++module_index) {
    const loom_link_module_map_t* module_map =
        &state->module_maps[module_index];
    for (iree_host_size_t source_symbol_id = 0;
         source_symbol_id < module_map->target_symbol_count;
         ++source_symbol_id) {
      iree_string_view_t source_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_link_source_symbol_name(
          module_map->source_module, (uint16_t)source_symbol_id, &source_name));
      if (!iree_string_view_equal(source_name, root_name)) {
        continue;
      }
      return loom_link_map_source_symbol(state, module_map->source_module,
                                         (uint16_t)source_symbol_id,
                                         out_target_ref);
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "root symbol '@%.*s' was not found",
                          (int)root_name.size, root_name.data);
}

static iree_status_t loom_link_target_symbol_has_defining_op(
    const loom_link_state_t* state, uint16_t target_symbol_id,
    bool* out_has_defining_op) {
  *out_has_defining_op = false;
  iree_string_view_t target_name = loom_link_target_symbol_name(
      state->target_module,
      (loom_symbol_ref_t){.module_id = 0, .symbol_id = target_symbol_id});
  for (iree_host_size_t module_index = 0;
       module_index < state->module_map_count; ++module_index) {
    const loom_link_module_map_t* module_map =
        &state->module_maps[module_index];
    for (iree_host_size_t source_symbol_id = 0;
         source_symbol_id < module_map->target_symbol_count;
         ++source_symbol_id) {
      iree_string_view_t source_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_link_source_symbol_name(
          module_map->source_module, (uint16_t)source_symbol_id, &source_name));
      if (!iree_string_view_equal(source_name, target_name)) {
        continue;
      }
      const loom_symbol_t* source_symbol =
          &module_map->source_module->symbols.entries[source_symbol_id];
      if (source_symbol->defining_op) {
        *out_has_defining_op = true;
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_link_mark_root_symbols_live(
    loom_link_state_t* state, const loom_link_options_t* options) {
  for (iree_host_size_t i = 0; i < options->root_symbols.count; ++i) {
    loom_symbol_ref_t target_ref = loom_symbol_ref_null();
    IREE_RETURN_IF_ERROR(loom_link_find_root_target_symbol(
        state, options->root_symbols.values[i], &target_ref));
    bool has_defining_op = false;
    IREE_RETURN_IF_ERROR(loom_link_target_symbol_has_defining_op(
        state, target_ref.symbol_id, &has_defining_op));
    if (!has_defining_op) {
      iree_string_view_t root_name =
          loom_link_normalize_root_name(options->root_symbols.values[i]);
      return iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "root symbol '@%.*s' has no materialized definition or declaration",
          (int)root_name.size, root_name.data);
    }
    IREE_RETURN_IF_ERROR(loom_link_mark_target_symbol_live(state, target_ref));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_resolve_live_symbols(loom_link_state_t* state) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (uint16_t target_symbol_id = 0;
         target_symbol_id < state->materialization_count; ++target_symbol_id) {
      if (!state->live_target_symbols[target_symbol_id] ||
          state->scanned_target_symbols[target_symbol_id]) {
        continue;
      }
      state->scanned_target_symbols[target_symbol_id] = 1;
      changed = true;
      IREE_RETURN_IF_ERROR(loom_link_record_target_symbol_materializations(
          state, target_symbol_id));
      IREE_RETURN_IF_ERROR(loom_link_mark_materialization_dependencies_live(
          state, target_symbol_id));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_link_validate_inputs(
    const loom_module_t* const* source_modules,
    iree_host_size_t source_module_count) {
  if (!source_modules || source_module_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at least one source module is required");
  }
  if (!source_modules[0]) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source module 0 is NULL");
  }
  loom_context_t* context = source_modules[0]->context;
  if (!context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source module context is required");
  }
  for (iree_host_size_t i = 0; i < source_module_count; ++i) {
    if (!source_modules[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source module %" PRIhsz " is NULL", i);
    }
    if (source_modules[i]->context != context) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source modules must share one context");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_link_validate_options(
    const loom_link_options_t* options) {
  if (!options || options->root_symbols.count == 0) {
    return iree_ok_status();
  }
  if (!options->root_symbols.values) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "root_symbols count is non-zero but values is NULL");
  }
  return iree_ok_status();
}

static iree_host_size_t loom_link_root_symbol_count(
    const loom_link_options_t* options) {
  return options ? options->root_symbols.count : 0;
}

static loom_module_size_hints_t loom_link_size_hints(
    const loom_module_t* const* source_modules,
    iree_host_size_t source_module_count) {
  loom_module_size_hints_t hints = {0};
  for (iree_host_size_t i = 0; i < source_module_count; ++i) {
    hints.value_count += source_modules[i]->values.count;
    hints.string_count += source_modules[i]->strings.count;
    hints.type_count += source_modules[i]->types.count;
    hints.symbol_count += source_modules[i]->symbols.count;
  }
  return hints;
}

iree_status_t loom_link_materialized_modules(
    const loom_module_t* const* source_modules,
    iree_host_size_t source_module_count, const loom_link_options_t* options,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  IREE_RETURN_IF_ERROR(
      loom_link_validate_inputs(source_modules, source_module_count));
  IREE_RETURN_IF_ERROR(loom_link_validate_options(options));

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(block_pool, &scratch_arena);

  const iree_host_size_t root_symbol_count =
      loom_link_root_symbol_count(options);
  const bool materialize_all_symbols = root_symbol_count == 0;
  iree_string_view_t module_name =
      options && !iree_string_view_is_empty(options->module_name)
          ? options->module_name
          : IREE_SV("linked");
  loom_module_size_hints_t hints =
      loom_link_size_hints(source_modules, source_module_count);
  loom_module_t* target_module = NULL;
  iree_status_t status =
      loom_module_allocate(source_modules[0]->context, module_name, block_pool,
                           &hints, allocator, &target_module);

  loom_link_state_t state = {
      .target_module = target_module,
      .scratch_arena = &scratch_arena,
      .module_map_count = source_module_count,
  };
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&scratch_arena, source_module_count,
                                       sizeof(*state.module_maps),
                                       (void**)&state.module_maps);
  }
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&scratch_arena, source_module_count,
                                       sizeof(*state.module_remaps),
                                       (void**)&state.module_remaps);
  }
  if (iree_status_is_ok(status)) {
    memset(state.module_maps, 0,
           source_module_count * sizeof(*state.module_maps));
    memset(state.module_remaps, 0,
           source_module_count * sizeof(*state.module_remaps));
    for (iree_host_size_t i = 0; i < source_module_count; ++i) {
      state.module_maps[i].source_module = source_modules[i];
      status = loom_link_map_module_symbols(&state, &state.module_maps[i],
                                            materialize_all_symbols);
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
  }
  if (iree_status_is_ok(status) && !materialize_all_symbols) {
    status = iree_arena_allocate_array(&scratch_arena, hints.symbol_count,
                                       sizeof(*state.live_target_symbols),
                                       (void**)&state.live_target_symbols);
  }
  if (iree_status_is_ok(status) && !materialize_all_symbols) {
    status = iree_arena_allocate_array(&scratch_arena, hints.symbol_count,
                                       sizeof(*state.scanned_target_symbols),
                                       (void**)&state.scanned_target_symbols);
  }
  if (iree_status_is_ok(status) && !materialize_all_symbols) {
    memset(state.live_target_symbols, 0,
           hints.symbol_count * sizeof(*state.live_target_symbols));
    memset(state.scanned_target_symbols, 0,
           hints.symbol_count * sizeof(*state.scanned_target_symbols));
    status = loom_link_mark_root_symbols_live(&state, options);
  }
  if (iree_status_is_ok(status) && !materialize_all_symbols) {
    status = loom_link_resolve_live_symbols(&state);
  }
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&scratch_arena, hints.symbol_count,
                                       sizeof(*state.cloned_target_symbols),
                                       (void**)&state.cloned_target_symbols);
  }
  if (iree_status_is_ok(status)) {
    memset(state.cloned_target_symbols, 0,
           hints.symbol_count * sizeof(*state.cloned_target_symbols));
  }
  for (iree_host_size_t i = 0;
       i < source_module_count && iree_status_is_ok(status); ++i) {
    status = loom_link_clone_module_body(&state, source_modules[i]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_module_compute_uses(target_module);
  }

  if (iree_status_is_ok(status)) {
    *out_module = target_module;
  } else {
    loom_module_free(target_module);
  }
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
