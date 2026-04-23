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
  // Chosen source materialization for each target symbol.
  loom_link_materialization_t* materializations;
  // Number of entries allocated in materializations.
  iree_host_size_t materialization_count;
  // Number of entries available in materializations.
  iree_host_size_t materialization_capacity;
} loom_link_state_t;

static bool loom_link_symbol_is_declaration(const loom_symbol_t* symbol) {
  return symbol->kind == LOOM_SYMBOL_FUNC_DECL;
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
    loom_link_state_t* state, loom_link_module_map_t* module_map) {
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
  const loom_link_module_map_t* module_map =
      loom_link_find_module_map(state, source_module);
  if (!module_map) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source module is not part of this link");
  }
  if (source_ref.symbol_id >= module_map->target_symbol_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source symbol ref {module=%u, symbol=%u} is out of range",
        (unsigned)source_ref.module_id, (unsigned)source_ref.symbol_id);
  }
  *out_target_ref = module_map->target_symbols[source_ref.symbol_id];
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

static bool loom_link_should_clone_symbol_op(const loom_link_state_t* state,
                                             const loom_module_t* source_module,
                                             loom_symbol_ref_t source_ref) {
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
  const loom_link_materialization_t* materialization =
      &state->materializations[target_ref.symbol_id];
  return materialization->source_module == source_module &&
         materialization->source_symbol_id == source_ref.symbol_id;
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
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      source_module, state->target_module, state->scratch_arena,
      &(loom_ir_remap_options_t){
          .remap_symbol = loom_link_remap_symbol,
          .remap_symbol_user_data = state,
      },
      &remap));

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
    if (has_symbol_ref) {
      if (!loom_link_should_clone_symbol_op(state, source_module, source_ref)) {
        continue;
      }
    }
    loom_op_t* cloned_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_ir_clone_op(&builder, source_op, &remap, &cloned_op));
    if (has_symbol_ref) {
      const loom_link_module_map_t* module_map =
          loom_link_find_module_map(state, source_module);
      loom_symbol_ref_t target_ref =
          module_map->target_symbols[source_ref.symbol_id];
      IREE_RETURN_IF_ERROR(
          loom_link_merge_func_contract(state, target_ref, cloned_op));
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

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(block_pool, &scratch_arena);

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
    memset(state.module_maps, 0,
           source_module_count * sizeof(*state.module_maps));
    for (iree_host_size_t i = 0; i < source_module_count; ++i) {
      state.module_maps[i].source_module = source_modules[i];
      status = loom_link_map_module_symbols(&state, &state.module_maps[i]);
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
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
