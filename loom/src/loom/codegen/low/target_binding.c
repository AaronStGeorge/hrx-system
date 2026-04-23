// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/target_binding.h"

#include <inttypes.h>

#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/error/error_defs.h"
#include "loom/ir/module.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/facts.h"
#include "loom/ops/target/ops.h"
#include "loom/target/function_contract.h"

static iree_status_t loom_low_emit(iree_diagnostic_emitter_t emitter,
                                   const loom_op_t* op,
                                   const loom_error_def_t* error,
                                   const loom_diagnostic_param_t* params,
                                   iree_host_size_t param_count,
                                   const loom_diagnostic_related_op_t* related,
                                   iree_host_size_t related_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
      .related_ops = related,
      .related_op_count = related_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static const loom_symbol_t* loom_low_lookup_defined_symbol(
    const loom_module_t* module, loom_symbol_ref_t ref) {
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return NULL;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->definition == NULL || symbol->defining_op == NULL) {
    return NULL;
  }
  return symbol;
}

static iree_string_view_t loom_low_symbol_name(const loom_module_t* module,
                                               loom_symbol_ref_t ref) {
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->name_id < module->strings.count) {
    return module->strings.entries[symbol->name_id];
  }
  return IREE_SV("<unnamed>");
}

static iree_string_view_t loom_low_function_name(const loom_module_t* module,
                                                 const loom_op_t* low_func_op) {
  if (loom_low_func_def_isa(low_func_op)) {
    return loom_low_symbol_name(module, loom_low_func_def_callee(low_func_op));
  }
  if (loom_low_func_decl_isa(low_func_op)) {
    return loom_low_symbol_name(module, loom_low_func_decl_callee(low_func_op));
  }
  return IREE_SV("<unnamed>");
}

static iree_string_view_t loom_low_string_or_empty(const loom_module_t* module,
                                                   loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_string_view_t loom_low_symbol_definition_name(
    const loom_symbol_t* symbol) {
  if (!symbol || !symbol->definition) {
    return IREE_SV("unresolved");
  }
  return loom_symbol_definition_descriptor_name(symbol->definition);
}

static iree_status_t loom_low_emit_symbol_kind_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_module_t* module,
    const loom_op_t* op, loom_symbol_ref_t ref, const loom_symbol_t* symbol,
    uint16_t attr_index, iree_string_view_t expected_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(loom_low_symbol_name(module, ref)),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attr_index)),
      loom_param_string(loom_low_symbol_definition_name(symbol)),
      loom_param_string(expected_kind),
  };
  loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("defined here"),
      .op = symbol ? symbol->defining_op : NULL,
  }};
  return loom_low_emit(
      emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3), params,
      IREE_ARRAYSIZE(params), related,
      symbol && symbol->defining_op ? IREE_ARRAYSIZE(related) : 0);
}

static iree_status_t loom_low_emit_unresolved_symbol(
    iree_diagnostic_emitter_t emitter, const loom_module_t* module,
    const loom_op_t* op, loom_symbol_ref_t ref, uint16_t attr_index) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(loom_low_symbol_name(module, ref)),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attr_index)),
  };
  return loom_low_emit(emitter, op,
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 2),
                       params, IREE_ARRAYSIZE(params), NULL, 0);
}

static iree_status_t loom_low_emit_missing_descriptor_set(
    iree_diagnostic_emitter_t emitter, const loom_module_t* module,
    const loom_op_t* low_func_op, const loom_op_t* target_context_op,
    uint16_t target_attr_index, iree_string_view_t descriptor_set_key,
    iree_string_view_t target_name) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_function_name(module, low_func_op)),
      loom_param_with_field_ref(
          loom_param_string(target_name),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    target_attr_index)),
      loom_param_string(descriptor_set_key),
  };
  loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("descriptor set selected here"),
      .op = target_context_op,
      .field_ref = loom_diagnostic_field_ref_none(),
  }};
  return loom_low_emit(emitter, low_func_op,
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 3),
                       params, IREE_ARRAYSIZE(params), related,
                       target_context_op ? IREE_ARRAYSIZE(related) : 0);
}

static bool loom_low_get_function_target_ref(const loom_op_t* low_func_op,
                                             loom_symbol_ref_t* out_target_ref,
                                             uint16_t* out_target_attr_index) {
  if (loom_low_func_def_isa(low_func_op)) {
    *out_target_ref = loom_low_func_def_target(low_func_op);
    *out_target_attr_index = loom_low_func_def_target_ATTR_INDEX;
    return true;
  }
  if (loom_low_func_decl_isa(low_func_op)) {
    *out_target_ref = loom_low_func_decl_target(low_func_op);
    *out_target_attr_index = loom_low_func_decl_target_ATTR_INDEX;
    return true;
  }
  return false;
}

static iree_status_t loom_low_resolve_func_target(
    const loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_descriptor_registry_t* registry,
    iree_diagnostic_emitter_t emitter, uint16_t target_attr_index,
    loom_low_resolved_target_t* out_target) {
  if (registry == NULL || registry->target_bundle_count == 0 ||
      registry->target_bundles == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "target.profile low function targets require a descriptor registry "
        "with target-bundle presets");
  }

  loom_target_preset_registry_t preset_registry = {
      .target_bundles = registry->target_bundles,
      .target_bundle_count = registry->target_bundle_count,
  };
  loom_symbol_fact_resource_t resource =
      loom_target_profile_preset_registry_resource(&preset_registry);
  const loom_symbol_fact_table_options_t fact_options = {
      .resources = loom_make_symbol_fact_resource_list(&resource, 1),
  };

  iree_arena_allocator_t arena;
  iree_arena_initialize(module->arena.block_pool, &arena);
  loom_symbol_fact_table_t fact_table = {0};
  loom_symbol_fact_table_initialize_with_options(&fact_table, &fact_options,
                                                 &arena);

  loom_symbol_ref_t func_ref = loom_symbol_ref_null();
  if (loom_low_func_def_isa(low_func_op)) {
    func_ref = loom_low_func_def_callee(low_func_op);
  } else {
    func_ref = loom_low_func_decl_callee(low_func_op);
  }

  const loom_symbol_facts_base_t* base_facts = NULL;
  iree_status_t status = loom_symbol_fact_table_lookup_ref(
      &fact_table, module, func_ref, &base_facts);
  const loom_func_symbol_facts_t* func_facts =
      iree_status_is_ok(status) ? loom_func_symbol_facts_cast(base_facts)
                                : NULL;
  if (iree_status_is_ok(status) && func_facts == NULL) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low function symbol must resolve to func symbol facts");
  }
  if (iree_status_is_ok(status) &&
      !loom_symbol_ref_is_valid(func_facts->target_symbol)) {
    status = loom_low_emit_symbol_kind_mismatch(
        emitter, module, low_func_op, func_facts->target_symbol,
        out_target->target_symbol, target_attr_index,
        IREE_SV("target profile"));
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_function_contract_resolve(
        module, &fact_table, func_facts, &out_target->bundle_storage);
  }
  if (iree_status_is_ok(status)) {
    out_target->descriptor_set_key =
        out_target->bundle_storage.config.contract_set_key;
    out_target->feature_bits =
        out_target->bundle_storage.config.contract_feature_bits;
  }

  iree_arena_deinitialize(&arena);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  const loom_low_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_registry_lookup(
      registry, out_target->descriptor_set_key, &descriptor_set));
  if (!descriptor_set) {
    return loom_low_emit_missing_descriptor_set(
        emitter, module, low_func_op, out_target->target_op, target_attr_index,
        out_target->descriptor_set_key, out_target->target_name);
  }
  out_target->descriptor_set = descriptor_set;
  return iree_ok_status();
}

iree_status_t loom_low_resolve_function_target(
    const loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_descriptor_registry_t* registry,
    iree_diagnostic_emitter_t emitter, loom_low_resolved_target_t* out_target) {
  if (!module || !low_func_op || !out_target) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module, low function op, and output target are required");
  }
  *out_target = (loom_low_resolved_target_t){0};
  loom_symbol_ref_t target_ref = loom_symbol_ref_null();
  uint16_t target_attr_index = 0;
  if (!loom_low_get_function_target_ref(low_func_op, &target_ref,
                                        &target_attr_index)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected low.func.def or low.func.decl");
  }

  const loom_symbol_t* target_symbol =
      loom_low_lookup_defined_symbol(module, target_ref);
  if (!target_symbol) {
    return loom_low_emit_unresolved_symbol(emitter, module, low_func_op,
                                           target_ref, target_attr_index);
  }

  out_target->target_symbol = target_symbol;
  out_target->target_op = target_symbol->defining_op;
  out_target->target_name = loom_low_symbol_name(module, target_ref);

  if (loom_target_profile_isa(target_symbol->defining_op)) {
    return loom_low_resolve_func_target(module, low_func_op, registry, emitter,
                                        target_attr_index, out_target);
  }
  return loom_low_emit_symbol_kind_mismatch(
      emitter, module, low_func_op, target_ref, target_symbol,
      target_attr_index, IREE_SV("target profile"));
}

iree_status_t loom_low_resolve_descriptor_packet(
    const loom_module_t* module, const loom_low_resolved_target_t* target,
    const loom_op_t* op, loom_low_resolved_descriptor_packet_t* out_packet) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(target->descriptor_set);
  IREE_ASSERT_ARGUMENT(op);
  IREE_ASSERT_ARGUMENT(out_packet);
  *out_packet = (loom_low_resolved_descriptor_packet_t){
      .op = op,
      .kind = LOOM_LOW_DESCRIPTOR_PACKET_NONE,
      .stable_id = LOOM_LOW_DESCRIPTOR_ID_NONE,
      .descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
  };

  loom_string_id_t key_id = LOOM_STRING_ID_INVALID;
  int64_t stable_id = (int64_t)LOOM_LOW_DESCRIPTOR_ID_NONE;
  if (loom_low_op_isa(op)) {
    out_packet->kind = LOOM_LOW_DESCRIPTOR_PACKET_OP;
    out_packet->key_attr_index = loom_low_op_opcode_ATTR_INDEX;
    key_id = loom_low_op_opcode(op);
    stable_id = loom_low_op_descriptor_id(op);
  } else if (loom_low_const_isa(op)) {
    out_packet->kind = LOOM_LOW_DESCRIPTOR_PACKET_CONST;
    out_packet->key_attr_index = loom_low_const_opcode_ATTR_INDEX;
    key_id = loom_low_const_opcode(op);
    stable_id = loom_low_const_descriptor_id(op);
  } else {
    return iree_ok_status();
  }

  out_packet->key = loom_low_string_or_empty(module, key_id);
  if (stable_id <= 0) {
    return iree_ok_status();
  }
  out_packet->stable_id = (uint64_t)stable_id;
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_lookup_descriptor_by_id(target->descriptor_set,
                                                      out_packet->stable_id);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_ok_status();
  }

  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(target->descriptor_set,
                                            descriptor_ordinal);
  if (descriptor == NULL) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor ordinal %" PRIu32
                            " is out of range",
                            descriptor_ordinal);
  }
  IREE_ASSERT(descriptor->stable_id == out_packet->stable_id);
  out_packet->descriptor_ordinal = descriptor_ordinal;
  out_packet->descriptor = descriptor;
  return iree_ok_status();
}
