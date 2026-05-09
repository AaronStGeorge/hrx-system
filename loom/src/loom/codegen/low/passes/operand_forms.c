// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/passes/operand_forms.h"

#include <inttypes.h>

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/pass_environment.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/math.h"

enum {
  LOOM_LOW_SELECT_OPERAND_FORMS_STAT_FORMS_SELECTED = 0,
};

static const loom_pass_statistic_def_t kLowSelectOperandFormsStatistics[] = {
    {IREE_SVL("forms-selected"),
     IREE_SVL("Number of low packets rewritten to descriptor operand forms.")},
};

static const loom_pass_info_t loom_low_select_operand_forms_pass_info_storage =
    {
        .name = IREE_SVL("low-select-operand-forms"),
        .description = IREE_SVL(
            "Rewrite low packets to descriptor-selected operand forms."),
        .kind = LOOM_PASS_FUNCTION,
        .statistic_defs = kLowSelectOperandFormsStatistics,
        .statistic_count = IREE_ARRAYSIZE(kLowSelectOperandFormsStatistics),
};

const loom_pass_info_t* loom_low_select_operand_forms_pass_info(void) {
  return &loom_low_select_operand_forms_pass_info_storage;
}

typedef struct loom_low_select_operand_forms_state_t {
  // Pass invocation used for statistics and scratch arena allocation.
  loom_pass_t* pass;
  // Module being rewritten.
  loom_module_t* module;
  // Resolved target for the low function.
  const loom_low_resolved_target_t* target;
  // Scoped function facts used to match descriptor operand-form predicates.
  loom_value_fact_table_t* value_facts;
  // True once this pass has rewritten at least one packet.
  bool changed;
} loom_low_select_operand_forms_state_t;

static bool loom_low_select_operand_form_matches(
    const loom_value_fact_table_t* value_facts, loom_value_id_t value_id,
    const loom_low_operand_form_t* form, int64_t* out_matched_value) {
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(value_facts, value_id);
  switch (form->match_kind) {
    case LOOM_LOW_OPERAND_FORM_MATCH_ALL_EQUAL_I64: {
      loom_value_facts_t element = loom_value_facts_unknown();
      if (!loom_value_facts_query_all_equal_element(&value_facts->context,
                                                    facts, &element)) {
        return false;
      }
      int64_t value = 0;
      if (!loom_value_facts_as_exact_i64(element, &value) ||
          value != form->match_i64) {
        return false;
      }
      *out_matched_value = value;
      return true;
    }
    case LOOM_LOW_OPERAND_FORM_MATCH_ALL_EQUAL_EXACT_I64: {
      loom_value_facts_t element = loom_value_facts_unknown();
      if (!loom_value_facts_query_all_equal_element(&value_facts->context,
                                                    facts, &element)) {
        return false;
      }
      return loom_value_facts_as_exact_i64(element, out_matched_value);
    }
    default:
      return false;
  }
}

static bool loom_low_packet_has_operand_forms(
    const loom_low_resolved_descriptor_packet_t* packet) {
  return packet->kind == LOOM_LOW_DESCRIPTOR_PACKET_OP &&
         packet->descriptor != NULL &&
         packet->descriptor->operand_form_count != 0;
}

static iree_status_t loom_low_select_operand_forms_function_has_candidate(
    loom_module_t* module, loom_func_like_t function,
    const loom_low_resolved_target_t* target, bool* out_has_candidate) {
  *out_has_candidate = false;
  loom_region_t* body = loom_func_like_body(function);
  if (!body) {
    return iree_ok_status();
  }
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    loom_block_t* block = body->blocks[block_index];
    for (loom_op_t* op = block->first_op; op != NULL; op = op->next_op) {
      if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
        continue;
      }
      loom_low_resolved_descriptor_packet_t packet = {0};
      IREE_RETURN_IF_ERROR(
          loom_low_resolve_descriptor_packet(module, target, op, &packet));
      if (loom_low_packet_has_operand_forms(&packet)) {
        *out_has_candidate = true;
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static uint16_t loom_low_descriptor_packet_operand_index(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    uint16_t descriptor_operand_index) {
  uint16_t packet_operand_index = 0;
  for (uint16_t i = descriptor->result_count; i < descriptor_operand_index;
       ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + i];
    if (loom_low_operand_role_is_packet_operand(operand->role) &&
        !iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_IMPLICIT)) {
      ++packet_operand_index;
    }
  }
  return packet_operand_index;
}

static iree_status_t loom_low_descriptor_build_tied_results(
    iree_arena_allocator_t* arena,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    const loom_tied_result_t** out_tied_results,
    iree_host_size_t* out_tied_result_count) {
  *out_tied_results = NULL;
  iree_host_size_t tied_result_count = 0;
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind == LOOM_LOW_CONSTRAINT_KIND_TIED) {
      ++tied_result_count;
    }
  }
  *out_tied_result_count = tied_result_count;
  if (tied_result_count == 0) {
    return iree_ok_status();
  }

  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, tied_result_count, sizeof(*tied_results), (void**)&tied_results));
  iree_host_size_t tied_result_index = 0;
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_TIED) {
      continue;
    }
    tied_results[tied_result_index++] = (loom_tied_result_t){
        .result_index = constraint->lhs_operand_index,
        .operand_index = loom_low_descriptor_packet_operand_index(
            descriptor_set, descriptor, constraint->rhs_operand_index),
        .has_type_change = false,
    };
  }
  *out_tied_results = tied_results;
  return iree_ok_status();
}

static bool loom_low_immediate_accepts_i64(
    const loom_low_immediate_t* immediate, int64_t value) {
  switch (immediate->kind) {
    case LOOM_LOW_IMMEDIATE_KIND_SIGNED: {
      const int64_t signed_max = immediate->unsigned_max > (uint64_t)INT64_MAX
                                     ? INT64_MAX
                                     : (int64_t)immediate->unsigned_max;
      return value >= immediate->signed_min && value <= signed_max;
    }
    case LOOM_LOW_IMMEDIATE_KIND_UNSIGNED:
    case LOOM_LOW_IMMEDIATE_KIND_ORDINAL:
      return value >= 0 && (uint64_t)value <= immediate->unsigned_max;
    default:
      return false;
  }
}

static bool loom_low_immediate_has_default(
    const loom_low_immediate_t* immediate) {
  return iree_any_bit_set(immediate->flags,
                          LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE);
}

static const loom_named_attr_t* loom_low_select_operand_form_find_attr(
    loom_named_attr_slice_t attrs, loom_string_id_t name_id) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    if (attrs.entries[i].name_id == name_id) {
      return &attrs.entries[i];
    }
  }
  return NULL;
}

static iree_status_t loom_low_select_operand_form_read_i64_immediate(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t immediate_index,
    loom_named_attr_slice_t attrs, bool* out_has_value, int64_t* out_value) {
  *out_has_value = false;
  *out_value = 0;
  if (immediate_index == LOOM_LOW_DESCRIPTOR_SET_ORDINAL_NONE) {
    return iree_ok_status();
  }

  IREE_ASSERT(immediate_index < descriptor->immediate_count);
  const loom_low_immediate_t* immediate =
      &descriptor_set
           ->immediates[descriptor->immediate_start + immediate_index];
  iree_string_view_t immediate_name = loom_low_descriptor_set_string(
      descriptor_set, immediate->field_name_string_offset);
  loom_string_id_t immediate_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, immediate_name, &immediate_name_id));
  const loom_named_attr_t* attr =
      loom_low_select_operand_form_find_attr(attrs, immediate_name_id);
  if (attr) {
    if (attr->value.kind != LOOM_ATTR_I64) {
      return iree_ok_status();
    }
    *out_value = loom_attr_as_i64(attr->value);
    *out_has_value = true;
    return iree_ok_status();
  }
  if (loom_low_immediate_has_default(immediate)) {
    *out_value = immediate->default_value;
    *out_has_value = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_select_operand_form_build_attrs(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* source_descriptor,
    const loom_low_descriptor_t* replacement_descriptor,
    const loom_low_operand_form_t* form, int64_t matched_value,
    loom_named_attr_slice_t source_attrs, loom_named_attr_slice_t* out_attrs,
    bool* out_can_rewrite) {
  *out_attrs = source_attrs;
  *out_can_rewrite = true;

  int64_t replacement_value = 0;
  switch (form->immediate_action) {
    case LOOM_LOW_OPERAND_FORM_IMMEDIATE_NONE:
      return iree_ok_status();
    case LOOM_LOW_OPERAND_FORM_IMMEDIATE_SET_MATCHED_I64:
      replacement_value = matched_value;
      break;
    case LOOM_LOW_OPERAND_FORM_IMMEDIATE_ADD_MATCHED_I64: {
      bool has_source_value = false;
      int64_t source_value = 0;
      IREE_RETURN_IF_ERROR(loom_low_select_operand_form_read_i64_immediate(
          module, descriptor_set, source_descriptor,
          form->source_immediate_index, source_attrs, &has_source_value,
          &source_value));
      if (!has_source_value ||
          !loom_checked_add_i64(source_value, matched_value,
                                &replacement_value)) {
        *out_can_rewrite = false;
        return iree_ok_status();
      }
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unknown low operand-form immediate action %u",
                              (unsigned)form->immediate_action);
  }

  IREE_ASSERT(form->replacement_immediate_index <
              replacement_descriptor->immediate_count);
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[replacement_descriptor->immediate_start +
                                  form->replacement_immediate_index];
  if (!loom_low_immediate_accepts_i64(immediate, replacement_value)) {
    *out_can_rewrite = false;
    return iree_ok_status();
  }

  iree_string_view_t immediate_name = loom_low_descriptor_set_string(
      descriptor_set, immediate->field_name_string_offset);
  loom_string_id_t immediate_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, immediate_name, &immediate_name_id));
  const bool remove_default = loom_low_immediate_has_default(immediate) &&
                              replacement_value == immediate->default_value;
  loom_named_attr_update_t update =
      remove_default ? loom_named_attr_remove(immediate_name_id)
                     : loom_named_attr_replace(
                           immediate_name_id, loom_attr_i64(replacement_value));
  loom_attribute_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_module_replace_canonical_attr_dict(
      module, source_attrs, loom_make_named_attr_update_slice(&update, 1),
      &attr));
  *out_attrs = loom_attr_as_dict(attr);
  return iree_ok_status();
}

static iree_status_t loom_low_select_operand_form_rewrite_packet(
    loom_low_select_operand_forms_state_t* state, loom_rewriter_t* rewriter,
    loom_op_t* op, const loom_low_descriptor_t* source_descriptor,
    const loom_low_operand_form_t* form, int64_t matched_value) {
  const loom_low_descriptor_set_t* descriptor_set =
      state->target->descriptor_set;
  const loom_low_descriptor_t* replacement_descriptor =
      &descriptor_set->descriptors[form->replacement_descriptor_ordinal];

  iree_string_view_t replacement_key = loom_low_descriptor_set_string(
      descriptor_set, replacement_descriptor->key_string_offset);
  loom_string_id_t replacement_key_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(
      &rewriter->builder, replacement_key, &replacement_key_id));

  loom_value_id_t* operands = NULL;
  if (form->operand_map_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->pass->arena, form->operand_map_count,
                                  sizeof(*operands), (void**)&operands));
    for (uint16_t i = 0; i < form->operand_map_count; ++i) {
      const uint16_t source_packet_operand_index =
          descriptor_set
              ->operand_form_operand_indices[form->operand_map_start + i];
      operands[i] = loom_op_operands(op)[source_packet_operand_index];
    }
  }

  loom_type_t* result_types = NULL;
  if (op->result_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->pass->arena, op->result_count, sizeof(*result_types),
        (void**)&result_types));
    const loom_value_id_t* results = loom_op_results(op);
    for (uint16_t i = 0; i < op->result_count; ++i) {
      result_types[i] = loom_module_value_type(state->module, results[i]);
    }
  }

  const loom_tied_result_t* tied_results = NULL;
  iree_host_size_t tied_result_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_build_tied_results(
      state->pass->arena, descriptor_set, replacement_descriptor, &tied_results,
      &tied_result_count));

  loom_named_attr_slice_t replacement_attrs = {0};
  bool can_rewrite = false;
  IREE_RETURN_IF_ERROR(loom_low_select_operand_form_build_attrs(
      state->module, descriptor_set, source_descriptor, replacement_descriptor,
      form, matched_value, loom_low_op_attrs(op), &replacement_attrs,
      &can_rewrite));
  if (!can_rewrite) {
    return iree_ok_status();
  }

  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_builder_ip_t saved_ip = loom_builder_save(&rewriter->builder);
  loom_builder_set_before(&rewriter->builder, op);
  loom_op_t* replacement_op = NULL;
  iree_status_t status = loom_low_op_build(
      &rewriter->builder, replacement_key_id, operands, form->operand_map_count,
      replacement_attrs, result_types, op->result_count, tied_results,
      tied_result_count, op->location, &replacement_op);
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  const loom_value_id_t* replacements = loom_op_results(replacement_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, replacement_op->result_count,
      value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements, replacement_op->result_count));
  state->changed = true;
  loom_pass_mark_changed(state->pass);
  if (state->pass->statistics) {
    loom_pass_statistic_add(
        state->pass, LOOM_LOW_SELECT_OPERAND_FORMS_STAT_FORMS_SELECTED, 1);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_select_operand_forms_try_rewrite_packet(
    loom_low_select_operand_forms_state_t* state, loom_rewriter_t* rewriter,
    loom_op_t* op) {
  loom_low_resolved_descriptor_packet_t packet = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_descriptor_packet(
      state->module, state->target, op, &packet));
  if (!loom_low_packet_has_operand_forms(&packet)) {
    return iree_ok_status();
  }

  const loom_low_descriptor_set_t* descriptor_set =
      state->target->descriptor_set;
  for (uint16_t i = 0; i < packet.descriptor->operand_form_count; ++i) {
    const loom_low_operand_form_t* form =
        &descriptor_set
             ->operand_forms[packet.descriptor->operand_form_start + i];
    const loom_value_id_t value_id =
        loom_op_operands(op)[form->source_packet_operand_index];
    int64_t matched_value = 0;
    if (!loom_low_select_operand_form_matches(state->value_facts, value_id,
                                              form, &matched_value)) {
      continue;
    }
    return loom_low_select_operand_form_rewrite_packet(
        state, rewriter, op, packet.descriptor, form, matched_value);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_select_operand_forms_function(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t emitter) {
  loom_op_t* low_func_op = function.op;
  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      module, low_func_op, descriptor_registry, emitter, &target));
  if (!target.descriptor_set ||
      target.descriptor_set->operand_form_count == 0) {
    return iree_ok_status();
  }

  bool has_candidate = false;
  IREE_RETURN_IF_ERROR(loom_low_select_operand_forms_function_has_candidate(
      module, function, &target, &has_candidate));
  if (!has_candidate) {
    return iree_ok_status();
  }

  loom_value_fact_table_t* value_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
      pass, module,
      loom_pass_value_fact_scope_function_for_target(
          function, &target.bundle_storage.bundle),
      &value_facts));

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  loom_low_select_operand_forms_state_t state = {
      .pass = pass,
      .module = module,
      .target = &target,
      .value_facts = value_facts,
  };

  loom_region_t* body = loom_func_like_body(function);
  iree_status_t status = iree_ok_status();
  for (uint16_t block_index = 0;
       iree_status_is_ok(status) && block_index < body->block_count;
       ++block_index) {
    loom_block_t* block = body->blocks[block_index];
    for (loom_op_t* op = block->first_op; iree_status_is_ok(status) && op;) {
      loom_op_t* next_op = op->next_op;
      if (!iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
        status = loom_low_select_operand_forms_try_rewrite_packet(
            &state, &rewriter, op);
      }
      op = next_op;
    }
  }
  loom_rewriter_deinitialize(&rewriter);
  if (state.changed) {
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
  }
  return status;
}

iree_status_t loom_low_select_operand_forms_run(loom_pass_t* pass,
                                                loom_module_t* module,
                                                loom_func_like_t function) {
  if (!loom_low_function_def_isa(function.op) ||
      !loom_func_like_isa(function) || !loom_func_like_body(function)) {
    return iree_ok_status();
  }
  const loom_low_pass_capability_t* low_capability =
      loom_low_pass_capability_from_pass(pass);
  const loom_low_descriptor_registry_t* descriptor_registry =
      loom_low_pass_capability_descriptor_registry(low_capability);
  return loom_low_select_operand_forms_function(
      pass, module, function, descriptor_registry, pass->diagnostic_emitter);
}
