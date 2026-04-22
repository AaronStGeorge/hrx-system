// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower_rules.h"

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

static const loom_low_lower_rule_span_t* loom_low_lower_rule_set_find_span(
    const loom_low_lower_rule_set_t* rule_set, loom_op_kind_t source_op_kind) {
  uint16_t low = 0;
  uint16_t high = rule_set->span_count;
  while (low < high) {
    uint16_t mid = low + (uint16_t)((high - low) / 2);
    const loom_low_lower_rule_span_t* span = &rule_set->spans[mid];
    if (span->source_op_kind == source_op_kind) {
      return span;
    }
    if (span->source_op_kind < source_op_kind) {
      low = (uint16_t)(mid + 1);
    } else {
      high = mid;
    }
  }
  return NULL;
}

static iree_status_t loom_low_lower_rule_emit_no_mapping(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("op"),
      loom_op_name(loom_low_lower_context_module(context), source_op),
      IREE_SV("the selected target-low lowering policy has no descriptor "
              "mapping for this op"));
}

static iree_status_t loom_low_lower_rule_emit_diagnostic(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t diagnostic_index) {
  if (diagnostic_index == LOOM_LOW_LOWER_DIAGNOSTIC_NONE ||
      diagnostic_index >= rule_set->diagnostic_count) {
    return loom_low_lower_rule_emit_no_mapping(context, source_op);
  }
  const loom_low_lower_diagnostic_t* diagnostic =
      &rule_set->diagnostics[diagnostic_index];
  return loom_low_lower_emit_reject(
      context, source_op, diagnostic->subject_kind, diagnostic->subject_name,
      diagnostic->reason);
}

static const loom_low_lower_value_materializer_t*
loom_low_lower_rule_value_materializer(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_value_ref_t* value_ref) {
  IREE_ASSERT_GT(value_ref->materializer_index, 0);
  const uint16_t materializer_index =
      (uint16_t)(value_ref->materializer_index - 1);
  IREE_ASSERT_LT(materializer_index, rule_set->materializer_count);
  IREE_ASSERT(rule_set->materializers != NULL);
  return &rule_set->materializers[materializer_index];
}

static loom_value_id_t loom_low_lower_rule_source_value(
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index) {
  IREE_ASSERT_LT(value_ref_index, rule_set->value_ref_count);
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  switch (value_ref->kind) {
    case LOOM_LOW_LOWER_VALUE_REF_OPERAND:
      IREE_ASSERT_LT(value_ref->index, source_op->operand_count);
      return loom_op_const_operands(source_op)[value_ref->index];
    case LOOM_LOW_LOWER_VALUE_REF_RESULT:
      IREE_ASSERT_LT(value_ref->index, source_op->result_count);
      return loom_op_const_results(source_op)[value_ref->index];
    case LOOM_LOW_LOWER_VALUE_REF_TEMPORARY:
      IREE_ASSERT_UNREACHABLE();
      return LOOM_VALUE_ID_INVALID;
    default:
      IREE_ASSERT_UNREACHABLE();
      return LOOM_VALUE_ID_INVALID;
  }
}

static bool loom_low_lower_rule_can_materialize_value(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index) {
  IREE_ASSERT_LT(value_ref_index, rule_set->value_ref_count);
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  IREE_ASSERT_GT(value_ref->materializer_index, 0);
  const loom_low_lower_value_materializer_t* materializer =
      loom_low_lower_rule_value_materializer(rule_set, value_ref);
  IREE_ASSERT(materializer->can_materialize != NULL);
  return materializer->can_materialize(
      context, source_op,
      loom_low_lower_rule_source_value(rule_set, source_op, value_ref_index));
}

typedef struct loom_low_lower_rule_emit_state_t {
  // Rule-local low SSA values captured by earlier emit rows.
  loom_value_id_t* temporaries;
  // Number of entries in temporaries.
  uint16_t temporary_count;
} loom_low_lower_rule_emit_state_t;

static iree_status_t loom_low_lower_rule_emit_state_initialize(
    loom_low_lower_context_t* context, const loom_low_lower_rule_t* rule,
    loom_low_lower_rule_emit_state_t* out_state) {
  IREE_ASSERT_ARGUMENT(out_state);
  *out_state = (loom_low_lower_rule_emit_state_t){
      .temporaries = NULL,
      .temporary_count = rule->temporary_count,
  };
  if (rule->temporary_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, rule->temporary_count, sizeof(*out_state->temporaries),
      (void**)&out_state->temporaries));
  for (uint16_t i = 0; i < rule->temporary_count; ++i) {
    out_state->temporaries[i] = LOOM_VALUE_ID_INVALID;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_low_value(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_emit_state_t* state, uint16_t value_ref_index,
    loom_value_id_t* out_low_value_id) {
  IREE_ASSERT_ARGUMENT(out_low_value_id);
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_LT(value_ref_index, rule_set->value_ref_count);
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  switch (value_ref->kind) {
    case LOOM_LOW_LOWER_VALUE_REF_OPERAND:
    case LOOM_LOW_LOWER_VALUE_REF_RESULT: {
      loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
          rule_set, source_op, value_ref_index);
      if (value_ref->materializer_index != 0) {
        const loom_low_lower_value_materializer_t* materializer =
            loom_low_lower_rule_value_materializer(rule_set, value_ref);
        IREE_ASSERT(materializer->materialize != NULL);
        return materializer->materialize(context, source_op, source_value_id,
                                         out_low_value_id);
      }
      return loom_low_lower_lookup_value(context, source_value_id,
                                         out_low_value_id);
    }
    case LOOM_LOW_LOWER_VALUE_REF_TEMPORARY:
      IREE_ASSERT_LT(value_ref->index, state->temporary_count);
      IREE_ASSERT(state->temporaries != NULL);
      IREE_ASSERT(state->temporaries[value_ref->index] !=
                  LOOM_VALUE_ID_INVALID);
      *out_low_value_id = state->temporaries[value_ref->index];
      return iree_ok_status();
    default:
      IREE_ASSERT_UNREACHABLE();
      return iree_ok_status();
  }
}

static bool loom_low_lower_rule_type_matches(
    const loom_low_lower_type_pattern_t* pattern, loom_type_t type) {
  if (iree_any_bit_set(pattern->flags, LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND) &&
      loom_type_kind(type) != pattern->type_kind) {
    return false;
  }
  if (iree_any_bit_set(pattern->flags,
                       LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT) &&
      !iree_any_bit_set(
          pattern->element_type_mask,
          LOOM_LOW_LOWER_SCALAR_TYPE_BIT(loom_type_element_type(type)))) {
    return false;
  }
  if (iree_any_bit_set(pattern->flags, LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK) &&
      loom_type_rank(type) != pattern->rank) {
    return false;
  }
  if (iree_any_bit_set(pattern->flags,
                       LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0)) {
    if (loom_type_rank(type) == 0 || loom_type_dim_is_dynamic_at(type, 0)) {
      return false;
    }
    if (loom_type_dim_static_size_at(type, 0) != pattern->static_dim0) {
      return false;
    }
  }
  if (iree_any_bit_set(pattern->flags,
                       LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0_RANGE)) {
    if (loom_type_rank(type) == 0 || loom_type_dim_is_dynamic_at(type, 0)) {
      return false;
    }
    const int64_t static_dim0 = loom_type_dim_static_size_at(type, 0);
    if (static_dim0 < pattern->static_dim0_min ||
        static_dim0 > pattern->static_dim0_max) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_low_lower_rule_guard_matches(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_guard_t* guard, bool* out_matches) {
  IREE_ASSERT_ARGUMENT(out_matches);
  *out_matches = false;
  switch (guard->kind) {
    case LOOM_LOW_LOWER_GUARD_VALUE_TYPE: {
      IREE_ASSERT_LT(guard->type_pattern_index, rule_set->type_pattern_count);
      loom_value_id_t value_id = loom_low_lower_rule_source_value(
          rule_set, source_op, guard->value_ref_index);
      loom_type_t type = loom_module_value_type(
          loom_low_lower_context_module(context), value_id);
      *out_matches = loom_low_lower_rule_type_matches(
          &rule_set->type_patterns[guard->type_pattern_index], type);
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_ATTR_KIND:
      if (guard->attr_index >= source_op->attribute_count) {
        return iree_ok_status();
      }
      *out_matches = loom_op_const_attrs(source_op)[guard->attr_index].kind ==
                     guard->attr_kind;
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_ATTR_ENUM_EQ:
      if (guard->attr_index >= source_op->attribute_count) {
        return iree_ok_status();
      }
      *out_matches =
          loom_op_const_attrs(source_op)[guard->attr_index].kind ==
              LOOM_ATTR_ENUM &&
          loom_op_const_attrs(source_op)[guard->attr_index].raw == guard->u64;
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_ATTR_I64_RANGE:
      if (guard->attr_index >= source_op->attribute_count ||
          loom_op_const_attrs(source_op)[guard->attr_index].kind !=
              LOOM_ATTR_I64) {
        return iree_ok_status();
      }
      *out_matches = loom_op_const_attrs(source_op)[guard->attr_index].i64 >=
                         guard->minimum_i64 &&
                     loom_op_const_attrs(source_op)[guard->attr_index].i64 <=
                         guard->maximum_i64;
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE:
      *out_matches =
          loom_low_descriptor_set_lookup_descriptor_by_id(
              loom_low_lower_context_descriptor_set(context),
              guard->descriptor_id) != LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_MATERIALIZABLE:
      *out_matches = loom_low_lower_rule_can_materialize_value(
          context, rule_set, source_op, guard->value_ref_index);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_CLASS: {
      loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
          rule_set, source_op, guard->value_ref_index);
      loom_type_t low_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
          context, source_op, source_value_id, &low_type));
      if (!loom_type_is_register(low_type)) {
        return iree_ok_status();
      }
      loom_string_id_t expected_class_id = LOOM_STRING_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_low_lower_register_class_string_id(
          context, guard->register_class_id, &expected_class_id));
      *out_matches = loom_type_register_class_id(low_type) == expected_class_id;
      return iree_ok_status();
    }
    default:
      IREE_ASSERT_UNREACHABLE();
      return iree_ok_status();
  }
}

static iree_status_t loom_low_lower_rule_matches(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule, bool* out_matches,
    uint16_t* out_diagnostic_index, uint16_t* out_matched_guard_count) {
  IREE_ASSERT_ARGUMENT(out_matches);
  IREE_ASSERT_ARGUMENT(out_diagnostic_index);
  IREE_ASSERT_ARGUMENT(out_matched_guard_count);
  *out_matches = false;
  *out_diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  *out_matched_guard_count = 0;
  for (uint16_t i = 0; i < rule->guard_count; ++i) {
    uint16_t guard_index = (uint16_t)(rule->guard_start + i);
    IREE_ASSERT_LT(guard_index, rule_set->guard_count);
    const loom_low_lower_guard_t* guard = &rule_set->guards[guard_index];
    bool guard_matches = false;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_guard_matches(
        context, rule_set, source_op, guard, &guard_matches));
    if (!guard_matches) {
      *out_diagnostic_index = guard->diagnostic_index;
      *out_matched_guard_count = i;
      return iree_ok_status();
    }
  }
  *out_matches = true;
  *out_matched_guard_count = rule->guard_count;
  return iree_ok_status();
}

iree_status_t loom_low_lower_rule_set_select(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_selection_t* out_selection) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(rule_set);
  IREE_ASSERT_ARGUMENT(source_op);
  IREE_ASSERT_ARGUMENT(out_selection);
  *out_selection = (loom_low_lower_rule_selection_t){
      .rule = NULL,
      .has_source_op_span = false,
      .diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE,
  };

  const loom_low_lower_rule_span_t* span =
      loom_low_lower_rule_set_find_span(rule_set, source_op->kind);
  if (span == NULL) {
    return iree_ok_status();
  }
  out_selection->has_source_op_span = true;

  uint16_t best_diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  uint16_t best_matched_guard_count = 0;
  for (uint16_t i = 0; i < span->rule_count; ++i) {
    uint16_t rule_index = (uint16_t)(span->rule_start + i);
    IREE_ASSERT_LT(rule_index, rule_set->rule_count);
    const loom_low_lower_rule_t* rule = &rule_set->rules[rule_index];
    bool rule_matches = false;
    uint16_t diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
    uint16_t matched_guard_count = 0;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_matches(
        context, rule_set, source_op, rule, &rule_matches, &diagnostic_index,
        &matched_guard_count));
    if (rule_matches) {
      out_selection->rule = rule;
      return iree_ok_status();
    }
    if (best_diagnostic_index == LOOM_LOW_LOWER_DIAGNOSTIC_NONE ||
        matched_guard_count > best_matched_guard_count) {
      best_diagnostic_index = diagnostic_index;
      best_matched_guard_count = matched_guard_count;
    }
  }

  out_selection->diagnostic_index = best_diagnostic_index;
  return iree_ok_status();
}

iree_status_t loom_low_lower_rule_set_emit_selection_failure(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_selection_t selection) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(rule_set);
  IREE_ASSERT_ARGUMENT(source_op);
  IREE_ASSERT(selection.rule == NULL);
  if (!selection.has_source_op_span) {
    return loom_low_lower_rule_emit_no_mapping(context, source_op);
  }
  return loom_low_lower_rule_emit_diagnostic(context, rule_set, source_op,
                                             selection.diagnostic_index);
}

iree_status_t loom_low_lower_rule_set_select_op(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t** out_rule) {
  IREE_ASSERT_ARGUMENT(out_rule);
  *out_rule = NULL;
  loom_low_lower_rule_selection_t selection = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_set_select(context, rule_set, source_op, &selection));
  if (selection.rule == NULL) {
    return loom_low_lower_rule_set_emit_selection_failure(context, rule_set,
                                                          source_op, selection);
  }
  *out_rule = selection.rule;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_build_attrs(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_emit_t* emit, loom_named_attr_slice_t* out_attrs) {
  IREE_ASSERT_ARGUMENT(out_attrs);
  *out_attrs = loom_make_named_attr_slice(NULL, 0);
  if (emit->attr_copy_count == 0) {
    return iree_ok_status();
  }
  loom_named_attr_t* attrs = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, emit->attr_copy_count, sizeof(*attrs), (void**)&attrs));
  for (uint16_t i = 0; i < emit->attr_copy_count; ++i) {
    uint16_t attr_copy_index = (uint16_t)(emit->attr_copy_start + i);
    IREE_ASSERT_LT(attr_copy_index, rule_set->attr_copy_count);
    const loom_low_lower_attr_copy_t* attr_copy =
        &rule_set->attr_copies[attr_copy_index];
    IREE_ASSERT_LT(attr_copy->source_attr_index, source_op->attribute_count);
    IREE_RETURN_IF_ERROR(
        loom_module_intern_string(loom_low_lower_context_module(context),
                                  attr_copy->target_name, &attrs[i].name_id));
    attrs[i].value =
        loom_op_const_attrs(source_op)[attr_copy->source_attr_index];
  }
  *out_attrs = loom_make_named_attr_slice(attrs, emit->attr_copy_count);
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_map_result_type(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_type_t* out_type) {
  IREE_ASSERT_ARGUMENT(out_type);
  *out_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, source_value_id, out_type));
  IREE_ASSERT(loom_type_is_register(*out_type));
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_build_low_operands(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_emit_t* emit, loom_value_id_t** out_operands) {
  IREE_ASSERT_ARGUMENT(out_operands);
  *out_operands = NULL;
  if (emit->operand_ref_count == 0) {
    return iree_ok_status();
  }
  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, emit->operand_ref_count, sizeof(*low_operands),
      (void**)&low_operands));
  for (uint16_t i = 0; i < emit->operand_ref_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_low_value(
        context, rule_set, source_op, state,
        (uint16_t)(emit->operand_ref_start + i), &low_operands[i]));
  }
  *out_operands = low_operands;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_copy_low_operands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_emit_t* emit, loom_value_id_t* low_operands) {
  if (emit->copy_operand_mask == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_LE(emit->operand_ref_count, 16);
  const uint16_t valid_operand_mask =
      emit->operand_ref_count == 16
          ? UINT16_MAX
          : (uint16_t)(((uint16_t)1u << emit->operand_ref_count) - 1u);
  IREE_ASSERT_FALSE(
      iree_any_bit_set(emit->copy_operand_mask, (uint16_t)~valid_operand_mask));
  for (uint16_t i = 0; i < emit->operand_ref_count; ++i) {
    const uint16_t operand_bit = (uint16_t)((uint16_t)1u << i);
    if (!iree_any_bit_set(emit->copy_operand_mask, operand_bit)) {
      continue;
    }
    loom_type_t copy_type = loom_module_value_type(
        loom_low_lower_context_module(context), low_operands[i]);
    IREE_ASSERT(loom_type_is_register(copy_type));
    loom_op_t* copy_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_copy_build(
        loom_low_lower_context_builder(context), low_operands[i], copy_type,
        source_op->location, &copy_op));
    low_operands[i] = loom_low_copy_result(copy_op);
  }
  return iree_ok_status();
}

static void loom_low_lower_rule_apply_operand_flags(
    const loom_low_lower_emit_t* emit, loom_value_id_t* low_operands) {
  if (iree_any_bit_set(emit->flags,
                       LOOM_LOW_LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1)) {
    IREE_ASSERT_GE(emit->operand_ref_count, 2);
    const loom_value_id_t temporary = low_operands[0];
    low_operands[0] = low_operands[1];
    low_operands[1] = temporary;
  }
}

static iree_status_t loom_low_lower_rule_build_result_types(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_emit_t* emit, loom_type_t** out_result_types) {
  IREE_ASSERT_ARGUMENT(out_result_types);
  *out_result_types = NULL;
  if (emit->result_ref_count == 0) {
    return iree_ok_status();
  }
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, emit->result_ref_count, sizeof(*result_types),
      (void**)&result_types));
  for (uint16_t i = 0; i < emit->result_ref_count; ++i) {
    loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
        rule_set, source_op, (uint16_t)(emit->result_ref_start + i));
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_map_result_type(
        context, source_op, source_value_id, &result_types[i]));
  }
  *out_result_types = result_types;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_bind_results(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state, const loom_low_lower_emit_t* emit,
    const loom_value_id_t* low_results) {
  const uint16_t result_bind_ref_start =
      iree_any_bit_set(emit->flags,
                       LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS)
          ? emit->result_bind_ref_start
          : emit->result_ref_start;
  for (uint16_t i = 0; i < emit->result_ref_count; ++i) {
    const uint16_t value_ref_index = (uint16_t)(result_bind_ref_start + i);
    IREE_ASSERT_LT(value_ref_index, rule_set->value_ref_count);
    const loom_low_lower_value_ref_t* value_ref =
        &rule_set->value_refs[value_ref_index];
    switch (value_ref->kind) {
      case LOOM_LOW_LOWER_VALUE_REF_RESULT: {
        loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
            rule_set, source_op, value_ref_index);
        IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(context, source_value_id,
                                                       low_results[i]));
        break;
      }
      case LOOM_LOW_LOWER_VALUE_REF_TEMPORARY:
        IREE_ASSERT_LT(value_ref->index, state->temporary_count);
        IREE_ASSERT(state->temporaries != NULL);
        IREE_ASSERT(state->temporaries[value_ref->index] ==
                        LOOM_VALUE_ID_INVALID ||
                    state->temporaries[value_ref->index] == low_results[i]);
        state->temporaries[value_ref->index] = low_results[i];
        break;
      case LOOM_LOW_LOWER_VALUE_REF_OPERAND:
      default:
        IREE_ASSERT_UNREACHABLE();
        break;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_emit_descriptor_const(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_emit_t* emit) {
  IREE_ASSERT_EQ(emit->operand_ref_count, 0);
  IREE_ASSERT_EQ(emit->result_ref_count, 1);
  loom_value_id_t source_result = loom_low_lower_rule_source_value(
      rule_set, source_op, emit->result_ref_start);
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_map_result_type(
      context, source_op, source_result, &result_type));

  loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_attrs(
      context, rule_set, source_op, emit, &attrs));

  loom_op_t* low_const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_const(
      context, emit->descriptor_id, attrs, result_type, source_op->location,
      &low_const_op));
  const loom_value_id_t low_result = loom_low_const_result(low_const_op);
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, &low_result);
}

static iree_status_t loom_low_lower_rule_emit_descriptor_op(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_emit_t* emit) {
  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_low_operands(
      context, rule_set, source_op, state, emit, &low_operands));
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_copy_low_operands(
      context, source_op, emit, low_operands));
  loom_low_lower_rule_apply_operand_flags(emit, low_operands);

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
      context, rule_set, source_op, emit, &result_types));

  loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_attrs(
      context, rule_set, source_op, emit, &attrs));

  const loom_tied_result_t* tied_results = NULL;
  if (emit->tied_result_count != 0) {
    IREE_ASSERT_LE((uint32_t)emit->tied_result_start + emit->tied_result_count,
                   rule_set->tied_result_count);
    tied_results = &rule_set->tied_results[emit->tied_result_start];
  }

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_op(
      context, emit->descriptor_id, low_operands, emit->operand_ref_count,
      attrs, result_types, emit->result_ref_count, tied_results,
      emit->tied_result_count, source_op->location, &low_op));
  loom_value_slice_t low_results = loom_low_op_results(low_op);
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, low_results.values);
}

static iree_status_t loom_low_lower_rule_slice_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value_id, uint32_t lane_index, loom_type_t lane_type,
    loom_value_id_t* out_lane_value_id) {
  IREE_ASSERT_ARGUMENT(out_lane_value_id);
  *out_lane_value_id = LOOM_VALUE_ID_INVALID;
  const loom_type_t low_type = loom_module_value_type(
      loom_low_lower_context_module(context), low_value_id);
  IREE_ASSERT(loom_type_is_register(low_type));
  IREE_ASSERT_EQ(loom_type_register_class_id(low_type),
                 loom_type_register_class_id(lane_type));
  IREE_ASSERT_LT(lane_index, loom_type_register_unit_count(low_type));
  if (loom_type_register_unit_count(low_type) == 1) {
    *out_lane_value_id = low_value_id;
    return iree_ok_status();
  }
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(
      loom_low_lower_context_builder(context), low_value_id, lane_index,
      lane_type, source_op->location, &slice_op));
  *out_lane_value_id = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_emit_descriptor_op_per_lane(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_emit_state_t* state,
    const loom_low_lower_emit_t* emit) {
  IREE_ASSERT_GT(emit->operand_ref_count, 0);
  IREE_ASSERT_EQ(emit->result_ref_count, 1);
  IREE_ASSERT_EQ(emit->attr_copy_count, 0);
  IREE_ASSERT_EQ(emit->tied_result_count, 0);

  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_low_operands(
      context, rule_set, source_op, state, emit, &low_operands));
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_copy_low_operands(
      context, source_op, emit, low_operands));

  loom_value_id_t source_result = loom_low_lower_rule_source_value(
      rule_set, source_op, emit->result_ref_start);
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_map_result_type(
      context, source_op, source_result, &result_type));
  IREE_ASSERT(loom_type_is_register(result_type));
  const uint32_t lane_count = loom_type_register_unit_count(result_type);
  IREE_ASSERT_GT(lane_count, 0);
  const loom_type_t lane_type =
      loom_type_register(loom_type_register_class_id(result_type), 1);
  for (uint16_t i = 0; i < emit->operand_ref_count; ++i) {
    const loom_type_t operand_type = loom_module_value_type(
        loom_low_lower_context_module(context), low_operands[i]);
    IREE_ASSERT(loom_type_is_register(operand_type));
    IREE_ASSERT_EQ(loom_type_register_unit_count(operand_type), lane_count);
  }

  if (lane_count == 1) {
    loom_low_lower_rule_apply_operand_flags(emit, low_operands);
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_op(
        context, emit->descriptor_id, low_operands, emit->operand_ref_count,
        loom_make_named_attr_slice(NULL, 0), &result_type, 1, NULL, 0,
        source_op->location, &low_op));
    return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                            emit,
                                            loom_low_op_results(low_op).values);
  }

  loom_value_id_t* lane_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, emit->operand_ref_count, sizeof(*lane_operands),
      (void**)&lane_operands));
  loom_value_id_t* lane_results = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, lane_count, sizeof(*lane_results), (void**)&lane_results));
  for (uint32_t lane_index = 0; lane_index < lane_count; ++lane_index) {
    for (uint16_t operand_index = 0; operand_index < emit->operand_ref_count;
         ++operand_index) {
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_slice_lane(
          context, source_op, low_operands[operand_index], lane_index,
          lane_type, &lane_operands[operand_index]));
    }
    loom_low_lower_rule_apply_operand_flags(emit, lane_operands);
    loom_op_t* lane_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_op(
        context, emit->descriptor_id, lane_operands, emit->operand_ref_count,
        loom_make_named_attr_slice(NULL, 0), &lane_type, 1, NULL, 0,
        source_op->location, &lane_op));
    lane_results[lane_index] =
        loom_value_slice_get(loom_low_op_results(lane_op), 0);
  }

  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lane_results, lane_count,
      result_type, source_op->location, &concat_op));
  const loom_value_id_t low_result = loom_low_concat_result(concat_op);
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, state,
                                          emit, &low_result);
}

iree_status_t loom_low_lower_rule_set_emit_rule(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(rule_set);
  IREE_ASSERT_ARGUMENT(source_op);
  IREE_ASSERT_ARGUMENT(rule);

  loom_low_lower_rule_emit_state_t state = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_rule_emit_state_initialize(context, rule, &state));
  for (uint16_t i = 0; i < rule->emit_count; ++i) {
    uint16_t emit_index = (uint16_t)(rule->emit_start + i);
    IREE_ASSERT_LT(emit_index, rule_set->emit_count);
    const loom_low_lower_emit_t* emit = &rule_set->emits[emit_index];
    switch (emit->kind) {
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_descriptor_op(
            context, rule_set, source_op, &state, emit));
        break;
      }
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_CONST: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_descriptor_const(
            context, rule_set, source_op, &state, emit));
        break;
      }
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_descriptor_op_per_lane(
            context, rule_set, source_op, &state, emit));
        break;
      }
      default:
        IREE_ASSERT_UNREACHABLE();
        break;
    }
  }
  return iree_ok_status();
}
