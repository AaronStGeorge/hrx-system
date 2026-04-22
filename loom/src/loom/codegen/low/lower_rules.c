// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower_rules.h"

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

static iree_status_t loom_low_lower_rule_source_value(
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, loom_value_id_t* out_value_id) {
  IREE_ASSERT_ARGUMENT(out_value_id);
  *out_value_id = LOOM_VALUE_ID_INVALID;
  if (value_ref_index >= rule_set->value_ref_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "source-to-low lowering rule references a missing "
                            "value-ref row");
  }
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  switch (value_ref->kind) {
    case LOOM_LOW_LOWER_VALUE_REF_OPERAND:
      if (value_ref->index >= source_op->operand_count) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "source-to-low lowering rule references a missing source operand");
      }
      *out_value_id = loom_op_const_operands(source_op)[value_ref->index];
      return iree_ok_status();
    case LOOM_LOW_LOWER_VALUE_REF_RESULT:
      if (value_ref->index >= source_op->result_count) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "source-to-low lowering rule references a missing source result");
      }
      *out_value_id = loom_op_const_results(source_op)[value_ref->index];
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "source-to-low lowering rule has an invalid "
                              "value-ref kind");
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
      if (guard->type_pattern_index >= rule_set->type_pattern_count) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "source-to-low lowering rule references a missing type-pattern "
            "row");
      }
      loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_value(
          rule_set, source_op, guard->value_ref_index, &value_id));
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
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "source-to-low lowering rule has an invalid "
                              "guard kind");
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
    if (guard_index >= rule_set->guard_count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "source-to-low lowering rule references a "
                              "missing guard row");
    }
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

iree_status_t loom_low_lower_rule_set_select_op(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t** out_rule) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(rule_set);
  IREE_ASSERT_ARGUMENT(source_op);
  IREE_ASSERT_ARGUMENT(out_rule);
  *out_rule = NULL;

  const loom_low_lower_rule_span_t* span =
      loom_low_lower_rule_set_find_span(rule_set, source_op->kind);
  if (span == NULL) {
    return loom_low_lower_rule_emit_no_mapping(context, source_op);
  }

  uint16_t best_diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  uint16_t best_matched_guard_count = 0;
  for (uint16_t i = 0; i < span->rule_count; ++i) {
    uint16_t rule_index = (uint16_t)(span->rule_start + i);
    if (rule_index >= rule_set->rule_count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "source-to-low lowering span references a "
                              "missing rule row");
    }
    const loom_low_lower_rule_t* rule = &rule_set->rules[rule_index];
    bool rule_matches = false;
    uint16_t diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
    uint16_t matched_guard_count = 0;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_matches(
        context, rule_set, source_op, rule, &rule_matches, &diagnostic_index,
        &matched_guard_count));
    if (rule_matches) {
      *out_rule = rule;
      return iree_ok_status();
    }
    if (best_diagnostic_index == LOOM_LOW_LOWER_DIAGNOSTIC_NONE ||
        matched_guard_count > best_matched_guard_count) {
      best_diagnostic_index = diagnostic_index;
      best_matched_guard_count = matched_guard_count;
    }
  }

  return loom_low_lower_rule_emit_diagnostic(context, rule_set, source_op,
                                             best_diagnostic_index);
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
    if (attr_copy_index >= rule_set->attr_copy_count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "source-to-low lowering emit references a "
                              "missing attribute-copy row");
    }
    const loom_low_lower_attr_copy_t* attr_copy =
        &rule_set->attr_copies[attr_copy_index];
    if (attr_copy->source_attr_index >= source_op->attribute_count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "source-to-low lowering emit references a "
                              "missing source attribute");
    }
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
  if (!loom_type_is_register(*out_type)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "source-to-low lowering rule mapped a result to a "
                            "non-register type");
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_build_low_operands(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
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
    loom_value_id_t source_value_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_value(
        rule_set, source_op, (uint16_t)(emit->operand_ref_start + i),
        &source_value_id));
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, source_value_id,
                                                     &low_operands[i]));
  }
  *out_operands = low_operands;
  return iree_ok_status();
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
    loom_value_id_t source_value_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_value(
        rule_set, source_op, (uint16_t)(emit->result_ref_start + i),
        &source_value_id));
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_map_result_type(
        context, source_op, source_value_id, &result_types[i]));
  }
  *out_result_types = result_types;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_bind_results(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_emit_t* emit, const loom_value_id_t* low_results) {
  for (uint16_t i = 0; i < emit->result_ref_count; ++i) {
    loom_value_id_t source_value_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_value(
        rule_set, source_op, (uint16_t)(emit->result_ref_start + i),
        &source_value_id));
    IREE_RETURN_IF_ERROR(
        loom_low_lower_bind_value(context, source_value_id, low_results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_emit_descriptor_const(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_emit_t* emit) {
  if (emit->operand_ref_count != 0 || emit->result_ref_count != 1) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "source-to-low const emit must have no operands "
                            "and exactly one result");
  }
  loom_value_id_t source_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_source_value(
      rule_set, source_op, emit->result_ref_start, &source_result));
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
  return loom_low_lower_bind_value(context, source_result,
                                   loom_low_const_result(low_const_op));
}

static iree_status_t loom_low_lower_rule_emit_descriptor_op(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_emit_t* emit) {
  loom_value_id_t* low_operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_low_operands(
      context, rule_set, source_op, emit, &low_operands));

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_result_types(
      context, rule_set, source_op, emit, &result_types));

  loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_build_attrs(
      context, rule_set, source_op, emit, &attrs));

  const loom_tied_result_t* tied_results = NULL;
  if (emit->tied_result_count != 0) {
    if ((uint32_t)emit->tied_result_start + emit->tied_result_count >
        rule_set->tied_result_count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "source-to-low emit references missing tied "
                              "result rows");
    }
    tied_results = &rule_set->tied_results[emit->tied_result_start];
  }

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_op(
      context, emit->descriptor_id, low_operands, emit->operand_ref_count,
      attrs, result_types, emit->result_ref_count, tied_results,
      emit->tied_result_count, source_op->location, &low_op));
  loom_value_slice_t low_results = loom_low_op_results(low_op);
  return loom_low_lower_rule_bind_results(context, rule_set, source_op, emit,
                                          low_results.values);
}

iree_status_t loom_low_lower_rule_set_emit_rule(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(rule_set);
  IREE_ASSERT_ARGUMENT(source_op);
  IREE_ASSERT_ARGUMENT(rule);

  for (uint16_t i = 0; i < rule->emit_count; ++i) {
    uint16_t emit_index = (uint16_t)(rule->emit_start + i);
    if (emit_index >= rule_set->emit_count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "source-to-low lowering rule references a "
                              "missing emit row");
    }
    const loom_low_lower_emit_t* emit = &rule_set->emits[emit_index];
    switch (emit->kind) {
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_descriptor_op(
            context, rule_set, source_op, emit));
        break;
      }
      case LOOM_LOW_LOWER_EMIT_DESCRIPTOR_CONST: {
        IREE_RETURN_IF_ERROR(loom_low_lower_rule_emit_descriptor_const(
            context, rule_set, source_op, emit));
        break;
      }
      default:
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "source-to-low lowering rule has an invalid "
                                "emit kind");
    }
  }
  return iree_ok_status();
}
